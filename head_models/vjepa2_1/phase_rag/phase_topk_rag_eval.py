#!/usr/bin/env python3
"""Evaluate phase prediction with a top-k embedding RAG index."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None

from phase_topk_embedding_index import (
    DEFAULT_OUTPUT_DIR,
    PHASE_NAMES,
    VAL_CSV,
    PhaseSample,
    build_dataset,
    load_phase_csv,
    phase_encoder_config,
    select_topk_tokens,
)
from phase_topk_rag_inference import aggregate_scores, args_from_index_metadata
from single_tool_threshold_sweep import collate_samples, prepare_batched_sample_inputs
from triplet_conditioned_inference import init_encoder_module, maybe_suppress_stdout, resolve_device


DEFAULT_INDEX = DEFAULT_OUTPUT_DIR / "embeddings.pt"
DEFAULT_OUTPUT_JSON = DEFAULT_OUTPUT_DIR / "eval_val_remaining.json"
DEFAULT_OUTPUT_CSV = DEFAULT_OUTPUT_DIR / "eval_val_remaining_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate phase RAG prediction on a CSV of held-out clips. By default "
            "this uses the native11 validation CSV, skips missing local clips, and "
            "excludes clips already present in the index."
        )
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--eval-csv", type=Path, default=VAL_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--top-k-query", type=int, default=None)
    parser.add_argument("--retrieval-k", type=int, default=200)
    parser.add_argument("--checkpoint", "--encoder-checkpoint", dest="checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-key", "--encoder-checkpoint-key", dest="checkpoint_key", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=None)
    parser.add_argument("--frames-per-clip", type=int, default=None)
    parser.add_argument("--frame-step", type=int, default=None)
    parser.add_argument(
        "--aggregation",
        choices=("unique_clip_max", "all_matches"),
        default="unique_clip_max",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=None,
        help="Restrict evaluation to one or more video ids, e.g. yt_robotic_chole_Batch4.",
    )
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--skip-missing-clips", action="store_true", default=True)
    parser.add_argument("--include-indexed-clips", action="store_true")
    parser.add_argument("--store-neighbors", action="store_true")
    parser.add_argument("--no-store-predictions", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def load_eval_samples(
    eval_csv: Path,
    allowed_video_ids: set[str] | None,
    indexed_paths: set[str],
    include_indexed_clips: bool,
    skip_missing_clips: bool,
    max_clips: int | None,
) -> tuple[list[PhaseSample], dict[str, Any]]:
    samples = load_phase_csv(eval_csv, "eval")
    if allowed_video_ids is not None:
        samples = [sample for sample in samples if sample.video_id in allowed_video_ids]

    missing_samples = [sample for sample in samples if not sample.path.exists()]
    if missing_samples and not skip_missing_clips:
        preview = "\n".join(str(sample.path) for sample in missing_samples[:8])
        raise FileNotFoundError(f"{len(missing_samples)} eval clips do not exist. First paths:\n{preview}")
    if skip_missing_clips:
        samples = [sample for sample in samples if sample.path.exists()]

    indexed_eval_samples = [sample for sample in samples if str(sample.path) in indexed_paths]
    if not include_indexed_clips:
        samples = [sample for sample in samples if str(sample.path) not in indexed_paths]

    if max_clips is not None:
        if max_clips < 1:
            raise ValueError(f"--max-clips must be >= 1, got {max_clips}")
        samples = samples[:max_clips]

    if not samples:
        raise RuntimeError("No eval clips selected")

    metadata = {
        "eval_csv": str(eval_csv.resolve()),
        "requested_clip_count": len(load_phase_csv(eval_csv, "eval")),
        "allowed_video_ids": sorted(allowed_video_ids) if allowed_video_ids is not None else None,
        "skip_missing_clips": bool(skip_missing_clips),
        "missing_clip_count": len(missing_samples),
        "missing_videos": sorted({sample.video_id for sample in missing_samples}),
        "include_indexed_clips": bool(include_indexed_clips),
        "excluded_indexed_clip_count": 0 if include_indexed_clips else len(indexed_eval_samples),
        "selected_clip_count": len(samples),
        "selected_videos": sorted({sample.video_id for sample in samples}),
        "max_clips": max_clips,
    }
    return samples, metadata


def write_runtime_csv(samples: list[PhaseSample]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix="_phase_rag_eval.csv",
        delete=False,
    )
    with handle:
        for sample in samples:
            handle.write(f"{sample.path} {sample.label}\n")
    return Path(handle.name)


def compute_per_class(confusion: torch.Tensor) -> list[dict[str, Any]]:
    rows = []
    total = int(confusion.sum().item())
    for class_id, class_name in enumerate(PHASE_NAMES):
        tp = int(confusion[class_id, class_id].item())
        support = int(confusion[class_id, :].sum().item())
        predicted = int(confusion[:, class_id].sum().item())
        fp = predicted - tp
        fn = support - tp
        tn = total - tp - fp - fn
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "phase_id": int(class_id),
                "phase_name": class_name,
                "support": support,
                "predicted": predicted,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return rows


def write_predictions_csv(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_index",
        "video_id",
        "true_phase_id",
        "true_phase_name",
        "pred_phase_id",
        "pred_phase_name",
        "correct",
        "top1_score_share",
        "top2_phase_id",
        "top2_score_share",
        "top3_phase_id",
        "top3_score_share",
        "path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in predictions:
            top = row["ranked_phases"]
            writer.writerow(
                {
                    "clip_index": row["clip_index"],
                    "video_id": row["video_id"],
                    "true_phase_id": row["true_phase_id"],
                    "true_phase_name": row["true_phase_name"],
                    "pred_phase_id": row["pred_phase_id"],
                    "pred_phase_name": row["pred_phase_name"],
                    "correct": int(row["correct"]),
                    "top1_score_share": top[0]["score_share"] if len(top) > 0 else 0.0,
                    "top2_phase_id": top[1]["phase_id"] if len(top) > 1 else "",
                    "top2_score_share": top[1]["score_share"] if len(top) > 1 else "",
                    "top3_phase_id": top[2]["phase_id"] if len(top) > 2 else "",
                    "top3_score_share": top[2]["score_share"] if len(top) > 2 else "",
                    "path": row["path"],
                }
            )


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    if args.retrieval_k < 1:
        raise ValueError(f"--retrieval-k must be >= 1, got {args.retrieval_k}")

    index = torch.load(args.index, map_location="cpu", weights_only=False)
    embeddings = index["embeddings"].float()
    index_labels = index["labels"].long()
    index_paths = [str(path) for path in index["paths"]]
    metadata = index.get("metadata", {})
    if embeddings.ndim != 3:
        raise ValueError(f"Expected index embeddings [clips, top_k, dim], got {tuple(embeddings.shape)}")

    db_clip_count, db_top_k, dim = embeddings.shape
    db = F.normalize(embeddings.reshape(db_clip_count * db_top_k, dim), p=2, dim=-1)

    samples, selection_metadata = load_eval_samples(
        eval_csv=args.eval_csv,
        allowed_video_ids=set(args.video_id) if args.video_id else None,
        indexed_paths=set(index_paths),
        include_indexed_clips=args.include_indexed_clips,
        skip_missing_clips=args.skip_missing_clips,
        max_clips=args.max_clips,
    )
    sample_by_path = {str(sample.path): sample for sample in samples}
    runtime_csv = write_runtime_csv(samples)

    encoder_args = args_from_index_metadata(args, metadata)
    config = phase_encoder_config(encoder_args)
    device = resolve_device(args.device)
    try:
        with maybe_suppress_stdout(args.verbose):
            encoder = init_encoder_module(
                module_name=config["model_kwargs"]["module_name"],
                frames_per_clip=int(encoder_args.frames_per_clip),
                resolution=int(encoder_args.resolution),
                checkpoint=str(encoder_args.checkpoint),
                model_kwargs=config["model_kwargs"]["pretrain_kwargs"],
                wrapper_kwargs=config["model_kwargs"]["wrapper_kwargs"],
                device=device,
            )
        encoder.eval()

        dataset = build_dataset(config, runtime_csv)
        loader = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_samples,
        )

        confusion = torch.zeros((len(PHASE_NAMES), len(PHASE_NAMES)), dtype=torch.int64)
        predictions: list[dict[str, Any]] = []
        correct = 0
        total = 0

        iterable = loader
        progress_bar = None
        if not args.no_progress and tqdm is not None:
            progress_bar = tqdm(loader, total=len(loader), desc="phase rag eval", unit="batch")
            iterable = progress_bar

        with torch.no_grad():
            for batch in iterable:
                batch_buffers = [item[0] for item in batch]
                batch_labels = [int(item[1]) for item in batch]
                batch_clip_indices = [item[2] for item in batch]
                batch_paths = [str(item[3]) for item in batch]
                clips, prepared_indices = prepare_batched_sample_inputs(
                    batch_buffers,
                    batch_clip_indices,
                    device,
                )

                with torch.amp.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    encoded_clip = encoder(clips, prepared_indices)[0]

                query_embeddings, _ = select_topk_tokens(
                    encoded_clip=encoded_clip,
                    top_k=args.top_k_query or int(metadata.get("top_k", 8)),
                    normalize=True,
                )
                query_embeddings = F.normalize(query_embeddings.float().cpu(), p=2, dim=-1)
                scores = torch.matmul(query_embeddings, db.T)
                flat_scores = scores.reshape(scores.shape[0], -1)
                retrieval_k = min(int(args.retrieval_k), int(flat_scores.shape[1]))
                top_scores, top_indices = torch.topk(flat_scores, k=retrieval_k, dim=1, largest=True)

                for local_idx, (truth, path_text) in enumerate(zip(batch_labels, batch_paths)):
                    ranked_phases, neighbors = aggregate_scores(
                        similarities=top_scores[local_idx],
                        flat_indices=top_indices[local_idx],
                        labels=index_labels,
                        db_top_k=int(db_top_k),
                        db_vector_count=int(db.shape[0]),
                        aggregation=args.aggregation,
                    )
                    pred = int(ranked_phases[0]["phase_id"])
                    truth = int(truth)
                    confusion[truth, pred] += 1
                    is_correct = pred == truth
                    correct += int(is_correct)
                    total += 1

                    if not args.no_store_predictions:
                        sample = sample_by_path[path_text]
                        prediction_row = {
                            "clip_index": total - 1,
                            "video_id": sample.video_id,
                            "path": path_text,
                            "true_phase_id": truth,
                            "true_phase_name": PHASE_NAMES[truth],
                            "pred_phase_id": pred,
                            "pred_phase_name": PHASE_NAMES[pred],
                            "correct": bool(is_correct),
                            "ranked_phases": ranked_phases[:5],
                        }
                        if args.store_neighbors:
                            for neighbor in neighbors:
                                clip_idx = int(neighbor["clip_index"])
                                neighbor["path"] = index_paths[clip_idx]
                                neighbor["video_id"] = index["video_ids"][clip_idx]
                            prediction_row["neighbors"] = neighbors
                        predictions.append(prediction_row)

        if progress_bar is not None:
            progress_bar.close()
    finally:
        runtime_csv.unlink(missing_ok=True)

    accuracy = correct / total if total else 0.0
    per_class = compute_per_class(confusion)
    macro_f1 = sum(row["f1"] for row in per_class) / len(per_class)
    macro_recall = sum(row["recall"] for row in per_class) / len(per_class)

    payload = {
        "accuracy": accuracy,
        "correct": int(correct),
        "total": int(total),
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "per_class": per_class,
        "confusion_matrix_rows_true_cols_pred": confusion.tolist(),
        "selection": selection_metadata,
        "settings": {
            "index": str(args.index.resolve()),
            "retrieval_k": int(args.retrieval_k),
            "aggregation": args.aggregation,
            "top_k_query": args.top_k_query or int(metadata.get("top_k", 8)),
            "db_clip_count": int(db_clip_count),
            "db_top_k": int(db_top_k),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "device": str(device),
            "checkpoint": str(encoder_args.checkpoint),
            "checkpoint_key": encoder_args.checkpoint_key,
            "resolution": int(encoder_args.resolution),
            "frames_per_clip": int(encoder_args.frames_per_clip),
            "frame_step": int(encoder_args.frame_step),
        },
        "index_metadata": metadata,
        "predictions": predictions if not args.no_store_predictions else None,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    if args.output_csv is not None and not args.no_store_predictions:
        write_predictions_csv(args.output_csv, predictions)

    print(f"Evaluated {total} phase clips")
    print(f"Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    print(f"Macro F1: {macro_f1 * 100:.2f}%")
    print(f"Macro recall: {macro_recall * 100:.2f}%")
    print(f"Wrote JSON: {args.output_json}")
    if args.output_csv is not None and not args.no_store_predictions:
        print(f"Wrote CSV: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
