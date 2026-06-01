#!/usr/bin/env python3
"""Infer a phase label for one clip with the phase top-k embedding index."""

from __future__ import annotations

import argparse
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

from phase_topk_embedding_index import (
    CHECKPOINT,
    DEFAULT_OUTPUT_DIR,
    PHASE_NAMES,
    build_dataset,
    phase_encoder_config,
    resolve_encoder_settings,
    select_topk_tokens,
)
from single_tool_threshold_sweep import prepare_batched_sample_inputs
from triplet_conditioned_inference import init_encoder_module, maybe_suppress_stdout, resolve_device


DEFAULT_INDEX = DEFAULT_OUTPUT_DIR / "embeddings.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed one clip and infer its phase by cosine RAG over a top-k embedding index."
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
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
        help="How retrieved token matches vote into phase scores.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def args_from_index_metadata(args: argparse.Namespace, metadata: dict[str, Any]) -> argparse.Namespace:
    return resolve_encoder_settings(argparse.Namespace(
        checkpoint=args.checkpoint or Path(metadata.get("checkpoint", CHECKPOINT)),
        checkpoint_key=args.checkpoint_key or metadata.get("checkpoint_key"),
        resolution=args.resolution or metadata.get("resolution"),
        frames_per_clip=args.frames_per_clip or int(metadata.get("frames_per_clip", 16)),
        frame_step=args.frame_step or int(metadata.get("frame_step", 1)),
    ))


def write_query_runtime_csv(input_video: Path) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix="_phase_rag_query.csv",
        delete=False,
    )
    with handle:
        handle.write(f"{input_video} 0\n")
    return Path(handle.name)


def embed_query_clip(args: argparse.Namespace, metadata: dict[str, Any]) -> torch.Tensor:
    encoder_args = args_from_index_metadata(args, metadata)
    if not encoder_args.checkpoint.exists():
        raise FileNotFoundError(encoder_args.checkpoint)
    if not args.input_video.exists():
        raise FileNotFoundError(args.input_video)

    config = phase_encoder_config(encoder_args)
    runtime_csv = write_query_runtime_csv(args.input_video.resolve())
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
        buffer, _, clip_indices, _ = dataset[0]
        clips, prepared_indices = prepare_batched_sample_inputs([buffer], [clip_indices], device)
        with torch.no_grad():
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                encoded_clip = encoder(clips, prepared_indices)[0]
            top_k = args.top_k_query or int(metadata.get("top_k", 8))
            query_embeddings, _ = select_topk_tokens(
                encoded_clip=encoded_clip,
                top_k=top_k,
                normalize=True,
            )
        return query_embeddings[0].float().cpu()
    finally:
        runtime_csv.unlink(missing_ok=True)


def aggregate_scores(
    similarities: torch.Tensor,
    flat_indices: torch.Tensor,
    labels: torch.Tensor,
    db_top_k: int,
    db_vector_count: int,
    aggregation: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    phase_scores = torch.zeros(len(PHASE_NAMES), dtype=torch.float64)
    neighbors: list[dict[str, Any]] = []
    best_by_clip: dict[int, tuple[float, int, int]] = {}

    for rank, (score, flat_idx) in enumerate(zip(similarities.tolist(), flat_indices.tolist()), start=1):
        query_token_rank = int(flat_idx // db_vector_count)
        db_vector_idx = int(flat_idx % db_vector_count)
        clip_idx = int(db_vector_idx // db_top_k)
        db_token_rank = int(db_vector_idx % db_top_k)
        phase_id = int(labels[clip_idx].item())
        score = float(score)
        neighbors.append(
            {
                "rank": rank,
                "score": score,
                "query_token_rank": query_token_rank,
                "clip_index": clip_idx,
                "db_token_rank": db_token_rank,
                "phase_id": phase_id,
                "phase_name": PHASE_NAMES[phase_id],
            }
        )
        if aggregation == "all_matches":
            phase_scores[phase_id] += score
            continue
        previous = best_by_clip.get(clip_idx)
        if previous is None or score > previous[0]:
            best_by_clip[clip_idx] = (score, phase_id, db_token_rank)

    if aggregation == "unique_clip_max":
        for score, phase_id, _ in best_by_clip.values():
            phase_scores[phase_id] += score

    ranked_phases = []
    total = float(phase_scores.sum().item())
    for phase_id in torch.argsort(phase_scores, descending=True).tolist():
        score = float(phase_scores[phase_id].item())
        ranked_phases.append(
            {
                "phase_id": int(phase_id),
                "phase_name": PHASE_NAMES[phase_id],
                "score": score,
                "score_share": (score / total) if total > 0 else 0.0,
            }
        )
    return ranked_phases, neighbors


def main() -> int:
    args = parse_args()
    if args.retrieval_k < 1:
        raise ValueError(f"--retrieval-k must be >= 1, got {args.retrieval_k}")

    index = torch.load(args.index, map_location="cpu", weights_only=False)
    embeddings = index["embeddings"].float()
    labels = index["labels"].long()
    paths = index["paths"]
    video_ids = index["video_ids"]
    metadata = index.get("metadata", {})

    if embeddings.ndim != 3:
        raise ValueError(f"Expected index embeddings [clips, top_k, dim], got {tuple(embeddings.shape)}")
    db_clip_count, db_top_k, dim = embeddings.shape
    db = F.normalize(embeddings.reshape(db_clip_count * db_top_k, dim), p=2, dim=-1)
    query = embed_query_clip(args, metadata)
    if query.shape[-1] != dim:
        raise ValueError(f"Query dim {query.shape[-1]} does not match index dim {dim}")
    query = F.normalize(query, p=2, dim=-1)

    scores = query @ db.T
    flat_scores = scores.reshape(-1)
    retrieval_k = min(int(args.retrieval_k), int(flat_scores.numel()))
    top_scores, top_indices = torch.topk(flat_scores, k=retrieval_k, largest=True)
    ranked_phases, neighbors = aggregate_scores(
        similarities=top_scores,
        flat_indices=top_indices,
        labels=labels,
        db_top_k=int(db_top_k),
        db_vector_count=int(db.shape[0]),
        aggregation=args.aggregation,
    )

    for neighbor in neighbors:
        clip_idx = int(neighbor["clip_index"])
        neighbor["path"] = paths[clip_idx]
        neighbor["video_id"] = video_ids[clip_idx]

    payload = {
        "input_video": str(args.input_video.resolve()),
        "index": str(args.index.resolve()),
        "prediction": ranked_phases[0],
        "ranked_phases": ranked_phases,
        "neighbors": neighbors,
        "settings": {
            "top_k_query": int(query.shape[0]),
            "db_clip_count": int(db_clip_count),
            "db_top_k": int(db_top_k),
            "retrieval_k": int(retrieval_k),
            "aggregation": args.aggregation,
        },
    }

    print(
        "Prediction: "
        f"{payload['prediction']['phase_id']} "
        f"{payload['prediction']['phase_name']} "
        f"(score_share={payload['prediction']['score_share']:.3f})"
    )
    print("Top phases:")
    for row in ranked_phases[:5]:
        print(
            f"  {row['phase_id']:>2}  "
            f"{row['score_share'] * 100:>6.2f}%  "
            f"{row['phase_name']}"
        )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"Wrote JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
