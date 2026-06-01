#!/usr/bin/env python3
"""Build a top-k V-JEPA2 token embedding index for SurgNet phase clips.

The defaults intentionally mirror the single-tool heads documented in
globus/surgenet_triplets/AGENTS_SINGLE_TOOL.md:

- surg-finetuned phase3 ViT-L checkpoint
- target_encoder checkpoint key
- 16 frames per clip
- 512px resolution
- one 4-second clip per CSV row

The output is meant for clip-level RAG: embed an incoming clip with the same
encoder, retrieve nearest stored token embeddings with cosine similarity, then
vote/aggregate over phase labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
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

from single_tool_threshold_sweep import collate_samples, prepare_batched_sample_inputs
from src.datasets.video_dataset import VideoDataset
from triplet_conditioned_inference import (
    build_transform,
    init_encoder_module,
    maybe_suppress_stdout,
    resolve_device,
)


PHASE_ROOT = Path("/path/to/phase_triplet_heads_bundle/globus/surgenet_phases")
TRAIN_CSV = PHASE_ROOT / "yt_robotic_chole_native11_phases_train.csv"
VAL_CSV = PHASE_ROOT / "yt_robotic_chole_native11_phases_val.csv"
OLD_PHASE_ROOT = (
    "/path/to/home/Desktop/PhdLeo/annotator_fellow/"
    "phase_recog/surgenet_phases"
)
CHECKPOINT = Path(
    "/path/to/phase_triplet_heads_bundle/vjepa2_1/chekpoints_vjepar_surg_finetuned/"
    "single4_all_datasets_phase3_higher_res_g8_latest.pth.tar"
)
DISTILLED_VITL_DIST_VITG_384_CHECKPOINT = Path(
    "/path/to/phase_triplet_heads_bundle/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt"
)
DEFAULT_OUTPUT_DIR = Path(
    "/path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag_embeddings/"
    "native11_phase3_topk8_train75"
)

VIDEO_ID_RE = re.compile(r"(yt_robotic_chole_Batch\d+)")

PHASE_NAMES = [
    "Cholecystectomy | Clipping of the cystic duct",
    "Cholecystectomy | Clipping the cystic artery",
    "Cholecystectomy | Dissection of the gallbladder from the liver",
    "Cholecystectomy | Division of the cystic duct and artery",
    "Cholecystectomy | Exposure of the working area",
    "Cholecystectomy | Isolation of the cystic artery",
    "Cholecystectomy | Isolation of the cystic duct",
    "Cholecystectomy | Opening the anterior peritoneal layer of the triangle of Calot",
    "Cholecystectomy | Opening the posterior peritoneal layer of the triangle of Calot",
    "Cholecystectomy | Retraction of the gallbladder neck",
    "Cholecystectomy | Specimen retrieval",
]


@dataclass(frozen=True)
class PhaseSample:
    path: Path
    label: int
    video_id: str
    source_split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export top-k normalized V-JEPA2 token embeddings for phase RAG."
    )
    parser.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    parser.add_argument("--val-csv", type=Path, default=VAL_CSV)
    parser.add_argument(
        "--source-splits",
        choices=("train", "val", "all"),
        default="train",
        help=(
            "Which split(s) to index. The default train split contains 7/9 "
            "SurgNet videos, i.e. the requested 75%% rounded to whole videos."
        ),
    )
    parser.add_argument(
        "--video-fraction",
        type=float,
        default=1.0,
        help="Fraction of selected split videos to use. Whole-video count is ceil(n * fraction).",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", "--encoder-checkpoint", dest="checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--checkpoint-key", "--encoder-checkpoint-key", dest="checkpoint_key", type=str, default=None)
    parser.add_argument(
        "--resolution",
        type=int,
        default=None,
        help=(
            "Encoder crop resolution. Defaults to 512 for the surgical phase3 checkpoint "
            "and 384 for vjepa2_1_vitl_dist_vitG_384.pt."
        ),
    )
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument(
        "--skip-missing-clips",
        action="store_true",
        help="Drop selected CSV rows whose remapped local video clip path does not exist.",
    )
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def is_distilled_vitl_dist_vitg_384_checkpoint(path: Path) -> bool:
    return Path(path).name == DISTILLED_VITL_DIST_VITG_384_CHECKPOINT.name


def resolve_encoder_settings(args: argparse.Namespace) -> argparse.Namespace:
    checkpoint = Path(args.checkpoint)
    checkpoint_key = args.checkpoint_key
    resolution = args.resolution
    if checkpoint_key is None:
        checkpoint_key = "ema_encoder" if is_distilled_vitl_dist_vitg_384_checkpoint(checkpoint) else "target_encoder"
    if resolution is None:
        resolution = 384 if is_distilled_vitl_dist_vitg_384_checkpoint(checkpoint) else 512
    return argparse.Namespace(
        checkpoint=checkpoint,
        checkpoint_key=str(checkpoint_key),
        resolution=int(resolution),
        frames_per_clip=int(args.frames_per_clip),
        frame_step=int(args.frame_step),
    )


def phase_encoder_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_kwargs": {
            "checkpoint": str(args.checkpoint),
            "module_name": "evals.triplet_recog_frozen.modelcustom.vit_encoder_multiclip",
            "pretrain_kwargs": {
                "encoder": {
                    "model_name": "vit_large",
                    "patch_size": 16,
                    "tubelet_size": 2,
                    "checkpoint_key": args.checkpoint_key,
                    "use_sdpa": True,
                    "use_rope": True,
                }
            },
            "wrapper_kwargs": {
                "max_frames": 128,
                "use_pos_embed": False,
            },
        },
        "experiment": {
            "data": {
                "frames_per_clip": int(args.frames_per_clip),
                "frame_step": int(args.frame_step),
                "resolution": int(args.resolution),
                "num_segments": 1,
                "num_views_per_segment": 1,
                "normalization": None,
            }
        },
    }


def remap_phase_path(raw_path: str) -> Path:
    if raw_path.startswith(OLD_PHASE_ROOT):
        return PHASE_ROOT / raw_path[len(OLD_PHASE_ROOT) :].lstrip("/")

    marker = "/surgenet_phases/"
    if marker in raw_path:
        return PHASE_ROOT / raw_path.split(marker, 1)[1]

    return Path(raw_path)


def video_id_from_path(path: Path) -> str:
    match = VIDEO_ID_RE.search(str(path))
    if not match:
        raise ValueError(f"Could not parse video id from path: {path}")
    return match.group(1)


def load_phase_csv(path: Path, source_split: str) -> list[PhaseSample]:
    samples: list[PhaseSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw_path, label_text = line.rsplit(" ", 1)
            except ValueError as exc:
                raise ValueError(f"Malformed row {line_number} in {path}: {line!r}") from exc
            clip_path = remap_phase_path(raw_path)
            label = int(label_text)
            samples.append(
                PhaseSample(
                    path=clip_path,
                    label=label,
                    video_id=video_id_from_path(clip_path),
                    source_split=source_split,
                )
            )
    return samples


def select_samples(args: argparse.Namespace) -> tuple[list[PhaseSample], dict[str, Any]]:
    if not (0.0 < args.video_fraction <= 1.0):
        raise ValueError(f"--video-fraction must be in (0, 1], got {args.video_fraction}")

    split_samples: list[PhaseSample] = []
    if args.source_splits in ("train", "all"):
        split_samples.extend(load_phase_csv(args.train_csv, "train"))
    if args.source_splits in ("val", "all"):
        split_samples.extend(load_phase_csv(args.val_csv, "val"))

    all_videos = sorted({sample.video_id for sample in split_samples})
    selected_video_count = max(1, int(math.ceil(len(all_videos) * args.video_fraction)))
    selected_videos = set(all_videos[:selected_video_count])
    selected = [sample for sample in split_samples if sample.video_id in selected_videos]
    if args.max_clips is not None:
        if args.max_clips < 1:
            raise ValueError(f"--max-clips must be >= 1, got {args.max_clips}")
        selected = selected[: args.max_clips]

    missing_samples = [sample for sample in selected if not sample.path.exists()]
    missing = [str(sample.path) for sample in missing_samples]
    if missing:
        if not args.skip_missing_clips:
            preview = "\n".join(missing[:8])
            raise FileNotFoundError(f"{len(missing)} selected clips do not exist. First paths:\n{preview}")
        selected = [sample for sample in selected if sample.path.exists()]
        if not selected:
            raise RuntimeError("All selected clips were missing after path remapping")

    final_selected_videos = sorted({sample.video_id for sample in selected})

    metadata = {
        "source_splits": args.source_splits,
        "video_fraction": float(args.video_fraction),
        "available_videos": all_videos,
        "available_video_count": len(all_videos),
        "requested_selected_videos": sorted(selected_videos),
        "requested_selected_video_count": len(selected_videos),
        "selected_videos": final_selected_videos,
        "selected_video_count": len(final_selected_videos),
        "selected_clip_count": len(selected),
        "max_clips": args.max_clips,
        "skip_missing_clips": bool(args.skip_missing_clips),
        "missing_clip_count": len(missing_samples),
        "missing_videos": sorted({sample.video_id for sample in missing_samples}),
    }
    return selected, metadata


def write_runtime_csv(path: Path, samples: list[PhaseSample]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        for sample in samples:
            handle.write(f"{sample.path} {sample.label}\n")


def write_clip_manifest(path: Path, samples: list[PhaseSample]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["clip_index", "video_id", "source_split", "phase_id", "phase_name", "path"],
        )
        writer.writeheader()
        for idx, sample in enumerate(samples):
            writer.writerow(
                {
                    "clip_index": idx,
                    "video_id": sample.video_id,
                    "source_split": sample.source_split,
                    "phase_id": sample.label,
                    "phase_name": PHASE_NAMES[sample.label],
                    "path": str(sample.path),
                }
            )


def build_dataset(config: dict[str, Any], runtime_csv: Path) -> VideoDataset:
    data_cfg = config["experiment"]["data"]
    return VideoDataset(
        data_paths=[str(runtime_csv)],
        frames_per_clip=int(data_cfg.get("frames_per_clip", 16)),
        frame_step=int(data_cfg.get("frame_step", 1)),
        num_clips=int(data_cfg.get("num_segments", 1)),
        transform=build_transform(config),
        random_clip_sampling=False,
        allow_clip_overlap=True,
        return_sample_path=True,
    )


def select_topk_tokens(encoded_clip: torch.Tensor, top_k: int, normalize: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if encoded_clip.ndim != 3:
        raise ValueError(f"Expected encoded clip shape [B, tokens, dim], got {tuple(encoded_clip.shape)}")
    encoded_clip = encoded_clip.float()
    scores = torch.linalg.vector_norm(encoded_clip, dim=-1)
    k = min(int(top_k), int(encoded_clip.shape[1]))
    token_indices = torch.topk(scores, k=k, dim=1, largest=True).indices
    gathered = torch.gather(
        encoded_clip,
        dim=1,
        index=token_indices.unsqueeze(-1).expand(-1, -1, encoded_clip.shape[-1]),
    )
    if normalize:
        gathered = F.normalize(gathered, p=2, dim=-1)
    return gathered, token_indices.to(torch.int32)


def main() -> int:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError(f"--top-k must be >= 1, got {args.top_k}")
    if args.batch_size < 1:
        raise ValueError(f"--batch-size must be >= 1, got {args.batch_size}")
    encoder_args = resolve_encoder_settings(args)
    if not encoder_args.checkpoint.exists():
        raise FileNotFoundError(encoder_args.checkpoint)

    output_dir = args.output_dir
    embeddings_path = output_dir / "embeddings.pt"
    metadata_path = output_dir / "metadata.json"
    clips_path = output_dir / "clips.csv"
    runtime_csv = output_dir / "selected_runtime.csv"
    if output_dir.exists() and embeddings_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {embeddings_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, selection_metadata = select_samples(args)
    if not samples:
        raise RuntimeError("No phase samples selected")
    write_runtime_csv(runtime_csv, samples)
    write_clip_manifest(clips_path, samples)

    config = phase_encoder_config(encoder_args)
    device = resolve_device(args.device)
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

    embedding_chunks: list[torch.Tensor] = []
    token_index_chunks: list[torch.Tensor] = []
    labels: list[int] = []
    paths: list[str] = []
    video_ids: list[str] = []
    source_splits: list[str] = []

    iterable = loader
    progress_bar = None
    if not args.no_progress and tqdm is not None:
        progress_bar = tqdm(loader, total=len(loader), desc="phase top-k embeddings", unit="batch")
        iterable = progress_bar

    sample_by_path = {str(sample.path): sample for sample in samples}
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

            topk_embeddings, token_indices = select_topk_tokens(
                encoded_clip=encoded_clip,
                top_k=int(args.top_k),
                normalize=not args.no_normalize,
            )
            if args.dtype == "fp16":
                topk_embeddings = topk_embeddings.to(torch.float16)
            else:
                topk_embeddings = topk_embeddings.to(torch.float32)

            embedding_chunks.append(topk_embeddings.cpu())
            token_index_chunks.append(token_indices.cpu())
            labels.extend(batch_labels)
            paths.extend(batch_paths)
            for batch_path in batch_paths:
                sample = sample_by_path[batch_path]
                video_ids.append(sample.video_id)
                source_splits.append(sample.source_split)

    if progress_bar is not None:
        progress_bar.close()

    embeddings = torch.cat(embedding_chunks, dim=0)
    token_indices = torch.cat(token_index_chunks, dim=0)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    payload = {
        "embeddings": embeddings,
        "labels": label_tensor,
        "token_indices": token_indices,
        "paths": paths,
        "video_ids": video_ids,
        "source_splits": source_splits,
        "phase_names": PHASE_NAMES,
        "metadata": {
            **selection_metadata,
            "embedding_shape": list(embeddings.shape),
            "embedding_dtype": str(embeddings.dtype).replace("torch.", ""),
            "embeddings_l2_normalized": not args.no_normalize,
            "token_selection": "largest_l2_norm_before_embedding_normalization",
            "top_k": int(args.top_k),
            "checkpoint": str(encoder_args.checkpoint.resolve()),
            "checkpoint_key": encoder_args.checkpoint_key,
            "module_name": config["model_kwargs"]["module_name"],
            "model_name": config["model_kwargs"]["pretrain_kwargs"]["encoder"]["model_name"],
            "resolution": int(encoder_args.resolution),
            "frames_per_clip": int(encoder_args.frames_per_clip),
            "frame_step": int(encoder_args.frame_step),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "device": str(device),
            "output_dir": str(output_dir.resolve()),
            "runtime_csv": str(runtime_csv.resolve()),
            "clips_csv": str(clips_path.resolve()),
        },
    }
    torch.save(payload, embeddings_path)

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(payload["metadata"], handle, indent=2)
        handle.write("\n")

    storage_mib = embeddings.numel() * embeddings.element_size() / (1024 * 1024)
    print(f"Wrote embeddings: {embeddings_path}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Wrote clips manifest: {clips_path}")
    print(
        "Shape: "
        f"{tuple(embeddings.shape)} {embeddings.dtype}, "
        f"approx tensor storage {storage_mib:.2f} MiB"
    )
    print(
        "Selected videos: "
        f"{selection_metadata['selected_video_count']}/"
        f"{selection_metadata['available_video_count']} "
        f"({', '.join(selection_metadata['selected_videos'])})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
