"""Build one-vs-rest CSVs and configs for surgical phase probes.

This script creates one binary train/val CSV pair per target phase and writes
matching `configs/heads_ovr/*.yaml` configs that can be trained with the
existing `evals.main` entrypoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from head_phases_ovr.label_spaces import get_class_names, remap_label, slugify_phase_name


DEFAULT_OUTPUT_ROOT = Path("/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/ovr_phase_heads")
DEFAULT_CONFIG_ROOT = Path("/path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads_ovr")
DEFAULT_CHECKPOINT = Path("/path/to/phase_triplet_heads_bundle/vjepa2_1/latest.pt")
DEFAULT_TRAIN_FOLDER = Path("/path/to/phase_triplet_heads_bundle/vjepa2_sophia/trained_heads/sitl_phase_video_segments_crcd_vjepa2_1_ovr")
DEFAULT_AUDIT_SCRIPT_ROOT = Path("/path/to/phase_triplet_heads_bundle/vjepa2_1/head_phases_ovr/audits")
DEFAULT_AUDIT_OUTPUT_ROOT = Path("/path/to/phase_triplet_heads_bundle/vjepa2_1/app/phase_clip_audit_ovr")


@dataclass(frozen=True)
class SourceSplit:
    source_name: str
    split_name: str
    csv_path: Path


DEFAULT_SOURCE_SPLITS = {
    "sitl": {
        "train": Path("/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/sitl_phases_train.csv"),
        "val": Path("/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/sitl_phases_val.csv"),
    },
    "surgenet": {
        "train": Path("/path/to/phase_triplet_heads_bundle/globus/surgenet_phases/yt_robotic_chole_native11_phases_train.csv"),
        "val": Path("/path/to/phase_triplet_heads_bundle/globus/surgenet_phases/yt_robotic_chole_native11_phases_val.csv"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-space",
        choices=("reduced6", "native11"),
        default="reduced6",
        help="Target label space for the one-vs-rest heads.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=tuple(DEFAULT_SOURCE_SPLITS.keys()),
        default=["sitl", "surgenet"],
        help="Which datasets to include when building the one-vs-rest heads.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-root", type=Path, default=DEFAULT_CONFIG_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-folder", type=Path, default=DEFAULT_TRAIN_FOLDER)
    parser.add_argument("--audit-script-root", type=Path, default=DEFAULT_AUDIT_SCRIPT_ROOT)
    parser.add_argument("--audit-output-root", type=Path, default=DEFAULT_AUDIT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=12)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument(
        "--backbone-trainable-last-n-blocks",
        type=int,
        default=2,
        help="How many ViT blocks to finetune for each binary head config.",
    )
    parser.add_argument(
        "--positive-weight-scale",
        type=float,
        default=2.0,
        help="Extra multiplicative boost applied to the positive-class loss weight after inverse-frequency balancing.",
    )
    return parser.parse_args()


def iter_rows(csv_path: Path):
    with csv_path.open() as handle:
        for row in csv.reader(handle, delimiter=" "):
            if not row:
                continue
            yield Path(row[0]), int(row[1])


def load_multiclass_rows(source_splits: list[SourceSplit], label_space: str):
    rows_by_split: dict[str, list[tuple[Path, int, str]]] = {"train": [], "val": []}
    skipped = Counter()
    for source_split in source_splits:
        for sample_path, raw_label in iter_rows(source_split.csv_path):
            mapped_label = remap_label(raw_label, label_space=label_space)
            if mapped_label is None:
                skipped[f"{source_split.source_name}:{source_split.split_name}"] += 1
                continue
            rows_by_split[source_split.split_name].append((sample_path, mapped_label, source_split.source_name))
    return rows_by_split, skipped


def compute_binary_weights(pos_count: int, neg_count: int, *, positive_weight_scale: float) -> list[float]:
    if pos_count <= 0 or neg_count <= 0:
        raise ValueError(f"Binary split must contain both classes, got pos={pos_count}, neg={neg_count}")
    raw = {0: 1.0 / neg_count, 1: positive_weight_scale / pos_count}
    mean = (raw[0] + raw[1]) / 2.0
    return [round(raw[0] / mean, 6), round(raw[1] / mean, 6)]


def make_config_text(
    *,
    class_id: int,
    class_name: str,
    train_csv: Path,
    val_csv: Path,
    checkpoint: Path,
    train_folder: Path,
    batch_size: int,
    num_epochs: int,
    resolution: int,
    frames_per_clip: int,
    frame_step: int,
    backbone_trainable_last_n_blocks: int,
    loss_class_weights: list[float],
    target_positive_prior: float,
    label_space: str,
) -> str:
    class_slug = slugify_phase_name(class_name)
    return f"""eval_name: 'video_classification_frozen_ovr'

folder: '{train_folder}'
tag: 'vitl-{resolution}-phase-ovr-{label_space}-class-{class_id:02d}-{class_slug}'
resume_checkpoint: False
val_only: False
num_workers: 0

wandb:
  enabled: False
  project: 'vjepa2_1_phase_ovr'
  entity: null
  name: null
  group: 'phase_ovr_{label_space}'
  tags: ['ovr', '{label_space}', 'class_{class_id:02d}']
  mode: 'online'
  dir: '/path/to/phase_triplet_heads_bundle/vjepa2_1/wandb'
  log_interval: 10

model_kwargs:
  checkpoint: '{checkpoint}'
  module_name: 'evals.video_classification_frozen.modelcustom.vit_encoder_multiclip'
  pretrain_kwargs:
    encoder:
      model_name: 'vit_large'
      patch_size: 16
      tubelet_size: 2
      checkpoint_key: 'target_encoder'
      use_sdpa: True
      use_rope: True
  wrapper_kwargs:
    max_frames: 128
    use_pos_embed: False

experiment:
  classifier:
    num_probe_blocks: 4
    num_heads: 16
    use_activation_checkpointing: False

  data:
    dataset_type: 'VideoDataset'
    dataset_train: '{train_csv}'
    dataset_val: '{val_csv}'
    train_cache_root: null
    val_cache_root: null
    cache_num_workers: 1
    cache_require_complete: True
    num_classes: 2
    frames_per_clip: {frames_per_clip}
    frame_step: {frame_step}
    resolution: {resolution}
    num_segments: 1
    num_views_per_segment: 1
    num_workers: 8
    pin_memory: True
    normalization: null

  optimization:
    batch_size: {batch_size}
    num_epochs: {num_epochs}
    use_bfloat16: True
    confusion_matrix_log_interval: 100
    loss_class_weights: [{loss_class_weights[0]}, {loss_class_weights[1]}]
    collapse_penalty_weight: 1.0
    collapse_penalty_mode: 'kl'
    target_positive_prior: {target_positive_prior}
    hard_collapse_penalty_weight: 5.0
    hard_collapse_penalty_mode: 'kl'
    hard_collapse_target_prior: {target_positive_prior}
    hard_collapse_threshold: 0.5
    hard_collapse_temperature: 0.01
    backbone_trainable_last_n_blocks: {backbone_trainable_last_n_blocks}
    backbone_trainable_include_norm: True
    backbone_kwargs:
      warmup: 0.05
      start_lr: 0.000005
      lr: 0.00005
      final_lr: 0.000001
      weight_decay: 0.05
      final_weight_decay: 0.4
    profile_timing: False
    profile_log_interval: 5
    profile_warmup_iters: 1
    profile_cuda_sync: True
    multihead_kwargs:
      - warmup: 0.05
        start_lr: 0.00005
        lr: 0.0005
        final_lr: 0.000005
        weight_decay: 0.05
        final_weight_decay: 0.4

nodes: 1
tasks_per_node: 1
mem_per_gpu: '40G'
cpus_per_task: 16
"""


def make_audit_wrapper_text(
    *,
    repo_root: Path,
    train_csv: Path,
    val_csv: Path,
    config_path: Path,
    output_dir: Path,
    class_id: int,
    class_name: str,
    label_space: str,
) -> str:
    return f"""#!python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(r\"{repo_root}\")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from head_phases_ovr.analyze_ovr_phase_clip_mosaics import main as audit_main


if __name__ == \"__main__\":
    raise SystemExit(
        audit_main(
            [
                \"--train-csv\",
                r\"{train_csv}\",
                \"--val-csv\",
                r\"{val_csv}\",
                \"--target-class-id\",
                \"{class_id}\",
                \"--target-class-name\",
                \"{class_name}\",
                \"--label-space\",
                \"{label_space}\",
                \"--config\",
                r\"{config_path}\",
                \"--output-dir\",
                r\"{output_dir}\",
                *sys.argv[1:],
            ]
        )
    )
"""


def main() -> None:
    args = parse_args()

    class_names = get_class_names(args.label_space)
    source_splits = [
        SourceSplit(source_name=source_name, split_name=split_name, csv_path=DEFAULT_SOURCE_SPLITS[source_name][split_name])
        for source_name in args.sources
        for split_name in ("train", "val")
    ]

    rows_by_split, skipped = load_multiclass_rows(source_splits, label_space=args.label_space)

    dataset_output_dir = args.output_root / args.label_space
    config_output_dir = args.config_root / args.label_space
    audit_script_dir = args.audit_script_root / args.label_space
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    config_output_dir.mkdir(parents=True, exist_ok=True)
    audit_script_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "label_space": args.label_space,
        "sources": args.sources,
        "class_names": {str(class_id): class_name for class_id, class_name in sorted(class_names.items())},
        "checkpoint": str(args.checkpoint),
        "train_folder": str(args.train_folder),
        "positive_weight_scale": args.positive_weight_scale,
        "skipped_source_rows": dict(skipped),
        "per_class": {},
    }

    for class_id, class_name in sorted(class_names.items()):
        class_slug = f"{class_id:02d}_{slugify_phase_name(class_name)}"
        train_csv = dataset_output_dir / f"{class_slug}_train.csv"
        val_csv = dataset_output_dir / f"{class_slug}_val.csv"
        config_path = config_output_dir / f"phase_ovr_{args.label_space}_{class_slug}.yaml"
        audit_output_dir = args.audit_output_root / args.label_space / class_slug
        audit_script_path = audit_script_dir / f"analyze_phase_ovr_{args.label_space}_{class_slug}.py"

        binary_rows = {}
        counts_by_split = {}
        for split_name, rows in rows_by_split.items():
            split_binary_rows = [(sample_path, 1 if mapped_label == class_id else 0) for sample_path, mapped_label, _source_name in rows]
            binary_rows[split_name] = split_binary_rows
            counts_by_split[split_name] = {
                "positive": sum(label for _path, label in split_binary_rows),
                "negative": sum(1 for _path, label in split_binary_rows if label == 0),
            }

        with train_csv.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter=" ", lineterminator="\n")
            writer.writerows((str(path), label) for path, label in binary_rows["train"])
        with val_csv.open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter=" ", lineterminator="\n")
            writer.writerows((str(path), label) for path, label in binary_rows["val"])

        train_weights = compute_binary_weights(
            pos_count=counts_by_split["train"]["positive"],
            neg_count=counts_by_split["train"]["negative"],
            positive_weight_scale=args.positive_weight_scale,
        )
        train_total_count = counts_by_split["train"]["positive"] + counts_by_split["train"]["negative"]
        target_positive_prior = round(counts_by_split["train"]["positive"] / train_total_count, 6)
        config_text = make_config_text(
            class_id=class_id,
            class_name=class_name,
            train_csv=train_csv,
            val_csv=val_csv,
            checkpoint=args.checkpoint,
            train_folder=args.train_folder,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            resolution=args.resolution,
            frames_per_clip=args.frames_per_clip,
            frame_step=args.frame_step,
            backbone_trainable_last_n_blocks=args.backbone_trainable_last_n_blocks,
            loss_class_weights=train_weights,
            target_positive_prior=target_positive_prior,
            label_space=args.label_space,
        )
        config_path.write_text(config_text)
        audit_script_path.write_text(
            make_audit_wrapper_text(
                repo_root=REPO_ROOT,
                train_csv=train_csv,
                val_csv=val_csv,
                config_path=config_path,
                output_dir=audit_output_dir,
                class_id=class_id,
                class_name=class_name,
                label_space=args.label_space,
            )
        )
        os.chmod(audit_script_path, 0o755)

        metadata["per_class"][str(class_id)] = {
            "class_name": class_name,
            "train_csv": str(train_csv),
            "val_csv": str(val_csv),
            "config_path": str(config_path),
            "audit_script_path": str(audit_script_path),
            "audit_output_dir": str(audit_output_dir),
            "train_counts": counts_by_split["train"],
            "val_counts": counts_by_split["val"],
            "loss_class_weights": train_weights,
            "target_positive_prior": target_positive_prior,
        }

    metadata_path = dataset_output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Wrote one-vs-rest datasets to {dataset_output_dir}")
    print(f"Wrote one-vs-rest configs to {config_output_dir}")
    print(f"Summary: {metadata_path}")


if __name__ == "__main__":
    main()
