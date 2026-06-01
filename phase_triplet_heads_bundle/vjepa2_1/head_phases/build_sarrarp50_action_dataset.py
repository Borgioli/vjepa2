#!/usr/bin/env python3
"""
Build V-JEPA probe CSVs and short RGB clips for SAR_RARP50 action recognition.

SAR_RARP50 stores per-frame action labels in each video directory as:

    action_discrete.txt

with comma-separated rows of:

    <rgb_frame_id>,<action_id>

The frozen/partially-finetuned phase probe in this repo expects one clip path
and one integer label per row. This script converts the SAR frame labels into
centered 16-frame clips, then writes train/val/test CSVs compatible with the
existing VideoDataset loader.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2


DEFAULT_DATA_ROOT = Path("/path/to/surg_vid/sarrarp50/globus/SAR_RARP50")
DEFAULT_CSV_DIR = Path("/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models")

VARIANT_DEFAULTS = {
    "10hz": {
        "output_root": Path("/path/to/surg_vid/sarrarp50/globus/sarrarp50_action_clips_16f_10hz"),
        "train_name": "sarrarp50_actions_train.csv",
        "val_name": "sarrarp50_actions_val.csv",
        "test_name": "sarrarp50_actions_test.csv",
        "metadata_name": "sarrarp50_actions_metadata.json",
        "output_fps": 10.0,
        "class_weight_mode": "balanced",
    },
    "4fps": {
        "output_root": Path("/path/to/surg_vid/sarrarp50/globus/sarrarp50_action_clips_16f_4fps"),
        "train_name": "sarrarp50_actions_4fps_train.csv",
        "val_name": "sarrarp50_actions_4fps_val.csv",
        "test_name": "sarrarp50_actions_4fps_test.csv",
        "metadata_name": "sarrarp50_actions_4fps_metadata.json",
        "output_fps": 4.0,
        "class_weight_mode": "proportional",
    },
}

ACTION_ID_TO_NAME = {
    0: "other",
    1: "picking_up_the_needle",
    2: "positioning_the_needle_tip",
    3: "pushing_the_needle_through_the_tissue",
    4: "pulling_the_needle_out_of_the_tissue",
    5: "tying_a_knot",
    6: "cutting_the_suture",
    7: "returning_dropping_the_needle",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class ActionRow:
    frame_id: str
    label: int
    row_index: int


@dataclass(frozen=True)
class VideoSpec:
    source_split: str
    video_dir: Path


@dataclass(frozen=True)
class ClipRow:
    clip_path: Path
    label: int
    sequence_labels: tuple[int, ...] | None
    source_split: str
    video_name: str
    frame_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=sorted(VARIANT_DEFAULTS),
        default="10hz",
        help="Named output/CSV preset. Use 4fps for the temporally subsampled SAR action clips.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--train-name", default=None)
    parser.add_argument("--val-name", default=None)
    parser.add_argument("--test-name", default=None)
    parser.add_argument("--metadata-name", default=None)
    parser.add_argument(
        "--train-splits",
        nargs="+",
        default=["train1", "train2"],
        help="Dataset subdirectories or video directories used for training.",
    )
    parser.add_argument(
        "--val-splits",
        nargs="*",
        default=None,
        help="Optional dataset subdirectories or video directories used for validation.",
    )
    parser.add_argument(
        "--test-splits",
        nargs="*",
        default=None,
        help="Optional labeled dataset subdirectories or video directories used for test CSV export.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.125,
        help="Video-level validation ratio if --val-splits is not provided.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames-per-clip", type=int, default=16)
    parser.add_argument(
        "--context-clips",
        type=int,
        default=1,
        help=(
            "Number of consecutive V-JEPA clips to write into each sample video. "
            "Use 3 for previous/target/future context; frames-per-clip remains the per-V-JEPA-clip length."
        ),
    )
    parser.add_argument(
        "--sequence-labels",
        action="store_true",
        help="Write one label per temporal tubelet token after the clip path, for ASFormer-style heads.",
    )
    parser.add_argument(
        "--tubelet-size",
        type=int,
        default=2,
        help="Temporal tubelet size used to derive sequence labels when --sequence-labels is enabled.",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=10,
        help="Keep every Nth action-label row. SAR labels are normally 10Hz, so 10 means 1Hz.",
    )
    parser.add_argument(
        "--source-rgb-fps",
        type=float,
        default=10.0,
        help="Frame rate of the prepared SAR rgb/ folders. The default 10Hz matches the toolkit extraction.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=None,
        help="FPS of the generated clips and temporal sampling rate. Defaults come from --variant.",
    )
    parser.add_argument(
        "--class-weight-mode",
        choices=["balanced", "sqrt_balanced", "proportional", "none"],
        default=None,
        help=(
            "Which train-set class weights to report in metadata. "
            "balanced is inverse-frequency, sqrt_balanced is a reduced inverse-frequency weighting, "
            "proportional follows the observed class proportions, and none disables weights."
        ),
    )
    parser.add_argument(
        "--resize-short-side",
        type=int,
        default=384,
        help="Resize clip frames before writing. Use 0 to keep original resolution.",
    )
    parser.add_argument(
        "--drop-action-labels",
        nargs="*",
        type=int,
        default=[],
        help="Optional action ids to drop, e.g. --drop-action-labels 5.",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="Debug knob: only process the first N collected videos.",
    )
    parser.add_argument(
        "--limit-samples-per-video",
        type=int,
        default=None,
        help="Debug knob: only process the first N kept samples per video.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
        help="Parallel video directories to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove --output-root before rebuilding clips.",
    )
    parser.add_argument(
        "--unpack-missing-rgb",
        action="store_true",
        help="If a video has video_left.avi but no rgb folder, sample it at 10Hz first.",
    )
    return parser.parse_args()


def apply_variant_defaults(args: argparse.Namespace) -> None:
    defaults = VARIANT_DEFAULTS[args.variant]
    for attr in ("output_root", "train_name", "val_name", "test_name", "metadata_name", "output_fps", "class_weight_mode"):
        if getattr(args, attr) is None:
            setattr(args, attr, defaults[attr])


def normalize_frame_id(value: str) -> str:
    value = str(value).strip()
    if not value:
        return value
    try:
        return f"{int(float(value)):09d}"
    except ValueError:
        return Path(value).stem


def numeric_stem(path: Path) -> int:
    try:
        return int(path.stem)
    except ValueError:
        return 0


def collect_video_specs(data_root: Path, split_names: list[str] | None) -> list[VideoSpec]:
    if not split_names:
        return []

    specs: list[VideoSpec] = []
    for split_name in split_names:
        split_path = Path(split_name)
        if not split_path.is_absolute():
            split_path = data_root / split_name
        if not split_path.exists():
            raise FileNotFoundError(
                f"Split/video path does not exist: {split_path}\n"
                "Expected SAR_RARP50 data root layout like:\n"
                "  <data-root>/train1/video_*/action_discrete.txt\n"
                "  <data-root>/train2/video_*/action_discrete.txt\n"
                "The GitHub repo SAR_RARP50-evaluation is only the toolkit/evaluator; "
                "it does not include the train/test video data."
            )

        if split_path.name.startswith("video_"):
            specs.append(VideoSpec(source_split=split_path.parent.name, video_dir=split_path))
            continue

        video_dirs = sorted(path for path in split_path.glob("video_*") if path.is_dir())
        if not video_dirs:
            raise ValueError(
                f"No video_* directories found under {split_path}\n"
                "Expected video directories containing action_discrete.txt. "
                "If this path is SAR_RARP50-evaluation, that is the toolkit repo, not the dataset."
            )
        specs.extend(VideoSpec(source_split=split_path.name, video_dir=path) for path in video_dirs)

    return sorted(specs, key=lambda spec: (spec.source_split, spec.video_dir.name))


def load_action_rows(video_dir: Path, drop_action_labels: set[int], sample_stride: int) -> list[ActionRow]:
    action_path = video_dir / "action_discrete.txt"
    if not action_path.exists():
        raise FileNotFoundError(f"Missing SAR action labels: {action_path}")

    rows: list[ActionRow] = []
    with action_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row_index, row in enumerate(reader):
            if len(row) < 2:
                continue
            if row_index % sample_stride != 0:
                continue
            label = int(float(row[1]))
            if label in drop_action_labels:
                continue
            if label not in ACTION_ID_TO_NAME:
                raise ValueError(f"Unexpected action label {label} in {action_path}:{row_index + 1}")
            rows.append(
                ActionRow(
                    frame_id=normalize_frame_id(row[0]),
                    label=label,
                    row_index=row_index,
                )
            )
    return rows


def load_action_label_map(video_dir: Path, drop_action_labels: set[int]) -> dict[str, int]:
    action_path = video_dir / "action_discrete.txt"
    if not action_path.exists():
        raise FileNotFoundError(f"Missing SAR action labels: {action_path}")

    labels: dict[str, int] = {}
    with action_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row_index, row in enumerate(reader):
            if len(row) < 2:
                continue
            label = int(float(row[1]))
            if label in drop_action_labels:
                continue
            if label not in ACTION_ID_TO_NAME:
                raise ValueError(f"Unexpected action label {label} in {action_path}:{row_index + 1}")
            labels[normalize_frame_id(row[0])] = label
    return labels


def sample_video_to_rgb(video_dir: Path, sampling_period: int = 6) -> None:
    video_path = video_dir / "video_left.avi"
    if not video_path.exists():
        raise FileNotFoundError(f"Cannot unpack missing rgb because video_left.avi is missing: {video_dir}")

    rgb_dir = video_dir / "rgb"
    rgb_dir.mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video for RGB unpacking: {video_path}")

    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_idx % sampling_period == 0:
                cv2.imwrite(str(rgb_dir / f"{frame_idx:09d}.png"), frame)
            frame_idx += 1
    finally:
        capture.release()


def load_rgb_index(video_dir: Path, unpack_missing_rgb: bool) -> tuple[list[Path], dict[str, int]]:
    rgb_dir = video_dir / "rgb"
    if (not rgb_dir.exists() or not any(rgb_dir.iterdir())) and unpack_missing_rgb:
        sample_video_to_rgb(video_dir)
    if not rgb_dir.exists():
        raise FileNotFoundError(
            f"Missing rgb directory for {video_dir}. Run the SAR toolkit unpack step first, "
            "or pass --unpack-missing-rgb."
        )

    frame_paths = sorted(
        [path for path in rgb_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS],
        key=numeric_stem,
    )
    if not frame_paths:
        raise ValueError(f"No RGB frame images found under {rgb_dir}")
    frame_to_index = {normalize_frame_id(path.stem): idx for idx, path in enumerate(frame_paths)}
    return frame_paths, frame_to_index


def centered_frame_paths(
    frame_paths: list[Path],
    center_index: int,
    frames_per_clip: int,
    source_fps: float,
    clip_fps: float,
) -> list[Path]:
    half_left = frames_per_clip // 2
    source_frames_per_output_frame = source_fps / clip_fps
    selected: list[Path] = []
    for offset in range(frames_per_clip):
        source_offset = int(round((offset - half_left) * source_frames_per_output_frame))
        idx = min(max(center_index + source_offset, 0), len(frame_paths) - 1)
        selected.append(frame_paths[idx])
    return selected


def resize_frame(frame, short_side: int):
    if short_side <= 0:
        return frame
    height, width = frame.shape[:2]
    current_short = min(height, width)
    if current_short == short_side:
        return frame
    scale = short_side / float(current_short)
    new_width = max(2, int(round(width * scale)))
    new_height = max(2, int(round(height * scale)))
    # yuv420p encoders prefer even dimensions.
    new_width += new_width % 2
    new_height += new_height % 2
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def write_clip(frame_paths: list[Path], output_path: Path, output_fps: float, resize_short_side: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read RGB frame: {frame_paths[0]}")
    first = resize_frame(first, resize_short_side)
    height, width = first.shape[:2]

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(output_fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create clip writer: {output_path}")

    try:
        writer.write(first)
        for frame_path in frame_paths[1:]:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read RGB frame: {frame_path}")
            frame = resize_frame(frame, resize_short_side)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()


def process_video(
    spec: VideoSpec,
    output_root: Path,
    frames_per_clip: int,
    sample_stride: int,
    source_rgb_fps: float,
    output_fps: float,
    resize_short_side: int,
    drop_action_labels: set[int],
    unpack_missing_rgb: bool,
    limit_samples_per_video: int | None,
    sequence_labels: bool,
    tubelet_size: int,
    context_clips: int,
) -> tuple[list[ClipRow], dict[str, int]]:
    action_rows = load_action_rows(spec.video_dir, drop_action_labels, sample_stride)
    if limit_samples_per_video is not None:
        action_rows = action_rows[:limit_samples_per_video]

    frame_paths, frame_to_index = load_rgb_index(spec.video_dir, unpack_missing_rgb)
    frame_label_map = load_action_label_map(spec.video_dir, drop_action_labels) if sequence_labels else {}
    clip_rows: list[ClipRow] = []
    frames_per_sample = frames_per_clip * context_clips
    clip_suffix = f"_ctx{context_clips}" if context_clips > 1 else ""
    stats = {
        "rows": len(action_rows),
        "written": 0,
        "reused": 0,
        "missing_frame_ids": 0,
    }

    for action_row in action_rows:
        center_index = frame_to_index.get(action_row.frame_id)
        if center_index is None:
            stats["missing_frame_ids"] += 1
            continue

        clip_path = (
            output_root
            / "clips"
            / spec.source_split
            / spec.video_dir.name
            / f"frame_{action_row.frame_id}{clip_suffix}.mp4"
        )
        if clip_path.exists():
            stats["reused"] += 1
            selected = None
        else:
            selected = centered_frame_paths(
                frame_paths=frame_paths,
                center_index=center_index,
                frames_per_clip=frames_per_sample,
                source_fps=source_rgb_fps,
                clip_fps=output_fps,
            )
            write_clip(
                frame_paths=selected,
                output_path=clip_path,
                output_fps=output_fps,
                resize_short_side=resize_short_side,
            )
            stats["written"] += 1

        clip_sequence_labels = None
        if sequence_labels:
            if selected is None:
                selected = centered_frame_paths(
                    frame_paths=frame_paths,
                    center_index=center_index,
                    frames_per_clip=frames_per_sample,
                    source_fps=source_rgb_fps,
                    clip_fps=output_fps,
                )
            clip_sequence = []
            for token_start in range(0, len(selected), tubelet_size):
                token_frames = selected[token_start : token_start + tubelet_size]
                token_frame = token_frames[min(len(token_frames) - 1, tubelet_size // 2)]
                token_label = frame_label_map.get(normalize_frame_id(token_frame.stem), action_row.label)
                clip_sequence.append(token_label)
            clip_sequence_labels = tuple(clip_sequence)

        clip_rows.append(
            ClipRow(
                clip_path=clip_path,
                label=action_row.label,
                sequence_labels=clip_sequence_labels,
                source_split=spec.source_split,
                video_name=spec.video_dir.name,
                frame_id=action_row.frame_id,
            )
        )
    return clip_rows, stats


def labels_by_video(specs: list[VideoSpec], drop_action_labels: set[int], sample_stride: int) -> dict[str, set[int]]:
    labels: dict[str, set[int]] = {}
    for spec in specs:
        key = f"{spec.source_split}/{spec.video_dir.name}"
        rows = load_action_rows(spec.video_dir, drop_action_labels, sample_stride)
        labels[key] = {row.label for row in rows}
    return labels


def split_train_val_specs(
    train_specs: list[VideoSpec],
    val_ratio: float,
    seed: int,
    drop_action_labels: set[int],
    sample_stride: int,
) -> tuple[list[VideoSpec], list[VideoSpec], str]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("--val-ratio must be between 0 and 1 when --val-splits is omitted")
    if len(train_specs) <= 1:
        return train_specs, [], "single_video_no_val"

    rng = random.Random(seed)
    specs = list(train_specs)
    rng.shuffle(specs)
    target_val = max(1, int(math.ceil(len(specs) * val_ratio)))
    target_val = min(target_val, len(specs) - 1)

    labels = labels_by_video(train_specs, drop_action_labels, sample_stride)
    video_key = lambda spec: f"{spec.source_split}/{spec.video_dir.name}"
    label_video_counts = Counter(label for video_labels in labels.values() for label in video_labels)

    must_train = {
        video_key(spec)
        for spec in train_specs
        if any(label_video_counts[label] == 1 for label in labels[video_key(spec)])
    }
    val_specs: list[VideoSpec] = []
    for spec in specs:
        if len(val_specs) >= target_val:
            break
        if video_key(spec) in must_train:
            continue
        val_specs.append(spec)

    if not val_specs:
        val_specs = [specs[0]]

    val_keys = {video_key(spec) for spec in val_specs}
    kept_train_specs = [spec for spec in train_specs if video_key(spec) not in val_keys]
    return kept_train_specs, sorted(val_specs, key=video_key), "video_level_label_preserving"


def write_probe_csv(rows: list[ClipRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: (str(item.clip_path), item.label)):
            labels = row.sequence_labels if row.sequence_labels is not None else (row.label,)
            handle.write(f"{row.clip_path} {' '.join(str(label) for label in labels)}\n")


def iter_row_labels(row: ClipRow):
    if row.sequence_labels is not None:
        yield from row.sequence_labels
    else:
        yield row.label


def class_counts(rows: list[ClipRow]) -> dict[str, int]:
    counts = Counter(label for row in rows for label in iter_row_labels(row))
    return {str(label_id): counts.get(label_id, 0) for label_id in sorted(ACTION_ID_TO_NAME)}


def class_proportions(rows: list[ClipRow]) -> dict[str, float]:
    counts = Counter(label for row in rows for label in iter_row_labels(row))
    total = sum(counts.values())
    if total <= 0:
        return {str(label_id): 0.0 for label_id in sorted(ACTION_ID_TO_NAME)}
    return {str(label_id): round(counts.get(label_id, 0) / total, 6) for label_id in sorted(ACTION_ID_TO_NAME)}


def balanced_class_weights(rows: list[ClipRow]) -> list[float]:
    counts = Counter(label for row in rows for label in iter_row_labels(row))
    total = sum(counts.values())
    num_classes = len(ACTION_ID_TO_NAME)
    weights = []
    for label_id in sorted(ACTION_ID_TO_NAME):
        count = counts.get(label_id, 0)
        weights.append(0.0 if count == 0 else round(total / (num_classes * count), 6))
    return weights


def sqrt_balanced_class_weights(rows: list[ClipRow]) -> list[float]:
    return [round(math.sqrt(weight), 6) if weight > 0.0 else 0.0 for weight in balanced_class_weights(rows)]


def proportional_class_weights(rows: list[ClipRow]) -> list[float]:
    counts = Counter(label for row in rows for label in iter_row_labels(row))
    total = sum(counts.values())
    num_classes = len(ACTION_ID_TO_NAME)
    if total <= 0:
        return [0.0 for _ in ACTION_ID_TO_NAME]
    uniform_count = total / float(num_classes)
    return [round(counts.get(label_id, 0) / uniform_count, 6) for label_id in sorted(ACTION_ID_TO_NAME)]


def selected_class_weights(rows: list[ClipRow], mode: str) -> list[float] | None:
    if mode == "none":
        return None
    if mode == "balanced":
        return balanced_class_weights(rows)
    if mode == "sqrt_balanced":
        return sqrt_balanced_class_weights(rows)
    if mode == "proportional":
        return proportional_class_weights(rows)
    raise ValueError(f"Unsupported class weight mode: {mode}")


def build_rows_for_specs(args: argparse.Namespace, specs: list[VideoSpec], split_label: str) -> tuple[list[ClipRow], list[dict]]:
    all_rows: list[ClipRow] = []
    stats_by_video: list[dict] = []
    drop_action_labels = set(args.drop_action_labels)

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {
            executor.submit(
                process_video,
                spec,
                args.output_root,
                args.frames_per_clip,
                args.sample_stride,
                args.source_rgb_fps,
                args.output_fps,
                args.resize_short_side,
                drop_action_labels,
                args.unpack_missing_rgb,
                args.limit_samples_per_video,
                args.sequence_labels,
                args.tubelet_size,
                args.context_clips,
            ): spec
            for spec in specs
        }
        for future in as_completed(futures):
            spec = futures[future]
            rows, stats = future.result()
            all_rows.extend(rows)
            stats_by_video.append(
                {
                    "split": split_label,
                    "source_split": spec.source_split,
                    "video": spec.video_dir.name,
                    **stats,
                }
            )
            print(
                f"[{split_label}] {spec.source_split}/{spec.video_dir.name}: "
                f"rows={stats['rows']} clips={len(rows)} "
                f"written={stats['written']} reused={stats['reused']} "
                f"missing_frame_ids={stats['missing_frame_ids']}"
            )

    return all_rows, sorted(stats_by_video, key=lambda item: (item["source_split"], item["video"]))


def maybe_limit_specs(specs: list[VideoSpec], limit_videos: int | None) -> list[VideoSpec]:
    if limit_videos is None:
        return specs
    return specs[:limit_videos]


def main() -> None:
    args = parse_args()
    apply_variant_defaults(args)
    if args.frames_per_clip <= 0:
        raise ValueError("--frames-per-clip must be positive")
    if args.context_clips <= 0:
        raise ValueError("--context-clips must be positive")
    if args.context_clips % 2 == 0:
        raise ValueError("--context-clips must be odd so there is a center target clip")
    if args.tubelet_size <= 0:
        raise ValueError("--tubelet-size must be positive")
    if args.sequence_labels and args.frames_per_clip % args.tubelet_size != 0:
        raise ValueError("--frames-per-clip must be divisible by --tubelet-size when --sequence-labels is used")
    if args.sample_stride <= 0:
        raise ValueError("--sample-stride must be positive")
    if args.source_rgb_fps <= 0:
        raise ValueError("--source-rgb-fps must be positive")
    if args.output_fps <= 0:
        raise ValueError("--output-fps must be positive")
    if args.output_fps > args.source_rgb_fps:
        raise ValueError("--output-fps cannot exceed --source-rgb-fps for pre-extracted SAR rgb frames")

    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)

    train_specs = collect_video_specs(args.data_root, args.train_splits)
    val_specs = collect_video_specs(args.data_root, args.val_splits)
    test_specs = collect_video_specs(args.data_root, args.test_splits)

    split_strategy = "explicit"
    if not val_specs:
        train_specs, val_specs, split_strategy = split_train_val_specs(
            train_specs,
            val_ratio=args.val_ratio,
            seed=args.seed,
            drop_action_labels=set(args.drop_action_labels),
            sample_stride=args.sample_stride,
        )

    train_specs = maybe_limit_specs(train_specs, args.limit_videos)
    val_specs = maybe_limit_specs(val_specs, args.limit_videos)
    test_specs = maybe_limit_specs(test_specs, args.limit_videos)

    print(f"Train videos: {len(train_specs)}")
    print(f"Val videos:   {len(val_specs)}")
    print(f"Test videos:  {len(test_specs)}")
    print(f"Split strategy: {split_strategy}")
    print(f"Variant: {args.variant}")
    print(f"Clip fps: {args.output_fps:g} from source rgb fps: {args.source_rgb_fps:g}")
    print(f"Class weight mode: {args.class_weight_mode}")

    train_rows, train_stats = build_rows_for_specs(args, train_specs, "train")
    val_rows, val_stats = build_rows_for_specs(args, val_specs, "val") if val_specs else ([], [])
    test_rows, test_stats = build_rows_for_specs(args, test_specs, "test") if test_specs else ([], [])

    train_csv = args.csv_dir / args.train_name
    val_csv = args.csv_dir / args.val_name
    test_csv = args.csv_dir / args.test_name
    metadata_path = args.csv_dir / args.metadata_name

    write_probe_csv(train_rows, train_csv)
    write_probe_csv(val_rows, val_csv)
    if test_rows:
        write_probe_csv(test_rows, test_csv)

    metadata = {
        "dataset": "SAR_RARP50",
        "source": "https://github.com/surgical-vision/SAR_RARP50-evaluation",
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv) if test_rows else None,
        "split_strategy": split_strategy,
        "train_splits": args.train_splits,
        "val_splits": args.val_splits,
        "test_splits": args.test_splits,
        "seed": args.seed,
        "variant": args.variant,
        "frames_per_clip": args.frames_per_clip,
        "context_clips": args.context_clips,
        "frames_per_sample": args.frames_per_clip * args.context_clips,
        "sequence_labels": args.sequence_labels,
        "tubelet_size": args.tubelet_size,
        "temporal_label_tokens": (
            args.frames_per_clip * args.context_clips // args.tubelet_size
            if args.sequence_labels
            else None
        ),
        "sample_stride": args.sample_stride,
        "source_rgb_fps": args.source_rgb_fps,
        "output_fps": args.output_fps,
        "source_frames_per_output_frame": args.source_rgb_fps / args.output_fps,
        "resize_short_side": args.resize_short_side,
        "dropped_action_labels": sorted(set(args.drop_action_labels)),
        "action_id_to_name": {str(key): value for key, value in ACTION_ID_TO_NAME.items()},
        "num_classes": len(ACTION_ID_TO_NAME),
        "train_counts": class_counts(train_rows),
        "val_counts": class_counts(val_rows),
        "test_counts": class_counts(test_rows) if test_rows else None,
        "train_label_proportions": class_proportions(train_rows),
        "val_label_proportions": class_proportions(val_rows),
        "test_label_proportions": class_proportions(test_rows) if test_rows else None,
        "class_weight_mode": args.class_weight_mode,
        "balanced_loss_class_weights_from_train": balanced_class_weights(train_rows),
        "sqrt_balanced_loss_class_weights_from_train": sqrt_balanced_class_weights(train_rows),
        "proportional_loss_class_weights_from_train": proportional_class_weights(train_rows),
        "selected_loss_class_weights_from_train": selected_class_weights(train_rows, args.class_weight_mode),
        "video_stats": train_stats + val_stats + test_stats,
    }
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"Wrote train CSV: {train_csv} ({len(train_rows)} rows)")
    print(f"Wrote val CSV:   {val_csv} ({len(val_rows)} rows)")
    if test_rows:
        print(f"Wrote test CSV:  {test_csv} ({len(test_rows)} rows)")
    print(f"Wrote metadata:  {metadata_path}")
    print(f"Train class proportions: {metadata['train_label_proportions']}")
    print(f"Selected train class weights ({args.class_weight_mode}): {metadata['selected_loss_class_weights_from_train']}")


if __name__ == "__main__":
    main()
