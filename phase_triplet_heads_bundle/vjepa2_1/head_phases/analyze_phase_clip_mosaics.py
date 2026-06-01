#!/usr/bin/env python3
"""Generate random GIF mosaics for manual auditing of phase clip labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
import textwrap
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import cv2
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This script expects the project Python environment. "
        "Run it with python3 "
        f"or install the missing dependency: {exc.name}"
    ) from exc


DEFAULT_CSV_PATHS = [
    Path("/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/sitl_surgenet_phases_train_6class_merge_01_56_78_drop_9_10.csv"),
    Path("/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/sitl_surgenet_phases_val_6class_merge_01_56_78_drop_9_10.csv"),
]
DEFAULT_METADATA_PATH = Path(
    "/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/sitl_surgenet_phases_6class_merge_01_56_78_drop_9_10_metadata.json"
)
DEFAULT_OUTPUT_DIR = Path("/path/to/phase_triplet_heads_bundle/vjepa2_1/app/phase_clip_audit_6class")


@dataclass(frozen=True)
class ClipEntry:
    sample_path: Path
    label: int
    source_name: str
    split_name: str
    caption_text: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        nargs="*",
        default=[str(path) for path in DEFAULT_CSV_PATHS],
        help="CSV files to audit. Defaults to the current 6-class SITL+Surgenet train+val CSVs.",
    )
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA_PATH),
        help="Metadata JSON containing class names.",
    )
    parser.add_argument(
        "--label-transform-json",
        default=None,
        help=(
            "Optional JSON file mapping source labels to display labels. "
            "Use null to drop a source label from the audit."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where GIF mosaics and the summary manifest will be written.",
    )
    parser.add_argument(
        "--num-mosaics",
        type=int,
        default=4,
        help="How many random audit GIFs to generate.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=4,
        help="How many clips to show per class in each GIF mosaic.",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=240,
        help="Tile width in pixels.",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=200,
        help="Tile height in pixels, including the caption area.",
    )
    parser.add_argument(
        "--caption-height",
        type=int,
        default=58,
        help="Caption height inside each tile.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=16,
        help="Maximum frames to use from each clip.",
    )
    parser.add_argument(
        "--gif-fps",
        type=float,
        default=4.0,
        help="Playback FPS for the output GIFs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for clip sampling.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of class ids to audit.",
    )
    return parser.parse_args()


def load_class_names(metadata_path: Path) -> dict[int, str]:
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    phase_to_id = metadata.get("reduced_phase_to_id") or metadata.get("phase_to_id")
    if not phase_to_id:
        raise ValueError(f"Could not find 'reduced_phase_to_id' or 'phase_to_id' in {metadata_path}")
    return {int(label_id): phase_name for phase_name, label_id in phase_to_id.items()}


def load_label_transform(transform_path: Path | None) -> dict[int, int | None] | None:
    if transform_path is None:
        return None
    with transform_path.open() as handle:
        raw_mapping = json.load(handle)
    if not isinstance(raw_mapping, dict):
        raise ValueError(f"Expected a JSON object in label transform file: {transform_path}")
    return {int(source_id): (None if target_id is None else int(target_id)) for source_id, target_id in raw_mapping.items()}


def infer_source_name(sample_path: Path) -> str:
    sample_str = str(sample_path)
    if "/surgenet_phases/" in sample_str:
        return "surgenet"
    if "/sitl_phases" in sample_str:
        return "sitl"
    return "unknown"


def infer_split_name(csv_path: Path) -> str:
    stem = csv_path.stem.lower()
    if "train" in stem:
        return "train"
    if "val" in stem or "valid" in stem:
        return "val"
    return stem


def load_entries(csv_paths: Iterable[Path], label_transform: dict[int, int | None] | None = None) -> list[ClipEntry]:
    entries: list[ClipEntry] = []
    for csv_path in csv_paths:
        split_name = infer_split_name(csv_path)
        with csv_path.open() as handle:
            for row in csv.reader(handle, delimiter=" "):
                if not row:
                    continue
                sample_path = Path(row[0])
                label = int(row[1])
                if label_transform is not None and label in label_transform:
                    label = label_transform[label]
                    if label is None:
                        continue
                entries.append(
                    ClipEntry(
                        sample_path=sample_path,
                        label=label,
                        source_name=infer_source_name(sample_path),
                        split_name=split_name,
                    )
                )
    return entries


def build_summary(entries: list[ClipEntry], class_names: dict[int, str]) -> dict:
    class_counts = Counter(entry.label for entry in entries)
    source_class_counts: dict[str, Counter] = defaultdict(Counter)
    split_class_counts: dict[str, Counter] = defaultdict(Counter)
    for entry in entries:
        source_class_counts[entry.source_name][entry.label] += 1
        split_class_counts[entry.split_name][entry.label] += 1

    return {
        "total_entries": len(entries),
        "class_names": {str(class_id): class_names[class_id] for class_id in sorted(class_names)},
        "class_counts": {str(class_id): class_counts[class_id] for class_id in sorted(class_names)},
        "source_class_counts": {
            source_name: {str(class_id): counts[class_id] for class_id in sorted(class_names)}
            for source_name, counts in sorted(source_class_counts.items())
        },
        "split_class_counts": {
            split_name: {str(class_id): counts[class_id] for class_id in sorted(class_names)}
            for split_name, counts in sorted(split_class_counts.items())
        },
    }


def resize_with_padding(frame_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    source_height, source_width = frame_rgb.shape[:2]
    scale = min(width / source_width, height / source_height)
    new_width = max(1, int(round(source_width * scale)))
    new_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(frame_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x_offset = (width - new_width) // 2
    y_offset = (height - new_height) // 2
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized
    return canvas


def read_clip_frames(video_path: Path, max_frames: int, frame_width: int, frame_height: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open clip: {video_path}")

    frames: list[np.ndarray] = []
    try:
        while len(frames) < max_frames:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(resize_with_padding(frame_rgb, frame_width, frame_height))
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"Could not decode any frames from clip: {video_path}")
    return frames


def draw_text_block(draw: ImageDraw.ImageDraw, x: int, y: int, lines: list[str], font: ImageFont.ImageFont) -> None:
    line_height = 12
    for line in lines:
        draw.text((x, y), line, fill=(255, 255, 255), font=font)
        y += line_height


def make_tile(
    frame_rgb: np.ndarray,
    entry: ClipEntry | None,
    class_name: str | None,
    class_id: int | None,
    tile_width: int,
    tile_height: int,
    caption_height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    tile = Image.new("RGB", (tile_width, tile_height), color=(18, 18, 18))
    frame_height = tile_height - caption_height
    frame_image = Image.fromarray(frame_rgb if entry is not None else np.zeros((frame_height, tile_width, 3), dtype=np.uint8))
    tile.paste(frame_image, (0, 0))

    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, frame_height, tile_width, tile_height), fill=(0, 0, 0))

    if entry is None or class_name is None or class_id is None:
        draw_text_block(draw, 6, frame_height + 5, ["empty"], font)
        return tile

    if entry.caption_text:
        wrapped_caption = textwrap.wrap(entry.caption_text, width=28)[:4]
        draw_text_block(draw, 6, frame_height + 4, wrapped_caption or ["-"], font)
        return tile

    title = f"[{class_id}] {class_name}"
    wrapped_title = textwrap.wrap(title, width=28)[:2]
    basename = entry.sample_path.name
    source_line = f"{entry.source_name} | {entry.split_name}"
    file_line = textwrap.shorten(basename, width=34, placeholder="...")
    draw_text_block(draw, 6, frame_height + 4, wrapped_title + [source_line, file_line], font)
    return tile


def choose_samples(
    entries_by_class: dict[int, list[ClipEntry]],
    class_ids: list[int],
    samples_per_class: int,
    rng: random.Random,
) -> dict[int, list[ClipEntry]]:
    chosen: dict[int, list[ClipEntry]] = {}
    for class_id in class_ids:
        entries = list(entries_by_class[class_id])
        if len(entries) <= samples_per_class:
            chosen[class_id] = entries
        else:
            chosen[class_id] = rng.sample(entries, samples_per_class)
    return chosen


def save_mosaic_round(
    round_index: int,
    sampled_entries: dict[int, list[ClipEntry]],
    class_names: dict[int, str],
    output_dir: Path,
    tile_width: int,
    tile_height: int,
    caption_height: int,
    max_frames: int,
    gif_fps: float,
) -> dict:
    font = ImageFont.load_default()
    class_ids = sorted(sampled_entries)
    num_rows = len(class_ids)
    num_cols = max(len(sampled_entries[class_id]) for class_id in class_ids)
    frame_height = tile_height - caption_height

    clip_frames: dict[tuple[int, int], list[np.ndarray]] = {}
    clip_lengths: list[int] = []
    for row_idx, class_id in enumerate(class_ids):
        for col_idx, entry in enumerate(sampled_entries[class_id]):
            frames = read_clip_frames(entry.sample_path, max_frames=max_frames, frame_width=tile_width, frame_height=frame_height)
            clip_frames[(row_idx, col_idx)] = frames
            clip_lengths.append(len(frames))

    max_round_frames = max(clip_lengths) if clip_lengths else 1
    mosaic_frames: list[Image.Image] = []
    for frame_idx in range(max_round_frames):
        canvas = Image.new("RGB", (num_cols * tile_width, num_rows * tile_height), color=(8, 8, 8))
        for row_idx, class_id in enumerate(class_ids):
            row_entries = sampled_entries[class_id]
            for col_idx in range(num_cols):
                x_offset = col_idx * tile_width
                y_offset = row_idx * tile_height
                if col_idx >= len(row_entries):
                    tile = make_tile(
                        frame_rgb=np.zeros((frame_height, tile_width, 3), dtype=np.uint8),
                        entry=None,
                        class_name=None,
                        class_id=None,
                        tile_width=tile_width,
                        tile_height=tile_height,
                        caption_height=caption_height,
                        font=font,
                    )
                else:
                    entry = row_entries[col_idx]
                    frames = clip_frames[(row_idx, col_idx)]
                    current_frame = frames[min(frame_idx, len(frames) - 1)]
                    tile = make_tile(
                        frame_rgb=current_frame,
                        entry=entry,
                        class_name=class_names[class_id],
                        class_id=class_id,
                        tile_width=tile_width,
                        tile_height=tile_height,
                        caption_height=caption_height,
                        font=font,
                    )
                canvas.paste(tile, (x_offset, y_offset))
        mosaic_frames.append(canvas)

    round_dir = output_dir / f"round_{round_index:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    gif_path = round_dir / f"class_audit_round_{round_index:02d}.gif"
    png_path = round_dir / f"class_audit_round_{round_index:02d}_poster.png"
    mosaic_frames[0].save(
        gif_path,
        save_all=True,
        append_images=mosaic_frames[1:],
        duration=max(1, int(round(1000.0 / gif_fps))),
        loop=0,
        optimize=False,
    )
    mosaic_frames[0].save(png_path)

    return {
        "round_index": round_index,
        "gif_path": str(gif_path),
        "poster_path": str(png_path),
        "samples": {
            str(class_id): [
                {
                    "sample_path": str(entry.sample_path),
                    "source_name": entry.source_name,
                    "split_name": entry.split_name,
                }
                for entry in sampled_entries[class_id]
            ]
            for class_id in class_ids
        },
    }


def main() -> None:
    args = parse_args()
    csv_paths = [Path(path) for path in args.csv]
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label_transform = load_label_transform(Path(args.label_transform_json)) if args.label_transform_json else None

    class_names = load_class_names(metadata_path)
    class_ids = sorted(class_names)
    if args.class_id is not None:
        requested = sorted(set(args.class_id))
        missing = [class_id for class_id in requested if class_id not in class_names]
        if missing:
            raise ValueError(f"Unknown class ids requested: {missing}")
        class_ids = requested

    entries = load_entries(csv_paths, label_transform=label_transform)
    filtered_entries = [entry for entry in entries if entry.label in class_ids]
    entries_by_class: dict[int, list[ClipEntry]] = defaultdict(list)
    for entry in filtered_entries:
        entries_by_class[entry.label].append(entry)

    empty_classes = [class_id for class_id in class_ids if not entries_by_class[class_id]]
    if empty_classes:
        raise ValueError(f"No clips found for class ids: {empty_classes}")

    rng = random.Random(args.seed)
    rounds = []
    for round_index in range(args.num_mosaics):
        sampled_entries = choose_samples(
            entries_by_class=entries_by_class,
            class_ids=class_ids,
            samples_per_class=args.samples_per_class,
            rng=rng,
        )
        rounds.append(
            save_mosaic_round(
                round_index=round_index,
                sampled_entries=sampled_entries,
                class_names=class_names,
                output_dir=output_dir,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                caption_height=args.caption_height,
                max_frames=args.max_frames,
                gif_fps=args.gif_fps,
            )
        )

    summary = build_summary(filtered_entries, {class_id: class_names[class_id] for class_id in class_ids})
    summary.update(
        {
            "csv_paths": [str(path) for path in csv_paths],
            "metadata_path": str(metadata_path),
            "label_transform_json": None if args.label_transform_json is None else str(Path(args.label_transform_json)),
            "output_dir": str(output_dir),
            "num_mosaics": args.num_mosaics,
            "samples_per_class": args.samples_per_class,
            "seed": args.seed,
            "tile_width": args.tile_width,
            "tile_height": args.tile_height,
            "caption_height": args.caption_height,
            "gif_fps": args.gif_fps,
            "rounds": rounds,
        }
    )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Wrote {len(rounds)} GIF mosaics to {output_dir}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
