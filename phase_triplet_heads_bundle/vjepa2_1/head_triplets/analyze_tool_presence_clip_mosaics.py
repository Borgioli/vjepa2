#!/usr/bin/env python3
"""Generate random GIF mosaics for auditing multi-label tool presence clips."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from head_phases.analyze_phase_clip_mosaics import (  # noqa: E402
    ClipEntry,
    build_summary,
    choose_samples,
    infer_split_name,
    save_mosaic_round,
)
from head_triplets.audit_utils import (  # noqa: E402
    DEFAULT_MULTILABEL_TRAIN_CSV,
    DEFAULT_MULTILABEL_VAL_CSV,
    DEFAULT_TRIPLET_METADATA,
    DEFAULT_TOOL_AUDIT_OUTPUT_ROOT,
    TOOL_ID_TO_NAME,
    build_triplet_label_lookup,
    infer_triplet_source_name,
    slugify,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        default=str(DEFAULT_MULTILABEL_TRAIN_CSV),
        help="Grouped multi-label train CSV to audit.",
    )
    parser.add_argument(
        "--val-csv",
        default=str(DEFAULT_MULTILABEL_VAL_CSV),
        help="Grouped multi-label val CSV to audit.",
    )
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_TRIPLET_METADATA),
        help="Metadata JSON used to translate active triplet ids into readable labels.",
    )
    parser.add_argument("--tool-id", type=int, required=True, help="Tool class id to audit as a binary presence task.")
    parser.add_argument(
        "--tool-name",
        default=None,
        help="Optional override for the target tool name used in captions and summary.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where GIFs, posters, and summary.json will be written. Defaults to app/tool_clip_audit/<tool-name>/",
    )
    parser.add_argument("--num-mosaics", type=int, default=4, help="How many random audit mosaics to create.")
    parser.add_argument("--samples-per-class", type=int, default=4, help="How many clips to sample per binary class.")
    parser.add_argument("--tile-width", type=int, default=240, help="Tile width in pixels.")
    parser.add_argument("--tile-height", type=int, default=200, help="Tile height in pixels, including the caption area.")
    parser.add_argument("--caption-height", type=int, default=58, help="Caption height inside each tile.")
    parser.add_argument("--max-frames", type=int, default=16, help="Maximum frames to decode from each clip.")
    parser.add_argument("--gif-fps", type=float, default=4.0, help="Playback FPS for the output GIFs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for clip sampling.")
    parser.add_argument(
        "--class-id",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of binary class ids to audit. Defaults to both 0 and 1.",
    )
    return parser.parse_args(argv)


def build_binary_class_names(tool_id: int, tool_name: str) -> dict[int, str]:
    target_suffix = f"[{tool_id}] {tool_name}"
    return {
        0: f"Rest (tool absent: {target_suffix})",
        1: f"Target tool present: {target_suffix}",
    }


def parse_multihot_field(value: str) -> list[int]:
    return [int(float(token)) for token in str(value).split()]


def normalize_triplet_caption(value: str | None, triplet_label_lookup: dict[str, str] | None = None) -> str | None:
    if value is None:
        return None
    cleaned_parts = []
    for part in str(value).split(";"):
        stripped = part.strip()
        if not stripped:
            continue
        cleaned_parts.append(triplet_label_lookup.get(stripped, stripped) if triplet_label_lookup else stripped)
    cleaned = " ; ".join(cleaned_parts)
    return cleaned or None


def load_tool_presence_entries(
    csv_paths: list[Path],
    tool_id: int,
    triplet_label_lookup: dict[str, str] | None = None,
) -> list[ClipEntry]:
    entries: list[ClipEntry] = []
    for csv_path in csv_paths:
        split_name = infer_split_name(csv_path)
        with csv_path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            required_fields = {"clip_path", "tool_multihot"}
            missing_fields = sorted(required_fields - set(reader.fieldnames or []))
            if missing_fields:
                raise ValueError(f"Missing required fields in {csv_path}: {missing_fields}")

            for row in reader:
                sample_path = Path(row["clip_path"])
                tool_multihot = parse_multihot_field(row["tool_multihot"])
                if tool_id < 0 or tool_id >= len(tool_multihot):
                    raise ValueError(
                        f"tool_id={tool_id} is out of range for {csv_path}; expected [0, {len(tool_multihot) - 1}]"
                    )
                label = int(tool_multihot[tool_id] > 0)
                entries.append(
                    ClipEntry(
                        sample_path=sample_path,
                        label=label,
                        source_name=infer_triplet_source_name(sample_path),
                        split_name=split_name,
                        caption_text=normalize_triplet_caption(
                            row.get("active_triplet_labels"),
                            triplet_label_lookup=triplet_label_lookup,
                        ),
                    )
                )
    return entries


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tool_name = args.tool_name or TOOL_ID_TO_NAME.get(args.tool_id, f"tool_{args.tool_id}")
    if args.tool_id not in TOOL_ID_TO_NAME:
        raise ValueError(f"Unknown tool id {args.tool_id}; supported ids are {sorted(TOOL_ID_TO_NAME)}")

    csv_paths = [Path(args.train_csv), Path(args.val_csv)]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else DEFAULT_TOOL_AUDIT_OUTPUT_ROOT / slugify(tool_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = build_binary_class_names(tool_id=args.tool_id, tool_name=tool_name)
    class_ids = sorted(class_names)
    if args.class_id is not None:
        requested = sorted(set(args.class_id))
        missing = [class_id for class_id in requested if class_id not in class_names]
        if missing:
            raise ValueError(f"Unknown binary class ids requested: {missing}")
        class_ids = requested

    triplet_label_lookup = build_triplet_label_lookup(Path(args.metadata))
    entries = load_tool_presence_entries(
        csv_paths,
        tool_id=args.tool_id,
        triplet_label_lookup=triplet_label_lookup,
    )
    filtered_entries = [entry for entry in entries if entry.label in class_ids]
    entries_by_class: dict[int, list[ClipEntry]] = defaultdict(list)
    for entry in filtered_entries:
        entries_by_class[entry.label].append(entry)

    empty_classes = [class_id for class_id in class_ids if not entries_by_class[class_id]]
    if empty_classes:
        raise ValueError(f"No clips found for binary class ids: {empty_classes}")

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
            "audit_type": "tool_presence_binary",
            "tool_id": args.tool_id,
            "tool_name": tool_name,
            "binary_class_names": {str(class_id): class_names[class_id] for class_id in sorted(class_names)},
            "csv_paths": [str(path) for path in csv_paths],
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

    print(f"Wrote {len(rounds)} tool audit GIF mosaics to {output_dir}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
