#!/usr/bin/env python3
"""Generate random GIF mosaics for auditing one-vs-rest phase head datasets."""

from __future__ import annotations

import argparse
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
    build_summary,
    choose_samples,
    load_entries,
    save_mosaic_round,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True, help="Binary OVR train CSV.")
    parser.add_argument("--val-csv", required=True, help="Binary OVR val CSV.")
    parser.add_argument("--target-class-id", type=int, required=True, help="Original multiclass id represented by the positive OVR class.")
    parser.add_argument("--target-class-name", required=True, help="Human-readable name of the positive OVR class.")
    parser.add_argument("--label-space", default="reduced6", help="Source label space name, e.g. reduced6 or native11.")
    parser.add_argument("--config", default=None, help="Optional OVR config path for provenance in the summary.")
    parser.add_argument("--output-dir", required=True, help="Where GIFs, posters, and summary.json will be written.")
    parser.add_argument("--num-mosaics", type=int, default=4, help="How many random audit mosaics to create.")
    parser.add_argument("--samples-per-class", type=int, default=4, help="How many clips to sample per binary class in each round.")
    parser.add_argument("--tile-width", type=int, default=240, help="Tile width in pixels.")
    parser.add_argument("--tile-height", type=int, default=200, help="Tile height in pixels, including the caption area.")
    parser.add_argument("--caption-height", type=int, default=58, help="Caption height inside each tile.")
    parser.add_argument("--max-frames", type=int, default=16, help="Maximum frames to decode from each clip.")
    parser.add_argument("--gif-fps", type=float, default=4.0, help="Playback FPS for the output GIFs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for clip sampling.")
    parser.add_argument("--class-id", type=int, nargs="*", default=None, help="Optional subset of binary class ids to audit. Defaults to both 0 and 1.")
    return parser.parse_args(argv)


def build_binary_class_names(target_class_id: int, target_class_name: str) -> dict[int, str]:
    target_suffix = f"[{target_class_id}] {target_class_name}"
    return {
        0: f"Rest (not {target_suffix})",
        1: f"Target {target_suffix}",
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    csv_paths = [Path(args.train_csv), Path(args.val_csv)]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = build_binary_class_names(
        target_class_id=args.target_class_id,
        target_class_name=args.target_class_name,
    )
    class_ids = sorted(class_names)
    if args.class_id is not None:
        requested = sorted(set(args.class_id))
        missing = [class_id for class_id in requested if class_id not in class_names]
        if missing:
            raise ValueError(f"Unknown binary class ids requested: {missing}")
        class_ids = requested

    entries = load_entries(csv_paths)
    filtered_entries = [entry for entry in entries if entry.label in class_ids]
    entries_by_class: dict[int, list] = defaultdict(list)
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
            "audit_type": "ovr_binary",
            "label_space": args.label_space,
            "target_class_id": args.target_class_id,
            "target_class_name": args.target_class_name,
            "binary_class_names": {str(class_id): class_names[class_id] for class_id in sorted(class_names)},
            "csv_paths": [str(path) for path in csv_paths],
            "config_path": args.config,
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

    print(f"Wrote {len(rounds)} OVR GIF mosaics to {output_dir}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
