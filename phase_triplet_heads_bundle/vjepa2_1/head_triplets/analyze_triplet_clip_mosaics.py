#!/usr/bin/env python3
"""Generate random GIF mosaics for manual auditing of supported triplet labels."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

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
    DEFAULT_TRIPLET_AUDIT_OUTPUT_DIR,
    DEFAULT_TRIPLET_METADATA,
    DEFAULT_TRIPLET_TRAIN_CSV,
    DEFAULT_TRIPLET_VAL_CSV,
    build_triplet_class_names,
    infer_triplet_source_name,
    load_triplet_tuple_to_id,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        nargs="*",
        default=[str(DEFAULT_TRIPLET_TRAIN_CSV), str(DEFAULT_TRIPLET_VAL_CSV)],
        help="Triplet multiclass CSVs to audit. Defaults to the supported train+val files.",
    )
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_TRIPLET_METADATA),
        help="Metadata JSON containing the supported triplet id mapping.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_TRIPLET_AUDIT_OUTPUT_DIR),
        help="Directory where GIF mosaics and summary.json will be written.",
    )
    parser.add_argument(
        "--num-mosaics",
        type=int,
        default=4,
        help="How many random audit GIFs to create.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=4,
        help="How many clips to sample per triplet class in each round.",
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
        help="Maximum frames to decode from each clip.",
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
        help="Random seed used for class and clip sampling.",
    )
    parser.add_argument(
        "--triplet-id",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of supported triplet ids to audit.",
    )
    parser.add_argument(
        "--top-k-classes",
        type=int,
        default=12,
        help=(
            "If --triplet-id is not provided, audit the top-K most frequent triplet classes. "
            "Use 0 or a negative value to audit every supported class."
        ),
    )
    return parser.parse_args(argv)


def parse_triplet_row(row: list[str]) -> tuple[Path, tuple[int, int, int]] | None:
    if len(row) < 2:
        return None
    sample_path = Path(row[0])
    label = " ".join(row[1:]).strip().strip('"').strip("'")
    if not label:
        return None
    parts = label.split()
    if len(parts) != 3:
        raise ValueError(f"Expected 3 triplet ids in row: {row}")
    triplet = tuple(int(float(part)) for part in parts)
    return sample_path, triplet


def load_triplet_entries(
    csv_paths: Iterable[Path],
    triplet_to_id: dict[tuple[int, int, int], int],
) -> list[ClipEntry]:
    entries: list[ClipEntry] = []
    for csv_path in csv_paths:
        split_name = infer_split_name(csv_path)
        with csv_path.open() as handle:
            for row in csv.reader(handle, delimiter=" "):
                if not row:
                    continue
                parsed = parse_triplet_row(row)
                if parsed is None:
                    continue
                sample_path, triplet = parsed
                if triplet not in triplet_to_id:
                    raise ValueError(f"Triplet {triplet} from {csv_path} not found in metadata mapping")
                entries.append(
                    ClipEntry(
                        sample_path=sample_path,
                        label=triplet_to_id[triplet],
                        source_name=infer_triplet_source_name(sample_path),
                        split_name=split_name,
                    )
                )
    return entries


def choose_triplet_ids(
    entries: list[ClipEntry],
    requested_triplet_ids: list[int] | None,
    top_k_classes: int,
    class_names: dict[int, str],
) -> list[int]:
    if requested_triplet_ids is not None:
        requested = sorted(set(requested_triplet_ids))
        missing = [triplet_id for triplet_id in requested if triplet_id not in class_names]
        if missing:
            raise ValueError(f"Unknown supported triplet ids requested: {missing}")
        return requested

    class_counts = Counter(entry.label for entry in entries)
    if top_k_classes <= 0:
        return sorted(class_names)
    ranked = sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))
    return [class_id for class_id, _ in ranked[:top_k_classes]]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    csv_paths = [Path(path) for path in args.csv]
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    triplet_to_id = load_triplet_tuple_to_id(metadata_path)
    class_names = build_triplet_class_names(metadata_path)
    entries = load_triplet_entries(csv_paths, triplet_to_id=triplet_to_id)
    selected_triplet_ids = choose_triplet_ids(
        entries=entries,
        requested_triplet_ids=args.triplet_id,
        top_k_classes=args.top_k_classes,
        class_names=class_names,
    )

    filtered_entries = [entry for entry in entries if entry.label in selected_triplet_ids]
    entries_by_class: dict[int, list[ClipEntry]] = defaultdict(list)
    for entry in filtered_entries:
        entries_by_class[entry.label].append(entry)

    empty_classes = [class_id for class_id in selected_triplet_ids if not entries_by_class[class_id]]
    if empty_classes:
        raise ValueError(f"No clips found for selected triplet ids: {empty_classes}")

    rng = random.Random(args.seed)
    rounds = []
    for round_index in range(args.num_mosaics):
        sampled_entries = choose_samples(
            entries_by_class=entries_by_class,
            class_ids=selected_triplet_ids,
            samples_per_class=args.samples_per_class,
            rng=rng,
        )
        rounds.append(
            save_mosaic_round(
                round_index=round_index,
                sampled_entries=sampled_entries,
                class_names={class_id: class_names[class_id] for class_id in selected_triplet_ids},
                output_dir=output_dir,
                tile_width=args.tile_width,
                tile_height=args.tile_height,
                caption_height=args.caption_height,
                max_frames=args.max_frames,
                gif_fps=args.gif_fps,
            )
        )

    summary = build_summary(
        filtered_entries,
        {class_id: class_names[class_id] for class_id in selected_triplet_ids},
    )
    full_summary = build_summary(entries, class_names)
    summary.update(
        {
            "audit_type": "triplet_multiclass",
            "csv_paths": [str(path) for path in csv_paths],
            "metadata_path": str(metadata_path),
            "output_dir": str(output_dir),
            "num_mosaics": args.num_mosaics,
            "samples_per_class": args.samples_per_class,
            "seed": args.seed,
            "tile_width": args.tile_width,
            "tile_height": args.tile_height,
            "caption_height": args.caption_height,
            "gif_fps": args.gif_fps,
            "selected_triplet_ids": selected_triplet_ids,
            "selected_triplet_names": {str(class_id): class_names[class_id] for class_id in selected_triplet_ids},
            "full_total_entries": full_summary["total_entries"],
            "full_class_counts": full_summary["class_counts"],
            "full_split_class_counts": full_summary["split_class_counts"],
            "rounds": rounds,
        }
    )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Wrote {len(rounds)} triplet audit GIF mosaics to {output_dir}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
