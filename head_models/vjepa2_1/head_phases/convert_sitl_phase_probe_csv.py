"""
Convert SITL surgical phase clip annotations into probe-training CSVs.

The frozen video-classification probe in this repo expects space-separated
files with one sample per line:

    /absolute/path/to/clip.mp4 <integer_label>

This script reads the SITL clip dataset rooted at `globus/sitl_phases_4s_4fps`,
normalizes phase-label variants, drops unlabeled clips, splits at the video
folder level to avoid leakage, and writes train/val CSVs in the expected
format.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from head_phases.map_sitl_to_cholec80 import CHOLEC80_PHASE_TO_ID, map_sitl_probe_id_to_cholec80_id


DEFAULT_INPUT_ROOT = Path("/path/to/phase_triplet_heads_bundle/globus/sitl_phases_4s_4fps")
DEFAULT_OUTPUT_DIR = Path("/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models")

# Match the existing phase probe setup in this repo: labels 1..11 with
# num_classes=12 and 0 left unused.
PHASE_TO_ID = {
    "Exposure of the working area": 1,
    "Retraction of the gallbladder neck": 2,
    "Opening the anterior peritoneal layer of the triangle of Calot": 3,
    "Opening the posterior peritoneal layer of the triangle of Calot": 4,
    "Isolation of the cystic duct": 5,
    "Isolation of the cystic artery": 6,
    "clipping of the cystic duct": 7,
    "clipping of the cystic artery": 8,
    "Division of the cystic duct": 9,
    "dissection of the gallbladder from the liver": 10,
    "specimen retrieval": 11,
}

EXACT_LABEL_MAP = {
    "": None,
    "/": None,
    "None": None,
    "Phase": None,
    "Exposure of the working area": "Exposure of the working area",
    "exposure of the working area": "Exposure of the working area",
    "1: exposure of the working area": "Exposure of the working area",
    "Retraction of the gallbladder neck": "Retraction of the gallbladder neck",
    "Opening the anterior peritoneal layer of the triangle of Calot": (
        "Opening the anterior peritoneal layer of the triangle of Calot"
    ),
    "Opening the posterior peritoneal layer of the triangle of Calot": (
        "Opening the posterior peritoneal layer of the triangle of Calot"
    ),
    "Isolation of the cystic duct": "Isolation of the cystic duct",
    "Isolation of the cystic artery": "Isolation of the cystic artery",
    "isolation of the cystic artery": "Isolation of the cystic artery",
    "Isolation of the cystic artery ( second branch)": "Isolation of the cystic artery",
    "6-Isolation of the cystic duct and cystic artery": "Isolation of the cystic duct",
    "clipping of the cystic duct": "clipping of the cystic duct",
    "Clipping of the cystic duct": "clipping of the cystic duct",
    "clipping of the cystic artery": "clipping of the cystic artery",
    "clipping of the cystic artery ( second branch)": "clipping of the cystic artery",
    "clipping of the cystic artery (second branch)": "clipping of the cystic artery",
    "8- clipping of the cystic duct and cystic artery": "clipping of the cystic duct",
    "Division of the cystic duct": "Division of the cystic duct",
    "Division of the cystic artery": "Division of the cystic duct",
    "Division of the cystic artery ( second branch)": "Division of the cystic duct",
    "Division of the cystic artery (second branch)": "Division of the cystic duct",
    "Division of the cystic duct and artery": "Division of the cystic duct",
    "Division of the cystic duct and cystic artery": "Division of the cystic duct",
    "dissection of the gallbladder from the liver": "dissection of the gallbladder from the liver",
    "11: specimen retrieval": "specimen retrieval",
    "specimen retrieval": "specimen retrieval",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument(
        "--annotations-csv",
        type=Path,
        default=None,
        help="Defaults to <input-root>/annotations.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-name", default="sitl_phases_train.csv")
    parser.add_argument("--val-name", default="sitl_phases_val.csv")
    parser.add_argument("--metadata-name", default="sitl_phases_metadata.json")
    parser.add_argument("--coarse-train-name", default="sitl_phases_cholec80_train.csv")
    parser.add_argument("--coarse-val-name", default="sitl_phases_cholec80_val.csv")
    parser.add_argument("--coarse-metadata-name", default="sitl_phases_cholec80_metadata.json")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Video-level validation split ratio when --val-videos is not set.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--val-videos",
        nargs="*",
        default=None,
        help="Optional explicit validation video folders, e.g. video01 video13.",
    )
    parser.add_argument(
        "--allow-missing-clips",
        action="store_true",
        help="Keep rows even if the referenced clip file is missing.",
    )
    parser.add_argument(
        "--skip-coarse-export",
        action="store_true",
        help="Do not also export the 5-class Cholec80-mapped legacy SITL CSVs.",
    )
    return parser.parse_args()


def canonicalize_label(raw_label: str | None) -> str | None:
    if raw_label is None:
        return None

    label = str(raw_label).strip()
    if label in EXACT_LABEL_MAP:
        return EXACT_LABEL_MAP[label]

    lowered = " ".join(label.lower().split())

    if lowered in {"none", "/", "phase"}:
        return None
    if "specimen" in lowered or "retrieval" in lowered or "retrival" in lowered:
        return "specimen retrieval"
    if "dissection" in lowered and "gallbladder" in lowered:
        return "dissection of the gallbladder from the liver"
    if "division" in lowered:
        # Collapse duct/artery variants to the single division class used by the
        # existing phase probe in this repo.
        return "Division of the cystic duct"
    if "clipping" in lowered or lowered.startswith("8- clipping"):
        if "duct" in lowered:
            return "clipping of the cystic duct"
        if "artery" in lowered:
            return "clipping of the cystic artery"
    if "isolation" in lowered:
        if "duct" in lowered:
            return "Isolation of the cystic duct"
        if "artery" in lowered:
            return "Isolation of the cystic artery"
    if "anterior" in lowered and "peritoneal" in lowered:
        return "Opening the anterior peritoneal layer of the triangle of Calot"
    if "posterior" in lowered and "peritoneal" in lowered:
        return "Opening the posterior peritoneal layer of the triangle of Calot"
    if "retraction" in lowered:
        return "Retraction of the gallbladder neck"
    if "exposure" in lowered or "working area" in lowered:
        return "Exposure of the working area"

    return None


def choose_val_videos(
    samples_by_video: dict[str, list[tuple[Path, int]]], args: argparse.Namespace
) -> tuple[list[str], list[str], str]:
    video_ids = sorted(samples_by_video)
    if args.val_videos:
        requested = set(args.val_videos)
        available = set(video_ids)
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"Requested validation videos are missing from the dataset: {missing}")
        val_videos = sorted(requested)
        strategy = "explicit"
    else:
        rng = random.Random(args.seed)
        shuffled = list(video_ids)
        rng.shuffle(shuffled)

        if len(shuffled) <= 1:
            val_videos = shuffled
        else:
            target_num_val = max(1, int(math.ceil(len(shuffled) * args.val_ratio)))
            target_num_val = min(target_num_val, len(shuffled) - 1)

            labels_by_video = {
                video_id: {label_id for _, label_id in samples}
                for video_id, samples in samples_by_video.items()
            }
            video_count_by_label = Counter(label for labels in labels_by_video.values() for label in labels)
            must_train = {
                video_id
                for video_id, labels in labels_by_video.items()
                if any(video_count_by_label[label_id] == 1 for label_id in labels)
            }

            uncovered = {label_id for label_id, count in video_count_by_label.items() if count > 1}
            remaining = set(video_ids) - must_train
            val_set: set[str] = set()

            # Greedily seed the validation split with videos that cover rare labels,
            # but do not remove the only video containing a class from training.
            while uncovered and remaining:
                best_video = max(
                    remaining,
                    key=lambda video_id: (
                        len(labels_by_video[video_id] & uncovered),
                        sum(
                            1.0 / video_count_by_label[label_id]
                            for label_id in labels_by_video[video_id] & uncovered
                        ),
                        -len(samples_by_video[video_id]),
                        video_id,
                    ),
                )
                covered = labels_by_video[best_video] & uncovered
                if not covered:
                    break
                val_set.add(best_video)
                remaining.remove(best_video)
                uncovered -= covered

            # Then fill up to the requested validation size deterministically.
            for video_id in shuffled:
                if len(val_set) >= target_num_val:
                    break
                if video_id in remaining:
                    val_set.add(video_id)
                    remaining.remove(video_id)

            val_videos = sorted(val_set)
            strategy = "video_level_train_label_preserving"

    val_set = set(val_videos)
    train_videos = sorted(video_id for video_id in video_ids if video_id not in val_set)
    return train_videos, val_videos, strategy


def write_probe_csv(samples: list[tuple[Path, int]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for clip_path, label_id in sorted(samples, key=lambda x: (str(x[0]), x[1])):
            handle.write(f"{clip_path} {label_id}\n")


def phase_counts(samples: list[tuple[Path, int]]) -> dict[str, int]:
    counts = Counter(label for _, label in samples)
    return {str(label_id): counts.get(label_id, 0) for label_id in sorted(PHASE_TO_ID.values())}


def coarse_phase_counts(samples: list[tuple[Path, int]]) -> dict[str, int]:
    counts = Counter(label for _, label in samples)
    return {str(label_id): counts.get(label_id, 0) for label_id in sorted(CHOLEC80_PHASE_TO_ID.values())}


def main() -> None:
    args = parse_args()
    annotations_csv = args.annotations_csv or (args.input_root / "annotations.csv")

    if not annotations_csv.exists():
        raise FileNotFoundError(f"Could not find annotations CSV: {annotations_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / args.train_name
    val_path = args.output_dir / args.val_name
    metadata_path = args.output_dir / args.metadata_name
    coarse_train_path = args.output_dir / args.coarse_train_name
    coarse_val_path = args.output_dir / args.coarse_val_name
    coarse_metadata_path = args.output_dir / args.coarse_metadata_name

    samples_by_video: dict[str, list[tuple[Path, int]]] = defaultdict(list)
    raw_label_counts: Counter[str] = Counter()
    canonical_label_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()

    with annotations_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = {"path_to_the_clip", "label"}
        if not expected_columns.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"Unexpected annotations columns. Found {reader.fieldnames}, expected at least {sorted(expected_columns)}"
            )

        for row in reader:
            raw_path = (row.get("path_to_the_clip") or "").strip()
            raw_label = (row.get("label") or "").strip()
            raw_label_counts[raw_label] += 1

            if not raw_path:
                skipped_counts["missing_path"] += 1
                continue

            canonical_label = canonicalize_label(raw_label)
            if canonical_label is None:
                skipped_counts["unlabeled_or_filtered"] += 1
                continue

            label_id = PHASE_TO_ID.get(canonical_label)
            if label_id is None:
                skipped_counts["unmapped_label"] += 1
                continue

            clip_path = (args.input_root / raw_path).resolve()
            if not clip_path.exists() and not args.allow_missing_clips:
                skipped_counts["missing_clip_file"] += 1
                continue

            path_parts = Path(raw_path).parts
            if len(path_parts) < 2:
                skipped_counts["malformed_relative_path"] += 1
                continue

            video_id = path_parts[1]
            samples_by_video[video_id].append((clip_path, label_id))
            canonical_label_counts[canonical_label] += 1

    if not samples_by_video:
        raise RuntimeError("No labeled samples were collected. Check the input paths and label mapping.")

    train_videos, val_videos, split_strategy = choose_val_videos(samples_by_video, args)
    train_samples = [sample for video_id in train_videos for sample in samples_by_video[video_id]]
    val_samples = [sample for video_id in val_videos for sample in samples_by_video[video_id]]

    write_probe_csv(train_samples, train_path)
    write_probe_csv(val_samples, val_path)

    coarse_train_samples = [
        (clip_path, coarse_label_id)
        for clip_path, label_id in train_samples
        for coarse_label_id in [map_sitl_probe_id_to_cholec80_id(label_id)]
        if coarse_label_id is not None
    ]
    coarse_val_samples = [
        (clip_path, coarse_label_id)
        for clip_path, label_id in val_samples
        for coarse_label_id in [map_sitl_probe_id_to_cholec80_id(label_id)]
        if coarse_label_id is not None
    ]

    if not args.skip_coarse_export:
        write_probe_csv(coarse_train_samples, coarse_train_path)
        write_probe_csv(coarse_val_samples, coarse_val_path)

    metadata = {
        "input_root": str(args.input_root),
        "annotations_csv": str(annotations_csv),
        "train_csv": str(train_path),
        "val_csv": str(val_path),
        "split_seed": args.seed,
        "val_ratio": args.val_ratio,
        "split_strategy": split_strategy,
        "train_videos": train_videos,
        "val_videos": val_videos,
        "num_classes": 12,
        "phase_to_id": PHASE_TO_ID,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "raw_label_counts": dict(sorted(raw_label_counts.items())),
        "canonical_label_counts": dict(sorted(canonical_label_counts.items())),
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "train_phase_counts": phase_counts(train_samples),
        "val_phase_counts": phase_counts(val_samples),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not args.skip_coarse_export:
        coarse_metadata = {
            "source_train_csv": str(train_path),
            "source_val_csv": str(val_path),
            "train_csv": str(coarse_train_path),
            "val_csv": str(coarse_val_path),
            "num_classes": len(CHOLEC80_PHASE_TO_ID),
            "phase_to_id": CHOLEC80_PHASE_TO_ID,
            "source_num_classes": 11,
            "train_samples": len(coarse_train_samples),
            "val_samples": len(coarse_val_samples),
            "train_phase_counts": coarse_phase_counts(coarse_train_samples),
            "val_phase_counts": coarse_phase_counts(coarse_val_samples),
            "dropped_unmapped_probe_ids": [
                label_id
                for label_id in sorted({label_id for _, label_id in train_samples + val_samples})
                if map_sitl_probe_id_to_cholec80_id(label_id) is None
            ],
        }
        coarse_metadata_path.write_text(json.dumps(coarse_metadata, indent=2), encoding="utf-8")

    print(f"Wrote train CSV: {train_path}")
    print(f"Wrote val CSV:   {val_path}")
    print(f"Wrote metadata:  {metadata_path}")
    if not args.skip_coarse_export:
        print(f"Wrote coarse train CSV: {coarse_train_path}")
        print(f"Wrote coarse val CSV:   {coarse_val_path}")
        print(f"Wrote coarse metadata:  {coarse_metadata_path}")
    print()
    print(f"Videos: train={len(train_videos)} val={len(val_videos)} total={len(samples_by_video)}")
    print(f"Samples: train={len(train_samples)} val={len(val_samples)} total={len(train_samples) + len(val_samples)}")
    if skipped_counts:
        print(f"Skipped rows: {dict(sorted(skipped_counts.items()))}")
    print("Canonical label counts:")
    for phase_name, count in sorted(canonical_label_counts.items(), key=lambda x: PHASE_TO_ID[x[0]]):
        print(f"  {PHASE_TO_ID[phase_name]:>2} {phase_name}: {count}")


if __name__ == "__main__":
    main()
