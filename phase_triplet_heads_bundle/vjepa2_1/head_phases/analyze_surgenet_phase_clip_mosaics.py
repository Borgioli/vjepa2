#!/usr/bin/env python3
"""Run the phase clip audit tool on native Surgenet labels."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


PYTHON_BIN = "python3"
BASE_DIR = Path("/path/to/phase_triplet_heads_bundle/vjepa2_1")
SCRIPT_PATH = BASE_DIR / "head_phases" / "analyze_phase_clip_mosaics.py"
NATIVE_TRAIN_CSV = Path("/path/to/phase_triplet_heads_bundle/globus/surgenet_phases/yt_robotic_chole_native11_phases_train.csv")
NATIVE_VAL_CSV = Path("/path/to/phase_triplet_heads_bundle/globus/surgenet_phases/yt_robotic_chole_native11_phases_val.csv")
NATIVE_METADATA = BASE_DIR / "app" / "csv_head_models" / "yt_robotic_chole_native11_phases_native_metadata.json"
REDUCED6_METADATA = BASE_DIR / "app" / "csv_head_models" / "yt_robotic_chole_native11_phases_reduced6_metadata.json"
REDUCED6_TRANSFORM = BASE_DIR / "app" / "csv_head_models" / "yt_robotic_chole_native11_to_reduced6_transform.json"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-space",
        choices=("native11", "reduced6"),
        default="native11",
        help="Audit native Surgenet labels directly or through the reduced 6-class transform.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=NATIVE_TRAIN_CSV,
        help="Native Surgenet train CSV to audit.",
    )
    parser.add_argument(
        "--val-csv",
        type=Path,
        default=NATIVE_VAL_CSV,
        help="Native Surgenet val CSV to audit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for the audit output directory.",
    )
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = (
            BASE_DIR / "app" / "phase_clip_audit_surgenet"
            if args.label_space == "native11"
            else BASE_DIR / "app" / "phase_clip_audit_surgenet_reduced6"
        )

    cmd = [
        PYTHON_BIN,
        str(SCRIPT_PATH),
        "--csv",
        str(args.train_csv),
        str(args.val_csv),
        "--metadata",
        str(NATIVE_METADATA if args.label_space == "native11" else REDUCED6_METADATA),
        "--output-dir",
        str(output_dir),
    ]
    if args.label_space == "reduced6":
        cmd.extend(["--label-transform-json", str(REDUCED6_TRANSFORM)])

    os.execv(PYTHON_BIN, [*cmd, *passthrough])


if __name__ == "__main__":
    main()
