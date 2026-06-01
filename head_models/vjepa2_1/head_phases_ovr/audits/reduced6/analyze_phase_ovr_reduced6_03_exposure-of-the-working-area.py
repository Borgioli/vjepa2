#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(r"/path/to/phase_triplet_heads_bundle/vjepa2_1")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from head_phases_ovr.analyze_ovr_phase_clip_mosaics import main as audit_main


if __name__ == "__main__":
    raise SystemExit(
        audit_main(
            [
                "--train-csv",
                r"/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/ovr_phase_heads/reduced6/03_exposure-of-the-working-area_train.csv",
                "--val-csv",
                r"/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/ovr_phase_heads/reduced6/03_exposure-of-the-working-area_val.csv",
                "--target-class-id",
                "3",
                "--target-class-name",
                "Exposure of the working area",
                "--label-space",
                "reduced6",
                "--config",
                r"/path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads_ovr/reduced6/phase_ovr_reduced6_03_exposure-of-the-working-area.yaml",
                "--output-dir",
                r"/path/to/phase_triplet_heads_bundle/vjepa2_1/app/phase_clip_audit_ovr/reduced6/03_exposure-of-the-working-area",
                *sys.argv[1:],
            ]
        )
    )
