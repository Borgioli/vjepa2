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
                r"/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/ovr_phase_heads_surgenet_only/reduced6/04_isolation-of-the-cystic-artery-duct_train.csv",
                "--val-csv",
                r"/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/ovr_phase_heads_surgenet_only/reduced6/04_isolation-of-the-cystic-artery-duct_val.csv",
                "--target-class-id",
                "4",
                "--target-class-name",
                "Isolation of the cystic artery/duct",
                "--label-space",
                "reduced6",
                "--config",
                r"/path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads_ovr_surgenet_only/reduced6/phase_ovr_reduced6_04_isolation-of-the-cystic-artery-duct.yaml",
                "--output-dir",
                r"/path/to/phase_triplet_heads_bundle/vjepa2_1/app/phase_clip_audit_ovr_surgenet_only/reduced6/04_isolation-of-the-cystic-artery-duct",
                *sys.argv[1:],
            ]
        )
    )
