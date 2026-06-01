"""
Map SITL surgical phase labels into the Cholec80 phase taxonomy.

This module intentionally performs exact matching only. It does not
normalize spelling, casing, or whitespace. Any input that does not exactly
match one of the supported SITL labels returns UNMAPPED.
"""

from __future__ import annotations

import argparse
import json


UNMAPPED = "UNMAPPED"

CHOLEC80_PHASE_TO_ID = {
    "Preparation": 0,
    "GallbladderRetraction": 1,
    "CalotTriangleDissection": 2,
    "ClippingCutting": 3,
    "GallbladderDissection": 4,
}

SITL_PHASE_TO_ID = {
    "None": 0,
    "Exposure of the working area": 1,
    "Retraction of the gallbladder neck": 2,
    "Opening the anterior peritoneal layer of the triangle of Calot": 3,
    "Opening the posterior peritoneal layer of the triangle of Calot": 4,
    "Isolation of the cystic duct": 5,
    "Isolation of the cystic artery": 6,
    "clipping of the cystic duct": 7,
    "clipping of the cystic artery": 8,
    "Division of the cystic duct": 9,
    "Division of the cystic artery": 9,
    "dissection of the gallbladder from the liver": 10,
    "specimen retrieval": 11,
}

SITL_TO_CHOLEC80_PHASE = {
    "None": "Preparation",
    "Retraction of the gallbladder neck": "GallbladderRetraction",
    "Opening the anterior peritoneal layer of the triangle of Calot": "CalotTriangleDissection",
    "Opening the posterior peritoneal layer of the triangle of Calot": "CalotTriangleDissection",
    "Isolation of the cystic duct": "CalotTriangleDissection",
    "Isolation of the cystic artery": "CalotTriangleDissection",
    "clipping of the cystic duct": "ClippingCutting",
    "clipping of the cystic artery": "ClippingCutting",
    "Division of the cystic duct": "ClippingCutting",
    "Division of the cystic artery": "ClippingCutting",
    "dissection of the gallbladder from the liver": "GallbladderDissection",
    "Exposure of the working area": "Preparation",
}

SITL_ID_TO_CHOLEC80_PHASE = {
    SITL_PHASE_TO_ID[sitl_label]: cholec80_label
    for sitl_label, cholec80_label in SITL_TO_CHOLEC80_PHASE.items()
    if sitl_label in SITL_PHASE_TO_ID
}

SITL_ID_TO_CHOLEC80_ID = {
    sitl_id: CHOLEC80_PHASE_TO_ID[cholec80_label]
    for sitl_id, cholec80_label in SITL_ID_TO_CHOLEC80_PHASE.items()
}

# Probe-training CSVs in this repo use the labeled SITL phases 1..11.
# "specimen retrieval" has no coarse Cholec80 equivalent here, so the
# 5-class mapping intentionally excludes probe label 11.
SITL_PROBE_ID_TO_CHOLEC80_ID = {
    sitl_id: cholec80_id
    for sitl_id, cholec80_id in SITL_ID_TO_CHOLEC80_ID.items()
    if sitl_id != 0
}

CHOLEC80_ID_TO_SITL_PROBE_IDS = {}
for sitl_probe_id, cholec80_id in SITL_PROBE_ID_TO_CHOLEC80_ID.items():
    CHOLEC80_ID_TO_SITL_PROBE_IDS.setdefault(cholec80_id, []).append(sitl_probe_id)
CHOLEC80_ID_TO_SITL_PROBE_IDS = {
    cholec80_id: tuple(sorted(sitl_probe_ids))
    for cholec80_id, sitl_probe_ids in CHOLEC80_ID_TO_SITL_PROBE_IDS.items()
}

# Surgenet/native11 uses the zero-based version of the same 11-class fine
# phase space, so these are the allowed fine labels for each coarse class.
CHOLEC80_ID_TO_SITL_FINE_ZERO_BASED_IDS = {
    cholec80_id: tuple(sorted(sitl_probe_id - 1 for sitl_probe_id in sitl_probe_ids))
    for cholec80_id, sitl_probe_ids in CHOLEC80_ID_TO_SITL_PROBE_IDS.items()
}


def map_sitl_phase_label(label: str | None) -> str:
    """Map one SITL phase label to a Cholec80 phase label."""
    if label is None:
        return SITL_TO_CHOLEC80_PHASE["None"]
    return SITL_TO_CHOLEC80_PHASE.get(label, UNMAPPED)


def map_sitl_phase_id(label_id: int | None) -> int | None:
    """Map one SITL integer label id to a Cholec80 integer label id."""
    if label_id is None:
        return CHOLEC80_PHASE_TO_ID["Preparation"]
    return SITL_ID_TO_CHOLEC80_ID.get(int(label_id))


def map_sitl_probe_id_to_cholec80_id(label_id: int | None) -> int | None:
    """Map one probe-CSV SITL label id (1..11) to a 5-class Cholec80 id."""
    if label_id is None:
        return None
    return SITL_PROBE_ID_TO_CHOLEC80_ID.get(int(label_id))


def map_sitl_phase_labels(labels: list[str | None]) -> list[str]:
    """Map a list of SITL phase labels in order."""
    return [map_sitl_phase_label(label) for label in labels]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "labels",
        nargs="*",
        help="One or more SITL labels to map. When more than one label is provided, a JSON list is printed.",
    )
    parser.add_argument(
        "--json-list",
        type=str,
        default=None,
        help="Optional JSON list of SITL labels. Use null for None values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.json_list is not None:
        labels = json.loads(args.json_list)
        if not isinstance(labels, list):
            raise ValueError("--json-list must decode to a JSON list")
        print(json.dumps(map_sitl_phase_labels(labels)))
        return

    if not args.labels:
        raise SystemExit("Provide at least one label or pass --json-list.")

    if len(args.labels) == 1:
        print(map_sitl_phase_label(args.labels[0]))
        return

    print(json.dumps(map_sitl_phase_labels(args.labels)))


if __name__ == "__main__":
    main()
