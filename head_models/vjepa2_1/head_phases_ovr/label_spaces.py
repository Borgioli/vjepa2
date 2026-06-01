"""Shared label-space definitions for one-vs-rest phase experiments."""

from __future__ import annotations

import re
from typing import Dict

NATIVE11_PHASE_TO_ID: Dict[str, int] = {
    "Clipping of the cystic duct": 0,
    "Clipping the cystic artery": 1,
    "Dissection of the gallbladder from the liver": 2,
    "Division of the cystic duct and artery": 3,
    "Exposure of the working area": 4,
    "Isolation of the cystic artery": 5,
    "Isolation of the cystic duct": 6,
    "Opening the anterior peritoneal layer of the triangle of Calot": 7,
    "Opening the posterior peritoneal layer of the triangle of Calot": 8,
    "Retraction of the gallbladder neck": 9,
    "Specimen retrieval": 10,
}

NATIVE11_ID_TO_PHASE: Dict[int, str] = {label_id: name for name, label_id in NATIVE11_PHASE_TO_ID.items()}

REDUCED6_PHASE_TO_ID: Dict[str, int] = {
    "Clipping of the cystic duct/artery": 0,
    "Dissection of the gallbladder from the liver": 1,
    "Division of the cystic duct and artery": 2,
    "Exposure of the working area": 3,
    "Isolation of the cystic artery/duct": 4,
    "Opening the anterior/posterior peritoneal layer of the triangle of Calot": 5,
}

REDUCED6_ID_TO_PHASE: Dict[int, str] = {label_id: name for name, label_id in REDUCED6_PHASE_TO_ID.items()}

NATIVE11_TO_REDUCED6: Dict[int, int | None] = {
    0: 0,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 4,
    7: 5,
    8: 5,
    9: None,
    10: None,
}


def get_class_names(label_space: str) -> Dict[int, str]:
    if label_space == "native11":
        return dict(NATIVE11_ID_TO_PHASE)
    if label_space == "reduced6":
        return dict(REDUCED6_ID_TO_PHASE)
    raise ValueError(f"Unsupported label space: {label_space}")


def remap_label(raw_label: int, label_space: str) -> int | None:
    if label_space == "native11":
        if raw_label not in NATIVE11_ID_TO_PHASE:
            raise ValueError(f"Unknown native11 label id: {raw_label}")
        return raw_label
    if label_space == "reduced6":
        if raw_label not in NATIVE11_TO_REDUCED6:
            raise ValueError(f"Unknown native11 label id: {raw_label}")
        return NATIVE11_TO_REDUCED6[raw_label]
    raise ValueError(f"Unsupported label space: {label_space}")


def slugify_phase_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "phase"
