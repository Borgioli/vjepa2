#!/usr/bin/env python3
"""Shared helpers for triplet and tool audit scripts."""

from __future__ import annotations

import json
import re
from pathlib import Path


DEFAULT_TRIPLET_TRAIN_CSV = Path(
    "/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/triplet_train_supported.csv"
)
DEFAULT_TRIPLET_VAL_CSV = Path(
    "/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/triplet_val_supported.csv"
)
DEFAULT_MULTILABEL_TRAIN_CSV = Path(
    "/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/triplet_multilabel_train_supported.csv"
)
DEFAULT_MULTILABEL_VAL_CSV = Path(
    "/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/triplet_multilabel_val_supported.csv"
)
DEFAULT_TRIPLET_METADATA = Path(
    "/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/triplet_multilabel_supported_metadata.json"
)
DEFAULT_TRIPLET_AUDIT_OUTPUT_DIR = Path(
    "/path/to/phase_triplet_heads_bundle/vjepa2_1/app/triplet_clip_audit"
)
DEFAULT_TOOL_AUDIT_OUTPUT_ROOT = Path(
    "/path/to/phase_triplet_heads_bundle/vjepa2_1/app/tool_clip_audit"
)

TOOL_ID_TO_NAME = {
    0: "bipolar",
    1: "clipper",
    2: "grasper",
    3: "harmonic shears",
    4: "hook",
    5: "irrigator",
    6: "needle driver",
    7: "scissors",
    8: "stapler",
}

VERB_ID_TO_NAME = {
    0: "coagulation",
    1: "grasp/retract",
    2: "null",
    3: "clip",
    4: "cut",
    5: "dissect",
    6: "clean",
}

TARGET_ID_TO_NAME = {
    0: "connective tissue",
    1: "cystic duct",
    2: "adhesion",
    3: "cystic pedicle",
    4: "gallbladder",
    5: "gallbladder wall",
    6: "liver",
    7: "null",
    8: "peritoneum",
    9: "suture",
    10: "cystic artery",
    11: "fallciform ligament",
    12: "gut",
    13: "omentum",
    14: "specimen bag",
    15: "fluid",
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "unknown"


def infer_triplet_source_name(sample_path: Path) -> str:
    sample_str = str(sample_path)
    if "/globus/surgenet_triplets/" in sample_str:
        return "surgenet_triplets"
    if "/globus/triplet/" in sample_str:
        return "triplet"
    if "yt_robotic_chole" in sample_str:
        return "yt_robotic_chole"
    return "unknown"


def format_triplet_name(tool_id: int, verb_id: int, target_id: int) -> str:
    tool_name = TOOL_ID_TO_NAME.get(tool_id, f"tool_{tool_id}")
    verb_name = VERB_ID_TO_NAME.get(verb_id, f"verb_{verb_id}")
    target_name = TARGET_ID_TO_NAME.get(target_id, f"target_{target_id}")
    return f"{tool_name} | {verb_name} | {target_name}"


def _load_task_new_to_old(metadata: dict, task_name: str) -> dict[int, int]:
    task_mapping = metadata.get("class_id_mappings", {}).get(task_name, {})
    new_to_old = task_mapping.get("new_to_old", {})
    if not new_to_old:
        return {}
    return {int(new_id): int(old_id) for new_id, old_id in new_to_old.items()}


def build_triplet_label_lookup(metadata_path: Path) -> dict[str, str]:
    metadata = load_triplet_metadata(metadata_path)
    tool_new_to_old = _load_task_new_to_old(metadata, "tool")
    verb_new_to_old = _load_task_new_to_old(metadata, "verb")
    target_new_to_old = _load_task_new_to_old(metadata, "target")

    triplet_labels: dict[str, str] = {}
    for item in metadata.get("id_to_triplet", []):
        tool_id = int(item["tool_id"])
        verb_id = int(item["verb_id"])
        target_id = int(item["target_id"])
        original_tool_id = tool_new_to_old.get(tool_id, tool_id)
        original_verb_id = verb_new_to_old.get(verb_id, verb_id)
        original_target_id = target_new_to_old.get(target_id, target_id)
        readable_label = format_triplet_name(original_tool_id, original_verb_id, original_target_id)
        triplet_labels[str(item["triplet_label"]).strip()] = readable_label
    return triplet_labels


def load_triplet_metadata(metadata_path: Path) -> dict:
    with metadata_path.open() as handle:
        return json.load(handle)


def load_triplet_id_to_tuple(metadata_path: Path) -> dict[int, tuple[int, int, int]]:
    metadata = load_triplet_metadata(metadata_path)
    triplet_map: dict[int, tuple[int, int, int]] = {}
    for item in metadata.get("id_to_triplet", []):
        triplet_id = int(item["triplet_id"])
        triplet_map[triplet_id] = (
            int(item["tool_id"]),
            int(item["verb_id"]),
            int(item["target_id"]),
        )
    if not triplet_map:
        raise ValueError(f"No id_to_triplet mapping found in {metadata_path}")
    return triplet_map


def load_triplet_tuple_to_id(metadata_path: Path) -> dict[tuple[int, int, int], int]:
    return {triplet: triplet_id for triplet_id, triplet in load_triplet_id_to_tuple(metadata_path).items()}


def build_triplet_class_names(metadata_path: Path) -> dict[int, str]:
    class_names: dict[int, str] = {}
    for triplet_id, triplet in load_triplet_id_to_tuple(metadata_path).items():
        class_names[triplet_id] = f"[{triplet_id}] {format_triplet_name(*triplet)}"
    return class_names
