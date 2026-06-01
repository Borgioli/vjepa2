#!/usr/bin/env python3
"""Build single-tool multi-label sidecars from per-tool local triplet CSVs.

The source CSVs use the compact format exported beside the Globus data:

    <clip_path> <comma-separated-local-label-ids>

This script keeps those local labels as the source of truth, expands each local
id through the mapping JSON, and writes headered multi-hot lookup CSVs plus
runtime CSVs compatible with the existing frozen-head training wrappers.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, OrderedDict
from pathlib import Path


DEFAULT_TOOL_CSV_ROOT = Path("/path/to/data_root/globus/surgenet_triplets/tool_csv")
DEFAULT_MAPPING_JSON = DEFAULT_TOOL_CSV_ROOT / "yt_robotic_chole_per_tool_4s_stride1s_mapping.json"
DEFAULT_OUTPUT_ROOT = Path("/path/to/data_root/vjepa2_1/app/csv_head_models/single_tool")
DEFAULT_CLIPS_ROOT = Path(
    "/path/to/data_root/globus/surgenet_triplets/yt_robotic_chole_tool_windows_clips"
)

COMPONENTS = ("tool", "action", "target")
LABEL_FIELDS = {
    "tool": "tool_multihot",
    "action": "action_multihot",
    "target": "target_multihot",
}
SEQUENCE_LABEL_FIELDS = {
    "tool": "tool_multihots",
    "action": "action_multihots",
    "target": "target_multihots",
}
CONDITION_FIELDS = {
    "tool": "conditioning_tool_ids",
    "action": "conditioning_action_ids",
    "target": "conditioning_target_ids",
}
CONDITION_NAME_FIELDS = {
    "tool": "conditioning_tool_names",
    "action": "conditioning_action_names",
    "target": "conditioning_target_names",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", default="hook", help="Tool slug inside the mapping JSON.")
    parser.add_argument("--mapping-json", type=Path, default=DEFAULT_MAPPING_JSON)
    parser.add_argument("--tool-csv-root", type=Path, default=DEFAULT_TOOL_CSV_ROOT)
    parser.add_argument("--train-input", type=Path)
    parser.add_argument("--val-input", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--clips-root", type=Path, default=DEFAULT_CLIPS_ROOT)
    parser.add_argument(
        "--skip-clips-root-check",
        action="store_true",
        help="Do not require every source clip path to live under --clips-root.",
    )
    return parser.parse_args()


def parse_label_name(value: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in str(value).split("|")]
    if len(parts) != 3 or any(part == "" for part in parts):
        raise ValueError(f"Expected '<tool> | <action> | <target>', got: {value!r}")
    return parts[0], parts[1], parts[2]


def ordered_unique(values: list[str]) -> list[str]:
    seen = set()
    unique_values = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def load_tool_mapping(mapping_path: Path, tool: str) -> dict:
    with mapping_path.open() as handle:
        metadata = json.load(handle)

    tools = metadata.get("tools", {})
    if tool not in tools:
        raise ValueError(f"Tool {tool!r} not found in mapping JSON {mapping_path}")

    raw_label_id_to_name = tools[tool].get("label_id_to_name", {})
    if not raw_label_id_to_name:
        raise ValueError(f"No label_id_to_name mapping for tool {tool!r}")

    label_id_to_name = {
        int(label_id): str(label_name)
        for label_id, label_name in raw_label_id_to_name.items()
    }
    local_ids = sorted(label_id_to_name)
    label_id_to_components = {
        label_id: parse_label_name(label_id_to_name[label_id]) for label_id in local_ids
    }

    class_names = {}
    component_name_to_id = {}
    for component_index, component in enumerate(COMPONENTS):
        names = ordered_unique(
            [label_id_to_components[label_id][component_index] for label_id in local_ids]
        )
        class_names[component] = names
        component_name_to_id[component] = {name: idx for idx, name in enumerate(names)}

    label_id_to_component_ids = {}
    for label_id in local_ids:
        component_names = label_id_to_components[label_id]
        label_id_to_component_ids[label_id] = tuple(
            component_name_to_id[component][component_names[idx]]
            for idx, component in enumerate(COMPONENTS)
        )

    return {
        "source_metadata": metadata,
        "tool_metadata": tools[tool],
        "label_id_to_name": label_id_to_name,
        "label_id_to_components": label_id_to_components,
        "label_id_to_component_ids": label_id_to_component_ids,
        "class_names": class_names,
        "component_name_to_id": component_name_to_id,
    }


def default_input_path(tool_csv_root: Path, tool: str, split: str) -> Path:
    return tool_csv_root / f"yt_robotic_chole_tool_{tool}_{split}.csv"


def parse_source_csv(path: Path, known_label_ids: set[int], clips_root: Path, check_clips_root: bool) -> list[dict]:
    rows = []
    seen_paths = set()
    clips_root_text = str(clips_root)

    with path.open() as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                clip_path, label_blob = line.rsplit(maxsplit=1)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: expected '<clip_path> <label_ids>'") from exc

            if check_clips_root and not clip_path.startswith(clips_root_text):
                raise ValueError(
                    f"{path}:{line_number}: clip path is outside expected clips root "
                    f"{clips_root_text}: {clip_path}"
                )
            if clip_path in seen_paths:
                raise ValueError(f"{path}:{line_number}: duplicate clip path: {clip_path}")
            seen_paths.add(clip_path)

            label_ids = []
            for token in label_blob.split(","):
                token = token.strip()
                if not token:
                    continue
                label_id = int(token)
                if label_id not in known_label_ids:
                    raise ValueError(f"{path}:{line_number}: unknown local label id {label_id}")
                label_ids.append(label_id)
            if not label_ids:
                raise ValueError(f"{path}:{line_number}: no usable local label ids")

            rows.append(
                {
                    "clip_path": clip_path,
                    "local_label_ids": sorted(set(label_ids)),
                    "source_label_blob": label_blob,
                }
            )

    if not rows:
        raise ValueError(f"No rows found in source CSV: {path}")
    return rows


def encode_multihot(active_ids: list[int], length: int) -> str:
    vector = ["0"] * length
    for active_id in active_ids:
        if active_id < 0 or active_id >= length:
            raise ValueError(f"Class id {active_id} is outside multi-hot length {length}")
        vector[active_id] = "1"
    return " ".join(vector)


def format_id_list(values: list[int]) -> str:
    return " ".join(str(value) for value in values)


def format_name_list(values: list[str]) -> str:
    return ";".join(values)


def triplet_ids_to_text(triplet_ids: list[tuple[int, int, int]]) -> str:
    return ";".join(f"{tool_id} {action_id} {target_id}" for tool_id, action_id, target_id in triplet_ids)


def build_unconditioned_rows(source_rows: list[dict], mapping: dict) -> list[dict]:
    output_rows = []
    class_names = mapping["class_names"]
    label_id_to_component_ids = mapping["label_id_to_component_ids"]
    label_id_to_name = mapping["label_id_to_name"]

    for row in source_rows:
        local_label_ids = row["local_label_ids"]
        active_by_component = {component: set() for component in COMPONENTS}
        active_triplet_component_ids = []
        for local_id in local_label_ids:
            component_ids = label_id_to_component_ids[local_id]
            active_triplet_component_ids.append(component_ids)
            for idx, component in enumerate(COMPONENTS):
                active_by_component[component].add(component_ids[idx])

        output_row = {
            "clip_path": row["clip_path"],
            "active_local_label_ids": ",".join(str(label_id) for label_id in local_label_ids),
            "active_label_names": "|".join(label_id_to_name[label_id] for label_id in local_label_ids),
            "active_triplet_component_ids": triplet_ids_to_text(active_triplet_component_ids),
            "num_active_local_labels": len(local_label_ids),
        }
        for component in COMPONENTS:
            output_row[LABEL_FIELDS[component]] = encode_multihot(
                sorted(active_by_component[component]),
                len(class_names[component]),
            )
        output_rows.append(output_row)

    return output_rows


def build_conditioned_rows(
    source_rows: list[dict],
    mapping: dict,
    label_component: str,
    condition_components: tuple[str, str],
) -> list[dict]:
    label_component_index = COMPONENTS.index(label_component)
    condition_component_indices = [COMPONENTS.index(component) for component in condition_components]
    label_field = SEQUENCE_LABEL_FIELDS[label_component]
    class_names = mapping["class_names"]
    label_id_to_component_ids = mapping["label_id_to_component_ids"]
    label_id_to_name = mapping["label_id_to_name"]
    output_rows = []

    for row in source_rows:
        triplets_by_condition: OrderedDict[tuple[int, int], list[int]] = OrderedDict()
        for local_id in row["local_label_ids"]:
            component_ids = label_id_to_component_ids[local_id]
            condition_key = tuple(component_ids[idx] for idx in condition_component_indices)
            triplets_by_condition.setdefault(condition_key, []).append(local_id)

        condition_ids = {component: [] for component in condition_components}
        condition_names = {component: [] for component in condition_components}
        label_multihots = []
        active_label_ids_by_condition = []
        active_label_names_by_condition = []
        active_triplet_ids_by_condition = []
        active_local_ids_by_condition = []
        active_local_names_by_condition = []

        for condition_key, local_ids in triplets_by_condition.items():
            active_label_ids = sorted(
                {label_id_to_component_ids[local_id][label_component_index] for local_id in local_ids}
            )
            active_label_names = [class_names[label_component][label_id] for label_id in active_label_ids]
            label_multihots.append(
                encode_multihot(active_label_ids, len(class_names[label_component]))
            )
            active_label_ids_by_condition.append(format_id_list(active_label_ids))
            active_label_names_by_condition.append(format_name_list(active_label_names))
            active_triplet_ids_by_condition.append(
                triplet_ids_to_text([label_id_to_component_ids[local_id] for local_id in local_ids])
            )
            active_local_ids_by_condition.append(",".join(str(local_id) for local_id in local_ids))
            active_local_names_by_condition.append("|".join(label_id_to_name[local_id] for local_id in local_ids))

            for component, condition_id in zip(condition_components, condition_key):
                condition_ids[component].append(condition_id)
                condition_names[component].append(class_names[component][condition_id])

        output_row = {
            "clip_path": row["clip_path"],
            label_field: ";".join(label_multihots),
            f"active_{label_component}_ids_by_condition": "|".join(active_label_ids_by_condition),
            f"active_{label_component}_names_by_condition": "|".join(active_label_names_by_condition),
            "active_triplet_component_ids_by_condition": "|".join(active_triplet_ids_by_condition),
            "active_local_label_ids_by_condition": "|".join(active_local_ids_by_condition),
            "active_local_label_names_by_condition": "||".join(active_local_names_by_condition),
            "num_condition_rows": len(label_multihots),
            "rows_with_multiple_labels": sum(
                1 for value in active_label_ids_by_condition if len(value.split()) > 1
            ),
            "max_active_labels_per_condition": max(
                len(value.split()) for value in active_label_ids_by_condition
            ),
            "num_active_local_labels": len(row["local_label_ids"]),
        }
        for component in condition_components:
            output_row[CONDITION_FIELDS[component]] = format_id_list(condition_ids[component])
            output_row[CONDITION_NAME_FIELDS[component]] = format_name_list(condition_names[component])
        output_rows.append(output_row)

    return output_rows


def unconditioned_fieldnames() -> list[str]:
    return [
        "clip_path",
        "tool_multihot",
        "action_multihot",
        "target_multihot",
        "active_local_label_ids",
        "active_label_names",
        "active_triplet_component_ids",
        "num_active_local_labels",
    ]


def conditioned_fieldnames(label_component: str, condition_components: tuple[str, str]) -> list[str]:
    fieldnames = ["clip_path"]
    for component in condition_components:
        fieldnames.extend([CONDITION_FIELDS[component], CONDITION_NAME_FIELDS[component]])
    fieldnames.extend(
        [
            SEQUENCE_LABEL_FIELDS[label_component],
            f"active_{label_component}_ids_by_condition",
            f"active_{label_component}_names_by_condition",
            "active_triplet_component_ids_by_condition",
            "active_local_label_ids_by_condition",
            "active_local_label_names_by_condition",
            "num_condition_rows",
            "rows_with_multiple_labels",
            "max_active_labels_per_condition",
            "num_active_local_labels",
        ]
    )
    return fieldnames


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_runtime(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=" ", quoting=csv.QUOTE_MINIMAL)
        for row in rows:
            writer.writerow([row["clip_path"], 0])


def split_summary(rows: list[dict], mapping: dict) -> dict:
    component_counts = {component: Counter() for component in COMPONENTS}
    label_counts = Counter()
    label_id_to_component_ids = mapping["label_id_to_component_ids"]

    for row in rows:
        for local_id in row["local_label_ids"]:
            label_counts[str(local_id)] += 1
            component_ids = label_id_to_component_ids[local_id]
            for idx, component in enumerate(COMPONENTS):
                component_counts[component][str(component_ids[idx])] += 1

    summary = {
        "num_rows": len(rows),
        "num_unique_clips": len({row["clip_path"] for row in rows}),
        "local_label_row_counts": {
            str(label_id): label_counts.get(str(label_id), 0)
            for label_id in sorted(mapping["label_id_to_name"])
        },
    }
    for component in COMPONENTS:
        summary[f"{component}_positive_counts"] = {
            str(idx): component_counts[component].get(str(idx), 0)
            for idx in range(len(mapping["class_names"][component]))
        }
    return summary


def conditioned_summary(rows: list[dict], mapping: dict, label_component: str, condition_components: tuple[str, str]) -> dict:
    label_field = SEQUENCE_LABEL_FIELDS[label_component]
    label_counts = Counter()
    condition_counts = {component: Counter() for component in condition_components}
    combo_counts = Counter()
    condition_rows_per_clip = []
    active_labels_per_condition = []

    for row in rows:
        condition_columns = []
        for component in condition_components:
            values = [int(token) for token in row[CONDITION_FIELDS[component]].split()]
            condition_columns.append(values)

        num_condition_rows = len(condition_columns[0])
        condition_rows_per_clip.append(num_condition_rows)
        for condition_index in range(num_condition_rows):
            combo_parts = []
            for component, values in zip(condition_components, condition_columns):
                condition_id = values[condition_index]
                condition_counts[component][str(condition_id)] += 1
                combo_parts.append(f"{component}={condition_id}")
            combo_counts["|".join(combo_parts)] += 1

        for vector_text in row[label_field].split(";"):
            active_ids = [
                idx for idx, token in enumerate(vector_text.split()) if token == "1"
            ]
            active_labels_per_condition.append(len(active_ids))
            for active_id in active_ids:
                label_counts[str(active_id)] += 1

    summary = {
        "num_rows": len(rows),
        "num_unique_clips": len({row["clip_path"] for row in rows}),
        "num_condition_rows": sum(condition_rows_per_clip),
        "max_condition_rows_per_clip": max(condition_rows_per_clip),
        "avg_condition_rows_per_clip": round(sum(condition_rows_per_clip) / len(condition_rows_per_clip), 4),
        f"{label_component}_positive_counts": {
            str(idx): label_counts.get(str(idx), 0)
            for idx in range(len(mapping["class_names"][label_component]))
        },
        "condition_rows_with_multiple_labels": sum(count > 1 for count in active_labels_per_condition),
        "max_active_labels_per_condition": max(active_labels_per_condition),
        "avg_active_labels_per_condition": round(
            sum(active_labels_per_condition) / len(active_labels_per_condition),
            4,
        ),
        "conditioning_combo_counts": dict(sorted(combo_counts.items())),
    }
    for component in condition_components:
        summary[f"conditioning_{component}_counts"] = {
            str(idx): condition_counts[component].get(str(idx), 0)
            for idx in range(len(mapping["class_names"][component]))
        }
    return summary


def output_paths(output_root: Path, tool: str, label_component: str | None = None, condition_components: tuple[str, str] | None = None) -> dict[str, Path]:
    base = f"yt_robotic_chole_tool_{tool}"
    if label_component is None:
        prefix = f"{base}_multilabel"
    else:
        condition_suffix = "_".join(condition_components or ())
        prefix = f"{base}_{label_component}_multilabel_conditioned_on_{condition_suffix}"
    return {
        "train": output_root / f"{prefix}_train.csv",
        "val": output_root / f"{prefix}_val.csv",
        "train_runtime": output_root / f"{prefix}_train_runtime.csv",
        "val_runtime": output_root / f"{prefix}_val_runtime.csv",
    }


def write_metadata(
    path: Path,
    args: argparse.Namespace,
    mapping: dict,
    unconditioned_paths: dict[str, Path],
    conditioned_paths: dict[str, dict[str, Path]],
    train_rows: list[dict],
    val_rows: list[dict],
    conditioned_train_rows: dict[str, list[dict]],
    conditioned_val_rows: dict[str, list[dict]],
) -> None:
    class_names = mapping["class_names"]
    label_id_to_name = mapping["label_id_to_name"]
    label_id_to_component_ids = mapping["label_id_to_component_ids"]
    metadata = {
        "description": (
            "Single-tool hook sidecars derived from the per-tool 4s/stride-1s "
            "local triplet CSVs. Component class ids are ordered by first "
            "appearance in the tool mapping's numeric local-label order."
        ),
        "tool": args.tool,
        "source_files": {
            "mapping_json": str(args.mapping_json),
            "train_input": str(args.train_input),
            "val_input": str(args.val_input),
            "clips_root": str(args.clips_root),
        },
        "output_files": {
            "unconditioned": {key: str(value) for key, value in unconditioned_paths.items()},
            "conditioned": {
                key: {path_key: str(path_value) for path_key, path_value in paths.items()}
                for key, paths in conditioned_paths.items()
            },
        },
        "format": {
            "runtime_csv": "space-delimited clip_path dummy_label",
            "unconditioned_lookup": "one row per clip_path with tool/action/target multi-hot columns",
            "conditioned_lookup": (
                "one row per clip_path; condition id columns are space-delimited "
                "lists and label columns are semicolon-delimited multi-hot vectors "
                "aligned with the condition rows"
            ),
        },
        "component_class_names": {
            component: {str(idx): name for idx, name in enumerate(names)}
            for component, names in class_names.items()
        },
        "local_label_id_to_name": {
            str(label_id): label_id_to_name[label_id] for label_id in sorted(label_id_to_name)
        },
        "local_label_id_to_component_ids": {
            str(label_id): {
                component: label_id_to_component_ids[label_id][idx]
                for idx, component in enumerate(COMPONENTS)
            }
            for label_id in sorted(label_id_to_name)
        },
        "split_stats": {
            "train": split_summary(train_rows, mapping),
            "val": split_summary(val_rows, mapping),
        },
        "conditioned_split_stats": {},
    }
    for key, rows in conditioned_train_rows.items():
        label_component, _, condition_suffix = key.partition("__")
        condition_components = tuple(condition_suffix.split("_"))
        metadata["conditioned_split_stats"][key] = {
            "train": conditioned_summary(rows, mapping, label_component, condition_components),
            "val": conditioned_summary(conditioned_val_rows[key], mapping, label_component, condition_components),
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as json_file:
        json.dump(metadata, json_file, indent=2)
        json_file.write("\n")


def main() -> None:
    args = parse_args()
    args.train_input = args.train_input or default_input_path(args.tool_csv_root, args.tool, "train")
    args.val_input = args.val_input or default_input_path(args.tool_csv_root, args.tool, "val")

    mapping = load_tool_mapping(args.mapping_json, args.tool)
    known_label_ids = set(mapping["label_id_to_name"])
    check_clips_root = not args.skip_clips_root_check

    train_source_rows = parse_source_csv(args.train_input, known_label_ids, args.clips_root, check_clips_root)
    val_source_rows = parse_source_csv(args.val_input, known_label_ids, args.clips_root, check_clips_root)

    train_unconditioned_rows = build_unconditioned_rows(train_source_rows, mapping)
    val_unconditioned_rows = build_unconditioned_rows(val_source_rows, mapping)

    unconditioned_paths = output_paths(args.output_root, args.tool)
    write_csv(unconditioned_paths["train"], train_unconditioned_rows, unconditioned_fieldnames())
    write_csv(unconditioned_paths["val"], val_unconditioned_rows, unconditioned_fieldnames())
    write_runtime(unconditioned_paths["train_runtime"], train_unconditioned_rows)
    write_runtime(unconditioned_paths["val_runtime"], val_unconditioned_rows)

    conditioned_specs = {
        "tool__action_target": ("tool", ("action", "target")),
        "action__tool_target": ("action", ("tool", "target")),
        "target__tool_action": ("target", ("tool", "action")),
    }
    conditioned_paths = {}
    conditioned_train_rows = {}
    conditioned_val_rows = {}
    for key, (label_component, condition_components) in conditioned_specs.items():
        train_rows = build_conditioned_rows(
            train_source_rows,
            mapping,
            label_component,
            condition_components,
        )
        val_rows = build_conditioned_rows(
            val_source_rows,
            mapping,
            label_component,
            condition_components,
        )
        paths = output_paths(args.output_root, args.tool, label_component, condition_components)
        write_csv(paths["train"], train_rows, conditioned_fieldnames(label_component, condition_components))
        write_csv(paths["val"], val_rows, conditioned_fieldnames(label_component, condition_components))
        write_runtime(paths["train_runtime"], train_rows)
        write_runtime(paths["val_runtime"], val_rows)
        conditioned_paths[key] = paths
        conditioned_train_rows[key] = train_rows
        conditioned_val_rows[key] = val_rows

    metadata_path = args.output_root / f"yt_robotic_chole_tool_{args.tool}_single_tool_metadata.json"
    write_metadata(
        metadata_path,
        args,
        mapping,
        unconditioned_paths,
        conditioned_paths,
        train_source_rows,
        val_source_rows,
        conditioned_train_rows,
        conditioned_val_rows,
    )

    print(f"Wrote unconditioned train lookup: {unconditioned_paths['train']}")
    print(f"Wrote unconditioned val lookup:   {unconditioned_paths['val']}")
    print(f"Wrote unconditioned train runtime:{unconditioned_paths['train_runtime']}")
    print(f"Wrote unconditioned val runtime:  {unconditioned_paths['val_runtime']}")
    for key, paths in conditioned_paths.items():
        print(f"Wrote {key} train lookup: {paths['train']}")
        print(f"Wrote {key} val lookup:   {paths['val']}")
    print(f"Wrote metadata JSON:              {metadata_path}")
    print(f"Tool classes:                     {mapping['class_names']['tool']}")
    print(f"Action classes:                   {mapping['class_names']['action']}")
    print(f"Target classes:                   {mapping['class_names']['target']}")
    print(f"Train / val rows:                 {len(train_source_rows)} / {len(val_source_rows)}")


if __name__ == "__main__":
    main()
