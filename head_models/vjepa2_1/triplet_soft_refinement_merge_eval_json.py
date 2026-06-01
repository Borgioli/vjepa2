#!/path/to/data_root/vjepa2_1/.venv/bin/python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from triplet_soft_refinement_eval import (
    compute_bootstrap_ci,
    pack_bool_mask,
    unpack_bool_mask,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge sharded JSON outputs from triplet_soft_refinement_eval.py "
            "into one exact evaluation summary."
        )
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        nargs="+",
        required=True,
        help="One or more shard JSON files produced by triplet_soft_refinement_eval.py.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the merged summary as JSON.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help=(
            "Number of bootstrap iterations to run on the merged per-sample masks "
            "(if available across all shard JSONs). Set to 0 to disable. Default 1000."
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260101,
        help="Deterministic seed for the merged bootstrap resampling.",
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        help="Two-sided confidence level for merged bootstrap CIs.",
    )
    parser.add_argument(
        "--keep-per-sample-masks",
        action="store_true",
        help="Persist the merged per-sample masks in the output JSON.",
    )
    return parser.parse_args()


def summarize_multilabel_counts(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    exact_correct: int,
    total: int,
) -> dict[str, float]:
    tp = tp.detach().cpu().to(dtype=torch.float64).reshape(-1)
    fp = fp.detach().cpu().to(dtype=torch.float64).reshape(-1)
    fn = fn.detach().cpu().to(dtype=torch.float64).reshape(-1)

    per_class_denom = (2.0 * tp) + fp + fn
    per_class_f1 = torch.where(
        per_class_denom > 0,
        (2.0 * tp) / per_class_denom,
        torch.zeros_like(per_class_denom),
    )

    micro_tp = float(tp.sum().item())
    micro_fp = float(fp.sum().item())
    micro_fn = float(fn.sum().item())

    micro_precision = 0.0
    if (micro_tp + micro_fp) > 0:
        micro_precision = micro_tp / (micro_tp + micro_fp)

    micro_recall = 0.0
    if (micro_tp + micro_fn) > 0:
        micro_recall = micro_tp / (micro_tp + micro_fn)

    micro_f1 = 0.0
    if ((2.0 * micro_tp) + micro_fp + micro_fn) > 0:
        micro_f1 = (2.0 * micro_tp) / ((2.0 * micro_tp) + micro_fp + micro_fn)

    exact_acc = 0.0
    if total > 0:
        exact_acc = float(exact_correct) / float(total)

    return {
        "exact_acc": 100.0 * exact_acc,
        "macro_f1": 100.0 * float(per_class_f1.mean().item()),
        "micro_precision": 100.0 * micro_precision,
        "micro_recall": 100.0 * micro_recall,
        "micro_f1": 100.0 * micro_f1,
    }


def print_metrics_block(title: str, metrics: dict[str, float]) -> None:
    print(
        f"{title}: "
        f"Exact={metrics['exact_acc']:.4f}% "
        f"MacroF1={metrics['macro_f1']:.4f}% "
        f"MicroP={metrics['micro_precision']:.4f}% "
        f"MicroR={metrics['micro_recall']:.4f}% "
        f"MicroF1={metrics['micro_f1']:.4f}%"
    )


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def counts_from_payload(raw_counts: dict[str, Any], key: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    block = raw_counts[key]
    return (
        torch.tensor(block["tp"], dtype=torch.float64),
        torch.tensor(block["fp"], dtype=torch.float64),
        torch.tensor(block["fn"], dtype=torch.float64),
        int(block["exact_correct"]),
        int(block["total"]),
    )


def merge_metric_counts(shard_payloads: list[dict[str, Any]], key: str) -> dict[str, Any]:
    tp_total = None
    fp_total = None
    fn_total = None
    exact_correct_total = 0
    total_total = 0

    for payload in shard_payloads:
        tp, fp, fn, exact_correct, total = counts_from_payload(payload["raw_counts"], key)
        if tp_total is None:
            tp_total = tp
            fp_total = fp
            fn_total = fn
        else:
            if tp.shape != tp_total.shape:
                raise ValueError(f"Mismatched class count while merging '{key}' shard JSONs")
            tp_total += tp
            fp_total += fp
            fn_total += fn
        exact_correct_total += exact_correct
        total_total += total

    assert tp_total is not None
    summary = summarize_multilabel_counts(
        tp=tp_total,
        fp=fp_total,
        fn=fn_total,
        exact_correct=exact_correct_total,
        total=total_total,
    )
    return {
        "summary": summary,
        "raw_counts": {
            "tp": [float(value) for value in tp_total.tolist()],
            "fp": [float(value) for value in fp_total.tolist()],
            "fn": [float(value) for value in fn_total.tolist()],
            "exact_correct": int(exact_correct_total),
            "total": int(total_total),
        },
    }


def merge_per_sample_masks(
    shard_payloads: list[dict[str, Any]],
) -> dict[str, Any] | None:
    blocks = [payload.get("per_sample_masks") for payload in shard_payloads]
    if any(block is None for block in blocks):
        return None
    if not blocks:
        return None

    schema = blocks[0].get("schema")
    if schema != "packed_hex_v1":
        raise ValueError(f"Unsupported per_sample_masks schema across shards: {schema}")

    expected_num_classes = blocks[0]["num_classes"]
    for block in blocks[1:]:
        if block.get("schema") != schema:
            raise ValueError("Mismatched per_sample_masks schema across shards")
        if block.get("num_classes") != expected_num_classes:
            raise ValueError("Mismatched num_classes across shard per_sample_masks")

    seen: set[int] = set()
    merged_indices: list[int] = []
    merged: dict[str, dict[str, list[torch.Tensor]]] = {
        "predicted": {axis: [] for axis in ("tool", "verb", "target", "triplet")},
        "target": {axis: [] for axis in ("tool", "verb", "target", "triplet")},
    }

    for block in blocks:
        indices = block.get("global_indices") or []
        for local_idx, global_idx in enumerate(indices):
            global_idx = int(global_idx)
            if global_idx in seen:
                continue
            seen.add(global_idx)
            merged_indices.append(global_idx)
            for axis in ("tool", "verb", "target", "triplet"):
                num_bits = int(expected_num_classes[axis])
                pred_mask = unpack_bool_mask(
                    block["predicted"][axis][local_idx], num_bits=num_bits
                )
                target_mask = unpack_bool_mask(
                    block["target"][axis][local_idx], num_bits=num_bits
                )
                merged["predicted"][axis].append(pred_mask)
                merged["target"][axis].append(target_mask)

    return {
        "global_indices": merged_indices,
        "num_classes": expected_num_classes,
        "predicted": merged["predicted"],
        "target": merged["target"],
    }


def main() -> int:
    args = parse_args()
    input_paths = [path.resolve() for path in args.input_json]
    shard_payloads = []
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Shard JSON does not exist: {path}")
        shard_payloads.append(load_json(path))

    if not shard_payloads:
        raise RuntimeError("No shard JSON files were loaded")

    first = shard_payloads[0]
    settings_compare_keys = (
        "device",
        "refinement_steps",
        "blend_alpha",
        "route_topk_tool",
        "route_topk_verb",
        "route_topk_target",
        "support_aware_triplet_refinement",
        "calibration_json",
        "head_temperatures_is_identity",
        "tool_thresholds",
        "verb_thresholds",
        "target_thresholds",
        "route_tool_thresholds",
        "route_verb_thresholds",
        "route_target_thresholds",
    )
    for payload in shard_payloads[1:]:
        for key in ("config_paths", "checkpoint_paths", "eval_files"):
            if payload.get(key) != first.get(key):
                raise ValueError(f"Mismatched '{key}' across shard JSON files")
        first_settings = first.get("settings") or {}
        payload_settings = payload.get("settings") or {}
        for key in settings_compare_keys:
            if first_settings.get(key) != payload_settings.get(key):
                raise ValueError(
                    f"Mismatched setting '{key}' across shard JSON files: "
                    f"{first_settings.get(key)!r} vs {payload_settings.get(key)!r}"
                )

    merged_tool = merge_metric_counts(shard_payloads, "tool")
    merged_verb = merge_metric_counts(shard_payloads, "verb")
    merged_target = merge_metric_counts(shard_payloads, "target")
    merged_triplet = merge_metric_counts(shard_payloads, "triplet")

    total_samples = merged_tool["raw_counts"]["total"]
    joint_exact_correct = sum(int(payload["raw_counts"]["joint_exact_correct"]) for payload in shard_payloads)
    unsupported_predicted_triplets = sum(
        int(payload["raw_counts"]["unsupported_predicted_triplets"]) for payload in shard_payloads
    )
    joint_exact_acc = 0.0
    if total_samples > 0:
        joint_exact_acc = 100.0 * float(joint_exact_correct) / float(total_samples)

    shard_metadata = []
    all_indices: list[int] = []
    for payload, path in zip(shard_payloads, input_paths):
        sharding = payload.get("sharding", {})
        shard_metadata.append(
            {
                "input_json": str(path),
                "num_shards": sharding.get("num_shards"),
                "shard_index": sharding.get("shard_index"),
                "num_indices": len(sharding.get("evaluated_global_indices", [])),
            }
        )
        all_indices.extend(int(idx) for idx in sharding.get("evaluated_global_indices", []))

    merged_masks = merge_per_sample_masks(shard_payloads)
    bootstrap_block: dict[str, Any] = {}
    if args.bootstrap_iterations > 0 and merged_masks is not None:
        if not (0.0 < args.bootstrap_confidence < 1.0):
            raise ValueError(
                f"--bootstrap-confidence must be in (0, 1), got {args.bootstrap_confidence}"
            )
        for axis_idx, axis in enumerate(("tool", "verb", "target", "triplet")):
            bootstrap_block[axis] = compute_bootstrap_ci(
                merged_masks["predicted"][axis],
                merged_masks["target"][axis],
                args.bootstrap_iterations,
                args.bootstrap_seed + axis_idx,
                args.bootstrap_confidence,
            )

    per_sample_block: dict[str, Any] | None = None
    if args.keep_per_sample_masks and merged_masks is not None:
        per_sample_block = {
            "schema": "packed_hex_v1",
            "global_indices": merged_masks["global_indices"],
            "num_classes": merged_masks["num_classes"],
            "predicted": {
                axis: [pack_bool_mask(mask) for mask in merged_masks["predicted"][axis]]
                for axis in ("tool", "verb", "target", "triplet")
            },
            "target": {
                axis: [pack_bool_mask(mask) for mask in merged_masks["target"][axis]]
                for axis in ("tool", "verb", "target", "triplet")
            },
        }

    provenance_per_shard = [payload.get("provenance") for payload in shard_payloads]

    summary = {
        "num_samples": int(total_samples),
        "joint_exact_tool_verb_target_acc": float(joint_exact_acc),
        "unsupported_predicted_triplets": int(unsupported_predicted_triplets),
        "tool": merged_tool["summary"],
        "verb": merged_verb["summary"],
        "target": merged_target["summary"],
        "triplet": merged_triplet["summary"],
        "config_paths": first.get("config_paths"),
        "checkpoint_paths": first.get("checkpoint_paths"),
        "eval_files": first.get("eval_files"),
        "settings": first.get("settings"),
        "provenance_per_shard": provenance_per_shard,
        "bootstrap": bootstrap_block,
        "merged_from_shards": shard_metadata,
        "sharding": {
            "num_input_jsons": len(input_paths),
            "unique_global_indices": sorted(set(all_indices)),
        },
        "raw_counts": {
            "joint_exact_correct": int(joint_exact_correct),
            "unsupported_predicted_triplets": int(unsupported_predicted_triplets),
            "tool": merged_tool["raw_counts"],
            "verb": merged_verb["raw_counts"],
            "target": merged_target["raw_counts"],
            "triplet": merged_triplet["raw_counts"],
        },
        "per_sample_masks": per_sample_block,
    }

    print(f"Merged {len(input_paths)} shard JSON files")
    print(f"Evaluated {total_samples} validation clips")
    print(f"Joint exact tool/verb/target accuracy: {joint_exact_acc:.4f}%")
    print(f"Unsupported predicted triplets: {unsupported_predicted_triplets}")
    print_metrics_block("tool", summary["tool"])
    print_metrics_block("verb", summary["verb"])
    print_metrics_block("target", summary["target"])
    print_metrics_block("triplet", summary["triplet"])

    if bootstrap_block:
        for name in ("tool", "verb", "target", "triplet"):
            block = bootstrap_block.get(name)
            if not block:
                continue
            macro = block["macro_f1"]
            micro = block["micro_f1"]
            print(
                f"{name} merged bootstrap (n={block['num_samples']}, "
                f"iters={block['iterations']}, conf={block['confidence']}): "
                f"MacroF1 mean={macro['mean']:.4f}% [{macro['low']:.4f}%, {macro['high']:.4f}%] "
                f"MicroF1 mean={micro['mean']:.4f}% [{micro['low']:.4f}%, {micro['high']:.4f}%]"
            )

    if args.output_json is not None:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"Saved merged JSON summary to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
