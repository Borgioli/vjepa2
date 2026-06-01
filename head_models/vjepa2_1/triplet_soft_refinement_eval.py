#!/path/to/data_root/vjepa2_1/.venv/bin/python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch

from src.datasets.video_dataset import VideoDataset
from triplet_conditioned_inference import (
    FALLBACK_TARGET_NAMES,
    FALLBACK_TOOL_NAMES,
    FALLBACK_VERB_NAMES,
    build_classifier_from_config,
    build_transform,
    class_names_from_config,
    derive_checkpoint_path,
    init_encoder_module,
    load_classifier_weights,
    load_yaml_config,
    maybe_suppress_stdout,
    override_backbone_checkpoint,
    resolve_device,
)
from triplet_soft_refinement_inference import (
    DEFAULT_TARGET0_CONFIG,
    DEFAULT_TARGET_REFINE_CONFIG,
    DEFAULT_TOOL0_CONFIG,
    DEFAULT_TOOL_REFINE_CONFIG,
    DEFAULT_VERB0_CONFIG,
    DEFAULT_VERB_REFINE_CONFIG,
    HeadTemperatures,
    build_support_pair_constraints,
    compute_soft_refined_probabilities,
    decode_support_aware_triplets,
    load_triplet_support_index,
    map_study_verb_id_to_grouped_ids,
    resolve_support_constrained_routing_mode,
    resolve_soft_refinement_checkpoint_path,
    resolve_multilabel_threshold_vector,
    validate_soft_refinement_configs,
    validate_topk,
    uses_grouped_to_study_verb_adapter,
)
from triplet_soft_refinement_target_relabel import (
    SUPPORTED_RELABEL_MODES,
    apply_relabel_to_target_mask,
    apply_relabel_to_triplet_mask,
    build_relabel_plan,
    relabel_label_lookup,
)

warnings.filterwarnings(
    "ignore",
    message="Importing from timm.models.layers is deprecated.*",
    category=FutureWarning,
)

DEFAULT_TRIPLET_METADATA = Path(
    "/path/to/data_root/vjepa2_1/app/csv_head_models/triplet_multilabel_native_tool5_metadata.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the soft triplet refinement architecture over the full clip-level "
            "validation split used by the Tool5 phase-3 study."
        )
    )
    parser.add_argument("--tool0-config", type=Path, default=DEFAULT_TOOL0_CONFIG)
    parser.add_argument("--verb0-config", type=Path, default=DEFAULT_VERB0_CONFIG)
    parser.add_argument("--target0-config", type=Path, default=DEFAULT_TARGET0_CONFIG)
    parser.add_argument("--tool-refine-config", type=Path, default=DEFAULT_TOOL_REFINE_CONFIG)
    parser.add_argument("--verb-refine-config", type=Path, default=DEFAULT_VERB_REFINE_CONFIG)
    parser.add_argument("--target-refine-config", type=Path, default=DEFAULT_TARGET_REFINE_CONFIG)

    parser.add_argument("--tool0-checkpoint", type=Path, default=None)
    parser.add_argument("--verb0-checkpoint", type=Path, default=None)
    parser.add_argument("--target0-checkpoint", type=Path, default=None)
    parser.add_argument("--tool-refine-checkpoint", type=Path, default=None)
    parser.add_argument("--verb-refine-checkpoint", type=Path, default=None)
    parser.add_argument("--target-refine-checkpoint", type=Path, default=None)

    parser.add_argument(
        "--head-checkpoint-policy",
        choices=("best", "latest"),
        default="best",
        help=(
            "How to derive head checkpoints when explicit paths are not passed. "
            "'best' prefers exported best_epoch checkpoints and falls back to latest.pt."
        ),
    )
    parser.add_argument(
        "--strict-best-head-checkpoints",
        action="store_true",
        help=(
            "With --head-checkpoint-policy best, fail instead of falling back to latest.pt "
            "when no best checkpoint exists."
        ),
    )
    parser.add_argument(
        "--backbone-checkpoint",
        "--encoder-checkpoint",
        dest="backbone_checkpoint",
        type=Path,
        default=None,
        help="Override only the frozen encoder/backbone checkpoint for every loaded head.",
    )
    parser.add_argument(
        "--backbone-checkpoint-key",
        "--encoder-checkpoint-key",
        dest="backbone_checkpoint_key",
        type=str,
        default=None,
        help=(
            "Override the encoder key inside --backbone-checkpoint. If omitted, "
            "the script keeps the config key when present, otherwise tries target_encoder "
            "then encoder."
        ),
    )

    parser.add_argument(
        "--eval-runtime-csv",
        type=Path,
        default=None,
        help="Clip-level runtime CSV to evaluate. Defaults to tool0 config dataset_val.",
    )
    parser.add_argument(
        "--eval-label-csv",
        type=Path,
        default=None,
        help="Grouped clip-level multilabel CSV. Defaults to tool0 config multilabel_lookup_val.",
    )
    parser.add_argument(
        "--triplet-metadata",
        type=Path,
        default=DEFAULT_TRIPLET_METADATA,
        help="Metadata JSON with triplet id mappings.",
    )

    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--tool-threshold", type=float, default=None)
    parser.add_argument("--verb-threshold", type=float, default=None)
    parser.add_argument("--target-threshold", type=float, default=None)
    parser.add_argument(
        "--refinement-steps",
        type=int,
        default=1,
        choices=(1, 2),
        help="How many soft refinement rounds to run after the stage-0 heads.",
    )
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=0.75,
        help=(
            "Blend factor between previous-stage and refined probabilities. "
            "0.0 means pure refinement, 1.0 keeps the previous stage unchanged."
        ),
    )
    parser.add_argument(
        "--blend-alpha-step2",
        type=float,
        default=None,
        help=(
            "Optional override for the second refinement round only. When omitted, "
            "--blend-alpha is reused for every refinement step."
        ),
    )
    parser.add_argument(
        "--route-topk-tool",
        type=int,
        default=None,
        help="Optional top-k truncation for tool routing weights before pairwise mixing.",
    )
    parser.add_argument(
        "--route-topk-verb",
        type=int,
        default=None,
        help="Optional top-k truncation for verb routing weights before pairwise mixing.",
    )
    parser.add_argument(
        "--route-topk-target",
        type=int,
        default=None,
        help="Optional top-k truncation for target routing weights before pairwise mixing.",
    )
    parser.add_argument(
        "--routing-probability-source",
        type=str,
        choices=("current", "stage0"),
        default="current",
        help=(
            "Which marginals to use when building routing distributions across refinement "
            "steps. 'current' re-routes on the latest refined probabilities; 'stage0' "
            "keeps routing anchored to the original unconditioned probabilities."
        ),
    )
    parser.add_argument(
        "--support-constrained-routing",
        action="store_true",
        help=(
            "Compatibility alias for --support-constrained-routing-mode hard."
        ),
    )
    parser.add_argument(
        "--support-constrained-routing-mode",
        type=str,
        choices=("none", "soft", "hard"),
        default="none",
        help=(
            "How to apply the supported-triplet prior during pair-conditioned "
            "routing: none leaves pair weights unchanged, soft downweights "
            "unsupported pairs, and hard removes unsupported pairs before "
            "renormalization."
        ),
    )
    parser.add_argument(
        "--support-constrained-routing-unsupported-scale",
        type=float,
        default=0.1,
        help=(
            "In soft support-constrained routing mode, multiplicative weight "
            "applied to unsupported pairs before renormalization."
        ),
    )
    parser.add_argument(
        "--support-aware-triplet-refinement",
        action="store_true",
        help=(
            "Decode final predictions through the supported triplet set and promote "
            "compatible fallback triplets for high-confidence tool/verb/target labels."
        ),
    )
    parser.add_argument(
        "--decode-topk-tool",
        type=int,
        default=None,
        help=(
            "Optional top-k budget for support-aware triplet decoding over threshold-passing "
            "tools. Defaults to --route-topk-tool when omitted."
        ),
    )
    parser.add_argument(
        "--decode-topk-verb",
        type=int,
        default=None,
        help=(
            "Optional top-k budget for support-aware triplet decoding over threshold-passing "
            "verbs per tool. Defaults to --route-topk-verb when omitted."
        ),
    )
    parser.add_argument(
        "--decode-topk-target",
        type=int,
        default=None,
        help=(
            "Optional top-k budget for support-aware triplet decoding over threshold-passing "
            "targets per tool/verb pair. Defaults to --route-topk-target when omitted."
        ),
    )
    parser.add_argument(
        "--decode-max-triplets-total",
        type=int,
        default=None,
        help=(
            "Optional cap on the total number of support-aware triplets emitted per clip "
            "after ranking."
        ),
    )
    parser.add_argument(
        "--decode-score-min",
        type=float,
        default=None,
        help=(
            "Optional minimum support-aware triplet score in [0, 1]. Triplets below "
            "this score are discarded before final selection."
        ),
    )
    parser.add_argument(
        "--calibration-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file with per-head temperatures applied in logit space "
            "before sigmoid. See triplet_soft_refinement_inference.HeadTemperatures."
        ),
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help=(
            "Number of bootstrap resamples for confidence intervals on F1 metrics. "
            "Set to 0 to disable bootstrap. Defaults to 1000."
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=20260101,
        help="Deterministic seed for bootstrap resampling (default 20260101).",
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        help="Two-sided confidence level for bootstrap CIs in (0, 1). Default 0.95.",
    )
    parser.add_argument(
        "--no-store-per-sample-masks",
        action="store_true",
        help=(
            "Skip saving per-sample predicted/target masks in the JSON output. "
            "Per-sample masks are required for accurate cross-shard bootstrap CIs."
        ),
    )
    parser.add_argument(
        "--target-relabel-mode",
        type=str,
        default=None,
        choices=(None, *SUPPORTED_RELABEL_MODES),
        help=(
            "Optional eval-only relabeling of the target axis. The relabel is "
            "applied in-memory to the validation labels and to the predicted "
            "target/triplet masks at metric time only; CSV files, metadata "
            "JSON, configs, and head weights are not modified. Currently "
            "supported: tool5_chole_v1 (drop suture/gut/specimen bag, merge "
            "gallbladder wall into gallbladder)."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on the number of validation clips to evaluate.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help=(
            "Optional number of dataset shards for multi-machine evaluation. "
            "Each process evaluates the subset of sample indices where "
            "sample_idx %% num_shards == shard_index."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based shard index to evaluate when --num-shards > 1.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print a progress update every N evaluated samples.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save the evaluation summary as JSON.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print setup details.")
    return parser.parse_args()


def _get_num_classes(config: dict[str, Any]) -> int:
    data_cfg = config["experiment"]["data"]
    num_classes_per_task = data_cfg.get("num_classes_per_task")
    if num_classes_per_task is None:
        return int(data_cfg["num_classes"])
    return int(num_classes_per_task[0])


def resolve_threshold_vector(config: dict[str, Any], explicit_value: float | None) -> torch.Tensor:
    num_classes = _get_num_classes(config)
    if explicit_value is not None:
        return torch.full((num_classes,), float(explicit_value), dtype=torch.float32)

    data_cfg = config["experiment"]["data"]
    threshold_value = data_cfg.get("multilabel_thresholds", data_cfg.get("multilabel_threshold", 0.5))
    if isinstance(threshold_value, (int, float)):
        return torch.full((num_classes,), float(threshold_value), dtype=torch.float32)

    thresholds = torch.as_tensor(threshold_value, dtype=torch.float32).reshape(-1)
    if thresholds.numel() != num_classes:
        raise ValueError(
            f"Expected {num_classes} multilabel thresholds, got {thresholds.numel()} "
            f"from config {config!r}"
        )
    return thresholds


def resolve_grouped_verb_threshold_vector(
    verb0_cfg: dict[str, Any],
    verb_refine_cfg: dict[str, Any],
    explicit_value: float | None,
) -> torch.Tensor:
    if explicit_value is not None:
        return resolve_threshold_vector(verb0_cfg, explicit_value)

    # Final verb predictions live in the base 7-way grouped space, so use the
    # base head thresholds for evaluation rather than the 6-way study routing
    # thresholds used internally by the conditioned experts.
    return resolve_threshold_vector(verb0_cfg, None)


def parse_multihot_field(value: str) -> torch.Tensor:
    value = str(value).strip()
    if value == "":
        raise ValueError("Encountered empty multi-hot field while loading evaluation labels")
    return torch.tensor([float(token) for token in value.split()], dtype=torch.float32)


def load_eval_label_lookup(grouped_csv_path: Path) -> dict[str, dict[str, torch.Tensor]]:
    lookup: dict[str, dict[str, torch.Tensor]] = {}
    with grouped_csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required_fields = [
            "clip_path",
            "tool_multihot",
            "verb_multihot",
            "target_multihot",
            "triplet_multihot",
        ]
        missing_fields = [field for field in required_fields if field not in (reader.fieldnames or [])]
        if missing_fields:
            raise ValueError(
                f"Missing required fields in grouped evaluation CSV {grouped_csv_path}: {missing_fields}"
            )

        for row in reader:
            clip_path = str(row["clip_path"]).strip()
            if not clip_path:
                continue
            if clip_path in lookup:
                raise ValueError(f"Duplicate clip_path in grouped evaluation CSV: {clip_path}")
            lookup[clip_path] = {
                "tool": parse_multihot_field(row["tool_multihot"]),
                "verb": parse_multihot_field(row["verb_multihot"]),
                "target": parse_multihot_field(row["target_multihot"]),
                "triplet": parse_multihot_field(row["triplet_multihot"]),
            }
    return lookup


def load_triplet_mapping(metadata_path: Path) -> tuple[dict[tuple[int, int, int], int], int]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    id_to_triplet = metadata.get("id_to_triplet")
    if not isinstance(id_to_triplet, list):
        raise ValueError(f"Metadata file does not contain a usable id_to_triplet list: {metadata_path}")

    num_triplet_classes = int(metadata.get("num_triplet_classes", len(id_to_triplet)))
    triplet_to_id: dict[tuple[int, int, int], int] = {}
    for entry in id_to_triplet:
        key = (
            int(entry["tool_id"]),
            int(entry["verb_id"]),
            int(entry["target_id"]),
        )
        triplet_to_id[key] = int(entry["triplet_id"])
    return triplet_to_id, num_triplet_classes


def build_eval_dataset(config: dict[str, Any], runtime_csv_path: Path) -> VideoDataset:
    data_cfg = config["experiment"]["data"]
    transform = build_transform(config)
    return VideoDataset(
        data_paths=[str(runtime_csv_path)],
        frames_per_clip=int(data_cfg.get("frames_per_clip", 16)),
        frame_step=int(data_cfg.get("frame_step", 1)),
        num_clips=int(data_cfg.get("num_segments", 1)),
        transform=transform,
        random_clip_sampling=False,
        allow_clip_overlap=True,
        return_sample_path=True,
    )


def prepare_dataset_sample_inputs(
    buffer: list[Any],
    clip_indices: list[Any],
    device: torch.device,
) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
    clips: list[list[torch.Tensor]] = []
    for clip_views in buffer:
        if torch.is_tensor(clip_views):
            clip_views = [clip_views]
        prepared_views = []
        for view in clip_views:
            tensor = view if torch.is_tensor(view) else torch.as_tensor(view)
            prepared_views.append(tensor.unsqueeze(0).to(device=device, non_blocking=True))
        clips.append(prepared_views)

    prepared_indices = []
    for indices in clip_indices:
        index_tensor = torch.as_tensor(indices, dtype=torch.int64)
        if index_tensor.ndim == 0:
            index_tensor = index_tensor.unsqueeze(0)
        prepared_indices.append(index_tensor.unsqueeze(0).to(device=device, non_blocking=True))
    return clips, prepared_indices


def threshold_probabilities(probabilities: torch.Tensor, thresholds: torch.Tensor) -> torch.Tensor:
    return probabilities.detach().cpu().float() >= thresholds.detach().cpu().float()


def build_predicted_triplet_mask(
    refinement: dict[str, Any],
    predicted_tool_mask: torch.Tensor,
    predicted_verb_mask: torch.Tensor,
    target_thresholds: torch.Tensor,
    triplet_to_id: dict[tuple[int, int, int], int],
    num_triplet_classes: int,
) -> tuple[torch.Tensor, int]:
    triplet_mask = torch.zeros(num_triplet_classes, dtype=torch.bool)
    unsupported_count = 0

    final_target_pairs = refinement["final_target_pairs"].detach().cpu()
    final_target_probs = refinement["final_target_conditioned_probs"].detach().cpu().float()
    target_thresholds = target_thresholds.detach().cpu().float()
    use_verb_adapter = bool(refinement.get("use_grouped_to_study_verb_adapter", False))

    for pair_idx, pair in enumerate(final_target_pairs.tolist()):
        tool_id, study_verb_id = int(pair[0]), int(pair[1])
        if not bool(predicted_tool_mask[tool_id].item()):
            continue

        predicted_target_mask = final_target_probs[pair_idx] >= target_thresholds
        target_ids = predicted_target_mask.nonzero(as_tuple=False).reshape(-1).tolist()
        for verb_id in map_study_verb_id_to_grouped_ids(study_verb_id, use_verb_adapter):
            if not bool(predicted_verb_mask[verb_id].item()):
                continue
            for target_id in target_ids:
                triplet_id = triplet_to_id.get((tool_id, verb_id, int(target_id)))
                if triplet_id is None:
                    unsupported_count += 1
                    continue
                triplet_mask[triplet_id] = True

    return triplet_mask, unsupported_count


class MultiLabelMetricsAccumulator:
    def __init__(self, num_labels: int):
        self.num_labels = int(num_labels)
        self.tp = torch.zeros(self.num_labels, dtype=torch.float64)
        self.fp = torch.zeros(self.num_labels, dtype=torch.float64)
        self.fn = torch.zeros(self.num_labels, dtype=torch.float64)
        self.exact_correct = 0
        self.total = 0

    def update(self, predicted_mask: torch.Tensor, target_mask: torch.Tensor) -> None:
        predicted_mask = predicted_mask.detach().cpu().bool().reshape(-1)
        target_mask = target_mask.detach().cpu().bool().reshape(-1)
        if predicted_mask.numel() != self.num_labels or target_mask.numel() != self.num_labels:
            raise ValueError(
                f"Expected {self.num_labels} labels, got pred={predicted_mask.numel()} "
                f"target={target_mask.numel()}"
            )

        self.tp += (predicted_mask & target_mask).to(dtype=torch.float64)
        self.fp += (predicted_mask & ~target_mask).to(dtype=torch.float64)
        self.fn += (~predicted_mask & target_mask).to(dtype=torch.float64)
        self.exact_correct += int(torch.equal(predicted_mask, target_mask))
        self.total += 1

    def summary(self) -> dict[str, float]:
        return summarize_multilabel_counts(
            tp=self.tp,
            fp=self.fp,
            fn=self.fn,
            exact_correct=self.exact_correct,
            total=self.total,
        )

    def raw_counts(self) -> dict[str, Any]:
        return {
            "tp": [float(value) for value in self.tp.tolist()],
            "fp": [float(value) for value in self.fp.tolist()],
            "fn": [float(value) for value in self.fn.tolist()],
            "exact_correct": int(self.exact_correct),
            "total": int(self.total),
        }


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
    precision_denom = micro_tp + micro_fp
    if precision_denom > 0:
        micro_precision = micro_tp / precision_denom

    micro_recall = 0.0
    recall_denom = micro_tp + micro_fn
    if recall_denom > 0:
        micro_recall = micro_tp / recall_denom

    micro_f1 = 0.0
    micro_f1_denom = (2.0 * micro_tp) + micro_fp + micro_fn
    if micro_f1_denom > 0:
        micro_f1 = (2.0 * micro_tp) / micro_f1_denom

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


def summarize_multilabel_counts_active(
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    exact_correct: int,
    total: int,
    active_mask: torch.Tensor,
) -> dict[str, float]:
    """Summarize multilabel counts under a per-class active mask.

    Inactive classes are excluded from per-class averaging (they do not
    count as zero-F1 entries) and from Micro TP/FP/FN sums.
    """
    tp_v = tp.detach().cpu().to(dtype=torch.float64).reshape(-1)
    fp_v = fp.detach().cpu().to(dtype=torch.float64).reshape(-1)
    fn_v = fn.detach().cpu().to(dtype=torch.float64).reshape(-1)
    keep = active_mask.detach().cpu().bool().reshape(-1)
    if keep.numel() != tp_v.numel():
        raise ValueError(
            f"Active mask length {keep.numel()} != counts length {tp_v.numel()}"
        )

    tp_kept = tp_v[keep]
    fp_kept = fp_v[keep]
    fn_kept = fn_v[keep]

    per_class_denom = (2.0 * tp_kept) + fp_kept + fn_kept
    per_class_f1 = torch.where(
        per_class_denom > 0,
        (2.0 * tp_kept) / per_class_denom,
        torch.zeros_like(per_class_denom),
    )

    micro_tp = float(tp_kept.sum().item())
    micro_fp = float(fp_kept.sum().item())
    micro_fn = float(fn_kept.sum().item())

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

    macro_f1 = float(per_class_f1.mean().item()) if per_class_f1.numel() > 0 else 0.0
    return {
        "exact_acc": 100.0 * exact_acc,
        "macro_f1": 100.0 * macro_f1,
        "micro_precision": 100.0 * micro_precision,
        "micro_recall": 100.0 * micro_recall,
        "micro_f1": 100.0 * micro_f1,
    }


def resolve_shard_sample_indices(max_samples: int, num_shards: int, shard_index: int) -> list[int]:
    if num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"--shard-index must satisfy 0 <= shard_index < num_shards, "
            f"got shard_index={shard_index}, num_shards={num_shards}"
        )
    return list(range(shard_index, max_samples, num_shards))


def compute_file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def safe_file_hash(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    resolved = Path(path).resolve()
    if not resolved.exists():
        return {"path": str(resolved), "sha256": None, "size_bytes": None, "error": "missing"}
    try:
        stat = resolved.stat()
        return {
            "path": str(resolved),
            "sha256": compute_file_sha256(resolved),
            "size_bytes": int(stat.st_size),
            "mtime": float(stat.st_mtime),
        }
    except OSError as error:
        return {"path": str(resolved), "sha256": None, "size_bytes": None, "error": str(error)}


def detect_git_revision(start_dir: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"available": False}
    try:
        rev = subprocess.run(
            ["git", "-C", str(start_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        try:
            dirty = subprocess.run(
                ["git", "-C", str(start_dir), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except (subprocess.SubprocessError, FileNotFoundError):
            dirty = ""
        info.update({
            "available": True,
            "head_sha": rev,
            "dirty": bool(dirty),
        })
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return info


def collect_provenance(
    args: argparse.Namespace,
    checkpoint_paths: dict[str, Path],
    config_paths: dict[str, Path],
    eval_paths: dict[str, Path],
    backbone_checkpoint: Path | None,
    calibration_json: Path | None,
    head_temperatures: HeadTemperatures,
) -> dict[str, Any]:
    args_dict = {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()}
    args_blob = json.dumps(args_dict, sort_keys=True, default=str).encode("utf-8")
    args_sha = hashlib.sha256(args_blob).hexdigest()

    return {
        "schema_version": 1,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": os.uname().nodename if hasattr(os, "uname") else None,
        "python_executable": sys.executable,
        "torch_version": torch.__version__,
        "argv": list(sys.argv),
        "args": args_dict,
        "args_sha256": args_sha,
        "checkpoint_hashes": {name: safe_file_hash(path) for name, path in checkpoint_paths.items()},
        "config_hashes": {name: safe_file_hash(path) for name, path in config_paths.items()},
        "eval_file_hashes": {name: safe_file_hash(path) for name, path in eval_paths.items()},
        "backbone_checkpoint_hash": safe_file_hash(backbone_checkpoint),
        "calibration_json_hash": safe_file_hash(calibration_json),
        "head_temperatures": head_temperatures.to_serializable(),
        "head_temperatures_is_identity": head_temperatures.is_identity(),
        "git": detect_git_revision(Path(__file__).resolve().parent),
    }


def pack_bool_mask(mask: torch.Tensor) -> str:
    flat = mask.detach().cpu().bool().reshape(-1).tolist()
    if not flat:
        return ""
    bits = "".join("1" if value else "0" for value in flat)
    padding = (-len(bits)) % 8
    bits += "0" * padding
    byte_values = bytes(int(bits[idx : idx + 8], 2) for idx in range(0, len(bits), 8))
    return byte_values.hex()


def unpack_bool_mask(hex_string: str, num_bits: int) -> torch.Tensor:
    if num_bits == 0:
        return torch.zeros(0, dtype=torch.bool)
    raw = bytes.fromhex(hex_string)
    bits = "".join(f"{byte:08b}" for byte in raw)[:num_bits]
    if len(bits) != num_bits:
        raise ValueError(f"Packed mask has {len(bits)} bits, expected {num_bits}")
    return torch.tensor([bit == "1" for bit in bits], dtype=torch.bool)


def compute_bootstrap_ci(
    predicted: list[torch.Tensor],
    target: list[torch.Tensor],
    num_iterations: int,
    seed: int,
    confidence: float,
    active_mask: torch.Tensor | None = None,
) -> dict[str, Any] | None:
    if num_iterations <= 0 or not predicted:
        return None
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"--bootstrap-confidence must be in (0, 1), got {confidence}")

    num_samples = len(predicted)
    num_labels = predicted[0].numel()
    pred_matrix = torch.stack([mask.bool().reshape(-1) for mask in predicted]).to(torch.bool)
    target_matrix = torch.stack([mask.bool().reshape(-1) for mask in target]).to(torch.bool)

    if active_mask is not None:
        keep = active_mask.detach().cpu().bool().reshape(-1)
        if keep.numel() != num_labels:
            raise ValueError(
                f"Active mask length {keep.numel()} != num_labels {num_labels}"
            )
        column_idx = keep.nonzero(as_tuple=False).reshape(-1)
        pred_matrix = pred_matrix.index_select(1, column_idx)
        target_matrix = target_matrix.index_select(1, column_idx)
        num_active_labels = int(column_idx.numel())
    else:
        num_active_labels = int(num_labels)

    rng = random.Random(seed)
    macro_samples: list[float] = []
    micro_samples: list[float] = []

    for _ in range(num_iterations):
        idx = torch.tensor(
            [rng.randrange(num_samples) for _ in range(num_samples)],
            dtype=torch.long,
        )
        pred_bs = pred_matrix.index_select(0, idx)
        target_bs = target_matrix.index_select(0, idx)
        tp = (pred_bs & target_bs).sum(dim=0).to(torch.float64)
        fp = (pred_bs & ~target_bs).sum(dim=0).to(torch.float64)
        fn = (~pred_bs & target_bs).sum(dim=0).to(torch.float64)

        per_class_denom = (2.0 * tp) + fp + fn
        per_class_f1 = torch.where(
            per_class_denom > 0,
            (2.0 * tp) / per_class_denom,
            torch.zeros_like(per_class_denom),
        )
        macro_value = float(per_class_f1.mean().item()) if per_class_f1.numel() > 0 else 0.0
        macro_samples.append(macro_value)

        micro_tp = float(tp.sum().item())
        micro_fp = float(fp.sum().item())
        micro_fn = float(fn.sum().item())
        micro_denom = (2.0 * micro_tp) + micro_fp + micro_fn
        micro = (2.0 * micro_tp) / micro_denom if micro_denom > 0 else 0.0
        micro_samples.append(float(micro))

    alpha = (1.0 - confidence) / 2.0
    macro_sorted = sorted(macro_samples)
    micro_sorted = sorted(micro_samples)
    lo_idx = max(0, int(round(alpha * len(macro_sorted))) - 1)
    hi_idx = min(len(macro_sorted) - 1, int(round((1.0 - alpha) * len(macro_sorted))) - 1)

    return {
        "iterations": int(num_iterations),
        "confidence": float(confidence),
        "num_samples": int(num_samples),
        "num_labels": int(num_labels),
        "num_active_labels": int(num_active_labels),
        "macro_f1": {
            "mean": 100.0 * float(sum(macro_sorted) / len(macro_sorted)),
            "low": 100.0 * float(macro_sorted[lo_idx]),
            "high": 100.0 * float(macro_sorted[hi_idx]),
        },
        "micro_f1": {
            "mean": 100.0 * float(sum(micro_sorted) / len(micro_sorted)),
            "low": 100.0 * float(micro_sorted[lo_idx]),
            "high": 100.0 * float(micro_sorted[hi_idx]),
        },
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


def main() -> int:
    args = parse_args()

    validate_topk("route_topk_tool", args.route_topk_tool)
    validate_topk("route_topk_verb", args.route_topk_verb)
    validate_topk("route_topk_target", args.route_topk_target)
    validate_topk("decode_topk_tool", args.decode_topk_tool)
    validate_topk("decode_topk_verb", args.decode_topk_verb)
    validate_topk("decode_topk_target", args.decode_topk_target)
    validate_topk("decode_max_triplets_total", args.decode_max_triplets_total)
    if args.decode_score_min is not None and not (0.0 <= args.decode_score_min <= 1.0):
        raise ValueError(f"--decode-score-min must be in [0, 1], got {args.decode_score_min}")
    if not (0.0 <= args.blend_alpha <= 1.0):
        raise ValueError(f"--blend-alpha must be in [0, 1], got {args.blend_alpha}")
    if args.blend_alpha_step2 is not None and not (0.0 <= args.blend_alpha_step2 <= 1.0):
        raise ValueError(
            f"--blend-alpha-step2 must be in [0, 1], got {args.blend_alpha_step2}"
        )
    if not (0.0 <= args.support_constrained_routing_unsupported_scale <= 1.0):
        raise ValueError(
            "--support-constrained-routing-unsupported-scale must be in [0, 1], "
            f"got {args.support_constrained_routing_unsupported_scale}"
        )
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError(f"--max-samples must be >= 1 when provided, got {args.max_samples}")
    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            f"--shard-index must satisfy 0 <= shard-index < num-shards, "
            f"got shard_index={args.shard_index}, num_shards={args.num_shards}"
        )
    if args.progress_every < 1:
        raise ValueError(f"--progress-every must be >= 1, got {args.progress_every}")
    if args.bootstrap_iterations < 0:
        raise ValueError(f"--bootstrap-iterations must be >= 0, got {args.bootstrap_iterations}")
    if not (0.0 < args.bootstrap_confidence < 1.0):
        raise ValueError(
            f"--bootstrap-confidence must be in (0, 1), got {args.bootstrap_confidence}"
        )

    if args.calibration_json is not None:
        head_temperatures = HeadTemperatures.from_json(args.calibration_json.resolve())
    else:
        head_temperatures = HeadTemperatures.identity()

    device = resolve_device(args.device)
    if args.verbose:
        print(f"Using device: {device}")

    tool0_cfg = load_yaml_config(args.tool0_config)
    verb0_cfg = load_yaml_config(args.verb0_config)
    target0_cfg = load_yaml_config(args.target0_config)
    tool_refine_cfg = load_yaml_config(args.tool_refine_config)
    verb_refine_cfg = load_yaml_config(args.verb_refine_config)
    target_refine_cfg = load_yaml_config(args.target_refine_config)

    all_cfgs = [
        tool0_cfg,
        verb0_cfg,
        target0_cfg,
        tool_refine_cfg,
        verb_refine_cfg,
        target_refine_cfg,
    ]
    override_backbone_checkpoint(
        all_cfgs,
        args.backbone_checkpoint,
        checkpoint_key=args.backbone_checkpoint_key,
    )
    validate_soft_refinement_configs(
        tool0_cfg=tool0_cfg,
        verb0_cfg=verb0_cfg,
        target0_cfg=target0_cfg,
        tool_refine_cfg=tool_refine_cfg,
        verb_refine_cfg=verb_refine_cfg,
        target_refine_cfg=target_refine_cfg,
    )
    support_constrained_routing_mode = resolve_support_constrained_routing_mode(
        args.support_constrained_routing_mode,
        args.support_constrained_routing,
    )

    eval_runtime_csv = (
        args.eval_runtime_csv
        if args.eval_runtime_csv is not None
        else Path(tool0_cfg["experiment"]["data"]["dataset_val"])
    ).resolve()
    eval_label_csv = (
        args.eval_label_csv
        if args.eval_label_csv is not None
        else Path(tool0_cfg["experiment"]["data"]["multilabel_lookup_val"])
    ).resolve()
    triplet_metadata_path = args.triplet_metadata.resolve()

    if not eval_runtime_csv.exists():
        raise FileNotFoundError(f"Evaluation runtime CSV does not exist: {eval_runtime_csv}")
    if not eval_label_csv.exists():
        raise FileNotFoundError(f"Evaluation label CSV does not exist: {eval_label_csv}")
    if not triplet_metadata_path.exists():
        raise FileNotFoundError(f"Triplet metadata JSON does not exist: {triplet_metadata_path}")

    tool_thresholds = resolve_threshold_vector(tool_refine_cfg, args.tool_threshold)
    verb_thresholds = resolve_grouped_verb_threshold_vector(
        verb0_cfg,
        verb_refine_cfg,
        args.verb_threshold,
    )
    target_thresholds = resolve_threshold_vector(target_refine_cfg, args.target_threshold)
    route_tool_thresholds = resolve_multilabel_threshold_vector(tool_refine_cfg, args.tool_threshold)
    route_verb_thresholds = resolve_multilabel_threshold_vector(verb_refine_cfg, None)
    route_target_thresholds = resolve_multilabel_threshold_vector(target_refine_cfg, args.target_threshold)

    tool_names = class_names_from_config(tool0_cfg, FALLBACK_TOOL_NAMES, "tool")
    verb_names = class_names_from_config(verb0_cfg, FALLBACK_VERB_NAMES, "verb")
    target_names = class_names_from_config(target0_cfg, FALLBACK_TARGET_NAMES, "target")

    label_lookup = load_eval_label_lookup(eval_label_csv)
    triplet_to_id, num_triplet_classes, supported_triplets = load_triplet_support_index(
        triplet_metadata_path
    )
    support_pair_constraints = None
    if support_constrained_routing_mode != "none":
        support_pair_constraints = build_support_pair_constraints(
            supported_triplets,
            use_verb_adapter=uses_grouped_to_study_verb_adapter(verb0_cfg, verb_refine_cfg),
        )
    relabel_plan = build_relabel_plan(
        mode=args.target_relabel_mode,
        target_names=class_names_from_config(target0_cfg, FALLBACK_TARGET_NAMES, "target"),
        triplet_to_id=triplet_to_id,
        num_triplet_classes=num_triplet_classes,
    )
    if relabel_plan is not None:
        label_lookup = relabel_label_lookup(label_lookup, relabel_plan)
        if args.verbose:
            print(
                "Target relabel mode "
                f"'{relabel_plan.mode}' applied: "
                f"dropped target ids={list(relabel_plan.dropped_target_ids)}, "
                f"target merges={relabel_plan.target_merges}, "
                f"active target classes="
                f"{int(relabel_plan.target_active_mask.sum().item())}/"
                f"{relabel_plan.num_target_classes}, "
                f"active triplet classes="
                f"{int(relabel_plan.triplet_active_mask.sum().item())}/"
                f"{relabel_plan.num_triplet_classes}"
            )
    dataset = build_eval_dataset(tool0_cfg, eval_runtime_csv)

    tool0_checkpoint = resolve_soft_refinement_checkpoint_path(
        tool0_cfg,
        args.tool0_config,
        args.tool0_checkpoint,
        fallback_name="tool",
        checkpoint_policy=args.head_checkpoint_policy,
        strict_best=args.strict_best_head_checkpoints,
    )
    verb0_checkpoint = resolve_soft_refinement_checkpoint_path(
        verb0_cfg,
        args.verb0_config,
        args.verb0_checkpoint,
        fallback_name="verb",
        checkpoint_policy=args.head_checkpoint_policy,
        strict_best=args.strict_best_head_checkpoints,
    )
    target0_checkpoint = resolve_soft_refinement_checkpoint_path(
        target0_cfg,
        args.target0_config,
        args.target0_checkpoint,
        fallback_name="target",
        checkpoint_policy=args.head_checkpoint_policy,
        strict_best=args.strict_best_head_checkpoints,
    )
    tool_refine_checkpoint = resolve_soft_refinement_checkpoint_path(
        tool_refine_cfg,
        args.tool_refine_config,
        args.tool_refine_checkpoint,
        fallback_name="tool",
        checkpoint_policy=args.head_checkpoint_policy,
        strict_best=args.strict_best_head_checkpoints,
    )
    verb_refine_checkpoint = resolve_soft_refinement_checkpoint_path(
        verb_refine_cfg,
        args.verb_refine_config,
        args.verb_refine_checkpoint,
        fallback_name="verb",
        checkpoint_policy=args.head_checkpoint_policy,
        strict_best=args.strict_best_head_checkpoints,
    )
    target_refine_checkpoint = resolve_soft_refinement_checkpoint_path(
        target_refine_cfg,
        args.target_refine_config,
        args.target_refine_checkpoint,
        fallback_name="target",
        checkpoint_policy=args.head_checkpoint_policy,
        strict_best=args.strict_best_head_checkpoints,
    )

    frames_per_clip = int(tool0_cfg["experiment"]["data"].get("frames_per_clip", 16))
    resolution = int(tool0_cfg["experiment"]["data"].get("resolution", 224))
    if args.verbose:
        print(f"Loading encoder from {tool0_cfg['model_kwargs']['checkpoint']}")
        print(f"Head checkpoint policy: {args.head_checkpoint_policy}")
    with maybe_suppress_stdout(args.verbose):
        encoder = init_encoder_module(
            module_name=tool0_cfg["model_kwargs"]["module_name"],
            frames_per_clip=frames_per_clip,
            resolution=resolution,
            checkpoint=tool0_cfg["model_kwargs"]["checkpoint"],
            model_kwargs=tool0_cfg["model_kwargs"]["pretrain_kwargs"],
            wrapper_kwargs=tool0_cfg["model_kwargs"]["wrapper_kwargs"],
            device=device,
        )
    encoder.eval()

    tool0_head = build_classifier_from_config(tool0_cfg, encoder.embed_dim).to(device).eval()
    verb0_head = build_classifier_from_config(verb0_cfg, encoder.embed_dim).to(device).eval()
    target0_head = build_classifier_from_config(target0_cfg, encoder.embed_dim).to(device).eval()
    tool_refine_head = build_classifier_from_config(tool_refine_cfg, encoder.embed_dim).to(device).eval()
    verb_refine_head = build_classifier_from_config(verb_refine_cfg, encoder.embed_dim).to(device).eval()
    target_refine_head = build_classifier_from_config(target_refine_cfg, encoder.embed_dim).to(device).eval()

    load_classifier_weights(tool0_head, tool0_checkpoint)
    load_classifier_weights(verb0_head, verb0_checkpoint)
    load_classifier_weights(target0_head, target0_checkpoint)
    load_classifier_weights(tool_refine_head, tool_refine_checkpoint)
    load_classifier_weights(verb_refine_head, verb_refine_checkpoint)
    load_classifier_weights(target_refine_head, target_refine_checkpoint)

    if args.verbose:
        print(f"Loaded tool0 head from {tool0_checkpoint}")
        print(f"Loaded verb0 head from {verb0_checkpoint}")
        print(f"Loaded target0 head from {target0_checkpoint}")
        print(f"Loaded tool refinement head from {tool_refine_checkpoint}")
        print(f"Loaded verb refinement head from {verb_refine_checkpoint}")
        print(f"Loaded target refinement head from {target_refine_checkpoint}")
        print(f"Evaluation runtime CSV: {eval_runtime_csv}")
        print(f"Evaluation label CSV: {eval_label_csv}")
        print(f"Triplet metadata: {triplet_metadata_path}")
        print(
            "Threshold vectors loaded for classes: "
            f"tool={len(tool_names)} verb={len(verb_names)} target={len(target_names)}"
        )

    tool_metrics = MultiLabelMetricsAccumulator(len(tool_names))
    verb_metrics = MultiLabelMetricsAccumulator(len(verb_names))
    target_metrics = MultiLabelMetricsAccumulator(len(target_names))
    triplet_metrics = MultiLabelMetricsAccumulator(num_triplet_classes)

    per_sample_tool_pred: list[torch.Tensor] = []
    per_sample_verb_pred: list[torch.Tensor] = []
    per_sample_target_pred: list[torch.Tensor] = []
    per_sample_triplet_pred: list[torch.Tensor] = []
    per_sample_tool_target: list[torch.Tensor] = []
    per_sample_verb_target: list[torch.Tensor] = []
    per_sample_target_target: list[torch.Tensor] = []
    per_sample_triplet_target: list[torch.Tensor] = []
    per_sample_global_indices: list[int] = []

    joint_exact_correct = 0
    unsupported_predicted_triplets = 0
    suppressed_unsupported_triplets = 0
    support_search_tools = 0
    support_search_triplets = 0
    evaluated_samples = 0

    max_samples = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    shard_sample_indices = resolve_shard_sample_indices(
        max_samples=max_samples,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )

    if args.verbose:
        print(
            f"Shard selection: shard {args.shard_index + 1}/{args.num_shards} "
            f"covering {len(shard_sample_indices)} of {max_samples} candidate clips"
        )

    for shard_position, sample_idx in enumerate(shard_sample_indices, start=1):
        buffer, _, clip_indices, sample_path = dataset[sample_idx]
        sample_path = str(sample_path)
        if sample_path not in label_lookup:
            raise KeyError(f"Could not find grouped labels for validation clip: {sample_path}")

        sample_labels = label_lookup[sample_path]
        clips, prepared_indices = prepare_dataset_sample_inputs(buffer, clip_indices, device)

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                encoded_outputs = encoder(clips, prepared_indices)
                encoded_clip = encoded_outputs[0]
                refinement = compute_soft_refined_probabilities(
                    encoded_clip=encoded_clip,
                    tool0_head=tool0_head,
                    verb0_head=verb0_head,
                    target0_head=target0_head,
                    tool_refine_head=tool_refine_head,
                    verb_refine_head=verb_refine_head,
                    target_refine_head=target_refine_head,
                    device=device,
                    refinement_steps=args.refinement_steps,
                    blend_alpha=args.blend_alpha,
                    blend_alpha_step2=args.blend_alpha_step2,
                    route_topk_tool=args.route_topk_tool,
                    route_topk_verb=args.route_topk_verb,
                    route_topk_target=args.route_topk_target,
                    routing_probability_source=args.routing_probability_source,
                    route_threshold_tool=route_tool_thresholds,
                    route_threshold_verb=route_verb_thresholds,
                    route_threshold_target=route_target_thresholds,
                    head_temperatures=head_temperatures,
                    support_pair_constraints=support_pair_constraints,
                    support_constrained_routing_mode=support_constrained_routing_mode,
                    support_constrained_routing_unsupported_scale=(
                        args.support_constrained_routing_unsupported_scale
                    ),
                )

        raw_predicted_tool_mask = threshold_probabilities(refinement["final"]["tool"], tool_thresholds)
        raw_predicted_verb_mask = threshold_probabilities(refinement["final"]["verb"], verb_thresholds)
        raw_predicted_target_mask = threshold_probabilities(refinement["final"]["target"], target_thresholds)
        raw_predicted_triplet_mask, raw_unsupported_count = build_predicted_triplet_mask(
            refinement=refinement,
            predicted_tool_mask=raw_predicted_tool_mask,
            predicted_verb_mask=raw_predicted_verb_mask,
            target_thresholds=target_thresholds,
            triplet_to_id=triplet_to_id,
            num_triplet_classes=num_triplet_classes,
        )

        predicted_tool_mask = raw_predicted_tool_mask
        predicted_verb_mask = raw_predicted_verb_mask
        predicted_target_mask = raw_predicted_target_mask
        predicted_triplet_mask = raw_predicted_triplet_mask
        unsupported_count = raw_unsupported_count
        if args.support_aware_triplet_refinement:
            decoded = decode_support_aware_triplets(
                refinement=refinement,
                triplet_to_id=triplet_to_id,
                num_triplet_classes=num_triplet_classes,
                tool_thresholds=tool_thresholds,
                verb_thresholds=verb_thresholds,
                target_thresholds=target_thresholds,
                search_topk_tool=(
                    args.decode_topk_tool
                    if args.decode_topk_tool is not None
                    else args.route_topk_tool
                ),
                search_topk_verb=(
                    args.decode_topk_verb
                    if args.decode_topk_verb is not None
                    else args.route_topk_verb
                ),
                search_topk_target=(
                    args.decode_topk_target
                    if args.decode_topk_target is not None
                    else args.route_topk_target
                ),
                max_triplets_total=args.decode_max_triplets_total,
                min_triplet_score=args.decode_score_min,
                fallback_top1=False,
            )
            predicted_triplet_mask = decoded["triplet_mask"]
            unsupported_count = 0
            suppressed_unsupported_triplets += raw_unsupported_count
            support_search_tools += int(decoded["num_support_search_tools"])
            support_search_triplets += int(decoded["num_support_search_triplets"])

        target_tool_mask = sample_labels["tool"] > 0.5
        target_verb_mask = sample_labels["verb"] > 0.5
        target_target_mask = sample_labels["target"] > 0.5
        target_triplet_mask = sample_labels["triplet"] > 0.5

        if relabel_plan is not None:
            predicted_target_mask = apply_relabel_to_target_mask(predicted_target_mask, relabel_plan)
            predicted_triplet_mask = apply_relabel_to_triplet_mask(predicted_triplet_mask, relabel_plan)

        tool_metrics.update(predicted_tool_mask, target_tool_mask)
        verb_metrics.update(predicted_verb_mask, target_verb_mask)
        target_metrics.update(predicted_target_mask, target_target_mask)
        triplet_metrics.update(predicted_triplet_mask, target_triplet_mask)

        per_sample_tool_pred.append(predicted_tool_mask.detach().cpu().bool().reshape(-1))
        per_sample_verb_pred.append(predicted_verb_mask.detach().cpu().bool().reshape(-1))
        per_sample_target_pred.append(predicted_target_mask.detach().cpu().bool().reshape(-1))
        per_sample_triplet_pred.append(predicted_triplet_mask.detach().cpu().bool().reshape(-1))
        per_sample_tool_target.append(target_tool_mask.detach().cpu().bool().reshape(-1))
        per_sample_verb_target.append(target_verb_mask.detach().cpu().bool().reshape(-1))
        per_sample_target_target.append(target_target_mask.detach().cpu().bool().reshape(-1))
        per_sample_triplet_target.append(target_triplet_mask.detach().cpu().bool().reshape(-1))
        per_sample_global_indices.append(int(sample_idx))

        if (
            torch.equal(predicted_tool_mask, target_tool_mask)
            and torch.equal(predicted_verb_mask, target_verb_mask)
            and torch.equal(predicted_target_mask, target_target_mask)
        ):
            joint_exact_correct += 1

        unsupported_predicted_triplets += unsupported_count
        evaluated_samples += 1

        if (
            args.verbose
            and (
                (shard_position % args.progress_every == 0)
                or (shard_position == len(shard_sample_indices))
            )
        ):
            print(
                f"Evaluated {shard_position}/{len(shard_sample_indices)} shard clips "
                f"(global sample_idx={sample_idx})"
            )

    if evaluated_samples == 0:
        raise RuntimeError("No validation samples were evaluated")

    tool_summary = tool_metrics.summary()
    verb_summary = verb_metrics.summary()
    target_summary = target_metrics.summary()
    triplet_summary = triplet_metrics.summary()
    if relabel_plan is not None:
        target_summary = summarize_multilabel_counts_active(
            tp=target_metrics.tp,
            fp=target_metrics.fp,
            fn=target_metrics.fn,
            exact_correct=target_metrics.exact_correct,
            total=target_metrics.total,
            active_mask=relabel_plan.target_active_mask,
        )
        triplet_summary = summarize_multilabel_counts_active(
            tp=triplet_metrics.tp,
            fp=triplet_metrics.fp,
            fn=triplet_metrics.fn,
            exact_correct=triplet_metrics.exact_correct,
            total=triplet_metrics.total,
            active_mask=relabel_plan.triplet_active_mask,
        )
    joint_exact_acc = 100.0 * float(joint_exact_correct) / float(evaluated_samples)

    bootstrap_block: dict[str, Any] = {}
    if args.bootstrap_iterations > 0:
        if args.verbose:
            print(
                f"Running {args.bootstrap_iterations} bootstrap iterations "
                f"(seed={args.bootstrap_seed}, confidence={args.bootstrap_confidence})"
            )
        bootstrap_block = {
            "tool": compute_bootstrap_ci(
                per_sample_tool_pred,
                per_sample_tool_target,
                args.bootstrap_iterations,
                args.bootstrap_seed,
                args.bootstrap_confidence,
            ),
            "verb": compute_bootstrap_ci(
                per_sample_verb_pred,
                per_sample_verb_target,
                args.bootstrap_iterations,
                args.bootstrap_seed + 1,
                args.bootstrap_confidence,
            ),
            "target": compute_bootstrap_ci(
                per_sample_target_pred,
                per_sample_target_target,
                args.bootstrap_iterations,
                args.bootstrap_seed + 2,
                args.bootstrap_confidence,
                active_mask=(relabel_plan.target_active_mask if relabel_plan is not None else None),
            ),
            "triplet": compute_bootstrap_ci(
                per_sample_triplet_pred,
                per_sample_triplet_target,
                args.bootstrap_iterations,
                args.bootstrap_seed + 3,
                args.bootstrap_confidence,
                active_mask=(relabel_plan.triplet_active_mask if relabel_plan is not None else None),
            ),
        }

    provenance = collect_provenance(
        args=args,
        checkpoint_paths={
            "tool0": tool0_checkpoint,
            "verb0": verb0_checkpoint,
            "target0": target0_checkpoint,
            "tool_refine": tool_refine_checkpoint,
            "verb_refine": verb_refine_checkpoint,
            "target_refine": target_refine_checkpoint,
        },
        config_paths={
            "tool0": args.tool0_config,
            "verb0": args.verb0_config,
            "target0": args.target0_config,
            "tool_refine": args.tool_refine_config,
            "verb_refine": args.verb_refine_config,
            "target_refine": args.target_refine_config,
        },
        eval_paths={
            "runtime_csv": eval_runtime_csv,
            "grouped_csv": eval_label_csv,
            "triplet_metadata": triplet_metadata_path,
        },
        backbone_checkpoint=args.backbone_checkpoint,
        calibration_json=args.calibration_json,
        head_temperatures=head_temperatures,
    )

    per_sample_block: dict[str, Any] | None = None
    if not args.no_store_per_sample_masks:
        per_sample_block = {
            "schema": "packed_hex_v1",
            "global_indices": per_sample_global_indices,
            "num_classes": {
                "tool": len(tool_names),
                "verb": len(verb_names),
                "target": len(target_names),
                "triplet": int(num_triplet_classes),
            },
            "predicted": {
                "tool": [pack_bool_mask(mask) for mask in per_sample_tool_pred],
                "verb": [pack_bool_mask(mask) for mask in per_sample_verb_pred],
                "target": [pack_bool_mask(mask) for mask in per_sample_target_pred],
                "triplet": [pack_bool_mask(mask) for mask in per_sample_triplet_pred],
            },
            "target": {
                "tool": [pack_bool_mask(mask) for mask in per_sample_tool_target],
                "verb": [pack_bool_mask(mask) for mask in per_sample_verb_target],
                "target": [pack_bool_mask(mask) for mask in per_sample_target_target],
                "triplet": [pack_bool_mask(mask) for mask in per_sample_triplet_target],
            },
        }

    summary = {
        "num_samples": evaluated_samples,
        "joint_exact_tool_verb_target_acc": joint_exact_acc,
        "unsupported_predicted_triplets": int(unsupported_predicted_triplets),
        "tool": tool_summary,
        "verb": verb_summary,
        "target": target_summary,
        "triplet": triplet_summary,
        "config_paths": {
            "tool0": str(args.tool0_config.resolve()),
            "verb0": str(args.verb0_config.resolve()),
            "target0": str(args.target0_config.resolve()),
            "tool_refine": str(args.tool_refine_config.resolve()),
            "verb_refine": str(args.verb_refine_config.resolve()),
            "target_refine": str(args.target_refine_config.resolve()),
        },
        "checkpoint_paths": {
            "tool0": str(tool0_checkpoint.resolve()),
            "verb0": str(verb0_checkpoint.resolve()),
            "target0": str(target0_checkpoint.resolve()),
            "tool_refine": str(tool_refine_checkpoint.resolve()),
            "verb_refine": str(verb_refine_checkpoint.resolve()),
            "target_refine": str(target_refine_checkpoint.resolve()),
        },
        "eval_files": {
            "runtime_csv": str(eval_runtime_csv),
            "grouped_csv": str(eval_label_csv),
            "triplet_metadata": str(triplet_metadata_path),
        },
        "settings": {
            "device": str(device),
            "refinement_steps": int(args.refinement_steps),
            "blend_alpha": float(args.blend_alpha),
            "blend_alpha_step2": (
                None if args.blend_alpha_step2 is None else float(args.blend_alpha_step2)
            ),
            "route_topk_tool": args.route_topk_tool,
            "route_topk_verb": args.route_topk_verb,
            "route_topk_target": args.route_topk_target,
            "routing_probability_source": args.routing_probability_source,
            "support_constrained_routing": bool(support_pair_constraints),
            "support_constrained_routing_mode": support_constrained_routing_mode,
            "support_constrained_routing_unsupported_scale": float(
                args.support_constrained_routing_unsupported_scale
            ),
            "decode_topk_tool": args.decode_topk_tool,
            "decode_topk_verb": args.decode_topk_verb,
            "decode_topk_target": args.decode_topk_target,
            "decode_max_triplets_total": args.decode_max_triplets_total,
            "decode_score_min": args.decode_score_min,
            "support_aware_triplet_refinement": bool(args.support_aware_triplet_refinement),
            "target_relabel_mode": args.target_relabel_mode,
            "target_relabel_plan": (
                relabel_plan.to_serializable() if relabel_plan is not None else None
            ),
            "calibration_json": (
                str(args.calibration_json.resolve()) if args.calibration_json is not None else None
            ),
            "head_temperatures_is_identity": head_temperatures.is_identity(),
            "bootstrap_iterations": int(args.bootstrap_iterations),
            "bootstrap_seed": int(args.bootstrap_seed),
            "bootstrap_confidence": float(args.bootstrap_confidence),
            "tool_thresholds": [float(value) for value in tool_thresholds.tolist()],
            "verb_thresholds": [float(value) for value in verb_thresholds.tolist()],
            "target_thresholds": [float(value) for value in target_thresholds.tolist()],
            "route_tool_thresholds": [float(value) for value in route_tool_thresholds.tolist()],
            "route_verb_thresholds": [float(value) for value in route_verb_thresholds.tolist()],
            "route_target_thresholds": [float(value) for value in route_target_thresholds.tolist()],
        },
        "provenance": provenance,
        "bootstrap": bootstrap_block,
        "sharding": {
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "max_samples_limit": int(max_samples),
            "evaluated_global_indices": shard_sample_indices,
            "candidate_clips_before_sharding": int(max_samples),
        },
        "raw_counts": {
            "joint_exact_correct": int(joint_exact_correct),
            "unsupported_predicted_triplets": int(unsupported_predicted_triplets),
            "suppressed_unsupported_triplets": int(suppressed_unsupported_triplets),
            "support_search_tools": int(support_search_tools),
            "support_search_triplets": int(support_search_triplets),
            "tool": tool_metrics.raw_counts(),
            "verb": verb_metrics.raw_counts(),
            "target": target_metrics.raw_counts(),
            "triplet": triplet_metrics.raw_counts(),
        },
        "per_sample_masks": per_sample_block,
    }

    print(
        f"Evaluated {evaluated_samples} validation clips "
        f"(shard {args.shard_index + 1}/{args.num_shards})"
    )
    print(f"Joint exact tool/verb/target accuracy: {joint_exact_acc:.4f}%")
    if args.support_aware_triplet_refinement:
        print(f"Suppressed unsupported raw triplets: {suppressed_unsupported_triplets}")
        print(f"Support-aware searched tools: {support_search_tools}")
        print(f"Support-aware selected triplets: {support_search_triplets}")
    print(f"Unsupported predicted triplets: {unsupported_predicted_triplets}")
    print_metrics_block("tool", tool_summary)
    print_metrics_block("verb", verb_summary)
    print_metrics_block("target", target_summary)
    print_metrics_block("triplet", triplet_summary)

    if bootstrap_block:
        for name in ("tool", "verb", "target", "triplet"):
            block = bootstrap_block.get(name)
            if not block:
                continue
            macro = block["macro_f1"]
            micro = block["micro_f1"]
            print(
                f"{name} bootstrap (n={block['num_samples']}, "
                f"iters={block['iterations']}, conf={block['confidence']}): "
                f"MacroF1 mean={macro['mean']:.4f}% "
                f"[{macro['low']:.4f}%, {macro['high']:.4f}%] "
                f"MicroF1 mean={micro['mean']:.4f}% "
                f"[{micro['low']:.4f}%, {micro['high']:.4f}%]"
            )

    if args.output_json is not None:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"Saved JSON summary to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
