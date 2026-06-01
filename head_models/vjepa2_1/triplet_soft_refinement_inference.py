#!/path/to/data_root/vjepa2_1/.venv/bin/python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from triplet_conditioned_inference import (
    DEFAULT_CLIP_FPS,
    DEFAULT_CLIP_SECONDS,
    DEFAULT_INPUT_VIDEO,
    DEFAULT_STRIDE_SECONDS,
    FALLBACK_TARGET_NAMES,
    FALLBACK_TOOL_NAMES,
    FALLBACK_VERB_NAMES,
    SequentialVideoSampler,
    annotate_video,
    build_classifier_from_config,
    build_transform,
    build_window_starts,
    class_names_from_config,
    compare_encoder_settings,
    derive_checkpoint_path,
    init_encoder_module,
    load_classifier_weights,
    load_yaml_config,
    lookup_name,
    maybe_suppress_stdout,
    override_backbone_checkpoint,
    prepare_encoder_inputs,
    print_prediction_distribution,
    resolve_device,
    resolve_multilabel_threshold,
    select_multilabel_indices,
    validate_temporal_sampling,
)


DEFAULT_TOOL0_CONFIG = Path(
    "/path/to/data_root/vjepa2_1/configs/heads/surg_finetuned_phase3_conditioning_study/"
    "triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5_classic_surg_finetuned_phase3.yaml"
)
DEFAULT_VERB0_CONFIG = Path(
    "/path/to/data_root/vjepa2_1/configs/heads/surg_finetuned_phase3_conditioning_study/"
    "verb_multilabel_only_train_vitl_latest_token_aggregation_tool5_surg_finetuned_phase3.yaml"
)
DEFAULT_TARGET0_CONFIG = Path(
    "/path/to/data_root/vjepa2_1/configs/heads/surg_finetuned_phase3_conditioning_study/"
    "target_multilabel_only_train_vitl_latest_token_aggregation_tool5_surg_finetuned_phase3.yaml"
)
DEFAULT_TOOL_REFINE_CONFIG = Path(
    "/path/to/data_root/vjepa2_1/configs/heads/surg_finetuned_phase3_conditioning_study/"
    "tool_multilabel_conditioned_on_verb_target_train_vitl_latest_token_aggregation_tool5_surg_finetuned_phase3.yaml"
)
DEFAULT_VERB_REFINE_CONFIG = Path(
    "/path/to/data_root/vjepa2_1/configs/heads/surg_finetuned_phase3_conditioning_study/"
    "verb_multilabel_conditioned_on_tool_target_train_vitl_latest_token_aggregation_tool5_surg_finetuned_phase3.yaml"
)
DEFAULT_TARGET_REFINE_CONFIG = Path(
    "/path/to/data_root/vjepa2_1/configs/heads/surg_finetuned_phase3_conditioning_study/"
    "target_multilabel_conditioned_on_tool_verb_train_vitl_latest_token_aggregation_tool5_surg_finetuned_phase3.yaml"
)
DEFAULT_TRIPLET_METADATA = Path(
    "/path/to/data_root/vjepa2_1/app/csv_head_models/triplet_multilabel_native_tool5_metadata.json"
)
DEFAULT_TOOL0_PHASE2_CHECKPOINT = Path(
    "/path/to/data_root/vjepa2_1/trained_heads/"
    "triplet_multilabel_tool_only_tool5_native_classic_surg_finetuned_phase2/"
    "video_classification_frozen/"
    "triplet_multilabel_tool_only_head_training_vitl_latest_token_aggregation_"
    "tool5_native_classic_surg_finetuned_phase2/latest.pt"
)
DEFAULT_TOOL0_PHASE3_CHECKPOINT = Path(
    "/path/to/data_root/vjepa2_1/trained_heads/"
    "triplet_multilabel_tool_only_tool5_native_classic_surg_finetuned_phase3/"
    "video_classification_frozen/"
    "triplet_multilabel_tool_only_head_training_vitl_latest_token_aggregation_"
    "tool5_native_classic_surg_finetuned_phase3/latest.pt"
)
DEFAULT_TARGET_REFINE_PHASE3_CHECKPOINT = Path(
    "/path/to/data_root/vjepa2_1/trained_heads/"
    "target_multilabel_conditioned_on_tool_verb_tool5_native_surg_finetuned_phase3/"
    "video_classification_frozen/"
    "target_multilabel_conditioned_on_tool_verb_head_training_vitl_latest_token_aggregation_"
    "tool5_native_surg_finetuned_phase3/latest.pt"
)
SOFT_REFINEMENT_CHECKPOINT_FALLBACKS = {
    DEFAULT_TARGET_REFINE_CONFIG.name: DEFAULT_TARGET_REFINE_PHASE3_CHECKPOINT,
}

GROUPED_VERB_NAMES_7 = (
    "coagulation",
    "grasp/retract",
    "null",
    "clip",
    "cut",
    "dissect",
    "clean",
)
STUDY_VERB_NAMES_6 = (
    "coagulation",
    "grasp/retract",
    "null",
    "cut/dissect",
    "clean",
    "clip",
)
GROUPED_TO_STUDY_VERB_INDEX = {
    0: 0,
    1: 1,
    2: 2,
    4: 3,
    5: 3,
    6: 4,
    3: 5,
}
STUDY_TO_GROUPED_VERB_INDICES = {
    0: (0,),
    1: (1,),
    2: (2,),
    3: (4, 5),
    4: (6,),
    5: (3,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run soft triplet refinement inference on a single video using the existing "
            "tool-only / verb-only / target-only heads plus the pair-conditioned heads."
        )
    )
    parser.add_argument("--input-video", type=Path, default=DEFAULT_INPUT_VIDEO)
    parser.add_argument("--output-video", type=Path, default=None)

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
            "With --head-checkpoint-policy best, fail instead of falling back to "
            "latest.pt when no best checkpoint exists."
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
            "Override the encoder key inside --backbone-checkpoint. "
            "If omitted, the script keeps the config key when present, otherwise tries "
            "target_encoder then encoder."
        ),
    )

    parser.add_argument("--clip-seconds", type=float, default=DEFAULT_CLIP_SECONDS)
    parser.add_argument("--stride-seconds", type=float, default=DEFAULT_STRIDE_SECONDS)
    parser.add_argument("--clip-fps", type=float, default=DEFAULT_CLIP_FPS)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--max-windows", type=int, default=None)

    parser.add_argument("--tool-threshold", type=float, default=None)
    parser.add_argument("--verb-threshold", type=float, default=None)
    parser.add_argument("--target-threshold", type=float, default=None)
    parser.add_argument("--max-tools", type=int, default=None)
    parser.add_argument("--max-verbs-per-tool", type=int, default=None)
    parser.add_argument("--max-targets-per-pair", type=int, default=None)
    parser.add_argument("--max-triplets-display", type=int, default=3)
    parser.add_argument("--distribution-top-k", type=int, default=10)
    parser.add_argument(
        "--triplet-metadata",
        type=Path,
        default=DEFAULT_TRIPLET_METADATA,
        help="Metadata JSON with the supported Tool5 triplet set.",
    )

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
            "Decode final predictions through the supported triplet set and promote the "
            "best compatible alternative when a high-confidence tool/verb/target choice "
            "would otherwise produce an unsupported triplet."
        ),
    )
    parser.add_argument(
        "--decode-topk-tool",
        type=int,
        default=None,
        help=(
            "Optional top-k budget for support-aware triplet decoding over threshold-passing "
            "tools. Defaults to --max-tools when provided, otherwise --route-topk-tool."
        ),
    )
    parser.add_argument(
        "--decode-topk-verb",
        type=int,
        default=None,
        help=(
            "Optional top-k budget for support-aware triplet decoding over threshold-passing "
            "verbs per tool. Defaults to --max-verbs-per-tool when provided, otherwise "
            "--route-topk-verb."
        ),
    )
    parser.add_argument(
        "--decode-topk-target",
        type=int,
        default=None,
        help=(
            "Optional top-k budget for support-aware triplet decoding over threshold-passing "
            "targets per tool/verb pair. Defaults to --max-targets-per-pair when provided, "
            "otherwise --route-topk-target."
        ),
    )
    parser.add_argument(
        "--decode-max-triplets-total",
        type=int,
        default=None,
        help="Optional cap on the total number of support-aware triplets emitted per window.",
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
            "Optional JSON file with per-head temperatures, used to apply temperature "
            "scaling on the head logits before sigmoid. The file must expose a "
            "'temperatures' object keyed by head name (tool0, verb0, target0, "
            "tool_refine, verb_refine, target_refine); each value may be a scalar or a "
            "per-class list. Identity (1.0) is the default and matches uncalibrated runs."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Print setup and per-window progress.")
    return parser.parse_args()


def resolve_output_path(input_video: Path, output_video: Path | None) -> Path:
    if output_video is not None:
        return output_video
    return input_video.with_name(f"{input_video.stem}_soft_refinement_inference.mp4")


def _get_head_type(config: dict[str, Any]) -> str:
    return str(config["experiment"]["classifier"].get("head_type", "single_head_pooler"))


def _get_num_classes(config: dict[str, Any]) -> int:
    data_cfg = config["experiment"]["data"]
    num_classes_per_task = data_cfg.get("num_classes_per_task")
    if num_classes_per_task is None:
        return int(data_cfg["num_classes"])
    return int(num_classes_per_task[0])


def _normalize_class_name(name: str) -> str:
    return str(name).strip().lower()


def _get_class_names(config: dict[str, Any]) -> list[str]:
    data_cfg = config["experiment"]["data"]
    class_names_per_task = data_cfg.get("class_names_per_task") or []
    if class_names_per_task and class_names_per_task[0]:
        return [str(name) for name in class_names_per_task[0]]
    num_classes = _get_num_classes(config)
    return [f"class_{idx}" for idx in range(num_classes)]


def uses_grouped_to_study_verb_adapter(
    verb0_cfg: dict[str, Any],
    verb_refine_cfg: dict[str, Any],
) -> bool:
    if _get_num_classes(verb0_cfg) != 7 or _get_num_classes(verb_refine_cfg) != 6:
        return False

    grouped_names = tuple(_normalize_class_name(name) for name in _get_class_names(verb0_cfg))
    study_names = tuple(_normalize_class_name(name) for name in _get_class_names(verb_refine_cfg))
    return grouped_names == GROUPED_VERB_NAMES_7 and study_names == STUDY_VERB_NAMES_6


def grouped_verb_probs_to_study_probs(grouped_probs: torch.Tensor) -> torch.Tensor:
    if grouped_probs.ndim != 1 or grouped_probs.numel() != 7:
        raise ValueError(f"Expected grouped 7-way verb probabilities, got shape {tuple(grouped_probs.shape)}")

    study_probs = grouped_probs.new_zeros((6,))
    study_probs[0] = grouped_probs[0]
    study_probs[1] = grouped_probs[1]
    study_probs[2] = grouped_probs[2]
    study_probs[3] = 1.0 - ((1.0 - grouped_probs[4]) * (1.0 - grouped_probs[5]))
    study_probs[4] = grouped_probs[6]
    study_probs[5] = grouped_probs[3]
    return study_probs


def study_verb_probs_to_grouped_probs(
    study_probs: torch.Tensor,
    reference_grouped_probs: torch.Tensor,
) -> torch.Tensor:
    if study_probs.ndim != 1 or study_probs.numel() != 6:
        raise ValueError(f"Expected study 6-way verb probabilities, got shape {tuple(study_probs.shape)}")
    if reference_grouped_probs.ndim != 1 or reference_grouped_probs.numel() != 7:
        raise ValueError(
            f"Expected grouped 7-way reference probabilities, got shape {tuple(reference_grouped_probs.shape)}"
        )

    grouped_probs = reference_grouped_probs.new_zeros((7,))
    grouped_probs[0] = study_probs[0]
    grouped_probs[1] = study_probs[1]
    grouped_probs[2] = study_probs[2]
    grouped_probs[3] = study_probs[5]
    grouped_probs[6] = study_probs[4]

    cut_ref = reference_grouped_probs[4].clamp_min(0.0)
    dissect_ref = reference_grouped_probs[5].clamp_min(0.0)
    split_total = cut_ref + dissect_ref
    if float(split_total.item()) <= 1e-8:
        cut_ratio = 0.5
        dissect_ratio = 0.5
    else:
        cut_ratio = float((cut_ref / split_total).item())
        dissect_ratio = float((dissect_ref / split_total).item())
    grouped_probs[4] = study_probs[3] * cut_ratio
    grouped_probs[5] = study_probs[3] * dissect_ratio
    return grouped_probs


def map_study_verb_id_to_grouped_ids(study_verb_id: int, use_adapter: bool) -> tuple[int, ...]:
    if not use_adapter:
        return (int(study_verb_id),)
    return STUDY_TO_GROUPED_VERB_INDICES[int(study_verb_id)]


def grouped_verb_id_to_study_verb_id(verb_id: int, use_adapter: bool) -> int:
    if not use_adapter:
        return int(verb_id)
    return GROUPED_TO_STUDY_VERB_INDEX[int(verb_id)]


def load_triplet_support_index(
    metadata_path: Path,
) -> tuple[dict[tuple[int, int, int], int], int, tuple[tuple[int, int, int], ...]]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    id_to_triplet = metadata.get("id_to_triplet")
    if not isinstance(id_to_triplet, list):
        raise ValueError(f"Metadata file does not contain a usable id_to_triplet list: {metadata_path}")

    num_triplet_classes = int(metadata.get("num_triplet_classes", len(id_to_triplet)))
    triplet_to_id: dict[tuple[int, int, int], int] = {}
    supported_triplets: list[tuple[int, int, int]] = []
    for entry in id_to_triplet:
        key = (
            int(entry["tool_id"]),
            int(entry["verb_id"]),
            int(entry["target_id"]),
        )
        triplet_to_id[key] = int(entry["triplet_id"])
        supported_triplets.append(key)
    return triplet_to_id, num_triplet_classes, tuple(supported_triplets)


def build_support_pair_constraints(
    supported_triplets: tuple[tuple[int, int, int], ...],
    use_verb_adapter: bool,
) -> dict[str, frozenset[tuple[int, int]]]:
    tool_pairs: set[tuple[int, int]] = set()
    verb_pairs: set[tuple[int, int]] = set()
    target_pairs: set[tuple[int, int]] = set()

    for tool_id, verb_id, target_id in supported_triplets:
        study_verb_id = grouped_verb_id_to_study_verb_id(verb_id, use_verb_adapter)
        tool_pairs.add((int(study_verb_id), int(target_id)))
        verb_pairs.add((int(tool_id), int(target_id)))
        target_pairs.add((int(tool_id), int(study_verb_id)))

    return {
        "tool": frozenset(tool_pairs),
        "verb": frozenset(verb_pairs),
        "target": frozenset(target_pairs),
    }


def resolve_support_constrained_routing_mode(
    explicit_mode: str | None,
    legacy_flag: bool,
) -> str:
    mode = str(explicit_mode or "none").strip().lower()
    if mode not in {"none", "soft", "hard"}:
        raise ValueError(
            f"Unsupported support-constrained routing mode {explicit_mode!r}; "
            "expected one of: none, soft, hard"
        )
    if mode == "none" and legacy_flag:
        return "hard"
    return mode


def build_conditioned_probability_lookup(
    pair_indices: torch.Tensor,
    conditioned_probs: torch.Tensor,
) -> dict[tuple[int, int], torch.Tensor]:
    pair_indices = pair_indices.detach().cpu()
    conditioned_probs = conditioned_probs.detach().cpu().float()
    lookup: dict[tuple[int, int], torch.Tensor] = {}
    for pair_idx, pair in enumerate(pair_indices.tolist()):
        lookup[(int(pair[0]), int(pair[1]))] = conditioned_probs[pair_idx]
    return lookup


def build_grouped_verb_conditioned_lookup(
    refinement: dict[str, Any],
) -> dict[tuple[int, int], torch.Tensor]:
    verb_lookup = build_conditioned_probability_lookup(
        refinement["final_verb_pairs"],
        refinement["final_verb_conditioned_probs"],
    )
    if not bool(refinement.get("use_grouped_to_study_verb_adapter", False)):
        return verb_lookup

    final_grouped_verb_probs = refinement["final"]["verb"].detach().cpu().float()
    grouped_lookup: dict[tuple[int, int], torch.Tensor] = {}
    for key, study_probs in verb_lookup.items():
        grouped_lookup[key] = study_verb_probs_to_grouped_probs(
            study_probs,
            final_grouped_verb_probs,
        ).detach().cpu().float()
    return grouped_lookup


def resolve_support_search_budget(
    explicit_topk: int | None,
    raw_mask: torch.Tensor,
    fallback_top1: bool,
) -> int:
    if explicit_topk is not None:
        return max(0, int(explicit_topk))

    raw_count = int(raw_mask.detach().cpu().to(dtype=torch.int64).sum().item())
    if raw_count > 0:
        return raw_count
    return 1 if fallback_top1 else 0


def select_threshold_candidate_indices(
    probabilities: torch.Tensor,
    thresholds: torch.Tensor,
    explicit_topk: int | None,
    fallback_top1: bool,
) -> list[int]:
    probabilities = probabilities.detach().cpu().float().reshape(-1)
    thresholds = thresholds.detach().cpu().float().reshape(-1)
    active_indices = (probabilities >= thresholds).nonzero(as_tuple=False).reshape(-1)
    if active_indices.numel() > 0:
        k = int(active_indices.numel()) if explicit_topk is None else min(
            int(explicit_topk),
            int(active_indices.numel()),
        )
        if k <= 0:
            return []
        active_probs = probabilities.index_select(0, active_indices)
        _, order = torch.topk(active_probs, k=k)
        return [int(active_indices[idx].item()) for idx in order.tolist()]

    if not fallback_top1 or probabilities.numel() == 0:
        return []

    k = 1 if explicit_topk is None else min(int(explicit_topk), int(probabilities.numel()))
    if k <= 0:
        return []
    _, indices = torch.topk(probabilities, k=k)
    return [int(idx) for idx in indices.tolist()]


def compute_supported_triplet_score(
    tool_probability: float,
    verb_probability: float,
    target_probability: float,
    tool_conditioned_probability: float,
    verb_conditioned_probability: float,
    target_conditioned_probability: float,
) -> float:
    eps = 1e-8
    terms = (
        float(tool_probability),
        float(verb_probability),
        float(target_probability),
        float(tool_conditioned_probability),
        float(verb_conditioned_probability),
        float(target_conditioned_probability),
    )
    log_mean = sum(math.log(max(term, eps)) for term in terms) / float(len(terms))
    return float(math.exp(log_mean))


def decode_support_aware_triplets(
    refinement: dict[str, Any],
    triplet_to_id: dict[tuple[int, int, int], int],
    num_triplet_classes: int,
    tool_thresholds: torch.Tensor,
    verb_thresholds: torch.Tensor,
    target_thresholds: torch.Tensor,
    search_topk_tool: int | None = None,
    search_topk_verb: int | None = None,
    search_topk_target: int | None = None,
    max_triplets_total: int | None = None,
    min_triplet_score: float | None = None,
    fallback_top1: bool = False,
) -> dict[str, Any]:
    tool_probs = refinement["final"]["tool"].detach().cpu().float()
    verb_probs = refinement["final"]["verb"].detach().cpu().float()
    global_target_probs = refinement["final"]["target"].detach().cpu().float()

    tool_thresholds = tool_thresholds.detach().cpu().float().reshape(-1)
    verb_thresholds = verb_thresholds.detach().cpu().float().reshape(-1)
    target_thresholds = target_thresholds.detach().cpu().float().reshape(-1)

    raw_tool_mask = tool_probs >= tool_thresholds
    raw_verb_mask = verb_probs >= verb_thresholds
    raw_target_mask = global_target_probs >= target_thresholds

    use_verb_adapter = bool(refinement.get("use_grouped_to_study_verb_adapter", False))
    target_cond_lookup = build_conditioned_probability_lookup(
        refinement["final_target_pairs"],
        refinement["final_target_conditioned_probs"],
    )
    tool_cond_lookup = build_conditioned_probability_lookup(
        refinement["final_tool_pairs"],
        refinement["final_tool_conditioned_probs"],
    )
    verb_cond_lookup = build_grouped_verb_conditioned_lookup(refinement)

    selected_triplets: list[dict[str, Any]] = []
    selected_triplet_ids: set[int] = set()
    selected_tool_mask = torch.zeros_like(raw_tool_mask)
    selected_verb_mask = torch.zeros_like(raw_verb_mask)
    selected_target_mask = torch.zeros_like(raw_target_mask)
    triplet_mask = torch.zeros(num_triplet_classes, dtype=torch.bool)

    def register_candidate(candidate: dict[str, Any]) -> bool:
        triplet_id = int(candidate["triplet_id"])
        if triplet_id in selected_triplet_ids:
            return False
        selected_triplet_ids.add(triplet_id)
        selected_triplets.append(candidate)
        triplet_mask[triplet_id] = True
        selected_tool_mask[int(candidate["tool_id"])] = True
        selected_verb_mask[int(candidate["verb_id"])] = True
        selected_target_mask[int(candidate["target_id"])] = True
        return True

    verb_budget = max(1, resolve_support_search_budget(search_topk_verb, raw_verb_mask, fallback_top1))
    target_budget = max(1, resolve_support_search_budget(search_topk_target, raw_target_mask, fallback_top1))
    tool_candidate_indices = select_threshold_candidate_indices(
        tool_probs,
        tool_thresholds,
        search_topk_tool,
        fallback_top1,
    )

    supported_triplets_by_tool: dict[int, list[tuple[int, int, int, int]]] = {}
    for (tool_id, verb_id, target_id), triplet_id in triplet_to_id.items():
        supported_triplets_by_tool.setdefault(int(tool_id), []).append(
            (
                int(verb_id),
                int(target_id),
                int(triplet_id),
                grouped_verb_id_to_study_verb_id(int(verb_id), use_verb_adapter),
            )
        )

    support_search_tools = 0
    support_search_triplets = 0
    if tool_candidate_indices:
        for tool_id in tool_candidate_indices:
            rows = []
            for verb_id, target_id, triplet_id, study_verb_id in supported_triplets_by_tool.get(tool_id, []):
                target_cond_probs = target_cond_lookup.get((tool_id, study_verb_id))
                tool_cond_probs = tool_cond_lookup.get((study_verb_id, target_id))
                verb_cond_probs = verb_cond_lookup.get((tool_id, target_id))

                tool_probability = float(tool_probs[tool_id].item())
                verb_probability = float(verb_probs[verb_id].item())
                target_probability = float(global_target_probs[target_id].item())
                tool_conditioned_probability = (
                    float(tool_cond_probs[tool_id].item())
                    if tool_cond_probs is not None
                    else tool_probability
                )
                verb_conditioned_probability = (
                    float(verb_cond_probs[verb_id].item())
                    if verb_cond_probs is not None
                    else verb_probability
                )
                target_conditioned_probability = (
                    float(target_cond_probs[target_id].item())
                    if target_cond_probs is not None
                    else target_probability
                )
                verb_threshold = float(verb_thresholds[verb_id].item())
                target_threshold = float(target_thresholds[target_id].item())
                verb_is_active = (
                    bool(raw_verb_mask[verb_id].item())
                    or verb_conditioned_probability >= verb_threshold
                )
                target_is_active = (
                    bool(raw_target_mask[target_id].item())
                    or target_conditioned_probability >= target_threshold
                )
                rows.append(
                    {
                        "triplet_id": int(triplet_id),
                        "tool_id": tool_id,
                        "verb_id": int(verb_id),
                        "target_id": int(target_id),
                        "tool_probability": tool_probability,
                        "verb_probability": verb_probability,
                        "target_probability": target_probability,
                        "tool_conditioned_probability": tool_conditioned_probability,
                        "verb_conditioned_probability": verb_conditioned_probability,
                        "target_conditioned_probability": target_conditioned_probability,
                        "verb_is_active": verb_is_active,
                        "target_is_active": target_is_active,
                        "score": compute_supported_triplet_score(
                            tool_probability=tool_probability,
                            verb_probability=verb_probability,
                            target_probability=target_probability,
                            tool_conditioned_probability=tool_conditioned_probability,
                            verb_conditioned_probability=verb_conditioned_probability,
                            target_conditioned_probability=target_conditioned_probability,
                        ),
                    }
                )

            if not rows:
                continue

            candidate_rows = [
                row
                for row in rows
                if bool(row["verb_is_active"]) and bool(row["target_is_active"])
            ]
            if not candidate_rows and fallback_top1:
                candidate_rows = [
                    row
                    for row in rows
                    if bool(row["verb_is_active"]) or bool(row["target_is_active"])
                ]
            if not candidate_rows and fallback_top1:
                candidate_rows = rows
            if not candidate_rows:
                continue
            if min_triplet_score is not None:
                candidate_rows = [
                    row
                    for row in candidate_rows
                    if float(row["score"]) >= float(min_triplet_score)
                ]
            if not candidate_rows:
                continue

            support_search_tools += 1
            best_score_by_verb: dict[int, float] = {}
            for row in candidate_rows:
                verb_id = int(row["verb_id"])
                row_score = float(row["score"])
                if verb_id not in best_score_by_verb or row_score > best_score_by_verb[verb_id]:
                    best_score_by_verb[verb_id] = row_score

            ranked_verbs = sorted(best_score_by_verb.items(), key=lambda item: item[1], reverse=True)
            selected_verb_ids = [
                verb_id
                for verb_id, _ in ranked_verbs[: min(verb_budget, len(ranked_verbs))]
            ]
            for verb_id in selected_verb_ids:
                verb_rows = [
                    row for row in candidate_rows if int(row["verb_id"]) == int(verb_id)
                ]
                verb_rows.sort(key=lambda row: float(row["score"]), reverse=True)
                for row in verb_rows[: min(target_budget, len(verb_rows))]:
                    if register_candidate(row):
                        support_search_triplets += 1

    if not selected_triplets and fallback_top1:
        best_candidate = None
        best_score = None
        for tool_id, rows in supported_triplets_by_tool.items():
            for verb_id, target_id, triplet_id, study_verb_id in rows:
                target_cond_probs = target_cond_lookup.get((tool_id, study_verb_id))
                tool_cond_probs = tool_cond_lookup.get((study_verb_id, target_id))
                verb_cond_probs = verb_cond_lookup.get((tool_id, target_id))

                tool_probability = float(tool_probs[tool_id].item())
                verb_probability = float(verb_probs[verb_id].item())
                target_probability = float(global_target_probs[target_id].item())
                candidate_score = compute_supported_triplet_score(
                    tool_probability=tool_probability,
                    verb_probability=verb_probability,
                    target_probability=target_probability,
                    tool_conditioned_probability=(
                        float(tool_cond_probs[tool_id].item())
                        if tool_cond_probs is not None
                        else tool_probability
                    ),
                    verb_conditioned_probability=(
                        float(verb_cond_probs[verb_id].item())
                        if verb_cond_probs is not None
                        else verb_probability
                    ),
                    target_conditioned_probability=(
                        float(target_cond_probs[target_id].item())
                        if target_cond_probs is not None
                        else target_probability
                    ),
                )
                if (
                    min_triplet_score is not None
                    and candidate_score < float(min_triplet_score)
                ):
                    continue
                if best_score is None or candidate_score > best_score:
                    best_score = candidate_score
                    best_candidate = {
                        "triplet_id": int(triplet_id),
                        "tool_id": int(tool_id),
                        "verb_id": int(verb_id),
                        "target_id": int(target_id),
                        "tool_probability": tool_probability,
                        "verb_probability": verb_probability,
                        "target_probability": target_probability,
                        "score": candidate_score,
                    }
        if best_candidate is not None and register_candidate(best_candidate):
            support_search_triplets += 1

    selected_triplets.sort(key=lambda item: item["score"], reverse=True)
    if max_triplets_total is not None:
        selected_triplets = selected_triplets[: min(int(max_triplets_total), len(selected_triplets))]
        selected_tool_mask.zero_()
        selected_verb_mask.zero_()
        selected_target_mask.zero_()
        triplet_mask.zero_()
        for candidate in selected_triplets:
            triplet_mask[int(candidate["triplet_id"])] = True
            selected_tool_mask[int(candidate["tool_id"])] = True
            selected_verb_mask[int(candidate["verb_id"])] = True
            selected_target_mask[int(candidate["target_id"])] = True
    support_search_triplets = len(selected_triplets)

    return {
        "tool_mask": selected_tool_mask,
        "verb_mask": selected_verb_mask,
        "target_mask": selected_target_mask,
        "triplet_mask": triplet_mask,
        "triplets": selected_triplets,
        "raw_tool_mask": raw_tool_mask,
        "raw_verb_mask": raw_verb_mask,
        "raw_target_mask": raw_target_mask,
        "num_support_search_tools": int(support_search_tools),
        "num_support_search_triplets": int(support_search_triplets),
    }


def validate_soft_refinement_configs(
    tool0_cfg: dict[str, Any],
    verb0_cfg: dict[str, Any],
    target0_cfg: dict[str, Any],
    tool_refine_cfg: dict[str, Any],
    verb_refine_cfg: dict[str, Any],
    target_refine_cfg: dict[str, Any],
) -> None:
    reference_cfg = tool0_cfg
    for name, candidate_cfg in (
        ("verb0", verb0_cfg),
        ("target0", target0_cfg),
        ("tool_refine", tool_refine_cfg),
        ("verb_refine", verb_refine_cfg),
        ("target_refine", target_refine_cfg),
    ):
        compare_encoder_settings(reference_cfg, candidate_cfg, name)

    for name, cfg in (
        ("tool0", tool0_cfg),
        ("verb0", verb0_cfg),
        ("target0", target0_cfg),
    ):
        if _get_head_type(cfg) == "conditioned_token_aggregation":
            raise ValueError(f"{name} must be an unconditioned head, got conditioned_token_aggregation")

    use_verb_adapter = uses_grouped_to_study_verb_adapter(verb0_cfg, verb_refine_cfg)
    verb_condition_classes = 6 if use_verb_adapter else _get_num_classes(verb0_cfg)

    expected_pairs = {
        "tool_refine": (verb_condition_classes, _get_num_classes(target0_cfg)),
        "verb_refine": (_get_num_classes(tool0_cfg), _get_num_classes(target0_cfg)),
        "target_refine": (_get_num_classes(tool0_cfg), verb_condition_classes),
    }
    refine_cfgs = {
        "tool_refine": tool_refine_cfg,
        "verb_refine": verb_refine_cfg,
        "target_refine": target_refine_cfg,
    }
    for name, cfg in refine_cfgs.items():
        head_type = _get_head_type(cfg)
        if head_type != "conditioned_token_aggregation":
            raise ValueError(f"{name} must use conditioned_token_aggregation, got {head_type!r}")
        condition_num_classes = cfg["experiment"]["classifier"].get("condition_num_classes")
        if condition_num_classes is None:
            raise ValueError(f"{name} is missing classifier.condition_num_classes")
        if isinstance(condition_num_classes, int):
            actual = (int(condition_num_classes),)
        else:
            actual = tuple(int(value) for value in condition_num_classes)
        if actual != expected_pairs[name]:
            raise ValueError(
                f"{name} expected condition_num_classes={expected_pairs[name]}, got {actual}"
            )

    if _get_num_classes(tool0_cfg) != _get_num_classes(tool_refine_cfg):
        raise ValueError("tool0 and tool_refine predict different tool class counts")
    if not use_verb_adapter and _get_num_classes(verb0_cfg) != _get_num_classes(verb_refine_cfg):
        raise ValueError("verb0 and verb_refine predict different verb class counts")
    if _get_num_classes(target0_cfg) != _get_num_classes(target_refine_cfg):
        raise ValueError("target0 and target_refine predict different target class counts")


def validate_topk(name: str, value: int | None) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be >= 1 when provided, got {value}")


def resolve_tool0_resolution_fallback(config: dict[str, Any]) -> Path | None:
    resolution = int(config.get("experiment", {}).get("data", {}).get("resolution", 0))
    if resolution >= 512 and DEFAULT_TOOL0_PHASE3_CHECKPOINT.exists():
        return DEFAULT_TOOL0_PHASE3_CHECKPOINT
    if resolution > 0 and resolution <= 256 and DEFAULT_TOOL0_PHASE2_CHECKPOINT.exists():
        return DEFAULT_TOOL0_PHASE2_CHECKPOINT
    if DEFAULT_TOOL0_PHASE3_CHECKPOINT.exists():
        return DEFAULT_TOOL0_PHASE3_CHECKPOINT
    if DEFAULT_TOOL0_PHASE2_CHECKPOINT.exists():
        return DEFAULT_TOOL0_PHASE2_CHECKPOINT
    return None


def resolve_soft_refinement_checkpoint_path(
    config: dict[str, Any],
    config_path: Path,
    explicit_path: Path | None,
    fallback_name: str,
    checkpoint_policy: str = "best",
    strict_best: bool = False,
) -> Path:
    candidate_path = derive_checkpoint_path(
        config,
        explicit_path,
        fallback_name=fallback_name,
        checkpoint_policy=checkpoint_policy,
        strict_best=strict_best,
    )
    if candidate_path.exists():
        return candidate_path
    if explicit_path is not None:
        raise FileNotFoundError(f"Classifier checkpoint does not exist: {candidate_path}")

    fallback_path = None
    if Path(config_path).name == DEFAULT_TOOL0_CONFIG.name:
        fallback_path = resolve_tool0_resolution_fallback(config)
    if fallback_path is None:
        fallback_path = SOFT_REFINEMENT_CHECKPOINT_FALLBACKS.get(Path(config_path).name)
    if fallback_path is not None and fallback_path.exists():
        return fallback_path
    raise FileNotFoundError(f"Classifier checkpoint does not exist: {candidate_path}")


def resolve_multilabel_threshold_vector(
    config: dict[str, Any],
    explicit_value: float | None = None,
) -> torch.Tensor:
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


def normalize_routing_distribution(
    probabilities: torch.Tensor,
    topk: int | None = None,
    threshold_vector: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = probabilities.detach().float().clamp_min(0.0)
    if probs.ndim != 1:
        raise ValueError(f"Expected a 1D probability vector, got shape {tuple(probs.shape)}")
    if probs.numel() == 0:
        raise ValueError("Cannot build a routing distribution from an empty tensor")

    selected_indices = None
    selected_values = None
    if threshold_vector is not None:
        threshold_vector = threshold_vector.to(device=probs.device, dtype=torch.float32).reshape(-1)
        if threshold_vector.numel() != probs.numel():
            raise ValueError(
                f"Threshold vector length {threshold_vector.numel()} does not match "
                f"probability length {probs.numel()}"
            )
        support_mask = probs >= threshold_vector
        if support_mask.any():
            selected_indices = support_mask.nonzero(as_tuple=False).reshape(-1)
            selected_values = probs.index_select(0, selected_indices)

    if selected_indices is None or selected_values is None:
        if topk is not None and topk < probs.numel():
            selected_values, selected_indices = torch.topk(probs, k=topk)
        else:
            selected_indices = torch.arange(probs.numel(), device=probs.device, dtype=torch.long)
            selected_values = probs
    else:
        if topk is not None and topk < selected_values.numel():
            top_values, top_relative_indices = torch.topk(selected_values, k=topk)
            selected_indices = selected_indices.index_select(0, top_relative_indices)
            selected_values = top_values

    total = float(selected_values.sum().item())
    if total <= 0.0:
        selected_values = torch.full_like(selected_values, 1.0 / float(selected_values.numel()))
    else:
        selected_values = selected_values / total
    return selected_indices, selected_values


def build_pair_weights(
    indices_a: torch.Tensor,
    weights_a: torch.Tensor,
    indices_b: torch.Tensor,
    weights_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if indices_a.ndim != 1 or indices_b.ndim != 1:
        raise ValueError("Expected 1D index tensors for pairwise conditioning")
    if weights_a.ndim != 1 or weights_b.ndim != 1:
        raise ValueError("Expected 1D routing weight tensors for pairwise conditioning")
    if indices_a.numel() != weights_a.numel() or indices_b.numel() != weights_b.numel():
        raise ValueError("Index and weight tensor lengths must match")

    grid_a, grid_b = torch.meshgrid(indices_a, indices_b, indexing="ij")
    pair_indices = torch.stack((grid_a.reshape(-1), grid_b.reshape(-1)), dim=1).to(dtype=torch.long)
    pair_weights = (weights_a[:, None] * weights_b[None, :]).reshape(-1)
    pair_weights = pair_weights / pair_weights.sum().clamp_min(1e-8)
    return pair_indices, pair_weights


def constrain_pair_weights_to_supported_pairs(
    pair_indices: torch.Tensor,
    pair_weights: torch.Tensor,
    supported_pairs: frozenset[tuple[int, int]] | None,
    marginal_a: torch.Tensor,
    marginal_b: torch.Tensor,
    mode: str,
    unsupported_pair_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if supported_pairs is None or mode == "none":
        return pair_indices, pair_weights

    kept_positions = [
        idx
        for idx, pair in enumerate(pair_indices.tolist())
        if (int(pair[0]), int(pair[1])) in supported_pairs
    ]
    if mode == "soft":
        support_mask = torch.zeros(
            pair_weights.shape[0],
            dtype=torch.bool,
            device=pair_weights.device,
        )
        if kept_positions:
            keep_idx = torch.tensor(kept_positions, dtype=torch.long, device=pair_indices.device)
            support_mask.index_fill_(0, keep_idx.to(device=support_mask.device), True)
        pair_scale = torch.full_like(pair_weights, float(unsupported_pair_scale))
        pair_scale = torch.where(support_mask, torch.ones_like(pair_scale), pair_scale)
        adjusted_weights = pair_weights * pair_scale
        if float(adjusted_weights.sum().item()) > 0.0:
            adjusted_weights = adjusted_weights / adjusted_weights.sum().clamp_min(1e-8)
            return pair_indices, adjusted_weights
    elif mode == "hard":
        if kept_positions:
            keep_idx = torch.tensor(kept_positions, dtype=torch.long, device=pair_indices.device)
            constrained_indices = pair_indices.index_select(0, keep_idx)
            constrained_weights = pair_weights.index_select(0, keep_idx)
            constrained_weights = constrained_weights / constrained_weights.sum().clamp_min(1e-8)
            return constrained_indices, constrained_weights
    else:
        raise ValueError(f"Unsupported support-constrained routing mode: {mode!r}")

    probs_a = marginal_a.detach().float().clamp_min(0.0)
    probs_b = marginal_b.detach().float().clamp_min(0.0)
    fallback_pairs: list[list[int]] = []
    fallback_scores: list[float] = []
    for idx_a, idx_b in supported_pairs:
        if idx_a < 0 or idx_b < 0:
            continue
        if idx_a >= probs_a.numel() or idx_b >= probs_b.numel():
            continue
        fallback_pairs.append([int(idx_a), int(idx_b)])
        fallback_scores.append(float(probs_a[idx_a].item()) * float(probs_b[idx_b].item()))

    if not fallback_pairs:
        return pair_indices, pair_weights

    constrained_indices = torch.tensor(
        fallback_pairs,
        dtype=torch.long,
        device=pair_indices.device,
    )
    constrained_weights = torch.tensor(
        fallback_scores,
        dtype=pair_weights.dtype,
        device=pair_weights.device,
    )
    if float(constrained_weights.sum().item()) <= 0.0:
        constrained_weights = torch.full_like(
            constrained_weights,
            1.0 / float(constrained_weights.numel()),
        )
    else:
        constrained_weights = constrained_weights / constrained_weights.sum().clamp_min(1e-8)
    return constrained_indices, constrained_weights


def run_pair_conditioned_head(
    head: torch.nn.Module,
    encoded_clip: Any,
    pair_indices: torch.Tensor,
    device: torch.device,
    temperature: torch.Tensor | None = None,
) -> torch.Tensor:
    condition_tensor = pair_indices.unsqueeze(0).to(device=device, dtype=torch.long)
    logits = head(encoded_clip, condition_tensor)["task_0"][0]
    return temperature_scaled_sigmoid(logits, temperature)


def mix_conditioned_predictions(
    conditioned_probs: torch.Tensor,
    pair_weights: torch.Tensor,
) -> torch.Tensor:
    if conditioned_probs.ndim != 2:
        raise ValueError(
            f"Expected conditioned_probs with shape [num_pairs, num_classes], got {tuple(conditioned_probs.shape)}"
        )
    if pair_weights.ndim != 1 or pair_weights.shape[0] != conditioned_probs.shape[0]:
        raise ValueError("pair_weights must be 1D and match the number of conditioned rows")
    return (conditioned_probs * pair_weights.unsqueeze(-1)).sum(dim=0)


def maybe_blend_probabilities(
    previous_probs: torch.Tensor,
    refined_probs: torch.Tensor,
    blend_alpha: float,
) -> torch.Tensor:
    if blend_alpha <= 0.0:
        return refined_probs
    if blend_alpha >= 1.0:
        return previous_probs
    return (blend_alpha * previous_probs) + ((1.0 - blend_alpha) * refined_probs)


SOFT_HEAD_NAMES = (
    "tool0",
    "verb0",
    "target0",
    "tool_refine",
    "verb_refine",
    "target_refine",
)


class HeadTemperatures:
    """Per-head temperature scalars or per-class vectors applied in logit space.

    A temperature value of 1.0 (the default) is a no-op, so an instance with no
    overrides is equivalent to the un-calibrated pipeline. Values must be > 0.
    """

    def __init__(self, raw: dict[str, Any] | None = None) -> None:
        self._values: dict[str, torch.Tensor] = {}
        if raw is None:
            raw = {}
        for name in SOFT_HEAD_NAMES:
            value = raw.get(name, 1.0)
            tensor = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
            if tensor.numel() == 0:
                raise ValueError(f"Empty temperature vector for head '{name}'")
            if (tensor <= 0).any().item():
                raise ValueError(
                    f"Temperatures must be strictly positive (head='{name}', got {tensor.tolist()})"
                )
            self._values[name] = tensor

    @classmethod
    def identity(cls) -> "HeadTemperatures":
        return cls()

    @classmethod
    def from_json(cls, path: Path) -> "HeadTemperatures":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Calibration JSON must contain a dict, got {type(payload).__name__}")
        head_section = payload.get("temperatures", payload)
        if not isinstance(head_section, dict):
            raise ValueError(
                f"Calibration JSON must expose 'temperatures' as a dict (file: {path})"
            )
        return cls(head_section)

    def is_identity(self) -> bool:
        for tensor in self._values.values():
            if not torch.allclose(tensor, torch.ones_like(tensor)):
                return False
        return True

    def get(self, name: str, num_classes: int, device: torch.device) -> torch.Tensor:
        if name not in self._values:
            raise KeyError(f"Unknown head temperature '{name}'")
        tensor = self._values[name].to(device=device, dtype=torch.float32)
        if tensor.numel() == 1:
            return tensor.expand(num_classes).contiguous()
        if tensor.numel() != num_classes:
            raise ValueError(
                f"Temperature for head '{name}' has {tensor.numel()} entries, "
                f"expected 1 or {num_classes}"
            )
        return tensor

    def to_serializable(self) -> dict[str, list[float]]:
        return {name: [float(x) for x in tensor.tolist()] for name, tensor in self._values.items()}


def temperature_scaled_sigmoid(
    logits: torch.Tensor,
    temperatures: torch.Tensor | None,
) -> torch.Tensor:
    if temperatures is None:
        return torch.sigmoid(logits.float())
    temp = temperatures.to(device=logits.device, dtype=torch.float32)
    return torch.sigmoid(logits.float() / temp)


def compute_soft_refined_probabilities(
    encoded_clip: Any,
    tool0_head: torch.nn.Module,
    verb0_head: torch.nn.Module,
    target0_head: torch.nn.Module,
    tool_refine_head: torch.nn.Module,
    verb_refine_head: torch.nn.Module,
    target_refine_head: torch.nn.Module,
    device: torch.device,
    refinement_steps: int,
    blend_alpha: float,
    route_topk_tool: int | None,
    route_topk_verb: int | None,
    route_topk_target: int | None,
    blend_alpha_step2: float | None = None,
    routing_probability_source: str = "current",
    route_threshold_tool: torch.Tensor | None = None,
    route_threshold_verb: torch.Tensor | None = None,
    route_threshold_target: torch.Tensor | None = None,
    head_temperatures: HeadTemperatures | None = None,
    support_pair_constraints: dict[str, frozenset[tuple[int, int]]] | None = None,
    support_constrained_routing_mode: str = "none",
    support_constrained_routing_unsupported_scale: float = 0.1,
    collect_debug: bool = False,
) -> dict[str, Any]:
    if head_temperatures is None:
        head_temperatures = HeadTemperatures.identity()
    if routing_probability_source not in {"current", "stage0"}:
        raise ValueError(
            "routing_probability_source must be 'current' or 'stage0', "
            f"got {routing_probability_source!r}"
        )
    if blend_alpha_step2 is not None and not (0.0 <= blend_alpha_step2 <= 1.0):
        raise ValueError(
            f"blend_alpha_step2 must be in [0, 1], got {blend_alpha_step2}"
        )

    tool_logits0 = tool0_head(encoded_clip)["task_0"][0]
    verb_logits0 = verb0_head(encoded_clip)["task_0"][0]
    target_logits0 = target0_head(encoded_clip)["task_0"][0]

    tool0_temp = head_temperatures.get("tool0", int(tool_logits0.numel()), device)
    verb0_temp = head_temperatures.get("verb0", int(verb_logits0.numel()), device)
    target0_temp = head_temperatures.get("target0", int(target_logits0.numel()), device)
    tool_refine_temp = head_temperatures.get(
        "tool_refine", int(tool_refine_head.num_classes_per_task[0]), device
    )
    verb_refine_temp = head_temperatures.get(
        "verb_refine", int(verb_refine_head.num_classes_per_task[0]), device
    )
    target_refine_temp = head_temperatures.get(
        "target_refine", int(target_refine_head.num_classes_per_task[0]), device
    )

    tool_probs0 = temperature_scaled_sigmoid(tool_logits0, tool0_temp)
    verb_probs0 = temperature_scaled_sigmoid(verb_logits0, verb0_temp)
    target_probs0 = temperature_scaled_sigmoid(target_logits0, target0_temp)
    use_verb_adapter = (
        int(verb_probs0.numel()) == 7
        and int(verb_refine_head.num_classes_per_task[0]) == 6
    )
    verb_routing0 = grouped_verb_probs_to_study_probs(verb_probs0) if use_verb_adapter else verb_probs0

    current = {
        "tool": tool_probs0,
        "verb": verb_probs0,
        "verb_routing": verb_routing0,
        "target": target_probs0,
    }
    routing_reference = {
        "tool": tool_probs0,
        "verb": verb_probs0,
        "verb_routing": verb_routing0,
        "target": target_probs0,
    }
    supported_tool_pairs = None
    supported_verb_pairs = None
    supported_target_pairs = None
    if support_pair_constraints is not None:
        supported_tool_pairs = support_pair_constraints.get("tool")
        supported_verb_pairs = support_pair_constraints.get("verb")
        supported_target_pairs = support_pair_constraints.get("target")
    stages = [
        {
            "tool": tool_probs0.clone(),
            "verb": verb_probs0.clone(),
            "verb_routing": verb_routing0.clone(),
            "target": target_probs0.clone(),
        }
    ]
    step_debug: list[dict[str, Any]] = []

    for step_idx in range(refinement_steps):
        current_blend_alpha = (
            blend_alpha
            if step_idx == 0 or blend_alpha_step2 is None
            else float(blend_alpha_step2)
        )
        routing_state = (
            current if routing_probability_source == "current" else routing_reference
        )
        tool_indices, tool_weights = normalize_routing_distribution(
            routing_state["tool"],
            topk=route_topk_tool,
            threshold_vector=route_threshold_tool,
        )
        verb_indices, verb_weights = normalize_routing_distribution(
            routing_state["verb_routing"],
            topk=route_topk_verb,
            threshold_vector=route_threshold_verb,
        )
        target_indices, target_weights = normalize_routing_distribution(
            routing_state["target"],
            topk=route_topk_target,
            threshold_vector=route_threshold_target,
        )

        tool_pairs, tool_pair_weights = build_pair_weights(
            verb_indices,
            verb_weights,
            target_indices,
            target_weights,
        )
        tool_pairs, tool_pair_weights = constrain_pair_weights_to_supported_pairs(
            pair_indices=tool_pairs,
            pair_weights=tool_pair_weights,
            supported_pairs=supported_tool_pairs,
            marginal_a=routing_state["verb_routing"],
            marginal_b=routing_state["target"],
            mode=support_constrained_routing_mode,
            unsupported_pair_scale=support_constrained_routing_unsupported_scale,
        )
        verb_pairs, verb_pair_weights = build_pair_weights(
            tool_indices,
            tool_weights,
            target_indices,
            target_weights,
        )
        verb_pairs, verb_pair_weights = constrain_pair_weights_to_supported_pairs(
            pair_indices=verb_pairs,
            pair_weights=verb_pair_weights,
            supported_pairs=supported_verb_pairs,
            marginal_a=routing_state["tool"],
            marginal_b=routing_state["target"],
            mode=support_constrained_routing_mode,
            unsupported_pair_scale=support_constrained_routing_unsupported_scale,
        )
        target_pairs, target_pair_weights = build_pair_weights(
            tool_indices,
            tool_weights,
            verb_indices,
            verb_weights,
        )
        target_pairs, target_pair_weights = constrain_pair_weights_to_supported_pairs(
            pair_indices=target_pairs,
            pair_weights=target_pair_weights,
            supported_pairs=supported_target_pairs,
            marginal_a=routing_state["tool"],
            marginal_b=routing_state["verb_routing"],
            mode=support_constrained_routing_mode,
            unsupported_pair_scale=support_constrained_routing_unsupported_scale,
        )

        tool_cond_probs = run_pair_conditioned_head(
            tool_refine_head, encoded_clip, tool_pairs, device, temperature=tool_refine_temp,
        )
        verb_cond_probs = run_pair_conditioned_head(
            verb_refine_head, encoded_clip, verb_pairs, device, temperature=verb_refine_temp,
        )
        target_cond_probs = run_pair_conditioned_head(
            target_refine_head, encoded_clip, target_pairs, device, temperature=target_refine_temp,
        )
        mixed_tool = mix_conditioned_predictions(tool_cond_probs, tool_pair_weights)
        mixed_verb = mix_conditioned_predictions(verb_cond_probs, verb_pair_weights)
        mixed_target = mix_conditioned_predictions(target_cond_probs, target_pair_weights)

        next_tool = maybe_blend_probabilities(
            current["tool"],
            mixed_tool,
            current_blend_alpha,
        )
        next_verb = maybe_blend_probabilities(
            current["verb_routing"],
            mixed_verb,
            current_blend_alpha,
        )
        next_target = maybe_blend_probabilities(
            current["target"],
            mixed_target,
            current_blend_alpha,
        )
        next_verb_display = (
            study_verb_probs_to_grouped_probs(next_verb, current["verb"])
            if use_verb_adapter
            else next_verb
        )

        current = {
            "tool": next_tool,
            "verb": next_verb_display,
            "verb_routing": next_verb,
            "target": next_target,
        }
        if collect_debug:
            step_debug.append(
                {
                    "step_index": int(step_idx + 1),
                    "blend_alpha": float(current_blend_alpha),
                    "routing_tool_indices": tool_indices.clone(),
                    "routing_tool_weights": tool_weights.clone(),
                    "routing_verb_indices": verb_indices.clone(),
                    "routing_verb_weights": verb_weights.clone(),
                    "routing_target_indices": target_indices.clone(),
                    "routing_target_weights": target_weights.clone(),
                    "tool_pairs": tool_pairs.clone(),
                    "tool_pair_weights": tool_pair_weights.clone(),
                    "tool_conditioned_probs": tool_cond_probs.clone(),
                    "tool_mixed_probs": mixed_tool.clone(),
                    "verb_pairs": verb_pairs.clone(),
                    "verb_pair_weights": verb_pair_weights.clone(),
                    "verb_conditioned_probs": verb_cond_probs.clone(),
                    "verb_mixed_probs": mixed_verb.clone(),
                    "target_pairs": target_pairs.clone(),
                    "target_pair_weights": target_pair_weights.clone(),
                    "target_conditioned_probs": target_cond_probs.clone(),
                    "target_mixed_probs": mixed_target.clone(),
                    "next_tool_probs": next_tool.clone(),
                    "next_verb_routing_probs": next_verb.clone(),
                    "next_verb_display_probs": next_verb_display.clone(),
                    "next_target_probs": next_target.clone(),
                }
            )
        stages.append(
            {
                "tool": next_tool.clone(),
                "verb": next_verb_display.clone(),
                "verb_routing": next_verb.clone(),
                "target": next_target.clone(),
            }
        )

    final_routing_state = (
        current if routing_probability_source == "current" else routing_reference
    )
    final_tool_indices, final_tool_weights = normalize_routing_distribution(
        final_routing_state["tool"],
        topk=route_topk_tool,
        threshold_vector=route_threshold_tool,
    )
    final_verb_indices, final_verb_weights = normalize_routing_distribution(
        final_routing_state["verb_routing"],
        topk=route_topk_verb,
        threshold_vector=route_threshold_verb,
    )
    final_target_indices, final_target_weights = normalize_routing_distribution(
        final_routing_state["target"],
        topk=route_topk_target,
        threshold_vector=route_threshold_target,
    )
    final_tool_pairs, final_tool_pair_weights = build_pair_weights(
        final_verb_indices,
        final_verb_weights,
        final_target_indices,
        final_target_weights,
    )
    final_tool_pairs, final_tool_pair_weights = constrain_pair_weights_to_supported_pairs(
        pair_indices=final_tool_pairs,
        pair_weights=final_tool_pair_weights,
        supported_pairs=supported_tool_pairs,
        marginal_a=final_routing_state["verb_routing"],
        marginal_b=final_routing_state["target"],
        mode=support_constrained_routing_mode,
        unsupported_pair_scale=support_constrained_routing_unsupported_scale,
    )
    final_verb_pairs, final_verb_pair_weights = build_pair_weights(
        final_tool_indices,
        final_tool_weights,
        final_target_indices,
        final_target_weights,
    )
    final_verb_pairs, final_verb_pair_weights = constrain_pair_weights_to_supported_pairs(
        pair_indices=final_verb_pairs,
        pair_weights=final_verb_pair_weights,
        supported_pairs=supported_verb_pairs,
        marginal_a=final_routing_state["tool"],
        marginal_b=final_routing_state["target"],
        mode=support_constrained_routing_mode,
        unsupported_pair_scale=support_constrained_routing_unsupported_scale,
    )
    final_target_pairs, final_target_pair_weights = build_pair_weights(
        final_tool_indices,
        final_tool_weights,
        final_verb_indices,
        final_verb_weights,
    )
    final_target_pairs, final_target_pair_weights = constrain_pair_weights_to_supported_pairs(
        pair_indices=final_target_pairs,
        pair_weights=final_target_pair_weights,
        supported_pairs=supported_target_pairs,
        marginal_a=final_routing_state["tool"],
        marginal_b=final_routing_state["verb_routing"],
        mode=support_constrained_routing_mode,
        unsupported_pair_scale=support_constrained_routing_unsupported_scale,
    )
    final_tool_cond_probs = run_pair_conditioned_head(
        tool_refine_head,
        encoded_clip,
        final_tool_pairs,
        device,
        temperature=tool_refine_temp,
    )
    final_verb_cond_probs = run_pair_conditioned_head(
        verb_refine_head,
        encoded_clip,
        final_verb_pairs,
        device,
        temperature=verb_refine_temp,
    )
    final_target_cond_probs = run_pair_conditioned_head(
        target_refine_head,
        encoded_clip,
        final_target_pairs,
        device,
        temperature=target_refine_temp,
    )

    return {
        "final": current,
        "stages": stages,
        "step_debug": step_debug if collect_debug else None,
        "final_tool_indices": final_tool_indices,
        "final_tool_weights": final_tool_weights,
        "final_tool_pairs": final_tool_pairs,
        "final_tool_pair_weights": final_tool_pair_weights,
        "final_tool_conditioned_probs": final_tool_cond_probs,
        "final_verb_indices": final_verb_indices,
        "final_verb_weights": final_verb_weights,
        "final_verb_pairs": final_verb_pairs,
        "final_verb_pair_weights": final_verb_pair_weights,
        "final_verb_conditioned_probs": final_verb_cond_probs,
        "final_target_indices": final_target_indices,
        "final_target_weights": final_target_weights,
        "final_target_pairs": final_target_pairs,
        "final_target_pair_weights": final_target_pair_weights,
        "final_target_conditioned_probs": final_target_cond_probs,
        "use_grouped_to_study_verb_adapter": use_verb_adapter,
        "support_constrained_routing": (
            support_pair_constraints is not None and support_constrained_routing_mode != "none"
        ),
        "support_constrained_routing_mode": support_constrained_routing_mode,
        "support_constrained_routing_unsupported_scale": float(
            support_constrained_routing_unsupported_scale
        ),
        "routing_probability_source": str(routing_probability_source),
        "blend_alpha_step2": (
            None if blend_alpha_step2 is None else float(blend_alpha_step2)
        ),
    }


def run_soft_refinement_prediction(
    encoder: torch.nn.Module,
    tool0_head: torch.nn.Module,
    verb0_head: torch.nn.Module,
    target0_head: torch.nn.Module,
    tool_refine_head: torch.nn.Module,
    verb_refine_head: torch.nn.Module,
    target_refine_head: torch.nn.Module,
    clips: list[list[torch.Tensor]],
    clip_indices: list[torch.Tensor],
    start_time: float,
    clip_seconds: float,
    device: torch.device,
    tool_names: list[str],
    verb_names: list[str],
    target_names: list[str],
    tool_threshold: float,
    verb_threshold: float,
    target_threshold: float,
    max_tools: int | None,
    max_verbs_per_tool: int | None,
    max_targets_per_pair: int | None,
    refinement_steps: int,
    blend_alpha: float,
    blend_alpha_step2: float | None,
    route_topk_tool: int | None,
    route_topk_verb: int | None,
    route_topk_target: int | None,
    route_threshold_tool: torch.Tensor | None,
    route_threshold_verb: torch.Tensor | None,
    route_threshold_target: torch.Tensor | None,
    routing_probability_source: str = "current",
    support_aware_triplet_refinement: bool = False,
    decode_topk_tool: int | None = None,
    decode_topk_verb: int | None = None,
    decode_topk_target: int | None = None,
    decode_max_triplets_total: int | None = None,
    decode_score_min: float | None = None,
    triplet_to_id: dict[tuple[int, int, int], int] | None = None,
    num_triplet_classes: int | None = None,
    head_temperatures: HeadTemperatures | None = None,
    support_pair_constraints: dict[str, frozenset[tuple[int, int]]] | None = None,
    support_constrained_routing_mode: str = "none",
    support_constrained_routing_unsupported_scale: float = 0.1,
) -> dict[str, Any]:
    with torch.no_grad():
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            encoded_outputs = encoder(clips, clip_indices)
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
                refinement_steps=refinement_steps,
                blend_alpha=blend_alpha,
                blend_alpha_step2=blend_alpha_step2,
                route_topk_tool=route_topk_tool,
                route_topk_verb=route_topk_verb,
                route_topk_target=route_topk_target,
                routing_probability_source=routing_probability_source,
                route_threshold_tool=route_threshold_tool,
                route_threshold_verb=route_threshold_verb,
                route_threshold_target=route_threshold_target,
                head_temperatures=head_temperatures,
                support_pair_constraints=support_pair_constraints,
                support_constrained_routing_mode=support_constrained_routing_mode,
                support_constrained_routing_unsupported_scale=(
                    support_constrained_routing_unsupported_scale
                ),
            )

    final_tool_probs = refinement["final"]["tool"]
    final_verb_probs = refinement["final"]["verb"]
    final_target_probs = refinement["final"]["target"]
    final_target_pairs = refinement["final_target_pairs"]
    final_target_cond_probs = refinement["final_target_conditioned_probs"]
    use_verb_adapter = bool(refinement.get("use_grouped_to_study_verb_adapter", False))

    tool_thresholds = torch.full_like(final_tool_probs.detach().cpu().float(), float(tool_threshold))
    verb_thresholds = torch.full_like(final_verb_probs.detach().cpu().float(), float(verb_threshold))
    target_thresholds = torch.full_like(final_target_probs.detach().cpu().float(), float(target_threshold))

    if support_aware_triplet_refinement:
        if triplet_to_id is None or num_triplet_classes is None:
            raise ValueError(
                "Support-aware triplet refinement requires triplet_to_id and num_triplet_classes"
            )

        decoded = decode_support_aware_triplets(
            refinement=refinement,
            triplet_to_id=triplet_to_id,
            num_triplet_classes=num_triplet_classes,
            tool_thresholds=tool_thresholds,
            verb_thresholds=verb_thresholds,
            target_thresholds=target_thresholds,
            search_topk_tool=(
                decode_topk_tool
                if decode_topk_tool is not None
                else (max_tools if max_tools is not None else route_topk_tool)
            ),
            search_topk_verb=(
                decode_topk_verb
                if decode_topk_verb is not None
                else (max_verbs_per_tool if max_verbs_per_tool is not None else route_topk_verb)
            ),
            search_topk_target=(
                decode_topk_target
                if decode_topk_target is not None
                else (
                    max_targets_per_pair
                    if max_targets_per_pair is not None
                    else route_topk_target
                )
            ),
            max_triplets_total=decode_max_triplets_total,
            min_triplet_score=decode_score_min,
            fallback_top1=True,
        )

        selected_tool_ids = decoded["tool_mask"].nonzero(as_tuple=False).reshape(-1).tolist()
        selected_tool_ids.sort(key=lambda tool_id: float(final_tool_probs[tool_id].item()), reverse=True)
        if max_tools is not None:
            selected_tool_ids = selected_tool_ids[:max_tools]

        tools = [
            {
                "tool_id": tool_id,
                "tool_name": lookup_name(tool_names, tool_id, "tool"),
                "tool_probability": float(final_tool_probs[tool_id].item()),
            }
            for tool_id in selected_tool_ids
        ]

        triplets = []
        per_tool_verbs: dict[int, list[int]] = {}
        per_pair_targets: dict[tuple[int, int], list[int]] = {}
        for candidate in decoded["triplets"]:
            tool_id = int(candidate["tool_id"])
            verb_id = int(candidate["verb_id"])
            target_id = int(candidate["target_id"])

            if max_tools is not None and tool_id not in selected_tool_ids:
                continue

            if max_verbs_per_tool is not None:
                verb_bucket = per_tool_verbs.setdefault(tool_id, [])
                if verb_id not in verb_bucket and len(verb_bucket) >= max_verbs_per_tool:
                    continue
                if verb_id not in verb_bucket:
                    verb_bucket.append(verb_id)

            if max_targets_per_pair is not None:
                pair_key = (tool_id, verb_id)
                target_bucket = per_pair_targets.setdefault(pair_key, [])
                if target_id not in target_bucket and len(target_bucket) >= max_targets_per_pair:
                    continue
                if target_id not in target_bucket:
                    target_bucket.append(target_id)

            triplets.append(
                {
                    "tool_id": tool_id,
                    "tool_name": lookup_name(tool_names, tool_id, "tool"),
                    "tool_probability": float(candidate["tool_probability"]),
                    "verb_id": verb_id,
                    "verb_name": lookup_name(verb_names, verb_id, "verb"),
                    "verb_probability": float(candidate["verb_probability"]),
                    "target_id": target_id,
                    "target_name": lookup_name(target_names, target_id, "target"),
                    "target_probability": float(candidate["target_probability"]),
                    "score": float(candidate["score"]),
                    "triplet_name": (
                        f"{lookup_name(tool_names, tool_id, 'tool')} | "
                        f"{lookup_name(verb_names, verb_id, 'verb')} | "
                        f"{lookup_name(target_names, target_id, 'target')}"
                    ),
                }
            )

        return {
            "start_time": float(start_time),
            "end_time": float(start_time + clip_seconds),
            "tools": tools,
            "triplets": triplets,
            "tool_probabilities": final_tool_probs.detach().cpu(),
            "verb_probabilities": final_verb_probs.detach().cpu(),
            "target_probabilities": final_target_probs.detach().cpu(),
            "refinement_steps": int(refinement_steps),
        }

    tool_ids = select_multilabel_indices(
        final_tool_probs,
        threshold=tool_threshold,
        max_count=max_tools,
        fallback_top1=True,
    )
    verb_ids = select_multilabel_indices(
        final_verb_probs,
        threshold=verb_threshold,
        max_count=max_verbs_per_tool,
        fallback_top1=True,
    )
    tools = [
        {
            "tool_id": tool_id,
            "tool_name": lookup_name(tool_names, tool_id, "tool"),
            "tool_probability": float(final_tool_probs[tool_id].item()),
        }
        for tool_id in tool_ids
    ]

    allowed_tools = set(tool_ids)
    allowed_verbs = set(verb_ids)
    triplets = []
    for pair_idx, pair in enumerate(final_target_pairs.tolist()):
        tool_id, study_verb_id = int(pair[0]), int(pair[1])
        if allowed_tools and tool_id not in allowed_tools:
            continue

        target_probs = final_target_cond_probs[pair_idx]
        target_ids = select_multilabel_indices(
            target_probs,
            threshold=target_threshold,
            max_count=max_targets_per_pair,
            fallback_top1=True,
        )

        for verb_id in map_study_verb_id_to_grouped_ids(study_verb_id, use_verb_adapter):
            if allowed_verbs and verb_id not in allowed_verbs:
                continue

            verb_prob = float(final_verb_probs[verb_id].item())
            for target_id in target_ids:
                target_prob = float(target_probs[target_id].item())
                score = float(final_tool_probs[tool_id].item()) * verb_prob * target_prob
                triplets.append(
                    {
                        "tool_id": tool_id,
                        "tool_name": lookup_name(tool_names, tool_id, "tool"),
                        "tool_probability": float(final_tool_probs[tool_id].item()),
                        "verb_id": verb_id,
                        "verb_name": lookup_name(verb_names, verb_id, "verb"),
                        "verb_probability": verb_prob,
                        "target_id": target_id,
                        "target_name": lookup_name(target_names, target_id, "target"),
                        "target_probability": target_prob,
                        "score": score,
                        "triplet_name": (
                            f"{lookup_name(tool_names, tool_id, 'tool')} | "
                            f"{lookup_name(verb_names, verb_id, 'verb')} | "
                            f"{lookup_name(target_names, target_id, 'target')}"
                        ),
                    }
                )

    triplets.sort(key=lambda item: item["score"], reverse=True)

    return {
        "start_time": float(start_time),
        "end_time": float(start_time + clip_seconds),
        "tools": tools,
        "triplets": triplets,
        "tool_probabilities": final_tool_probs.detach().cpu(),
        "verb_probabilities": final_verb_probs.detach().cpu(),
        "target_probabilities": final_target_probs.detach().cpu(),
        "refinement_steps": int(refinement_steps),
    }


def main() -> int:
    args = parse_args()
    input_video = args.input_video.resolve()
    output_video = resolve_output_path(input_video, args.output_video).resolve()

    if not input_video.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")
    if input_video == output_video:
        raise ValueError("Refusing to overwrite the input video; choose a different output path")

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
    use_verb_adapter = uses_grouped_to_study_verb_adapter(verb0_cfg, verb_refine_cfg)
    support_constrained_routing_mode = resolve_support_constrained_routing_mode(
        args.support_constrained_routing_mode,
        args.support_constrained_routing,
    )

    tool_threshold = resolve_multilabel_threshold(tool0_cfg, args.tool_threshold)
    verb_threshold = resolve_multilabel_threshold(verb0_cfg, args.verb_threshold)
    target_threshold = resolve_multilabel_threshold(target0_cfg, args.target_threshold)
    route_tool_threshold = resolve_multilabel_threshold_vector(tool_refine_cfg, args.tool_threshold)
    route_verb_threshold = resolve_multilabel_threshold_vector(verb_refine_cfg, None)
    route_target_threshold = resolve_multilabel_threshold_vector(target_refine_cfg, args.target_threshold)
    tool_names = class_names_from_config(tool0_cfg, FALLBACK_TOOL_NAMES, "tool")
    verb_names = class_names_from_config(verb0_cfg, FALLBACK_VERB_NAMES, "verb")
    target_names = class_names_from_config(target0_cfg, FALLBACK_TARGET_NAMES, "target")
    triplet_to_id = None
    num_triplet_classes = None
    support_pair_constraints = None
    if support_constrained_routing_mode != "none" or args.support_aware_triplet_refinement:
        triplet_metadata_path = args.triplet_metadata.resolve()
        if not triplet_metadata_path.exists():
            raise FileNotFoundError(f"Triplet metadata JSON does not exist: {triplet_metadata_path}")
        triplet_to_id, num_triplet_classes, supported_triplets = load_triplet_support_index(
            triplet_metadata_path
        )
        if support_constrained_routing_mode != "none":
            support_pair_constraints = build_support_pair_constraints(
                supported_triplets,
                use_verb_adapter=use_verb_adapter,
            )

    frames_per_clip = int(tool0_cfg["experiment"]["data"].get("frames_per_clip", 16))
    validate_temporal_sampling(
        frames_per_clip=frames_per_clip,
        clip_fps=args.clip_fps,
        clip_seconds=args.clip_seconds,
    )
    resolution = int(tool0_cfg["experiment"]["data"].get("resolution", 224))
    transform = build_transform(tool0_cfg)

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

    if args.calibration_json is not None:
        head_temperatures = HeadTemperatures.from_json(args.calibration_json.resolve())
    else:
        head_temperatures = HeadTemperatures.identity()

    if args.verbose:
        print(f"Loaded tool0 head from {tool0_checkpoint}")
        print(f"Loaded verb0 head from {verb0_checkpoint}")
        print(f"Loaded target0 head from {target0_checkpoint}")
        print(f"Loaded tool refinement head from {tool_refine_checkpoint}")
        print(f"Loaded verb refinement head from {verb_refine_checkpoint}")
        print(f"Loaded target refinement head from {target_refine_checkpoint}")
        if args.calibration_json is not None:
            print(f"Loaded head temperatures from {args.calibration_json.resolve()}")
        else:
            print("Head temperatures: identity (no calibration applied)")
        print(
            "Using multi-label thresholds: "
            f"tool={tool_threshold:.3f} verb={verb_threshold:.3f} target={target_threshold:.3f}"
        )
        print(
            "Soft refinement settings: "
            f"steps={args.refinement_steps} blend_alpha={args.blend_alpha:.3f} "
            f"blend_alpha_step2={args.blend_alpha_step2 if args.blend_alpha_step2 is not None else 'default'} "
            f"route_topk_tool={args.route_topk_tool} route_topk_verb={args.route_topk_verb} "
            f"route_topk_target={args.route_topk_target} "
            f"routing_source={args.routing_probability_source}"
        )
        if args.support_aware_triplet_refinement:
            print(f"Support-aware triplet refinement: enabled ({args.triplet_metadata.resolve()})")

    sampler = SequentialVideoSampler(input_video)
    source_fps = sampler.source_fps
    num_frames = sampler.num_frames
    width = sampler.width
    height = sampler.height

    if args.verbose:
        print(
            f"Video info: frames={num_frames} fps={source_fps:.3f} size={width}x{height} "
            f"clip_seconds={args.clip_seconds:.2f} stride_seconds={args.stride_seconds:.2f} clip_fps={args.clip_fps:.2f}"
        )
        print(
            f"Temporal sampling: {frames_per_clip} frames at {args.clip_fps:.2f} fps "
            f"over {args.clip_seconds:.2f}s"
        )

    window_starts = build_window_starts(
        num_frames=num_frames,
        source_fps=source_fps,
        clip_seconds=args.clip_seconds,
        stride_seconds=args.stride_seconds,
        clip_fps=args.clip_fps,
    )
    if args.max_windows is not None:
        window_starts = window_starts[: args.max_windows]
    if args.verbose:
        print(f"Running soft refinement inference on {len(window_starts)} windows")

    predictions = []
    try:
        for window_idx, start_time in enumerate(window_starts, start=1):
            frames_rgb, sampled_indices = sampler.sample_window(
                start_time=start_time,
                frames_per_clip=frames_per_clip,
                clip_fps=args.clip_fps,
            )
            clips, clip_indices = prepare_encoder_inputs(
                frames_rgb=frames_rgb,
                frame_indices=sampled_indices,
                transform=transform,
                device=device,
            )
            prediction = run_soft_refinement_prediction(
                encoder=encoder,
                tool0_head=tool0_head,
                verb0_head=verb0_head,
                target0_head=target0_head,
                tool_refine_head=tool_refine_head,
                verb_refine_head=verb_refine_head,
                target_refine_head=target_refine_head,
                clips=clips,
                clip_indices=clip_indices,
                start_time=start_time,
                clip_seconds=args.clip_seconds,
                device=device,
                tool_names=tool_names,
                verb_names=verb_names,
                target_names=target_names,
                tool_threshold=tool_threshold,
                verb_threshold=verb_threshold,
                target_threshold=target_threshold,
                max_tools=args.max_tools,
                max_verbs_per_tool=args.max_verbs_per_tool,
                max_targets_per_pair=args.max_targets_per_pair,
                refinement_steps=args.refinement_steps,
                blend_alpha=args.blend_alpha,
                blend_alpha_step2=args.blend_alpha_step2,
                route_topk_tool=args.route_topk_tool,
                route_topk_verb=args.route_topk_verb,
                route_topk_target=args.route_topk_target,
                routing_probability_source=args.routing_probability_source,
                route_threshold_tool=route_tool_threshold,
                route_threshold_verb=route_verb_threshold,
                route_threshold_target=route_target_threshold,
                support_aware_triplet_refinement=args.support_aware_triplet_refinement,
                decode_topk_tool=args.decode_topk_tool,
                decode_topk_verb=args.decode_topk_verb,
                decode_topk_target=args.decode_topk_target,
                decode_max_triplets_total=args.decode_max_triplets_total,
                decode_score_min=args.decode_score_min,
                triplet_to_id=triplet_to_id,
                num_triplet_classes=num_triplet_classes,
                head_temperatures=head_temperatures,
                support_pair_constraints=support_pair_constraints,
                support_constrained_routing_mode=support_constrained_routing_mode,
                support_constrained_routing_unsupported_scale=(
                    args.support_constrained_routing_unsupported_scale
                ),
            )
            predictions.append(prediction)
            if args.verbose:
                top_triplet = prediction["triplets"][0]["triplet_name"] if prediction["triplets"] else "no triplet"
                top_score = prediction["triplets"][0]["score"] * 100.0 if prediction["triplets"] else 0.0
                print(
                    f"[{window_idx:03d}/{len(window_starts):03d}] "
                    f"{prediction['start_time']:.2f}-{prediction['end_time']:.2f}s "
                    f"=> {top_triplet} ({top_score:.1f}%) "
                    f"[{len(prediction['triplets'])} triplets]"
                )
    finally:
        sampler.close()

    if not predictions:
        raise RuntimeError("No predictions were generated")

    print_prediction_distribution(predictions, top_k=args.distribution_top_k)

    annotate_video(
        input_video=input_video,
        output_video=output_video,
        predictions=predictions,
        source_fps=source_fps,
        width=width,
        height=height,
        max_triplets_display=args.max_triplets_display,
        verbose=args.verbose,
    )
    print(f"Saved annotated video to {output_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
