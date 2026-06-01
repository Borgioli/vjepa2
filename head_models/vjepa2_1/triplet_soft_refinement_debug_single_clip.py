#!/path/to/data_root/vjepa2_1/.venv/bin/python
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import torch

from triplet_conditioned_inference import (
    FALLBACK_TARGET_NAMES,
    FALLBACK_TOOL_NAMES,
    FALLBACK_VERB_NAMES,
    build_classifier_from_config,
    class_names_from_config,
    init_encoder_module,
    load_classifier_weights,
    load_yaml_config,
    maybe_suppress_stdout,
    override_backbone_checkpoint,
    resolve_device,
)
from triplet_soft_refinement_eval import (
    build_eval_dataset,
    build_predicted_triplet_mask,
    load_eval_label_lookup,
    prepare_dataset_sample_inputs,
    resolve_grouped_verb_threshold_vector,
    resolve_threshold_vector,
    threshold_probabilities,
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
    resolve_soft_refinement_checkpoint_path,
    resolve_support_constrained_routing_mode,
    validate_soft_refinement_configs,
    validate_topk,
    uses_grouped_to_study_verb_adapter,
)

warnings.filterwarnings(
    "ignore",
    message="Importing from timm.models.layers is deprecated.*",
    category=FutureWarning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the soft triplet refinement architecture on one validation clip and "
            "print a step-by-step trace of routing, conditioned heads, blending, and decode."
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
        help="How to derive head checkpoints when explicit paths are not passed.",
    )
    parser.add_argument("--strict-best-head-checkpoints", action="store_true")
    parser.add_argument("--backbone-checkpoint", type=Path, default=None)
    parser.add_argument("--backbone-checkpoint-key", type=str, default=None)

    parser.add_argument("--eval-runtime-csv", type=Path, default=None)
    parser.add_argument("--eval-label-csv", type=Path, default=None)
    parser.add_argument(
        "--triplet-metadata",
        type=Path,
        default=Path(
            "/path/to/data_root/vjepa2_1/app/csv_head_models/"
            "triplet_multilabel_native_tool5_metadata.json"
        ),
    )

    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--sample-idx", type=int, default=0, help="Global validation sample index.")
    parser.add_argument(
        "--sample-path",
        type=str,
        default=None,
        help="Optional exact clip_path from the validation CSV. Overrides --sample-idx when set.",
    )

    parser.add_argument("--tool-threshold", type=float, default=None)
    parser.add_argument("--verb-threshold", type=float, default=None)
    parser.add_argument("--target-threshold", type=float, default=None)

    parser.add_argument("--refinement-steps", type=int, default=1, choices=(1, 2))
    parser.add_argument("--blend-alpha", type=float, default=0.75)
    parser.add_argument("--blend-alpha-step2", type=float, default=None)
    parser.add_argument("--route-topk-tool", type=int, default=None)
    parser.add_argument("--route-topk-verb", type=int, default=None)
    parser.add_argument("--route-topk-target", type=int, default=None)
    parser.add_argument(
        "--routing-probability-source",
        type=str,
        choices=("current", "stage0"),
        default="current",
    )
    parser.add_argument("--support-constrained-routing", action="store_true")
    parser.add_argument(
        "--support-constrained-routing-mode",
        type=str,
        choices=("none", "soft", "hard"),
        default="none",
    )
    parser.add_argument(
        "--support-constrained-routing-unsupported-scale",
        type=float,
        default=0.1,
    )
    parser.add_argument("--support-aware-triplet-refinement", action="store_true")
    parser.add_argument("--decode-topk-tool", type=int, default=None)
    parser.add_argument("--decode-topk-verb", type=int, default=None)
    parser.add_argument("--decode-topk-target", type=int, default=None)
    parser.add_argument("--decode-max-triplets-total", type=int, default=None)
    parser.add_argument("--decode-score-min", type=float, default=None)
    parser.add_argument("--calibration-json", type=Path, default=None)

    parser.add_argument(
        "--print-topk",
        type=int,
        default=8,
        help="How many classes to print per probability vector.",
    )
    parser.add_argument(
        "--print-pair-topk",
        type=int,
        default=6,
        help="How many routing pairs to print for each conditioned head.",
    )
    parser.add_argument(
        "--print-conditioned-topk",
        type=int,
        default=5,
        help="How many conditioned output classes to print per pair row.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def axis_top_rows(
    probabilities: torch.Tensor,
    class_names: list[str],
    thresholds: torch.Tensor | None,
    topk: int,
) -> list[dict[str, Any]]:
    probs = probabilities.detach().cpu().float().reshape(-1)
    order = torch.argsort(probs, descending=True)
    if topk > 0:
        order = order[: min(int(topk), int(order.numel()))]
    rows = []
    threshold_values = None
    if thresholds is not None:
        threshold_values = thresholds.detach().cpu().float().reshape(-1)
    for idx_tensor in order:
        idx = int(idx_tensor.item())
        row = {
            "id": idx,
            "name": class_names[idx],
            "probability": float(probs[idx].item()),
        }
        if threshold_values is not None:
            threshold = float(threshold_values[idx].item())
            row["threshold"] = threshold
            row["active"] = bool(probs[idx].item() >= threshold)
        rows.append(row)
    return rows


def active_names(mask: torch.Tensor, class_names: list[str]) -> list[str]:
    mask = mask.detach().cpu().bool().reshape(-1)
    return [class_names[idx] for idx, value in enumerate(mask.tolist()) if value]


def triplet_label(
    triplet: tuple[int, int, int],
    tool_names: list[str],
    verb_names: list[str],
    target_names: list[str],
) -> str:
    tool_id, verb_id, target_id = triplet
    return f"{tool_names[tool_id]} | {verb_names[verb_id]} | {target_names[target_id]}"


def build_id_to_triplet(
    triplet_to_id: dict[tuple[int, int, int], int],
) -> dict[int, tuple[int, int, int]]:
    return {int(triplet_id): tuple(triplet) for triplet, triplet_id in triplet_to_id.items()}


def mask_to_triplet_labels(
    mask: torch.Tensor,
    id_to_triplet: dict[int, tuple[int, int, int]],
    tool_names: list[str],
    verb_names: list[str],
    target_names: list[str],
) -> list[str]:
    mask = mask.detach().cpu().bool().reshape(-1)
    labels = []
    for triplet_id, active in enumerate(mask.tolist()):
        if not active:
            continue
        labels.append(
            triplet_label(id_to_triplet[triplet_id], tool_names, verb_names, target_names)
        )
    return labels


def pair_name_tool(pair: tuple[int, int], study_verb_names: list[str], target_names: list[str]) -> str:
    return f"({study_verb_names[pair[0]]}, {target_names[pair[1]]})"


def pair_name_verb(pair: tuple[int, int], tool_names: list[str], target_names: list[str]) -> str:
    return f"({tool_names[pair[0]]}, {target_names[pair[1]]})"


def pair_name_target(pair: tuple[int, int], tool_names: list[str], study_verb_names: list[str]) -> str:
    return f"({tool_names[pair[0]]}, {study_verb_names[pair[1]]})"


def print_axis_block(
    title: str,
    probabilities: torch.Tensor,
    class_names: list[str],
    thresholds: torch.Tensor | None,
    topk: int,
) -> None:
    print()
    print(title)
    for row in axis_top_rows(probabilities, class_names, thresholds, topk):
        suffix = ""
        if "threshold" in row:
            suffix = f" threshold={row['threshold']:.4f} active={row['active']}"
        print(
            f"  [{row['id']:>2}] {row['name']}: prob={row['probability']:.4f}{suffix}"
        )


def print_selected_distribution(
    title: str,
    indices: torch.Tensor,
    weights: torch.Tensor,
    class_names: list[str],
) -> None:
    print()
    print(title)
    indices = indices.detach().cpu().long().reshape(-1)
    weights = weights.detach().cpu().float().reshape(-1)
    if indices.numel() == 0:
        print("  <none>")
        return
    for idx_tensor, weight_tensor in zip(indices, weights):
        idx = int(idx_tensor.item())
        print(f"  [{idx:>2}] {class_names[idx]}: weight={float(weight_tensor.item()):.4f}")


def print_active_block(title: str, mask: torch.Tensor, class_names: list[str]) -> None:
    labels = active_names(mask, class_names)
    print()
    print(title)
    if not labels:
        print("  <none>")
        return
    for label in labels:
        print(f"  {label}")


def print_pair_block(
    title: str,
    pair_indices: torch.Tensor,
    pair_weights: torch.Tensor,
    conditioned_probs: torch.Tensor,
    pair_name_fn,
    class_names: list[str],
    pair_topk: int,
    class_topk: int,
) -> None:
    print()
    print(title)
    pair_indices = pair_indices.detach().cpu().long()
    pair_weights = pair_weights.detach().cpu().float().reshape(-1)
    conditioned_probs = conditioned_probs.detach().cpu().float()
    if pair_indices.numel() == 0:
        print("  <no pairs>")
        return

    order = torch.argsort(pair_weights, descending=True)
    order = order[: min(int(pair_topk), int(order.numel()))]
    for rank, pos_tensor in enumerate(order, start=1):
        pos = int(pos_tensor.item())
        pair = tuple(int(x) for x in pair_indices[pos].tolist())
        weight = float(pair_weights[pos].item())
        print(f"  Pair {rank}: {pair_name_fn(pair)} weight={weight:.4f}")
        top_rows = axis_top_rows(conditioned_probs[pos], class_names, None, class_topk)
        for row in top_rows:
            print(
                f"    [{row['id']:>2}] {row['name']}: prob={row['probability']:.4f}"
            )


def serialize_for_json(value: Any) -> Any:
    if torch.is_tensor(value):
        tensor = value.detach().cpu()
        if tensor.ndim == 0:
            return tensor.item()
        return tensor.tolist()
    if isinstance(value, dict):
        return {str(key): serialize_for_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def resolve_sample_index(
    dataset,
    requested_idx: int,
    requested_path: str | None,
) -> tuple[int, str]:
    if requested_path is None:
        if requested_idx < 0 or requested_idx >= len(dataset):
            raise ValueError(
                f"--sample-idx must be in [0, {len(dataset) - 1}], got {requested_idx}"
            )
        _, _, _, sample_path = dataset[requested_idx]
        return int(requested_idx), str(sample_path)

    for sample_idx in range(len(dataset)):
        _, _, _, sample_path = dataset[sample_idx]
        if str(sample_path) == str(requested_path):
            return int(sample_idx), str(sample_path)
    raise KeyError(f"Could not find validation clip path in dataset: {requested_path}")


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

    device = resolve_device(args.device)
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
    head_temperatures = (
        HeadTemperatures.from_json(args.calibration_json.resolve())
        if args.calibration_json is not None
        else HeadTemperatures.identity()
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

    tool_thresholds = resolve_threshold_vector(tool_refine_cfg, args.tool_threshold)
    verb_thresholds = resolve_grouped_verb_threshold_vector(
        verb0_cfg,
        verb_refine_cfg,
        args.verb_threshold,
    )
    target_thresholds = resolve_threshold_vector(target_refine_cfg, args.target_threshold)
    route_tool_thresholds = resolve_threshold_vector(tool_refine_cfg, args.tool_threshold)
    route_verb_thresholds = resolve_threshold_vector(verb_refine_cfg, None)
    route_target_thresholds = resolve_threshold_vector(target_refine_cfg, args.target_threshold)

    tool_names = class_names_from_config(tool0_cfg, FALLBACK_TOOL_NAMES, "tool")
    verb_names = class_names_from_config(verb0_cfg, FALLBACK_VERB_NAMES, "verb")
    target_names = class_names_from_config(target0_cfg, FALLBACK_TARGET_NAMES, "target")
    study_verb_names = class_names_from_config(verb_refine_cfg, FALLBACK_VERB_NAMES, "verb")

    label_lookup = load_eval_label_lookup(eval_label_csv)
    dataset = build_eval_dataset(tool0_cfg, eval_runtime_csv)
    sample_idx, sample_path = resolve_sample_index(dataset, args.sample_idx, args.sample_path)
    if sample_path not in label_lookup:
        raise KeyError(f"Could not find grouped labels for validation clip: {sample_path}")
    sample_labels = label_lookup[sample_path]

    triplet_to_id, num_triplet_classes, supported_triplets = load_triplet_support_index(
        triplet_metadata_path
    )
    id_to_triplet = build_id_to_triplet(triplet_to_id)
    support_pair_constraints = None
    if support_constrained_routing_mode != "none":
        support_pair_constraints = build_support_pair_constraints(
            supported_triplets,
            use_verb_adapter=uses_grouped_to_study_verb_adapter(verb0_cfg, verb_refine_cfg),
        )

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
    print("Single-clip soft refinement debug")
    print(f"  device: {device}")
    print(f"  sample_idx: {sample_idx}")
    print(f"  clip_path: {sample_path}")
    print(f"  eval_runtime_csv: {eval_runtime_csv}")
    print(f"  eval_label_csv: {eval_label_csv}")
    print(f"  triplet_metadata: {triplet_metadata_path}")
    print(f"  calibration_json: {args.calibration_json.resolve() if args.calibration_json else '<identity>'}")
    print(f"  tool0 checkpoint: {tool0_checkpoint}")
    print(f"  verb0 checkpoint: {verb0_checkpoint}")
    print(f"  target0 checkpoint: {target0_checkpoint}")
    print(f"  tool_refine checkpoint: {tool_refine_checkpoint}")
    print(f"  verb_refine checkpoint: {verb_refine_checkpoint}")
    print(f"  target_refine checkpoint: {target_refine_checkpoint}")

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

    buffer, _, clip_indices, _ = dataset[sample_idx]
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
                route_topk_tool=args.route_topk_tool,
                route_topk_verb=args.route_topk_verb,
                route_topk_target=args.route_topk_target,
                blend_alpha_step2=args.blend_alpha_step2,
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
                collect_debug=True,
            )

    target_tool_mask = sample_labels["tool"] > 0.5
    target_verb_mask = sample_labels["verb"] > 0.5
    target_target_mask = sample_labels["target"] > 0.5
    target_triplet_mask = sample_labels["triplet"] > 0.5

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

    decoded = None
    predicted_tool_mask = raw_predicted_tool_mask
    predicted_verb_mask = raw_predicted_verb_mask
    predicted_target_mask = raw_predicted_target_mask
    predicted_triplet_mask = raw_predicted_triplet_mask
    if args.support_aware_triplet_refinement:
        decoded = decode_support_aware_triplets(
            refinement=refinement,
            triplet_to_id=triplet_to_id,
            num_triplet_classes=num_triplet_classes,
            tool_thresholds=tool_thresholds,
            verb_thresholds=verb_thresholds,
            target_thresholds=target_thresholds,
            search_topk_tool=(
                args.decode_topk_tool if args.decode_topk_tool is not None else args.route_topk_tool
            ),
            search_topk_verb=(
                args.decode_topk_verb if args.decode_topk_verb is not None else args.route_topk_verb
            ),
            search_topk_target=(
                args.decode_topk_target if args.decode_topk_target is not None else args.route_topk_target
            ),
            max_triplets_total=args.decode_max_triplets_total,
            min_triplet_score=args.decode_score_min,
            fallback_top1=False,
        )
        predicted_tool_mask = decoded["tool_mask"]
        predicted_verb_mask = decoded["verb_mask"]
        predicted_target_mask = decoded["target_mask"]
        predicted_triplet_mask = decoded["triplet_mask"]

    print_active_block("Ground-truth tools", target_tool_mask, tool_names)
    print_active_block("Ground-truth verbs", target_verb_mask, verb_names)
    print_active_block("Ground-truth targets", target_target_mask, target_names)
    print()
    print("Ground-truth triplets")
    gt_triplets = mask_to_triplet_labels(
        target_triplet_mask, id_to_triplet, tool_names, verb_names, target_names
    )
    if not gt_triplets:
        print("  <none>")
    else:
        for label in gt_triplets:
            print(f"  {label}")

    print_axis_block(
        "Stage-0 tool probabilities",
        refinement["stages"][0]["tool"],
        tool_names,
        tool_thresholds,
        args.print_topk,
    )
    print_axis_block(
        "Stage-0 verb probabilities (grouped display space)",
        refinement["stages"][0]["verb"],
        verb_names,
        verb_thresholds,
        args.print_topk,
    )
    print_axis_block(
        "Stage-0 target probabilities",
        refinement["stages"][0]["target"],
        target_names,
        target_thresholds,
        args.print_topk,
    )

    for step in refinement["step_debug"] or []:
        step_idx = int(step["step_index"])
        print()
        print(f"========== Refinement step {step_idx} ==========")
        print(f"blend_alpha={float(step['blend_alpha']):.4f}")

        print_selected_distribution(
            f"Step {step_idx} routing tools",
            step["routing_tool_indices"],
            step["routing_tool_weights"],
            tool_names,
        )
        print_selected_distribution(
            f"Step {step_idx} routing verbs (study space)",
            step["routing_verb_indices"],
            step["routing_verb_weights"],
            study_verb_names,
        )
        print_selected_distribution(
            f"Step {step_idx} routing targets",
            step["routing_target_indices"],
            step["routing_target_weights"],
            target_names,
        )

        print_pair_block(
            title=f"Step {step_idx} tool head pairs (verb,target) -> tool probs",
            pair_indices=step["tool_pairs"],
            pair_weights=step["tool_pair_weights"],
            conditioned_probs=step["tool_conditioned_probs"],
            pair_name_fn=lambda pair: pair_name_tool(pair, study_verb_names, target_names),
            class_names=tool_names,
            pair_topk=args.print_pair_topk,
            class_topk=args.print_conditioned_topk,
        )
        print_axis_block(
            f"Step {step_idx} mixed tool probs before blend",
            step["tool_mixed_probs"],
            tool_names,
            None,
            args.print_topk,
        )
        print_axis_block(
            f"Step {step_idx} blended tool probs",
            step["next_tool_probs"],
            tool_names,
            tool_thresholds,
            args.print_topk,
        )

        print_pair_block(
            title=f"Step {step_idx} verb head pairs (tool,target) -> verb probs",
            pair_indices=step["verb_pairs"],
            pair_weights=step["verb_pair_weights"],
            conditioned_probs=step["verb_conditioned_probs"],
            pair_name_fn=lambda pair: pair_name_verb(pair, tool_names, target_names),
            class_names=study_verb_names,
            pair_topk=args.print_pair_topk,
            class_topk=args.print_conditioned_topk,
        )
        print_axis_block(
            f"Step {step_idx} mixed verb probs before blend (study space)",
            step["verb_mixed_probs"],
            study_verb_names,
            None,
            args.print_topk,
        )
        print_axis_block(
            f"Step {step_idx} blended verb probs (study routing space)",
            step["next_verb_routing_probs"],
            study_verb_names,
            None,
            args.print_topk,
        )
        print_axis_block(
            f"Step {step_idx} blended verb probs (grouped display space)",
            step["next_verb_display_probs"],
            verb_names,
            verb_thresholds,
            args.print_topk,
        )

        print_pair_block(
            title=f"Step {step_idx} target head pairs (tool,verb) -> target probs",
            pair_indices=step["target_pairs"],
            pair_weights=step["target_pair_weights"],
            conditioned_probs=step["target_conditioned_probs"],
            pair_name_fn=lambda pair: pair_name_target(pair, tool_names, study_verb_names),
            class_names=target_names,
            pair_topk=args.print_pair_topk,
            class_topk=args.print_conditioned_topk,
        )
        print_axis_block(
            f"Step {step_idx} mixed target probs before blend",
            step["target_mixed_probs"],
            target_names,
            None,
            args.print_topk,
        )
        print_axis_block(
            f"Step {step_idx} blended target probs",
            step["next_target_probs"],
            target_names,
            target_thresholds,
            args.print_topk,
        )

    print()
    print("========== Final state ==========")
    print_axis_block(
        "Final tool probabilities",
        refinement["final"]["tool"],
        tool_names,
        tool_thresholds,
        args.print_topk,
    )
    print_axis_block(
        "Final verb probabilities (grouped display space)",
        refinement["final"]["verb"],
        verb_names,
        verb_thresholds,
        args.print_topk,
    )
    print_axis_block(
        "Final target probabilities",
        refinement["final"]["target"],
        target_names,
        target_thresholds,
        args.print_topk,
    )
    print_selected_distribution(
        "Final routing tools",
        refinement["final_tool_indices"],
        refinement["final_tool_weights"],
        tool_names,
    )
    print_selected_distribution(
        "Final routing verbs (study space)",
        refinement["final_verb_indices"],
        refinement["final_verb_weights"],
        study_verb_names,
    )
    print_selected_distribution(
        "Final routing targets",
        refinement["final_target_indices"],
        refinement["final_target_weights"],
        target_names,
    )
    print_pair_block(
        title="Final tool routing pairs (verb,target) -> tool probs",
        pair_indices=refinement["final_tool_pairs"],
        pair_weights=refinement["final_tool_pair_weights"],
        conditioned_probs=refinement["final_tool_conditioned_probs"],
        pair_name_fn=lambda pair: pair_name_tool(pair, study_verb_names, target_names),
        class_names=tool_names,
        pair_topk=args.print_pair_topk,
        class_topk=args.print_conditioned_topk,
    )
    print_pair_block(
        title="Final verb routing pairs (tool,target) -> verb probs",
        pair_indices=refinement["final_verb_pairs"],
        pair_weights=refinement["final_verb_pair_weights"],
        conditioned_probs=refinement["final_verb_conditioned_probs"],
        pair_name_fn=lambda pair: pair_name_verb(pair, tool_names, target_names),
        class_names=study_verb_names,
        pair_topk=args.print_pair_topk,
        class_topk=args.print_conditioned_topk,
    )
    print_pair_block(
        title="Final target routing pairs (tool,verb) -> target probs",
        pair_indices=refinement["final_target_pairs"],
        pair_weights=refinement["final_target_pair_weights"],
        conditioned_probs=refinement["final_target_conditioned_probs"],
        pair_name_fn=lambda pair: pair_name_target(pair, tool_names, study_verb_names),
        class_names=target_names,
        pair_topk=args.print_pair_topk,
        class_topk=args.print_conditioned_topk,
    )

    print_active_block("Raw thresholded tool prediction", raw_predicted_tool_mask, tool_names)
    print_active_block("Raw thresholded verb prediction", raw_predicted_verb_mask, verb_names)
    print_active_block("Raw thresholded target prediction", raw_predicted_target_mask, target_names)
    print()
    print(f"Raw unsupported predicted triplets: {int(raw_unsupported_count)}")
    print("Raw predicted triplets")
    raw_triplets = mask_to_triplet_labels(
        raw_predicted_triplet_mask, id_to_triplet, tool_names, verb_names, target_names
    )
    if not raw_triplets:
        print("  <none>")
    else:
        for label in raw_triplets:
            print(f"  {label}")

    if decoded is not None:
        print_active_block("Support-aware final tool prediction", predicted_tool_mask, tool_names)
        print_active_block("Support-aware final verb prediction", predicted_verb_mask, verb_names)
        print_active_block("Support-aware final target prediction", predicted_target_mask, target_names)
        print()
        print("Support-aware decoded triplets")
        if not decoded["triplets"]:
            print("  <none>")
        else:
            for rank, row in enumerate(decoded["triplets"], start=1):
                label = triplet_label(
                    (int(row["tool_id"]), int(row["verb_id"]), int(row["target_id"])),
                    tool_names,
                    verb_names,
                    target_names,
                )
                print(
                    f"  {rank}. {label} "
                    f"score={float(row['score']):.4f} "
                    f"base=({float(row['tool_probability']):.4f}, "
                    f"{float(row['verb_probability']):.4f}, "
                    f"{float(row['target_probability']):.4f}) "
                    f"cond=({float(row.get('tool_conditioned_probability', row['tool_probability'])):.4f}, "
                    f"{float(row.get('verb_conditioned_probability', row['verb_probability'])):.4f}, "
                    f"{float(row.get('target_conditioned_probability', row['target_probability'])):.4f})"
                )
    else:
        print_active_block("Final tool prediction", predicted_tool_mask, tool_names)
        print_active_block("Final verb prediction", predicted_verb_mask, verb_names)
        print_active_block("Final target prediction", predicted_target_mask, target_names)

    print()
    print("Final predicted triplets")
    final_triplets = mask_to_triplet_labels(
        predicted_triplet_mask, id_to_triplet, tool_names, verb_names, target_names
    )
    if not final_triplets:
        print("  <none>")
    else:
        for label in final_triplets:
            print(f"  {label}")

    if args.output_json is not None:
        payload = {
            "sample_idx": int(sample_idx),
            "clip_path": sample_path,
            "checkpoint_paths": {
                "tool0": str(tool0_checkpoint.resolve()),
                "verb0": str(verb0_checkpoint.resolve()),
                "target0": str(target0_checkpoint.resolve()),
                "tool_refine": str(tool_refine_checkpoint.resolve()),
                "verb_refine": str(verb_refine_checkpoint.resolve()),
                "target_refine": str(target_refine_checkpoint.resolve()),
            },
            "ground_truth": {
                "tool_labels": active_names(target_tool_mask, tool_names),
                "verb_labels": active_names(target_verb_mask, verb_names),
                "target_labels": active_names(target_target_mask, target_names),
                "triplet_labels": gt_triplets,
            },
            "raw_prediction": {
                "tool_labels": active_names(raw_predicted_tool_mask, tool_names),
                "verb_labels": active_names(raw_predicted_verb_mask, verb_names),
                "target_labels": active_names(raw_predicted_target_mask, target_names),
                "triplet_labels": raw_triplets,
                "unsupported_triplets": int(raw_unsupported_count),
            },
            "final_prediction": {
                "tool_labels": active_names(predicted_tool_mask, tool_names),
                "verb_labels": active_names(predicted_verb_mask, verb_names),
                "target_labels": active_names(predicted_target_mask, target_names),
                "triplet_labels": final_triplets,
                "decoded_triplets": serialize_for_json(decoded["triplets"]) if decoded is not None else None,
            },
            "refinement": serialize_for_json(refinement),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print()
        print(f"Saved debug JSON to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
