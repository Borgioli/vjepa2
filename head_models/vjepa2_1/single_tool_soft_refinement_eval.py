#!/path/to/data_root/vjepa2_1/.venv/bin/python
"""Evaluate one single-tool six-head soft-refinement model on its val split.

The implementation reuses the generic soft-refinement machinery from
triplet_soft_refinement_inference.py. That module names the middle triplet axis
"verb"; here it is exposed as "action" because the single-tool configs use
tool/action/target component names.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments.
    tqdm = None

from triplet_conditioned_inference import (
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
    MultiLabelMetricsAccumulator,
    build_eval_dataset,
    compute_bootstrap_ci,
    pack_bool_mask,
    prepare_dataset_sample_inputs,
    print_metrics_block,
    resolve_shard_sample_indices,
    resolve_threshold_vector,
    threshold_probabilities,
)
from triplet_soft_refinement_inference import (
    HeadTemperatures,
    compute_soft_refined_probabilities,
    resolve_multilabel_threshold_vector,
    resolve_soft_refinement_checkpoint_path,
    resolve_support_constrained_routing_mode,
    temperature_scaled_sigmoid,
    validate_soft_refinement_configs,
    validate_topk,
)


CONFIG_DIR = Path("/path/to/data_root/vjepa2_1/configs/heads/single_tool")
SINGLE_TOOL_DATA_DIR = Path("/path/to/data_root/vjepa2_1/app/csv_head_models/single_tool")
SUPPORTED_TOOLS = ("hook", "clipper", "grasper")


def default_config_paths(tool: str) -> dict[str, Path]:
    return {
        "tool0": CONFIG_DIR / f"{tool}_tool_unconditioned.yaml",
        "action0": CONFIG_DIR / f"{tool}_action_unconditioned.yaml",
        "target0": CONFIG_DIR / f"{tool}_target_unconditioned.yaml",
        "tool_refine": CONFIG_DIR / f"{tool}_tool_conditioned.yaml",
        "action_refine": CONFIG_DIR / f"{tool}_action_conditioned.yaml",
        "target_refine": CONFIG_DIR / f"{tool}_target_conditioned.yaml",
    }


def default_data_paths(tool: str) -> dict[str, Path]:
    return {
        "runtime": SINGLE_TOOL_DATA_DIR / f"yt_robotic_chole_tool_{tool}_multilabel_val_runtime.csv",
        "labels": SINGLE_TOOL_DATA_DIR / f"yt_robotic_chole_tool_{tool}_multilabel_val.csv",
        "metadata": SINGLE_TOOL_DATA_DIR / f"yt_robotic_chole_tool_{tool}_single_tool_metadata.json",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate hook/clipper/grasper single-tool soft refinement on that "
            "tool's validation split."
        )
    )
    parser.add_argument("--tool", choices=SUPPORTED_TOOLS, default="hook")

    parser.add_argument("--tool0-config", type=Path, default=None)
    parser.add_argument("--action0-config", type=Path, default=None)
    parser.add_argument("--target0-config", type=Path, default=None)
    parser.add_argument("--tool-refine-config", type=Path, default=None)
    parser.add_argument("--action-refine-config", type=Path, default=None)
    parser.add_argument("--target-refine-config", type=Path, default=None)

    parser.add_argument("--tool0-checkpoint", type=Path, default=None)
    parser.add_argument("--action0-checkpoint", type=Path, default=None)
    parser.add_argument("--target0-checkpoint", type=Path, default=None)
    parser.add_argument("--tool-refine-checkpoint", type=Path, default=None)
    parser.add_argument("--action-refine-checkpoint", type=Path, default=None)
    parser.add_argument("--target-refine-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--head-checkpoint-policy",
        choices=("best", "latest"),
        default="best",
        help="'best' prefers best*.pt checkpoints and falls back to latest.pt.",
    )
    parser.add_argument("--strict-best-head-checkpoints", action="store_true")

    parser.add_argument("--eval-runtime-csv", type=Path, default=None)
    parser.add_argument("--eval-label-csv", type=Path, default=None)
    parser.add_argument("--metadata-json", type=Path, default=None)

    parser.add_argument("--backbone-checkpoint", "--encoder-checkpoint", dest="backbone_checkpoint", type=Path)
    parser.add_argument("--backbone-checkpoint-key", "--encoder-checkpoint-key", dest="backbone_checkpoint_key")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")

    parser.add_argument("--tool-threshold", type=float, default=None)
    parser.add_argument("--action-threshold", type=float, default=None)
    parser.add_argument("--target-threshold", type=float, default=None)
    parser.add_argument(
        "--stage0-only",
        action="store_true",
        help=(
            "Evaluate only the unconditioned tool0/action0/target0 heads. "
            "No conditioned heads, soft routing, or refinement pass is used."
        ),
    )
    parser.add_argument("--refinement-steps", type=int, default=2, choices=(1, 2))
    parser.add_argument("--blend-alpha", type=float, default=0.75)
    parser.add_argument("--blend-alpha-step2", type=float, default=0.9)
    parser.add_argument("--route-topk-tool", type=int, default=None)
    parser.add_argument("--route-topk-action", type=int, default=None)
    parser.add_argument("--route-topk-target", type=int, default=None)
    parser.add_argument(
        "--routing-probability-source",
        choices=("current", "stage0"),
        default="current",
    )
    parser.add_argument(
        "--support-constrained-routing",
        action="store_true",
        help="Compatibility alias for --support-constrained-routing-mode hard.",
    )
    parser.add_argument(
        "--support-constrained-routing-mode",
        choices=("none", "soft", "hard"),
        default="hard",
    )
    parser.add_argument("--support-constrained-routing-unsupported-scale", type=float, default=0.1)
    parser.add_argument("--calibration-json", type=Path, default=None)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260101)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--no-store-per-sample-masks", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=None,
        help="Evaluate the first fraction of the validation split, e.g. 0.1 for 10%%.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, Path]]:
    config_defaults = default_config_paths(args.tool)
    data_defaults = default_data_paths(args.tool)
    config_paths = {
        "tool0": args.tool0_config or config_defaults["tool0"],
        "action0": args.action0_config or config_defaults["action0"],
        "target0": args.target0_config or config_defaults["target0"],
        "tool_refine": args.tool_refine_config or config_defaults["tool_refine"],
        "action_refine": args.action_refine_config or config_defaults["action_refine"],
        "target_refine": args.target_refine_config or config_defaults["target_refine"],
    }
    data_paths = {
        "runtime": args.eval_runtime_csv or data_defaults["runtime"],
        "labels": args.eval_label_csv or data_defaults["labels"],
        "metadata": args.metadata_json or data_defaults["metadata"],
    }
    return config_paths, data_paths


def parse_multihot(value: str) -> torch.Tensor:
    value = str(value).strip()
    if not value:
        raise ValueError("Encountered empty multi-hot field")
    return torch.tensor([float(token) for token in value.split()], dtype=torch.float32)


def parse_local_label_ids(value: str, num_triplet_classes: int) -> torch.Tensor:
    mask = torch.zeros(num_triplet_classes, dtype=torch.bool)
    value = str(value).strip()
    if not value:
        return mask
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if idx < 0 or idx >= num_triplet_classes:
            raise ValueError(f"Local label id {idx} outside [0, {num_triplet_classes})")
        mask[idx] = True
    return mask


def load_single_tool_metadata(path: Path) -> tuple[dict[tuple[int, int, int], int], int, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    raw = metadata.get("local_label_id_to_component_ids")
    if not isinstance(raw, dict):
        raise ValueError(f"Missing local_label_id_to_component_ids in {path}")
    triplet_to_id: dict[tuple[int, int, int], int] = {}
    max_id = -1
    for label_id_text, components in raw.items():
        label_id = int(label_id_text)
        max_id = max(max_id, label_id)
        key = (
            int(components["tool"]),
            int(components["action"]),
            int(components["target"]),
        )
        triplet_to_id[key] = label_id
    return triplet_to_id, max_id + 1, metadata


def load_eval_lookup(path: Path, num_triplet_classes: int) -> dict[str, dict[str, torch.Tensor]]:
    lookup: dict[str, dict[str, torch.Tensor]] = {}
    with path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        required = ["clip_path", "tool_multihot", "action_multihot", "target_multihot", "active_local_label_ids"]
        missing = [field for field in required if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required fields in {path}: {missing}")
        for row in reader:
            clip_path = str(row["clip_path"]).strip()
            if not clip_path:
                continue
            if clip_path in lookup:
                raise ValueError(f"Duplicate clip_path in {path}: {clip_path}")
            lookup[clip_path] = {
                "tool": parse_multihot(row["tool_multihot"]),
                "action": parse_multihot(row["action_multihot"]),
                "target": parse_multihot(row["target_multihot"]),
                "triplet": parse_local_label_ids(row["active_local_label_ids"], num_triplet_classes),
            }
    return lookup


def build_support_pair_constraints(
    triplet_to_id: dict[tuple[int, int, int], int],
) -> dict[str, frozenset[tuple[int, int]]]:
    return {
        "tool": frozenset((action_id, target_id) for _, action_id, target_id in triplet_to_id),
        "verb": frozenset((tool_id, target_id) for tool_id, _, target_id in triplet_to_id),
        "target": frozenset((tool_id, action_id) for tool_id, action_id, _ in triplet_to_id),
    }


def build_predicted_triplet_mask(
    refinement: dict[str, Any],
    predicted_tool_mask: torch.Tensor,
    predicted_action_mask: torch.Tensor,
    target_thresholds: torch.Tensor,
    triplet_to_id: dict[tuple[int, int, int], int],
    num_triplet_classes: int,
) -> tuple[torch.Tensor, int]:
    triplet_mask = torch.zeros(num_triplet_classes, dtype=torch.bool)
    unsupported_count = 0
    final_target_pairs = refinement["final_target_pairs"].detach().cpu()
    final_target_probs = refinement["final_target_conditioned_probs"].detach().cpu().float()
    target_thresholds = target_thresholds.detach().cpu().float()
    for pair_idx, pair in enumerate(final_target_pairs.tolist()):
        tool_id, action_id = int(pair[0]), int(pair[1])
        if not bool(predicted_tool_mask[tool_id].item()):
            continue
        if not bool(predicted_action_mask[action_id].item()):
            continue
        target_ids = (final_target_probs[pair_idx] >= target_thresholds).nonzero(
            as_tuple=False
        ).reshape(-1).tolist()
        for target_id in target_ids:
            triplet_id = triplet_to_id.get((tool_id, action_id, int(target_id)))
            if triplet_id is None:
                unsupported_count += 1
                continue
            triplet_mask[triplet_id] = True
    return triplet_mask, unsupported_count


def build_triplet_mask_from_component_masks(
    predicted_tool_mask: torch.Tensor,
    predicted_action_mask: torch.Tensor,
    predicted_target_mask: torch.Tensor,
    triplet_to_id: dict[tuple[int, int, int], int],
    num_triplet_classes: int,
) -> tuple[torch.Tensor, int]:
    triplet_mask = torch.zeros(num_triplet_classes, dtype=torch.bool)
    unsupported_count = 0
    tool_ids = predicted_tool_mask.detach().cpu().bool().nonzero(as_tuple=False).reshape(-1).tolist()
    action_ids = predicted_action_mask.detach().cpu().bool().nonzero(as_tuple=False).reshape(-1).tolist()
    target_ids = predicted_target_mask.detach().cpu().bool().nonzero(as_tuple=False).reshape(-1).tolist()
    for tool_id in tool_ids:
        for action_id in action_ids:
            for target_id in target_ids:
                triplet_id = triplet_to_id.get((int(tool_id), int(action_id), int(target_id)))
                if triplet_id is None:
                    unsupported_count += 1
                    continue
                triplet_mask[triplet_id] = True
    return triplet_mask, unsupported_count


def compute_stage0_probabilities(
    encoded_clip: Any,
    heads: dict[str, torch.nn.Module],
    head_temperatures: HeadTemperatures,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tool_logits = heads["tool0"](encoded_clip)["task_0"][0]
    action_logits = heads["action0"](encoded_clip)["task_0"][0]
    target_logits = heads["target0"](encoded_clip)["task_0"][0]
    tool_temp = head_temperatures.get("tool0", int(tool_logits.numel()), device)
    action_temp = head_temperatures.get("verb0", int(action_logits.numel()), device)
    target_temp = head_temperatures.get("target0", int(target_logits.numel()), device)
    return {
        "tool": temperature_scaled_sigmoid(tool_logits, tool_temp),
        "action": temperature_scaled_sigmoid(action_logits, action_temp),
        "target": temperature_scaled_sigmoid(target_logits, target_temp),
    }


def main() -> int:
    args = parse_args()
    validate_topk("route_topk_tool", args.route_topk_tool)
    validate_topk("route_topk_action", args.route_topk_action)
    validate_topk("route_topk_target", args.route_topk_target)
    if not (0.0 <= args.blend_alpha <= 1.0):
        raise ValueError(f"--blend-alpha must be in [0, 1], got {args.blend_alpha}")
    if args.blend_alpha_step2 is not None and not (0.0 <= args.blend_alpha_step2 <= 1.0):
        raise ValueError(f"--blend-alpha-step2 must be in [0, 1], got {args.blend_alpha_step2}")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError(f"--max-samples must be >= 1, got {args.max_samples}")
    if args.eval_fraction is not None and not (0.0 < args.eval_fraction <= 1.0):
        raise ValueError(f"--eval-fraction must be in (0, 1], got {args.eval_fraction}")

    config_paths, data_paths = resolve_paths(args)
    required_config_keys = ["tool0", "action0", "target0"]
    if not args.stage0_only:
        required_config_keys.extend(["tool_refine", "action_refine", "target_refine"])
    required_paths = [config_paths[key] for key in required_config_keys]
    required_paths.extend(data_paths.values())
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    head_temperatures = (
        HeadTemperatures.from_json(args.calibration_json.resolve())
        if args.calibration_json is not None
        else HeadTemperatures.identity()
    )
    device = resolve_device(args.device)
    support_mode = resolve_support_constrained_routing_mode(
        args.support_constrained_routing_mode,
        args.support_constrained_routing,
    )

    tool0_cfg = load_yaml_config(config_paths["tool0"])
    action0_cfg = load_yaml_config(config_paths["action0"])
    target0_cfg = load_yaml_config(config_paths["target0"])
    if args.stage0_only:
        tool_refine_cfg = tool0_cfg
        action_refine_cfg = action0_cfg
        target_refine_cfg = target0_cfg
        all_cfgs = [tool0_cfg, action0_cfg, target0_cfg]
    else:
        tool_refine_cfg = load_yaml_config(config_paths["tool_refine"])
        action_refine_cfg = load_yaml_config(config_paths["action_refine"])
        target_refine_cfg = load_yaml_config(config_paths["target_refine"])
        all_cfgs = [tool0_cfg, action0_cfg, target0_cfg, tool_refine_cfg, action_refine_cfg, target_refine_cfg]
    override_backbone_checkpoint(all_cfgs, args.backbone_checkpoint, checkpoint_key=args.backbone_checkpoint_key)
    if not args.stage0_only:
        validate_soft_refinement_configs(
            tool0_cfg=tool0_cfg,
            verb0_cfg=action0_cfg,
            target0_cfg=target0_cfg,
            tool_refine_cfg=tool_refine_cfg,
            verb_refine_cfg=action_refine_cfg,
            target_refine_cfg=target_refine_cfg,
        )

    tool_thresholds = resolve_threshold_vector(tool_refine_cfg, args.tool_threshold)
    action_thresholds = resolve_threshold_vector(action_refine_cfg, args.action_threshold)
    target_thresholds = resolve_threshold_vector(target_refine_cfg, args.target_threshold)
    route_tool_thresholds = resolve_multilabel_threshold_vector(tool_refine_cfg, args.tool_threshold)
    route_action_thresholds = resolve_multilabel_threshold_vector(action_refine_cfg, args.action_threshold)
    route_target_thresholds = resolve_multilabel_threshold_vector(target_refine_cfg, args.target_threshold)

    tool_names = class_names_from_config(tool0_cfg, [], "tool")
    action_names = class_names_from_config(action0_cfg, [], "action")
    target_names = class_names_from_config(target0_cfg, [], "target")
    triplet_to_id, num_triplet_classes, metadata = load_single_tool_metadata(data_paths["metadata"])
    label_lookup = load_eval_lookup(data_paths["labels"], num_triplet_classes)
    support_pair_constraints = None
    if support_mode != "none" and not args.stage0_only:
        support_pair_constraints = build_support_pair_constraints(triplet_to_id)

    dataset = build_eval_dataset(tool0_cfg, data_paths["runtime"])
    checkpoint_paths = {
        "tool0": resolve_soft_refinement_checkpoint_path(
            tool0_cfg,
            config_paths["tool0"],
            args.tool0_checkpoint,
            fallback_name="tool",
            checkpoint_policy=args.head_checkpoint_policy,
            strict_best=args.strict_best_head_checkpoints,
        ),
        "action0": resolve_soft_refinement_checkpoint_path(
            action0_cfg,
            config_paths["action0"],
            args.action0_checkpoint,
            fallback_name="action",
            checkpoint_policy=args.head_checkpoint_policy,
            strict_best=args.strict_best_head_checkpoints,
        ),
        "target0": resolve_soft_refinement_checkpoint_path(
            target0_cfg,
            config_paths["target0"],
            args.target0_checkpoint,
            fallback_name="target",
            checkpoint_policy=args.head_checkpoint_policy,
            strict_best=args.strict_best_head_checkpoints,
        ),
    }
    if not args.stage0_only:
        checkpoint_paths.update(
            {
                "tool_refine": resolve_soft_refinement_checkpoint_path(
                    tool_refine_cfg,
                    config_paths["tool_refine"],
                    args.tool_refine_checkpoint,
                    fallback_name="tool",
                    checkpoint_policy=args.head_checkpoint_policy,
                    strict_best=args.strict_best_head_checkpoints,
                ),
                "action_refine": resolve_soft_refinement_checkpoint_path(
                    action_refine_cfg,
                    config_paths["action_refine"],
                    args.action_refine_checkpoint,
                    fallback_name="action",
                    checkpoint_policy=args.head_checkpoint_policy,
                    strict_best=args.strict_best_head_checkpoints,
                ),
                "target_refine": resolve_soft_refinement_checkpoint_path(
                    target_refine_cfg,
                    config_paths["target_refine"],
                    args.target_refine_checkpoint,
                    fallback_name="target",
                    checkpoint_policy=args.head_checkpoint_policy,
                    strict_best=args.strict_best_head_checkpoints,
                ),
            }
        )

    frames_per_clip = int(tool0_cfg["experiment"]["data"].get("frames_per_clip", 16))
    resolution = int(tool0_cfg["experiment"]["data"].get("resolution", 224))
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

    heads = {
        "tool0": build_classifier_from_config(tool0_cfg, encoder.embed_dim).to(device).eval(),
        "action0": build_classifier_from_config(action0_cfg, encoder.embed_dim).to(device).eval(),
        "target0": build_classifier_from_config(target0_cfg, encoder.embed_dim).to(device).eval(),
    }
    if not args.stage0_only:
        heads.update(
            {
                "tool_refine": build_classifier_from_config(tool_refine_cfg, encoder.embed_dim).to(device).eval(),
                "action_refine": build_classifier_from_config(action_refine_cfg, encoder.embed_dim).to(device).eval(),
                "target_refine": build_classifier_from_config(target_refine_cfg, encoder.embed_dim).to(device).eval(),
            }
        )
    for name, head in heads.items():
        load_classifier_weights(head, checkpoint_paths[name])

    if args.verbose:
        print(f"Tool: {args.tool}")
        print(f"Using device: {device}")
        print(f"Runtime CSV: {data_paths['runtime']}")
        print(f"Label CSV: {data_paths['labels']}")
        print(f"Metadata JSON: {data_paths['metadata']}")
        for name, path in checkpoint_paths.items():
            print(f"Loaded {name}: {path}")

    tool_metrics = MultiLabelMetricsAccumulator(len(tool_names))
    action_metrics = MultiLabelMetricsAccumulator(len(action_names))
    target_metrics = MultiLabelMetricsAccumulator(len(target_names))
    triplet_metrics = MultiLabelMetricsAccumulator(num_triplet_classes)
    per_sample_pred = {"tool": [], "action": [], "target": [], "triplet": []}
    per_sample_target = {"tool": [], "action": [], "target": [], "triplet": []}
    per_sample_global_indices: list[int] = []
    joint_exact_correct = 0
    unsupported_predicted_triplets = 0

    max_samples = len(dataset)
    if args.eval_fraction is not None:
        max_samples = min(max_samples, max(1, int(math.ceil(len(dataset) * args.eval_fraction))))
    if args.max_samples is not None:
        max_samples = min(max_samples, args.max_samples)
    shard_indices = resolve_shard_sample_indices(max_samples, args.num_shards, args.shard_index)

    progress_iterable = shard_indices
    progress_bar = None
    if not args.no_progress and tqdm is not None:
        progress_bar = tqdm(
            shard_indices,
            total=len(shard_indices),
            desc=f"{args.tool} val",
            unit="clip",
            dynamic_ncols=True,
        )
        progress_iterable = progress_bar

    for shard_position, sample_idx in enumerate(progress_iterable, start=1):
        buffer, _, clip_indices, sample_path = dataset[sample_idx]
        sample_path = str(sample_path)
        if sample_path not in label_lookup:
            raise KeyError(f"Missing labels for validation clip: {sample_path}")
        labels = label_lookup[sample_path]
        clips, prepared_indices = prepare_dataset_sample_inputs(buffer, clip_indices, device)

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                encoded_clip = encoder(clips, prepared_indices)[0]
                if args.stage0_only:
                    stage0_probs = compute_stage0_probabilities(
                        encoded_clip=encoded_clip,
                        heads=heads,
                        head_temperatures=head_temperatures,
                        device=device,
                    )
                    pred_tool = threshold_probabilities(stage0_probs["tool"], tool_thresholds)
                    pred_action = threshold_probabilities(stage0_probs["action"], action_thresholds)
                    pred_target = threshold_probabilities(stage0_probs["target"], target_thresholds)
                    pred_triplet, unsupported_count = build_triplet_mask_from_component_masks(
                        predicted_tool_mask=pred_tool,
                        predicted_action_mask=pred_action,
                        predicted_target_mask=pred_target,
                        triplet_to_id=triplet_to_id,
                        num_triplet_classes=num_triplet_classes,
                    )
                else:
                    refinement = compute_soft_refined_probabilities(
                        encoded_clip=encoded_clip,
                        tool0_head=heads["tool0"],
                        verb0_head=heads["action0"],
                        target0_head=heads["target0"],
                        tool_refine_head=heads["tool_refine"],
                        verb_refine_head=heads["action_refine"],
                        target_refine_head=heads["target_refine"],
                        device=device,
                        refinement_steps=args.refinement_steps,
                        blend_alpha=args.blend_alpha,
                        blend_alpha_step2=args.blend_alpha_step2,
                        route_topk_tool=args.route_topk_tool,
                        route_topk_verb=args.route_topk_action,
                        route_topk_target=args.route_topk_target,
                        routing_probability_source=args.routing_probability_source,
                        route_threshold_tool=route_tool_thresholds,
                        route_threshold_verb=route_action_thresholds,
                        route_threshold_target=route_target_thresholds,
                        head_temperatures=head_temperatures,
                        support_pair_constraints=support_pair_constraints,
                        support_constrained_routing_mode=support_mode,
                        support_constrained_routing_unsupported_scale=(
                            args.support_constrained_routing_unsupported_scale
                        ),
                    )
                    pred_tool = threshold_probabilities(refinement["final"]["tool"], tool_thresholds)
                    pred_action = threshold_probabilities(refinement["final"]["verb"], action_thresholds)
                    pred_target = threshold_probabilities(refinement["final"]["target"], target_thresholds)
                    pred_triplet, unsupported_count = build_predicted_triplet_mask(
                        refinement=refinement,
                        predicted_tool_mask=pred_tool,
                        predicted_action_mask=pred_action,
                        target_thresholds=target_thresholds,
                        triplet_to_id=triplet_to_id,
                        num_triplet_classes=num_triplet_classes,
                    )

        target_tool = labels["tool"] > 0.5
        target_action = labels["action"] > 0.5
        target_target = labels["target"] > 0.5
        target_triplet = labels["triplet"] > 0.5

        tool_metrics.update(pred_tool, target_tool)
        action_metrics.update(pred_action, target_action)
        target_metrics.update(pred_target, target_target)
        triplet_metrics.update(pred_triplet, target_triplet)
        unsupported_predicted_triplets += int(unsupported_count)
        per_sample_global_indices.append(int(sample_idx))

        for name, pred_mask, target_mask in (
            ("tool", pred_tool, target_tool),
            ("action", pred_action, target_action),
            ("target", pred_target, target_target),
            ("triplet", pred_triplet, target_triplet),
        ):
            per_sample_pred[name].append(pred_mask.detach().cpu().bool().reshape(-1))
            per_sample_target[name].append(target_mask.detach().cpu().bool().reshape(-1))

        if torch.equal(pred_tool, target_tool) and torch.equal(pred_action, target_action) and torch.equal(pred_target, target_target):
            joint_exact_correct += 1

        if args.verbose and ((shard_position % args.progress_every == 0) or shard_position == len(shard_indices)):
            message = (
                f"Evaluated {shard_position}/{len(shard_indices)} shard clips "
                f"(global sample_idx={sample_idx})"
            )
            if progress_bar is not None:
                progress_bar.write(message)
            else:
                print(message)

    if progress_bar is not None:
        progress_bar.close()

    if not shard_indices:
        raise RuntimeError("No validation samples were evaluated")

    tool_summary = tool_metrics.summary()
    action_summary = action_metrics.summary()
    target_summary = target_metrics.summary()
    triplet_summary = triplet_metrics.summary()
    joint_exact_acc = 100.0 * float(joint_exact_correct) / float(len(shard_indices))

    bootstrap_block: dict[str, Any] = {}
    if args.bootstrap_iterations > 0:
        for offset, name in enumerate(("tool", "action", "target", "triplet")):
            bootstrap_block[name] = compute_bootstrap_ci(
                per_sample_pred[name],
                per_sample_target[name],
                args.bootstrap_iterations,
                args.bootstrap_seed + offset,
                args.bootstrap_confidence,
            )

    per_sample_block = None
    if not args.no_store_per_sample_masks:
        per_sample_block = {
            "schema": "packed_hex_v1",
            "global_indices": per_sample_global_indices,
            "num_classes": {
                "tool": len(tool_names),
                "action": len(action_names),
                "target": len(target_names),
                "triplet": num_triplet_classes,
            },
            "predicted": {
                name: [pack_bool_mask(mask) for mask in masks]
                for name, masks in per_sample_pred.items()
            },
            "target": {
                name: [pack_bool_mask(mask) for mask in masks]
                for name, masks in per_sample_target.items()
            },
        }

    summary = {
        "tool_slug": args.tool,
        "num_samples": int(len(shard_indices)),
        "joint_exact_tool_action_target_acc": joint_exact_acc,
        "unsupported_predicted_triplets": int(unsupported_predicted_triplets),
        "tool": tool_summary,
        "action": action_summary,
        "target": target_summary,
        "triplet": triplet_summary,
        "class_names": {
            "tool": tool_names,
            "action": action_names,
            "target": target_names,
        },
        "config_paths": {name: str(path.resolve()) for name, path in config_paths.items()},
        "checkpoint_paths": {name: str(path.resolve()) for name, path in checkpoint_paths.items()},
        "eval_files": {name: str(path.resolve()) for name, path in data_paths.items()},
        "settings": {
            "device": str(device),
            "prediction_mode": "stage0_only" if args.stage0_only else "soft_refinement",
            "stage0_only": bool(args.stage0_only),
            "eval_fraction": None if args.eval_fraction is None else float(args.eval_fraction),
            "max_samples_arg": args.max_samples,
            "refinement_steps": int(args.refinement_steps),
            "blend_alpha": float(args.blend_alpha),
            "blend_alpha_step2": None if args.blend_alpha_step2 is None else float(args.blend_alpha_step2),
            "route_topk_tool": args.route_topk_tool,
            "route_topk_action": args.route_topk_action,
            "route_topk_target": args.route_topk_target,
            "routing_probability_source": args.routing_probability_source,
            "support_constrained_routing_mode": support_mode,
            "tool_thresholds": [float(value) for value in tool_thresholds.tolist()],
            "action_thresholds": [float(value) for value in action_thresholds.tolist()],
            "target_thresholds": [float(value) for value in target_thresholds.tolist()],
            "bootstrap_iterations": int(args.bootstrap_iterations),
            "bootstrap_seed": int(args.bootstrap_seed),
            "bootstrap_confidence": float(args.bootstrap_confidence),
            "calibration_json": (
                str(args.calibration_json.resolve()) if args.calibration_json is not None else None
            ),
            "head_temperatures": head_temperatures.to_serializable(),
            "head_temperatures_is_identity": head_temperatures.is_identity(),
        },
        "metadata_tool": metadata.get("tool"),
        "bootstrap": bootstrap_block,
        "raw_counts": {
            "joint_exact_correct": int(joint_exact_correct),
            "unsupported_predicted_triplets": int(unsupported_predicted_triplets),
            "tool": tool_metrics.raw_counts(),
            "action": action_metrics.raw_counts(),
            "target": target_metrics.raw_counts(),
            "triplet": triplet_metrics.raw_counts(),
        },
        "sharding": {
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "candidate_clips_before_sharding": int(max_samples),
            "evaluated_global_indices": shard_indices,
        },
        "per_sample_masks": per_sample_block,
    }

    mode_label = "stage0-only" if args.stage0_only else "soft-refinement"
    print(f"Evaluated {len(shard_indices)} {args.tool} validation clips with {mode_label} (shard {args.shard_index + 1}/{args.num_shards})")
    print(f"Joint exact tool/action/target accuracy: {joint_exact_acc:.4f}%")
    print(f"Unsupported predicted local triplets: {unsupported_predicted_triplets}")
    print_metrics_block("tool", tool_summary)
    print_metrics_block("action", action_summary)
    print_metrics_block("target", target_summary)
    print_metrics_block("triplet", triplet_summary)
    if bootstrap_block:
        for name in ("tool", "action", "target", "triplet"):
            block = bootstrap_block.get(name)
            if not block:
                continue
            macro = block["macro_f1"]
            micro = block["micro_f1"]
            print(
                f"{name} bootstrap (n={block['num_samples']}, iters={block['iterations']}, "
                f"conf={block['confidence']}): MacroF1 mean={macro['mean']:.4f}% "
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
