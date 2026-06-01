#!/path/to/data_root/vjepa2_1/.venv/bin/python
"""Fit per-head temperature scalars for the soft triplet refinement pipeline.

This script collects raw logits + ground-truth multi-label targets for each of
the six heads on the Tool5 clip-level validation split, then fits a single
positive scalar temperature per head by minimizing binary cross-entropy on
those logits. The output is a JSON file consumable by
``triplet_soft_refinement_inference.HeadTemperatures.from_json`` and the
``--calibration-json`` flag of both the inference and evaluation scripts.

Stage-0 heads (tool0 / verb0 / target0) are fit against the corresponding
multi-label ground truth. Conditioned heads (tool|v,t / verb|t,r / target|t,v)
are fit only on (clip, condition-pair) entries whose condition pair appears in
a ground-truth triplet for that clip; the supervision target is the remaining
axis of those triplets, in the head's own (study) space.

Run e.g.

    /path/to/data_root/vjepa2_1/.venv/bin/python \
      /path/to/data_root/vjepa2_1/triplet_soft_refinement_fit_temperatures.py \
      --device cuda \
      --output-json /path/to/data_root/vjepa2_1/soft_refinement_temperatures.json
"""
from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import torch

from triplet_conditioned_inference import (
    build_classifier_from_config,
    build_transform,
    init_encoder_module,
    load_classifier_weights,
    load_yaml_config,
    maybe_suppress_stdout,
    override_backbone_checkpoint,
    resolve_device,
)
from triplet_soft_refinement_eval import (
    DEFAULT_TRIPLET_METADATA,
    build_eval_dataset,
    load_eval_label_lookup,
    prepare_dataset_sample_inputs,
    resolve_shard_sample_indices,
)
from triplet_soft_refinement_target_relabel import (
    SUPPORTED_RELABEL_MODES,
    build_relabel_plan,
    relabel_label_lookup,
)
from triplet_soft_refinement_inference import (
    DEFAULT_TARGET0_CONFIG,
    DEFAULT_TARGET_REFINE_CONFIG,
    DEFAULT_TOOL0_CONFIG,
    DEFAULT_TOOL_REFINE_CONFIG,
    DEFAULT_VERB0_CONFIG,
    DEFAULT_VERB_REFINE_CONFIG,
    GROUPED_TO_STUDY_VERB_INDEX,
    SOFT_HEAD_NAMES,
    load_triplet_support_index,
    resolve_soft_refinement_checkpoint_path,
    validate_soft_refinement_configs,
)

warnings.filterwarnings(
    "ignore",
    message="Importing from timm.models.layers is deprecated.*",
    category=FutureWarning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit per-head temperature scalars on the Tool5 validation split for the "
            "soft triplet refinement pipeline."
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
    )
    parser.add_argument("--strict-best-head-checkpoints", action="store_true")
    parser.add_argument("--backbone-checkpoint", type=Path, default=None)
    parser.add_argument("--backbone-checkpoint-key", type=str, default=None)

    parser.add_argument("--eval-runtime-csv", type=Path, default=None)
    parser.add_argument("--eval-label-csv", type=Path, default=None)
    parser.add_argument("--triplet-metadata", type=Path, default=DEFAULT_TRIPLET_METADATA)
    parser.add_argument("--device", type=str, default="auto", choices=("auto", "cuda", "cpu"))

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on validation clips used for temperature fitting.",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Optional shard count for collecting logits in parallel; merge JSONs is not supported.",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--temperature-bracket",
        type=float,
        nargs=2,
        default=(0.1, 10.0),
        help="Search bracket [lo, hi] for each scalar temperature in linear scale.",
    )
    parser.add_argument(
        "--golden-section-iterations",
        type=int,
        default=80,
        help="Iterations of golden-section search per head.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Where to write the fitted temperatures JSON.",
    )
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument(
        "--target-relabel-mode",
        type=str,
        default=None,
        choices=(None, *SUPPORTED_RELABEL_MODES),
        help=(
            "Optional eval-only relabeling of the target axis applied before "
            "logit collection. Mirrors the same flag in "
            "triplet_soft_refinement_eval.py so calibration is internally "
            "consistent with the relabeled evaluation."
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def bce_with_logits_at_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> float:
    if temperature <= 0:
        return float("inf")
    scaled = logits.to(dtype=torch.float64) / float(temperature)
    return float(
        torch.nn.functional.binary_cross_entropy_with_logits(
            scaled,
            targets.to(dtype=torch.float64),
            reduction="mean",
        ).item()
    )


def golden_section_minimize(
    objective,
    lo: float,
    hi: float,
    iterations: int,
) -> tuple[float, float]:
    if hi <= lo:
        raise ValueError(f"Invalid bracket [{lo}, {hi}]")
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lo), float(hi)
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc = objective(c)
    fd = objective(d)
    for _ in range(int(iterations)):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = objective(d)
    if fc < fd:
        return c, fc
    return d, fd


def study_verb_id_for_grouped_id(grouped_verb_id: int) -> int:
    return int(GROUPED_TO_STUDY_VERB_INDEX[int(grouped_verb_id)])


def fit_temperature_for_head(
    name: str,
    logits: torch.Tensor,
    targets: torch.Tensor,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if logits.numel() == 0:
        if args.verbose:
            print(f"[{name}] no supervision rows collected; defaulting temperature to 1.0")
        return {"temperature": 1.0, "nll_at_T1": None, "nll_at_T*": None, "num_rows": 0}

    nll_at_one = bce_with_logits_at_temperature(logits, targets, 1.0)
    objective = lambda temp: bce_with_logits_at_temperature(logits, targets, temp)
    lo, hi = float(args.temperature_bracket[0]), float(args.temperature_bracket[1])
    best_temp, best_nll = golden_section_minimize(
        objective, lo=lo, hi=hi, iterations=args.golden_section_iterations,
    )
    if args.verbose:
        print(
            f"[{name}] rows={int(logits.shape[0])} "
            f"NLL(T=1)={nll_at_one:.6f} -> NLL(T={best_temp:.4f})={best_nll:.6f}"
        )
    return {
        "temperature": float(best_temp),
        "nll_at_T1": float(nll_at_one),
        "nll_at_T*": float(best_nll),
        "num_rows": int(logits.shape[0]),
        "search_bracket": [lo, hi],
        "iterations": int(args.golden_section_iterations),
    }


def main() -> int:
    args = parse_args()

    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            f"--shard-index must satisfy 0 <= shard-index < num-shards, "
            f"got shard_index={args.shard_index}, num_shards={args.num_shards}"
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
        tool0_cfg, verb0_cfg, target0_cfg,
        tool_refine_cfg, verb_refine_cfg, target_refine_cfg,
    ]
    override_backbone_checkpoint(
        all_cfgs,
        args.backbone_checkpoint,
        checkpoint_key=args.backbone_checkpoint_key,
    )
    validate_soft_refinement_configs(
        tool0_cfg=tool0_cfg, verb0_cfg=verb0_cfg, target0_cfg=target0_cfg,
        tool_refine_cfg=tool_refine_cfg, verb_refine_cfg=verb_refine_cfg, target_refine_cfg=target_refine_cfg,
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
    label_lookup = load_eval_label_lookup(eval_label_csv)
    triplet_to_id, num_triplet_classes, _ = load_triplet_support_index(triplet_metadata_path)

    with triplet_metadata_path.open("r", encoding="utf-8") as handle:
        triplet_metadata = json.load(handle)
    triplet_id_to_components: dict[int, tuple[int, int, int]] = {}
    for entry in triplet_metadata.get("id_to_triplet", []):
        triplet_id_to_components[int(entry["triplet_id"])] = (
            int(entry["tool_id"]),
            int(entry["verb_id"]),
            int(entry["target_id"]),
        )

    target_names_for_relabel: list[str] = []
    target_class_names_per_task = (
        target0_cfg.get("experiment", {}).get("data", {}).get("class_names_per_task")
    )
    if target_class_names_per_task and target_class_names_per_task[0]:
        target_names_for_relabel = [str(name) for name in target_class_names_per_task[0]]
    relabel_plan = build_relabel_plan(
        mode=args.target_relabel_mode,
        target_names=target_names_for_relabel,
        triplet_to_id=triplet_to_id,
        num_triplet_classes=num_triplet_classes,
    )
    label_lookup_original = label_lookup
    if relabel_plan is not None:
        label_lookup = relabel_label_lookup(label_lookup, relabel_plan)
        if args.verbose:
            print(
                f"Target relabel mode '{relabel_plan.mode}' applied: "
                f"dropped target ids={list(relabel_plan.dropped_target_ids)}, "
                f"target merges={relabel_plan.target_merges}"
            )

    dataset = build_eval_dataset(tool0_cfg, eval_runtime_csv)

    tool0_checkpoint = resolve_soft_refinement_checkpoint_path(
        tool0_cfg, args.tool0_config, args.tool0_checkpoint, fallback_name="tool",
        checkpoint_policy=args.head_checkpoint_policy, strict_best=args.strict_best_head_checkpoints,
    )
    verb0_checkpoint = resolve_soft_refinement_checkpoint_path(
        verb0_cfg, args.verb0_config, args.verb0_checkpoint, fallback_name="verb",
        checkpoint_policy=args.head_checkpoint_policy, strict_best=args.strict_best_head_checkpoints,
    )
    target0_checkpoint = resolve_soft_refinement_checkpoint_path(
        target0_cfg, args.target0_config, args.target0_checkpoint, fallback_name="target",
        checkpoint_policy=args.head_checkpoint_policy, strict_best=args.strict_best_head_checkpoints,
    )
    tool_refine_checkpoint = resolve_soft_refinement_checkpoint_path(
        tool_refine_cfg, args.tool_refine_config, args.tool_refine_checkpoint, fallback_name="tool",
        checkpoint_policy=args.head_checkpoint_policy, strict_best=args.strict_best_head_checkpoints,
    )
    verb_refine_checkpoint = resolve_soft_refinement_checkpoint_path(
        verb_refine_cfg, args.verb_refine_config, args.verb_refine_checkpoint, fallback_name="verb",
        checkpoint_policy=args.head_checkpoint_policy, strict_best=args.strict_best_head_checkpoints,
    )
    target_refine_checkpoint = resolve_soft_refinement_checkpoint_path(
        target_refine_cfg, args.target_refine_config, args.target_refine_checkpoint, fallback_name="target",
        checkpoint_policy=args.head_checkpoint_policy, strict_best=args.strict_best_head_checkpoints,
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

    verb_refine_classes = int(verb_refine_head.num_classes_per_task[0])
    use_verb_adapter = (
        int(verb0_head.num_classes_per_task[0]) == 7 and verb_refine_classes == 6
    )

    tool0_logits_rows: list[torch.Tensor] = []
    tool0_target_rows: list[torch.Tensor] = []
    verb0_logits_rows: list[torch.Tensor] = []
    verb0_target_rows: list[torch.Tensor] = []
    target0_logits_rows: list[torch.Tensor] = []
    target0_target_rows: list[torch.Tensor] = []

    tool_ref_logits_rows: list[torch.Tensor] = []
    tool_ref_target_rows: list[torch.Tensor] = []
    verb_ref_logits_rows: list[torch.Tensor] = []
    verb_ref_target_rows: list[torch.Tensor] = []
    target_ref_logits_rows: list[torch.Tensor] = []
    target_ref_target_rows: list[torch.Tensor] = []

    max_samples = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    shard_indices = resolve_shard_sample_indices(
        max_samples=max_samples, num_shards=args.num_shards, shard_index=args.shard_index,
    )

    if args.verbose:
        print(
            f"Collecting logits over {len(shard_indices)} of {max_samples} clips "
            f"(shard {args.shard_index + 1}/{args.num_shards})"
        )

    for position, sample_idx in enumerate(shard_indices, start=1):
        buffer, _, clip_indices, sample_path = dataset[sample_idx]
        sample_path = str(sample_path)
        if sample_path not in label_lookup:
            raise KeyError(f"Could not find labels for clip: {sample_path}")
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
                tool_logits = tool0_head(encoded_clip)["task_0"][0].float().cpu()
                verb_logits = verb0_head(encoded_clip)["task_0"][0].float().cpu()
                target_logits = target0_head(encoded_clip)["task_0"][0].float().cpu()

                original_sample_labels = label_lookup_original[sample_path]
                triplet_target = original_sample_labels["triplet"].bool()
                triplet_ids = triplet_target.nonzero(as_tuple=False).reshape(-1).tolist()
                triple_components_raw = [
                    triplet_id_to_components[int(tid)] for tid in triplet_ids
                ]
                if relabel_plan is not None:
                    transformed: list[tuple[int, int, int]] = []
                    for tool_id, verb_id, target_id in triple_components_raw:
                        if int(target_id) in set(relabel_plan.dropped_target_ids):
                            continue
                        new_target_id = int(
                            relabel_plan.target_merges.get(int(target_id), int(target_id))
                        )
                        transformed.append((int(tool_id), int(verb_id), int(new_target_id)))
                    triple_components = transformed
                else:
                    triple_components = triple_components_raw
                if triple_components:
                    pair_for_tool = []
                    pair_for_verb = []
                    pair_for_target = []
                    tool_targets_per_pair = []
                    verb_targets_per_pair = []
                    target_targets_per_pair = []
                    seen_tool_pairs: dict[tuple[int, int], int] = {}
                    seen_verb_pairs: dict[tuple[int, int], int] = {}
                    seen_target_pairs: dict[tuple[int, int], int] = {}
                    for tool_id, verb_id, target_id in triple_components:
                        study_verb = study_verb_id_for_grouped_id(verb_id) if use_verb_adapter else verb_id

                        # tool | (study_verb, target)
                        key_t = (int(study_verb), int(target_id))
                        if key_t not in seen_tool_pairs:
                            seen_tool_pairs[key_t] = len(pair_for_tool)
                            pair_for_tool.append([int(study_verb), int(target_id)])
                            tool_targets_per_pair.append(
                                torch.zeros(int(tool_refine_head.num_classes_per_task[0])),
                            )
                        tool_targets_per_pair[seen_tool_pairs[key_t]][int(tool_id)] = 1.0

                        # verb | (tool, target), in study verb space
                        key_v = (int(tool_id), int(target_id))
                        if key_v not in seen_verb_pairs:
                            seen_verb_pairs[key_v] = len(pair_for_verb)
                            pair_for_verb.append([int(tool_id), int(target_id)])
                            verb_targets_per_pair.append(torch.zeros(verb_refine_classes))
                        verb_targets_per_pair[seen_verb_pairs[key_v]][int(study_verb)] = 1.0

                        # target | (tool, study_verb)
                        key_r = (int(tool_id), int(study_verb))
                        if key_r not in seen_target_pairs:
                            seen_target_pairs[key_r] = len(pair_for_target)
                            pair_for_target.append([int(tool_id), int(study_verb)])
                            target_targets_per_pair.append(
                                torch.zeros(int(target_refine_head.num_classes_per_task[0])),
                            )
                        target_targets_per_pair[seen_target_pairs[key_r]][int(target_id)] = 1.0

                    if pair_for_tool:
                        cond = torch.tensor(pair_for_tool, dtype=torch.long, device=device).unsqueeze(0)
                        tool_cond_logits = tool_refine_head(encoded_clip, cond)["task_0"][0].float().cpu()
                        tool_ref_logits_rows.append(tool_cond_logits)
                        tool_ref_target_rows.append(torch.stack(tool_targets_per_pair, dim=0))
                    if pair_for_verb:
                        cond = torch.tensor(pair_for_verb, dtype=torch.long, device=device).unsqueeze(0)
                        verb_cond_logits = verb_refine_head(encoded_clip, cond)["task_0"][0].float().cpu()
                        verb_ref_logits_rows.append(verb_cond_logits)
                        verb_ref_target_rows.append(torch.stack(verb_targets_per_pair, dim=0))
                    if pair_for_target:
                        cond = torch.tensor(pair_for_target, dtype=torch.long, device=device).unsqueeze(0)
                        target_cond_logits = target_refine_head(encoded_clip, cond)["task_0"][0].float().cpu()
                        target_ref_logits_rows.append(target_cond_logits)
                        target_ref_target_rows.append(torch.stack(target_targets_per_pair, dim=0))

        tool0_logits_rows.append(tool_logits.reshape(1, -1))
        tool0_target_rows.append(sample_labels["tool"].reshape(1, -1))
        verb0_logits_rows.append(verb_logits.reshape(1, -1))
        verb0_target_rows.append(sample_labels["verb"].reshape(1, -1))
        target0_logits_rows.append(target_logits.reshape(1, -1))
        target0_target_rows.append(sample_labels["target"].reshape(1, -1))

        if args.verbose and (position % args.progress_every == 0 or position == len(shard_indices)):
            print(f"Collected {position}/{len(shard_indices)} clips")

    def stack_or_empty(rows: list[torch.Tensor]) -> torch.Tensor:
        if not rows:
            return torch.zeros(0, 0)
        return torch.cat(rows, dim=0)

    head_inputs = {
        "tool0": (stack_or_empty(tool0_logits_rows), stack_or_empty(tool0_target_rows)),
        "verb0": (stack_or_empty(verb0_logits_rows), stack_or_empty(verb0_target_rows)),
        "target0": (stack_or_empty(target0_logits_rows), stack_or_empty(target0_target_rows)),
        "tool_refine": (stack_or_empty(tool_ref_logits_rows), stack_or_empty(tool_ref_target_rows)),
        "verb_refine": (stack_or_empty(verb_ref_logits_rows), stack_or_empty(verb_ref_target_rows)),
        "target_refine": (stack_or_empty(target_ref_logits_rows), stack_or_empty(target_ref_target_rows)),
    }

    if relabel_plan is not None:
        keep_target_columns = relabel_plan.target_active_mask.bool()
        for head_name in ("target0", "target_refine"):
            logits, targets = head_inputs[head_name]
            if logits.numel() == 0:
                continue
            if logits.shape[1] != keep_target_columns.numel():
                continue
            cols = keep_target_columns.nonzero(as_tuple=False).reshape(-1)
            head_inputs[head_name] = (logits.index_select(1, cols), targets.index_select(1, cols))

    fit_results: dict[str, dict[str, Any]] = {}
    for head_name in SOFT_HEAD_NAMES:
        logits, targets = head_inputs[head_name]
        fit_results[head_name] = fit_temperature_for_head(head_name, logits, targets, args)

    payload = {
        "schema_version": 1,
        "temperatures": {name: result["temperature"] for name, result in fit_results.items()},
        "fit_details": fit_results,
        "fit_settings": {
            "max_samples": int(max_samples),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "temperature_bracket": [float(args.temperature_bracket[0]), float(args.temperature_bracket[1])],
            "golden_section_iterations": int(args.golden_section_iterations),
            "use_verb_adapter": bool(use_verb_adapter),
            "target_relabel_mode": args.target_relabel_mode,
            "target_relabel_plan": (
                relabel_plan.to_serializable() if relabel_plan is not None else None
            ),
        },
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
    }

    output_path = args.output_json.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"Saved fitted temperatures to {output_path}")
    for name in SOFT_HEAD_NAMES:
        result = fit_results[name]
        print(
            f"  {name}: T={result['temperature']:.4f} "
            f"(rows={result['num_rows']}, "
            f"NLL T=1: {result['nll_at_T1']}, NLL T*: {result['nll_at_T*']})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
