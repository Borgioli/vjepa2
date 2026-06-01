#!/path/to/data_root/vjepa2_1/.venv/bin/python
"""Fit per-head temperatures for one single-tool soft-refinement model.

The output JSON is consumed by ``single_tool_soft_refinement_eval.py`` through
``--calibration-json``. The generic soft-refinement runtime still names the
middle axis ``verb``, so the written temperature keys are ``verb0`` and
``verb_refine`` even though the single-tool configs call that axis ``action``.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path
from typing import Any

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments.
    tqdm = None

from single_tool_soft_refinement_eval import (
    SUPPORTED_TOOLS,
    default_config_paths,
    default_data_paths,
    load_eval_lookup,
    load_single_tool_metadata,
)
from triplet_conditioned_inference import (
    build_classifier_from_config,
    init_encoder_module,
    load_classifier_weights,
    load_yaml_config,
    maybe_suppress_stdout,
    override_backbone_checkpoint,
    resolve_device,
)
from triplet_soft_refinement_eval import (
    build_eval_dataset,
    prepare_dataset_sample_inputs,
    resolve_shard_sample_indices,
)
from triplet_soft_refinement_inference import (
    SOFT_HEAD_NAMES,
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
            "Fit scalar logit temperatures for hook/clipper/grasper single-tool "
            "soft-refinement heads."
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

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--temperature-bracket",
        type=float,
        nargs=2,
        default=(0.1, 10.0),
        help="Search bracket [lo, hi] for each scalar temperature.",
    )
    parser.add_argument("--golden-section-iterations", type=int, default=80)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, Path]]:
    config_defaults = default_config_paths(args.tool)
    data_defaults = default_data_paths(args.tool)
    config_paths = {
        "tool0": args.tool0_config or config_defaults["tool0"],
        "verb0": args.action0_config or config_defaults["action0"],
        "target0": args.target0_config or config_defaults["target0"],
        "tool_refine": args.tool_refine_config or config_defaults["tool_refine"],
        "verb_refine": args.action_refine_config or config_defaults["action_refine"],
        "target_refine": args.target_refine_config or config_defaults["target_refine"],
    }
    data_paths = {
        "runtime": args.eval_runtime_csv or data_defaults["runtime"],
        "labels": args.eval_label_csv or data_defaults["labels"],
        "metadata": args.metadata_json or data_defaults["metadata"],
    }
    return config_paths, data_paths


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
        objective,
        lo=lo,
        hi=hi,
        iterations=args.golden_section_iterations,
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


def stack_or_empty(rows: list[torch.Tensor]) -> torch.Tensor:
    if not rows:
        return torch.zeros(0, 0)
    return torch.cat(rows, dim=0)


def metadata_id_to_components(metadata: dict[str, Any]) -> dict[int, tuple[int, int, int]]:
    raw = metadata.get("local_label_id_to_component_ids")
    if not isinstance(raw, dict):
        raise ValueError("Metadata is missing local_label_id_to_component_ids")
    return {
        int(label_id): (
            int(components["tool"]),
            int(components["action"]),
            int(components["target"]),
        )
        for label_id, components in raw.items()
    }


def add_conditioned_rows(
    encoded_clip: Any,
    heads: dict[str, torch.nn.Module],
    components: list[tuple[int, int, int]],
    device: torch.device,
    rows: dict[str, list[torch.Tensor]],
    targets: dict[str, list[torch.Tensor]],
) -> None:
    if not components:
        return

    pair_for_tool: list[list[int]] = []
    pair_for_action: list[list[int]] = []
    pair_for_target: list[list[int]] = []
    tool_targets_per_pair: list[torch.Tensor] = []
    action_targets_per_pair: list[torch.Tensor] = []
    target_targets_per_pair: list[torch.Tensor] = []
    seen_tool_pairs: dict[tuple[int, int], int] = {}
    seen_action_pairs: dict[tuple[int, int], int] = {}
    seen_target_pairs: dict[tuple[int, int], int] = {}

    for tool_id, action_id, target_id in components:
        key_tool = (int(action_id), int(target_id))
        if key_tool not in seen_tool_pairs:
            seen_tool_pairs[key_tool] = len(pair_for_tool)
            pair_for_tool.append([int(action_id), int(target_id)])
            tool_targets_per_pair.append(
                torch.zeros(int(heads["tool_refine"].num_classes_per_task[0])),
            )
        tool_targets_per_pair[seen_tool_pairs[key_tool]][int(tool_id)] = 1.0

        key_action = (int(tool_id), int(target_id))
        if key_action not in seen_action_pairs:
            seen_action_pairs[key_action] = len(pair_for_action)
            pair_for_action.append([int(tool_id), int(target_id)])
            action_targets_per_pair.append(
                torch.zeros(int(heads["verb_refine"].num_classes_per_task[0])),
            )
        action_targets_per_pair[seen_action_pairs[key_action]][int(action_id)] = 1.0

        key_target = (int(tool_id), int(action_id))
        if key_target not in seen_target_pairs:
            seen_target_pairs[key_target] = len(pair_for_target)
            pair_for_target.append([int(tool_id), int(action_id)])
            target_targets_per_pair.append(
                torch.zeros(int(heads["target_refine"].num_classes_per_task[0])),
            )
        target_targets_per_pair[seen_target_pairs[key_target]][int(target_id)] = 1.0

    if pair_for_tool:
        cond = torch.tensor(pair_for_tool, dtype=torch.long, device=device).unsqueeze(0)
        logits = heads["tool_refine"](encoded_clip, cond)["task_0"][0].float().cpu()
        rows["tool_refine"].append(logits)
        targets["tool_refine"].append(torch.stack(tool_targets_per_pair, dim=0))
    if pair_for_action:
        cond = torch.tensor(pair_for_action, dtype=torch.long, device=device).unsqueeze(0)
        logits = heads["verb_refine"](encoded_clip, cond)["task_0"][0].float().cpu()
        rows["verb_refine"].append(logits)
        targets["verb_refine"].append(torch.stack(action_targets_per_pair, dim=0))
    if pair_for_target:
        cond = torch.tensor(pair_for_target, dtype=torch.long, device=device).unsqueeze(0)
        logits = heads["target_refine"](encoded_clip, cond)["task_0"][0].float().cpu()
        rows["target_refine"].append(logits)
        targets["target_refine"].append(torch.stack(target_targets_per_pair, dim=0))


def main() -> int:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {args.num_shards}")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError(
            f"--shard-index must satisfy 0 <= shard-index < num-shards, "
            f"got shard_index={args.shard_index}, num_shards={args.num_shards}"
        )
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError(f"--max-samples must be >= 1, got {args.max_samples}")

    config_paths, data_paths = resolve_paths(args)
    for path in (*config_paths.values(), *data_paths.values()):
        if not path.exists():
            raise FileNotFoundError(path)

    device = resolve_device(args.device)
    cfgs = {name: load_yaml_config(path) for name, path in config_paths.items()}
    override_backbone_checkpoint(
        list(cfgs.values()),
        args.backbone_checkpoint,
        checkpoint_key=args.backbone_checkpoint_key,
    )
    validate_soft_refinement_configs(
        tool0_cfg=cfgs["tool0"],
        verb0_cfg=cfgs["verb0"],
        target0_cfg=cfgs["target0"],
        tool_refine_cfg=cfgs["tool_refine"],
        verb_refine_cfg=cfgs["verb_refine"],
        target_refine_cfg=cfgs["target_refine"],
    )

    _, num_triplet_classes, metadata = load_single_tool_metadata(data_paths["metadata"])
    id_to_components = metadata_id_to_components(metadata)
    label_lookup = load_eval_lookup(data_paths["labels"], num_triplet_classes)
    dataset = build_eval_dataset(cfgs["tool0"], data_paths["runtime"])

    checkpoint_paths = {
        "tool0": resolve_soft_refinement_checkpoint_path(
            cfgs["tool0"], config_paths["tool0"], args.tool0_checkpoint,
            fallback_name="tool", checkpoint_policy=args.head_checkpoint_policy,
            strict_best=args.strict_best_head_checkpoints,
        ),
        "verb0": resolve_soft_refinement_checkpoint_path(
            cfgs["verb0"], config_paths["verb0"], args.action0_checkpoint,
            fallback_name="action", checkpoint_policy=args.head_checkpoint_policy,
            strict_best=args.strict_best_head_checkpoints,
        ),
        "target0": resolve_soft_refinement_checkpoint_path(
            cfgs["target0"], config_paths["target0"], args.target0_checkpoint,
            fallback_name="target", checkpoint_policy=args.head_checkpoint_policy,
            strict_best=args.strict_best_head_checkpoints,
        ),
        "tool_refine": resolve_soft_refinement_checkpoint_path(
            cfgs["tool_refine"], config_paths["tool_refine"], args.tool_refine_checkpoint,
            fallback_name="tool", checkpoint_policy=args.head_checkpoint_policy,
            strict_best=args.strict_best_head_checkpoints,
        ),
        "verb_refine": resolve_soft_refinement_checkpoint_path(
            cfgs["verb_refine"], config_paths["verb_refine"], args.action_refine_checkpoint,
            fallback_name="action", checkpoint_policy=args.head_checkpoint_policy,
            strict_best=args.strict_best_head_checkpoints,
        ),
        "target_refine": resolve_soft_refinement_checkpoint_path(
            cfgs["target_refine"], config_paths["target_refine"], args.target_refine_checkpoint,
            fallback_name="target", checkpoint_policy=args.head_checkpoint_policy,
            strict_best=args.strict_best_head_checkpoints,
        ),
    }

    frames_per_clip = int(cfgs["tool0"]["experiment"]["data"].get("frames_per_clip", 16))
    resolution = int(cfgs["tool0"]["experiment"]["data"].get("resolution", 224))
    with maybe_suppress_stdout(args.verbose):
        encoder = init_encoder_module(
            module_name=cfgs["tool0"]["model_kwargs"]["module_name"],
            frames_per_clip=frames_per_clip,
            resolution=resolution,
            checkpoint=cfgs["tool0"]["model_kwargs"]["checkpoint"],
            model_kwargs=cfgs["tool0"]["model_kwargs"]["pretrain_kwargs"],
            wrapper_kwargs=cfgs["tool0"]["model_kwargs"]["wrapper_kwargs"],
            device=device,
        )
    encoder.eval()

    heads = {
        name: build_classifier_from_config(config, encoder.embed_dim).to(device).eval()
        for name, config in cfgs.items()
    }
    for name, head in heads.items():
        load_classifier_weights(head, checkpoint_paths[name])

    if args.verbose:
        print(f"Tool: {args.tool}")
        print(f"Using device: {device}")
        print(f"Runtime CSV: {data_paths['runtime']}")
        print(f"Label CSV: {data_paths['labels']}")
        for name, path in checkpoint_paths.items():
            print(f"Loaded {name}: {path}")

    logits_rows: dict[str, list[torch.Tensor]] = {name: [] for name in SOFT_HEAD_NAMES}
    target_rows: dict[str, list[torch.Tensor]] = {name: [] for name in SOFT_HEAD_NAMES}

    max_samples = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)
    shard_indices = resolve_shard_sample_indices(max_samples, args.num_shards, args.shard_index)
    if not shard_indices:
        raise RuntimeError("No validation samples were selected for temperature fitting")

    progress_iterable = shard_indices
    progress_bar = None
    if not args.no_progress and tqdm is not None:
        progress_bar = tqdm(
            shard_indices,
            total=len(shard_indices),
            desc=f"{args.tool} temp fit",
            unit="clip",
            dynamic_ncols=True,
        )
        progress_iterable = progress_bar

    for position, sample_idx in enumerate(progress_iterable, start=1):
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
                encoded_clip = encoder(clips, prepared_indices)[0]
                tool_logits = heads["tool0"](encoded_clip)["task_0"][0].float().cpu()
                action_logits = heads["verb0"](encoded_clip)["task_0"][0].float().cpu()
                target_logits = heads["target0"](encoded_clip)["task_0"][0].float().cpu()

                triplet_ids = (
                    sample_labels["triplet"].bool().nonzero(as_tuple=False).reshape(-1).tolist()
                )
                components = [id_to_components[int(label_id)] for label_id in triplet_ids]
                add_conditioned_rows(
                    encoded_clip=encoded_clip,
                    heads=heads,
                    components=components,
                    device=device,
                    rows=logits_rows,
                    targets=target_rows,
                )

        logits_rows["tool0"].append(tool_logits.reshape(1, -1))
        target_rows["tool0"].append(sample_labels["tool"].reshape(1, -1))
        logits_rows["verb0"].append(action_logits.reshape(1, -1))
        target_rows["verb0"].append(sample_labels["action"].reshape(1, -1))
        logits_rows["target0"].append(target_logits.reshape(1, -1))
        target_rows["target0"].append(sample_labels["target"].reshape(1, -1))

        if args.verbose and (position % args.progress_every == 0 or position == len(shard_indices)):
            message = f"Collected {position}/{len(shard_indices)} clips"
            if progress_bar is not None:
                progress_bar.write(message)
            else:
                print(message)

    if progress_bar is not None:
        progress_bar.close()

    head_inputs = {
        name: (stack_or_empty(logits_rows[name]), stack_or_empty(target_rows[name]))
        for name in SOFT_HEAD_NAMES
    }
    fit_results = {
        name: fit_temperature_for_head(name, logits, targets, args)
        for name, (logits, targets) in head_inputs.items()
    }

    output_path = (
        args.output_json
        or Path(f"/path/to/data_root/vjepa2_1/single_tool_soft_refinement_temperatures_{args.tool}.json")
    ).resolve()
    payload = {
        "schema_version": 1,
        "tool_slug": args.tool,
        "axis_aliases": {"verb": "action"},
        "temperatures": {name: result["temperature"] for name, result in fit_results.items()},
        "fit_details": fit_results,
        "fit_settings": {
            "max_samples": int(max_samples),
            "num_samples": int(len(shard_indices)),
            "num_shards": int(args.num_shards),
            "shard_index": int(args.shard_index),
            "temperature_bracket": [float(args.temperature_bracket[0]), float(args.temperature_bracket[1])],
            "golden_section_iterations": int(args.golden_section_iterations),
        },
        "config_paths": {name: str(path.resolve()) for name, path in config_paths.items()},
        "checkpoint_paths": {name: str(path.resolve()) for name, path in checkpoint_paths.items()},
        "eval_files": {name: str(path.resolve()) for name, path in data_paths.items()},
    }
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
