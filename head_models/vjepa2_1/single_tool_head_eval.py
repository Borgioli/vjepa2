#!/path/to/data_root/vjepa2_1/.venv/bin/python
"""Evaluate one single-tool multilabel head with its configured thresholds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments.
    tqdm = None

from single_tool_threshold_sweep import (
    collate_samples,
    load_label_lookup,
    parse_condition_label,
    prepare_batched_sample_inputs,
)
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
from triplet_soft_refinement_eval import build_eval_dataset
from triplet_soft_refinement_inference import resolve_soft_refinement_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one single-tool multilabel head on validation using the "
            "multilabel_thresholds stored in its YAML config."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--head-checkpoint-policy",
        choices=("best", "latest"),
        default="best",
        help="'best' prefers best*.pt checkpoints and falls back to latest.pt.",
    )
    parser.add_argument("--strict-best-head-checkpoints", action="store_true")
    parser.add_argument("--backbone-checkpoint", "--encoder-checkpoint", dest="backbone_checkpoint", type=Path)
    parser.add_argument("--backbone-checkpoint-key", "--encoder-checkpoint-key", dest="backbone_checkpoint_key")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Validation decode workers. Defaults to the config num_workers value.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def resolve_threshold_vector(data_cfg: dict[str, Any], num_classes: int) -> torch.Tensor:
    value = data_cfg.get("multilabel_thresholds", data_cfg.get("multilabel_threshold", 0.5))
    if isinstance(value, (int, float)):
        return torch.full((num_classes,), float(value), dtype=torch.float32)
    thresholds = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if thresholds.numel() != num_classes:
        raise ValueError(f"Expected {num_classes} thresholds, got {thresholds.numel()}: {value!r}")
    return thresholds


def compute_prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def summarize_subset(
    pred: torch.Tensor,
    truth: torch.Tensor,
    class_indices: list[int],
) -> dict[str, Any]:
    if not class_indices:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "support": 0,
            "predicted_positive": 0,
        }
    selected_pred = pred[:, class_indices]
    selected_truth = truth[:, class_indices]
    tp = int((selected_pred & selected_truth).sum().item())
    fp = int((selected_pred & ~selected_truth).sum().item())
    fn = int((~selected_pred & selected_truth).sum().item())
    precision, recall, f1 = compute_prf(tp, fp, fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": int(selected_truth.sum().item()),
        "predicted_positive": int(selected_pred.sum().item()),
    }


def label_combo(mask: torch.Tensor, class_names: list[str]) -> str:
    active = [class_names[idx] for idx, value in enumerate(mask.tolist()) if bool(value)]
    return "+".join(active) if active else "<none>"


def top_label_combos(mask: torch.Tensor, class_names: list[str], top_k: int = 10) -> list[dict[str, Any]]:
    counts = Counter(label_combo(row, class_names) for row in mask)
    total = max(1, int(mask.shape[0]))
    return [
        {"labels": labels, "count": int(count), "rate": count / total}
        for labels, count in counts.most_common(top_k)
    ]


def print_summary(payload: dict[str, Any]) -> None:
    print(f"Config: {payload['config']}")
    print(f"Checkpoint: {payload['checkpoint']}")
    print(f"Evaluated clips: {payload['num_validation_clips']}")
    print(f"Scored rows: {payload['num_scored_rows']}")
    print(f"Thresholds: {payload['thresholds']}")
    print(f"Exact multilabel row match: {payload['exact_match'] * 100:.2f}%")

    micro = payload["micro_all"]
    print(
        "Micro all classes: "
        f"F1={micro['f1'] * 100:.2f}% "
        f"P={micro['precision'] * 100:.2f}% "
        f"R={micro['recall'] * 100:.2f}% "
        f"support={micro['support']} pred_pos={micro['predicted_positive']}"
    )
    non_null = payload["micro_non_null"]
    print(
        "Micro non-null classes: "
        f"F1={non_null['f1'] * 100:.2f}% "
        f"P={non_null['precision'] * 100:.2f}% "
        f"R={non_null['recall'] * 100:.2f}% "
        f"support={non_null['support']} pred_pos={non_null['predicted_positive']}"
    )

    print("\nPer-class")
    print("class_id  class_name          thresh  precision  recall   f1       support  pred_pos  tp    fp    fn    tn")
    for row in payload["per_class"]:
        print(
            f"{row['class_index']:>8}  {row['class_name'][:18]:<18}  "
            f"{row['threshold']:>6.3f}  {row['precision'] * 100:>8.2f}%  "
            f"{row['recall'] * 100:>6.2f}%  {row['f1'] * 100:>6.2f}%  "
            f"{row['support']:>7}  {row['predicted_positive']:>8}  "
            f"{row['tp']:>4}  {row['fp']:>5}  {row['fn']:>4}  {row['tn']:>5}"
        )

    print("\nTop predicted label combinations")
    for row in payload["predicted_label_combinations"]:
        print(f"{row['labels']:<40} {row['count']:>6} ({row['rate'] * 100:.2f}%)")

    print("\nTop target label combinations")
    for row in payload["target_label_combinations"]:
        print(f"{row['labels']:<40} {row['count']:>6} ({row['rate'] * 100:.2f}%)")


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")

    config_path = args.config.resolve()
    config = load_yaml_config(config_path)
    override_backbone_checkpoint([config], args.backbone_checkpoint, checkpoint_key=args.backbone_checkpoint_key)

    data_cfg = config["experiment"]["data"]
    classifier_cfg = config["experiment"]["classifier"]
    num_classes_per_task = data_cfg.get("num_classes_per_task")
    if not num_classes_per_task or len(num_classes_per_task) != 1:
        raise ValueError("Single-head eval expects a single-task head with num_classes_per_task length 1")
    num_classes = int(num_classes_per_task[0])
    task_name = str(data_cfg.get("task_names", ["task"])[0])
    class_names = class_names_from_config(config, [], task_name)
    if len(class_names) != num_classes:
        class_names = [f"class_{idx}" for idx in range(num_classes)]
    thresholds = resolve_threshold_vector(data_cfg, num_classes)

    conditioning_mode, label_lookup = load_label_lookup(config, num_classes)
    runtime_csv = Path(data_cfg["dataset_val"]).resolve()
    dataset = build_eval_dataset(config, runtime_csv)
    max_samples = len(dataset) if args.max_samples is None else min(len(dataset), int(args.max_samples))
    if max_samples < 1:
        raise ValueError("No validation samples selected")

    device = resolve_device(args.device)
    eval_dataset = Subset(dataset, range(max_samples))
    num_workers = int(data_cfg.get("num_workers", 0) if args.num_workers is None else args.num_workers)
    dataloader = DataLoader(
        eval_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_samples,
    )

    checkpoint_path = resolve_soft_refinement_checkpoint_path(
        config,
        config_path,
        args.checkpoint,
        fallback_name=task_name,
        checkpoint_policy=args.head_checkpoint_policy,
        strict_best=args.strict_best_head_checkpoints,
    )

    frames_per_clip = int(data_cfg.get("frames_per_clip", 16))
    resolution = int(data_cfg.get("resolution", 224))
    with maybe_suppress_stdout(args.verbose):
        encoder = init_encoder_module(
            module_name=config["model_kwargs"]["module_name"],
            frames_per_clip=frames_per_clip,
            resolution=resolution,
            checkpoint=config["model_kwargs"]["checkpoint"],
            model_kwargs=config["model_kwargs"]["pretrain_kwargs"],
            wrapper_kwargs=config["model_kwargs"]["wrapper_kwargs"],
            device=device,
        )
    encoder.eval()
    head = build_classifier_from_config(config, encoder.embed_dim).to(device).eval()
    load_classifier_weights(head, checkpoint_path)

    if args.verbose:
        print(f"Conditioning mode: {conditioning_mode}")
        print(f"Class names: {class_names}")
        print(f"Classifier: head_type={classifier_cfg.get('head_type')}")
        print(f"Batch size: {args.batch_size}")
        print(f"Decode workers: {num_workers}")

    probability_rows: list[torch.Tensor] = []
    target_rows: list[torch.Tensor] = []
    iterable = dataloader
    progress_bar = None
    if not args.no_progress and tqdm is not None:
        progress_bar = tqdm(iterable, total=max_samples, desc="single-head eval", unit="clip", dynamic_ncols=True)
        iterable = progress_bar

    for batch in iterable:
        batch_buffers = []
        batch_clip_indices = []
        batch_targets = []
        batch_condition_labels = []
        for buffer, runtime_label, clip_indices, sample_path in batch:
            sample_path = str(sample_path)
            batch_buffers.append(buffer)
            batch_clip_indices.append(clip_indices)

            if conditioning_mode == "unconditioned":
                targets = label_lookup[sample_path].reshape(1, -1)
            elif conditioning_mode == "multilabel_multi_condition_task":
                condition_labels, targets = label_lookup[sample_path]
                batch_condition_labels.append(condition_labels)
            elif conditioning_mode == "multilabel_conditioned_task":
                condition_fields = data_cfg.get(
                    "multilabel_condition_fields",
                    data_cfg.get("multilabel_condition_field", "conditioning_label"),
                )
                if isinstance(condition_fields, str):
                    condition_fields = [condition_fields]
                condition_tensor = parse_condition_label(runtime_label, len(condition_fields))
                lookup_key = (sample_path, *tuple(int(value.item()) for value in condition_tensor))
                targets = label_lookup[lookup_key].reshape(1, -1)
                batch_condition_labels.append(condition_tensor)
            else:
                raise ValueError(f"Unsupported conditioning mode: {conditioning_mode}")
            batch_targets.append(targets)

        clips, prepared_indices = prepare_batched_sample_inputs(batch_buffers, batch_clip_indices, device)
        condition_tensor = None
        if conditioning_mode != "unconditioned":
            condition_tensor = torch.stack(batch_condition_labels, dim=0).to(device=device, non_blocking=True)
        targets = torch.stack(batch_targets, dim=0)

        with torch.no_grad():
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                encoded_clip = encoder(clips, prepared_indices)[0]
                if condition_tensor is None:
                    logits = head(encoded_clip)["task_0"]
                else:
                    logits = head(encoded_clip, condition_tensor)["task_0"]
                probs = torch.sigmoid(logits.float()).detach().cpu()

        probability_rows.append(probs.reshape(-1, num_classes))
        target_rows.append(targets.reshape(-1, num_classes).float().cpu())

        if progress_bar is not None:
            progress_bar.update(len(batch) - 1)

    if progress_bar is not None:
        progress_bar.close()

    probabilities = torch.cat(probability_rows, dim=0)
    targets = torch.cat(target_rows, dim=0).bool()
    pred = probabilities >= thresholds.reshape(1, -1)
    exact_match = float(pred.eq(targets).all(dim=1).float().mean().item())

    per_class = []
    for class_index, class_name in enumerate(class_names):
        class_pred = pred[:, class_index]
        class_truth = targets[:, class_index]
        tp = int((class_pred & class_truth).sum().item())
        fp = int((class_pred & ~class_truth).sum().item())
        fn = int((~class_pred & class_truth).sum().item())
        tn = int((~class_pred & ~class_truth).sum().item())
        precision, recall, f1 = compute_prf(tp, fp, fn)
        per_class.append(
            {
                "class_index": class_index,
                "class_name": class_name,
                "threshold": float(thresholds[class_index].item()),
                "support": int(class_truth.sum().item()),
                "predicted_positive": int(class_pred.sum().item()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    all_indices = list(range(num_classes))
    non_null_indices = [idx for idx, name in enumerate(class_names) if name.lower() != "null"]
    payload = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "runtime_csv": str(runtime_csv),
        "multilabel_lookup_val": str(Path(data_cfg["multilabel_lookup_val"]).resolve()),
        "conditioning_mode": conditioning_mode,
        "task_name": task_name,
        "class_names": class_names,
        "thresholds": [float(value) for value in thresholds.tolist()],
        "num_validation_clips": int(max_samples),
        "num_scored_rows": int(probabilities.shape[0]),
        "exact_match": exact_match,
        "micro_all": summarize_subset(pred, targets, all_indices),
        "micro_non_null": summarize_subset(pred, targets, non_null_indices),
        "per_class": per_class,
        "predicted_label_combinations": top_label_combos(pred, class_names),
        "target_label_combinations": top_label_combos(targets, class_names),
    }

    print_summary(payload)

    if args.output_json is not None:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        print(f"\nSaved JSON to {output_path}")

    if args.output_csv is not None:
        output_path = args.output_csv.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "class_index",
            "class_name",
            "threshold",
            "precision",
            "recall",
            "f1",
            "support",
            "predicted_positive",
            "tp",
            "fp",
            "fn",
            "tn",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in per_class:
                writer.writerow({field: row[field] for field in fieldnames})
        print(f"Saved CSV to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
