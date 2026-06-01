#!/path/to/data_root/vjepa2_1/.venv/bin/python
"""Sweep validation thresholds for one single-tool multilabel head config."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

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
    build_eval_dataset,
    prepare_dataset_sample_inputs,
)
from triplet_soft_refinement_inference import resolve_soft_refinement_checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one single-tool multilabel head on its validation split and "
            "sweep sigmoid thresholds per class."
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
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--class-index", type=int, default=None)
    parser.add_argument("--thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--threshold-start", type=float, default=0.01)
    parser.add_argument("--threshold-end", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
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


def parse_multihot(value: str) -> torch.Tensor:
    value = str(value).strip()
    if not value:
        raise ValueError("Encountered empty multi-hot label field")
    return torch.tensor([float(token) for token in value.split()], dtype=torch.float32)


def parse_multihot_sequence(value: str) -> torch.Tensor:
    value = str(value).strip()
    if not value:
        raise ValueError("Encountered empty multi-hot sequence")
    parts = [part.strip() for part in value.split(";") if part.strip()]
    if not parts:
        parts = [value]
    return torch.stack([parse_multihot(part) for part in parts], dim=0)


def parse_condition_values(value: str) -> list[int]:
    value = str(value).strip()
    if not value:
        raise ValueError("Encountered empty conditioning field")
    return [int(float(token)) for token in value.split()]


def parse_condition_label(value: Any, num_condition_fields: int) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.clone().to(dtype=torch.long).reshape(-1)
    else:
        text = str(value).strip().strip('"').strip("'")
        tensor = torch.tensor([int(float(token)) for token in text.split()], dtype=torch.long)
    if tensor.numel() != num_condition_fields:
        raise ValueError(
            f"Expected {num_condition_fields} conditioning values, got {tensor.numel()}: {value!r}"
        )
    return tensor


def label_field_from_config(data_cfg: dict[str, Any]) -> str:
    fields = data_cfg.get("multilabel_label_fields")
    if not fields:
        raise ValueError("Config is missing experiment.data.multilabel_label_fields")
    if len(fields) != 1:
        raise ValueError(f"Threshold sweep expects one multilabel field, got {fields}")
    return str(fields[0])


def load_label_lookup(
    config: dict[str, Any],
    num_classes: int,
) -> tuple[str, dict[Any, Any]]:
    data_cfg = config["experiment"]["data"]
    label_field = label_field_from_config(data_cfg)
    lookup_path = Path(data_cfg["multilabel_lookup_val"])
    conditioning_mode = data_cfg.get("conditioning_mode")

    lookup: dict[Any, Any] = {}
    with lookup_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        if conditioning_mode == "multilabel_multi_condition_task":
            condition_fields = data_cfg.get("multilabel_condition_fields")
            if not condition_fields:
                raise ValueError("Multi-condition config is missing multilabel_condition_fields")
            condition_fields = [str(field) for field in condition_fields]
            required = ["clip_path", *condition_fields, label_field]
            missing = [field for field in required if field not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"Missing fields in {lookup_path}: {missing}")

            for row in reader:
                clip_path = str(row["clip_path"]).strip()
                condition_columns = [parse_condition_values(row[field]) for field in condition_fields]
                num_rows = len(condition_columns[0])
                if any(len(values) != num_rows for values in condition_columns):
                    raise ValueError(f"Condition fields have mismatched lengths for {clip_path}")
                condition_labels = torch.tensor(list(zip(*condition_columns)), dtype=torch.long)
                labels = parse_multihot_sequence(row[label_field])
                if labels.shape[0] != num_rows:
                    raise ValueError(
                        f"Label sequence length does not match condition rows for {clip_path}: "
                        f"{labels.shape[0]} vs {num_rows}"
                    )
                if labels.shape[1] != num_classes:
                    raise ValueError(
                        f"Expected {num_classes} classes in {label_field}, got {labels.shape[1]}"
                    )
                lookup[clip_path] = (condition_labels, labels)
            return conditioning_mode, lookup

        if conditioning_mode == "multilabel_conditioned_task":
            condition_fields = data_cfg.get(
                "multilabel_condition_fields",
                data_cfg.get("multilabel_condition_field", "conditioning_label"),
            )
            if isinstance(condition_fields, str):
                condition_fields = [condition_fields]
            condition_fields = [str(field) for field in condition_fields]
            required = ["clip_path", *condition_fields, label_field]
            missing = [field for field in required if field not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"Missing fields in {lookup_path}: {missing}")

            for row in reader:
                clip_path = str(row["clip_path"]).strip()
                condition_values = tuple(int(float(str(row[field]).strip())) for field in condition_fields)
                label = parse_multihot(row[label_field])
                if label.numel() != num_classes:
                    raise ValueError(
                        f"Expected {num_classes} classes in {label_field}, got {label.numel()}"
                    )
                lookup[(clip_path, *condition_values)] = label
            return conditioning_mode, lookup

        required = ["clip_path", label_field]
        missing = [field for field in required if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing fields in {lookup_path}: {missing}")

        for row in reader:
            clip_path = str(row["clip_path"]).strip()
            label = parse_multihot(row[label_field])
            if label.numel() != num_classes:
                raise ValueError(f"Expected {num_classes} classes in {label_field}, got {label.numel()}")
            lookup[clip_path] = label
    return "unconditioned", lookup


def build_thresholds(args: argparse.Namespace) -> list[float]:
    if args.thresholds is not None:
        values = sorted({float(value) for value in args.thresholds})
    else:
        if args.threshold_step <= 0:
            raise ValueError("--threshold-step must be positive")
        if args.threshold_end < args.threshold_start:
            raise ValueError("--threshold-end must be >= --threshold-start")
        values = []
        value = float(args.threshold_start)
        while value <= float(args.threshold_end) + 1e-12:
            values.append(round(value, 10))
            value += float(args.threshold_step)
    for value in values:
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Thresholds must be in [0, 1], got {value}")
    return values


def compute_threshold_stats(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    class_index: int,
    threshold: float,
) -> dict[str, Any]:
    scores = probabilities[:, class_index]
    truth = targets[:, class_index].bool()
    pred = scores >= float(threshold)
    tp = int((pred & truth).sum().item())
    fp = int((pred & ~truth).sum().item())
    fn = int((~pred & truth).sum().item())
    tn = int((~pred & ~truth).sum().item())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": int(truth.sum().item()),
        "predicted_positive": int(pred.sum().item()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_positive_rate": float(pred.float().mean().item()),
    }


def select_class_indices(args: argparse.Namespace, class_names: list[str]) -> list[int]:
    selected: list[int] = []
    if args.class_index is not None:
        if args.class_index < 0 or args.class_index >= len(class_names):
            raise ValueError(f"--class-index outside [0, {len(class_names)}): {args.class_index}")
        selected.append(int(args.class_index))
    if args.class_name is not None:
        matches = [idx for idx, name in enumerate(class_names) if name == args.class_name]
        if not matches:
            lowered = args.class_name.lower()
            matches = [idx for idx, name in enumerate(class_names) if name.lower() == lowered]
        if not matches:
            raise ValueError(f"Could not find class name {args.class_name!r} in {class_names}")
        selected.extend(matches)
    if not selected:
        selected = list(range(len(class_names)))
    return sorted(set(selected))


def best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            row["f1"],
            row["precision"],
            row["recall"],
            -abs(row["predicted_positive"] - row["support"]),
        ),
    )


def print_best_summary(best_rows: list[dict[str, Any]]) -> None:
    print("\nBest threshold by F1")
    print("class_id  class_name          threshold  precision  recall   f1       support  pred_pos")
    for row in best_rows:
        print(
            f"{row['class_index']:>8}  {row['class_name'][:18]:<18}  "
            f"{row['threshold']:>9.3f}  {row['precision']*100:>8.2f}%  "
            f"{row['recall']*100:>6.2f}%  {row['f1']*100:>6.2f}%  "
            f"{row['support']:>7}  {row['predicted_positive']:>8}"
        )


def print_full_table(rows: list[dict[str, Any]]) -> None:
    print("\nThreshold sweep")
    print("threshold  precision  recall   f1       support  pred_pos  tp    fp    fn    tn")
    for row in rows:
        print(
            f"{row['threshold']:>9.3f}  {row['precision']*100:>8.2f}%  "
            f"{row['recall']*100:>6.2f}%  {row['f1']*100:>6.2f}%  "
            f"{row['support']:>7}  {row['predicted_positive']:>8}  "
            f"{row['tp']:>4}  {row['fp']:>5}  {row['fn']:>4}  {row['tn']:>5}"
        )


def collate_samples(batch: list[Any]) -> list[Any]:
    return batch


def prepare_batched_sample_inputs(
    batch_buffers: list[list[Any]],
    batch_clip_indices: list[list[Any]],
    device: torch.device,
) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
    if not batch_buffers:
        raise ValueError("Cannot prepare an empty batch")
    num_clips = len(batch_buffers[0])
    if any(len(buffer) != num_clips for buffer in batch_buffers):
        raise ValueError("All samples in a batch must have the same number of clips")

    clips: list[list[torch.Tensor]] = []
    for clip_idx in range(num_clips):
        first_clip_views = batch_buffers[0][clip_idx]
        if torch.is_tensor(first_clip_views):
            first_clip_views = [first_clip_views]
        num_views = len(first_clip_views)
        prepared_views = []
        for view_idx in range(num_views):
            view_tensors = []
            for sample_buffer in batch_buffers:
                clip_views = sample_buffer[clip_idx]
                if torch.is_tensor(clip_views):
                    clip_views = [clip_views]
                if len(clip_views) != num_views:
                    raise ValueError("All samples in a batch must have the same number of views")
                view = clip_views[view_idx]
                view_tensors.append(view if torch.is_tensor(view) else torch.as_tensor(view))
            prepared_views.append(torch.stack(view_tensors, dim=0).to(device=device, non_blocking=True))
        clips.append(prepared_views)

    prepared_indices = []
    for clip_idx in range(num_clips):
        index_tensors = []
        for sample_indices in batch_clip_indices:
            index_tensor = torch.as_tensor(sample_indices[clip_idx], dtype=torch.int64)
            if index_tensor.ndim == 0:
                index_tensor = index_tensor.unsqueeze(0)
            index_tensors.append(index_tensor)
        prepared_indices.append(torch.stack(index_tensors, dim=0).to(device=device, non_blocking=True))
    return clips, prepared_indices


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
        raise ValueError("Threshold sweep expects a single-task head with num_classes_per_task length 1")
    num_classes = int(num_classes_per_task[0])
    task_name = str(data_cfg.get("task_names", ["task"])[0])
    class_names = class_names_from_config(config, [], task_name)
    if len(class_names) != num_classes:
        class_names = [f"class_{idx}" for idx in range(num_classes)]

    thresholds = build_thresholds(args)
    selected_class_indices = select_class_indices(args, class_names)
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
        print(f"Config: {config_path}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Runtime CSV: {runtime_csv}")
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
        progress_bar = tqdm(iterable, total=max_samples, desc="threshold sweep", unit="clip", dynamic_ncols=True)
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
    targets = torch.cat(target_rows, dim=0)

    all_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    for class_index in selected_class_indices:
        class_rows = []
        for threshold in thresholds:
            row = compute_threshold_stats(probabilities, targets, class_index, threshold)
            row["class_index"] = int(class_index)
            row["class_name"] = class_names[class_index]
            class_rows.append(row)
            all_rows.append(row)
        best_rows.append(best_row(class_rows))

    print(f"Evaluated {max_samples} validation clips from {config_path}")
    print(f"Expanded rows scored: {int(probabilities.shape[0])}")
    print(f"Checkpoint: {checkpoint_path}")
    print_best_summary(best_rows)
    if len(selected_class_indices) == 1:
        print_full_table([row for row in all_rows if row["class_index"] == selected_class_indices[0]])

    payload = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path.resolve()),
        "runtime_csv": str(runtime_csv),
        "multilabel_lookup_val": str(Path(data_cfg["multilabel_lookup_val"]).resolve()),
        "conditioning_mode": conditioning_mode,
        "task_name": task_name,
        "class_names": class_names,
        "num_validation_clips": int(max_samples),
        "num_scored_rows": int(probabilities.shape[0]),
        "thresholds": thresholds,
        "selected_class_indices": selected_class_indices,
        "best_by_f1": best_rows,
        "rows": all_rows,
    }

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
            "predicted_positive_rate",
            "tp",
            "fp",
            "fn",
            "tn",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_rows:
                writer.writerow({field: row[field] for field in fieldnames})
        print(f"Saved CSV to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
