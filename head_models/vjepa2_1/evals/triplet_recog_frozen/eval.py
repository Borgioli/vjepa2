# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os
import re

try:
    local_rank = os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID"))
    if local_rank is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = local_rank
except Exception:
    pass

import logging
import math
import pprint

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from evals.triplet_recog_frozen.models import (
    ConditionedTokenAggregationMultiTaskClassifier,
    init_module,
    PooledFeatureMultiTaskClassifier,
    SingleHeadMultiTaskClassifier,
    TokenAggregationClassifier,
    TokenAggregationMultiTaskClassifier,
)
from evals.triplet_recog_frozen.dataset_wrapper import (
    ConditionedMultiLabelTaskLabelWrapper,
    ConditionedTaskFromTripletWrapper,
    MultiConditionMultiLabelTaskLabelWrapper,
    MultiLabelTaskLabelWrapper,
    MultiTaskLabelWrapper,
)
from evals.triplet_recog_frozen.utils import multitask_collate_fn
from evals.video_classification_frozen.utils import make_transforms
from src.datasets.data_manager import init_data
from src.datasets.video_dataset import make_videodataset
from src.models.attentive_pooler import AttentiveClassifier
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.distributed import AllReduce, AllReduceSum, init_distributed
from src.utils.logging import AverageMeter, CSVLogger

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)

TASK_DISPLAY_NAMES = ("tool", "verb", "target", "triplet")
MULTILABEL_FIELD_NAMES = ("tool_multihot", "verb_multihot", "target_multihot", "triplet_multihot")


def get_task_display_names(num_tasks):
    if num_tasks <= len(TASK_DISPLAY_NAMES):
        return list(TASK_DISPLAY_NAMES[:num_tasks])
    return [f"task_{task_idx}" for task_idx in range(num_tasks)]


def get_multilabel_field_names(num_tasks):
    if num_tasks <= len(MULTILABEL_FIELD_NAMES):
        return list(MULTILABEL_FIELD_NAMES[:num_tasks])
    return [f"task_{task_idx}_multihot" for task_idx in range(num_tasks)]


def parse_csv_multitask_label(row, num_tasks):
    if len(row) < 2:
        return None
    label = " ".join(row[1:]).strip().strip('"').strip("'")
    if label == "":
        return None

    parts = label.split()
    if len(parts) != num_tasks:
        return None

    try:
        return [int(float(part)) for part in parts]
    except ValueError:
        return None


def compute_multitask_class_weights(dataset_paths, num_classes_per_task, power=0.5, max_weight=10.0):
    counts = [torch.zeros(num_classes, dtype=torch.float32) for num_classes in num_classes_per_task]

    for dataset_path in dataset_paths:
        with open(dataset_path, newline="") as csv_file:
            reader = csv.reader(csv_file, delimiter=" ")
            for row in reader:
                labels = parse_csv_multitask_label(row, len(num_classes_per_task))
                if labels is None:
                    continue
                for task_idx, class_idx in enumerate(labels):
                    if 0 <= class_idx < num_classes_per_task[task_idx]:
                        counts[task_idx][class_idx] += 1.0

    weights = []
    for task_counts in counts:
        observed_mask = task_counts > 0
        task_weights = torch.ones_like(task_counts)
        if observed_mask.any():
            task_weights[observed_mask] = task_counts[observed_mask].pow(-power)
            task_weights[observed_mask] *= observed_mask.sum() / task_weights[observed_mask].sum()
            task_weights = torch.clamp(task_weights, max=max_weight)
            task_weights[observed_mask] *= observed_mask.sum() / task_weights[observed_mask].sum()
        task_weights[~observed_mask] = 0.0
        weights.append(task_weights)

    return counts, weights


def log_multitask_class_weights(task_names, counts, weights):
    for task_name, task_counts, task_weights in zip(task_names, counts, weights):
        logger.info(
            "Task '%s' class counts: %s",
            task_name,
            [int(v) for v in task_counts.tolist()],
        )
        logger.info(
            "Task '%s' class weights: %s",
            task_name,
            [round(float(v), 4) for v in task_weights.tolist()],
        )


def parse_multihot_field(value):
    value = str(value).strip()
    if value == "":
        raise ValueError("Encountered empty multi-hot field while computing multi-label statistics")
    if ";" in value:
        vectors = []
        for vector in value.split(";"):
            vector = vector.strip()
            if not vector:
                continue
            vectors.append(torch.tensor([float(token) for token in vector.split()], dtype=torch.float32))
        if not vectors:
            raise ValueError("Encountered empty multi-hot sequence while computing multi-label statistics")
        return torch.stack(vectors, dim=0)
    return torch.tensor([float(token) for token in value.split()], dtype=torch.float32)


def compute_multilabel_pos_weights(multilabel_csv_path, label_fields, power=1.0, max_pos_weight=25.0):
    pos_counts = None
    num_samples = 0

    with open(multilabel_csv_path, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_fields = [field for field in label_fields if field not in reader.fieldnames]
        if missing_fields:
            raise ValueError(
                f"Missing required multi-label fields in {multilabel_csv_path}: {missing_fields}"
            )

        for row in reader:
            task_vectors = [parse_multihot_field(row[field]) for field in label_fields]
            if pos_counts is None:
                pos_counts = [
                    torch.zeros_like(task_vector[0] if task_vector.ndim == 2 else task_vector)
                    for task_vector in task_vectors
                ]
            for task_idx, task_vector in enumerate(task_vectors):
                pos_counts[task_idx] += task_vector.sum(dim=0) if task_vector.ndim == 2 else task_vector
            num_samples += max(
                task_vector.shape[0] if task_vector.ndim == 2 else 1
                for task_vector in task_vectors
            )

    if pos_counts is None:
        raise ValueError(f"No usable rows found in multi-label CSV: {multilabel_csv_path}")

    pos_weights = []
    for task_counts in pos_counts:
        observed_mask = task_counts > 0
        task_weights = torch.ones_like(task_counts)
        if observed_mask.any():
            neg_counts = float(num_samples) - task_counts[observed_mask]
            task_weights[observed_mask] = torch.clamp(
                (neg_counts / task_counts[observed_mask]).pow(power),
                max=max_pos_weight,
            )
        task_weights[~observed_mask] = 0.0
        pos_weights.append(task_weights)

    return pos_counts, pos_weights, num_samples


def log_multilabel_pos_weights(task_names, pos_counts, pos_weights):
    for task_name, task_counts, task_weights in zip(task_names, pos_counts, pos_weights):
        logger.info(
            "Task '%s' positive counts: %s",
            task_name,
            [int(v) for v in task_counts.tolist()],
        )
        logger.info(
            "Task '%s' pos_weight: %s",
            task_name,
            [round(float(v), 4) for v in task_weights.tolist()],
        )


def focal_loss(logits, targets, gamma=2.0, alpha=None):
    ce_loss = F.cross_entropy(logits, targets, reduction="none")
    pt = torch.exp(-ce_loss)
    loss = ((1.0 - pt) ** gamma) * ce_loss
    if alpha is not None:
        loss = alpha[targets] * loss
    return loss.mean()


def compute_task_loss(logits, targets, loss_name="ce", class_weights=None, focal_gamma=2.0, sample_mask=None):
    if class_weights is not None:
        class_weights = class_weights.to(device=logits.device, dtype=logits.dtype)
    if loss_name == "bce":
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets.to(dtype=logits.dtype),
            reduction="none" if sample_mask is not None else "mean",
        )
        if sample_mask is None:
            return loss
        mask = sample_mask.to(device=logits.device, dtype=loss.dtype)
        while mask.ndim < loss.ndim:
            mask = mask.unsqueeze(-1)
        loss = loss * mask
        denom = mask.expand_as(loss).sum().clamp_min(1.0)
        return loss.sum() / denom
    if loss_name == "weighted_bce":
        loss = F.binary_cross_entropy_with_logits(
            logits,
            targets.to(dtype=logits.dtype),
            pos_weight=class_weights,
            reduction="none" if sample_mask is not None else "mean",
        )
        if sample_mask is None:
            return loss
        mask = sample_mask.to(device=logits.device, dtype=loss.dtype)
        while mask.ndim < loss.ndim:
            mask = mask.unsqueeze(-1)
        loss = loss * mask
        denom = mask.expand_as(loss).sum().clamp_min(1.0)
        return loss.sum() / denom
    if loss_name in ("ce", "cross_entropy"):
        return F.cross_entropy(logits, targets.to(device=logits.device, dtype=torch.long))
    if loss_name == "weighted_ce":
        return F.cross_entropy(logits, targets.to(device=logits.device, dtype=torch.long), weight=class_weights)
    if loss_name == "focal":
        return focal_loss(logits, targets, gamma=focal_gamma)
    if loss_name == "weighted_focal":
        return focal_loss(logits, targets, gamma=focal_gamma, alpha=class_weights)
    raise ValueError(f"Unsupported loss_name '{loss_name}'")


def unwrap_classifier_module(classifier):
    return classifier.module if hasattr(classifier, "module") else classifier


def classifier_expects_conditioning(classifier):
    return getattr(unwrap_classifier_module(classifier), "expects_conditioning", False)


def forward_classifier(classifier, encoder_output, conditioning_labels=None):
    if classifier_expects_conditioning(classifier):
        if conditioning_labels is None:
            raise ValueError("Conditioning labels are required for the configured classifier")
        return classifier(encoder_output, conditioning_labels)
    return classifier(encoder_output)


def make_confusion_matrix(targets, predictions, num_classes):
    flat_indices = targets.to(torch.int64) * num_classes + predictions.to(torch.int64)
    return torch.bincount(flat_indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def make_multilabel_confusion_matrices(targets, predictions):
    targets = targets.to(dtype=torch.bool)
    predictions = predictions.to(dtype=torch.bool)
    tn = (~targets & ~predictions).sum(dim=0, dtype=torch.int64)
    fp = (~targets & predictions).sum(dim=0, dtype=torch.int64)
    fn = (targets & ~predictions).sum(dim=0, dtype=torch.int64)
    tp = (targets & predictions).sum(dim=0, dtype=torch.int64)
    return torch.stack(
        (
            torch.stack((tn, fp), dim=1),
            torch.stack((fn, tp), dim=1),
        ),
        dim=1,
    )


def write_confusion_matrix(path, matrix, class_labels=None):
    class_ids = list(range(matrix.shape[0])) if class_labels is None else list(class_labels)
    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true/pred"] + class_ids)
        for class_idx, row in zip(class_ids, matrix.tolist()):
            writer.writerow([class_idx] + row)


def normalize_class_names(class_names, num_classes):
    if class_names is None:
        return [f"class_{class_idx}" for class_idx in range(num_classes)]

    normalized = [str(class_name) for class_name in class_names[:num_classes]]
    if len(normalized) < num_classes:
        normalized.extend(
            [f"class_{class_idx}" for class_idx in range(len(normalized), num_classes)]
        )
    return normalized


def normalize_task_modes(label_mode, task_modes, num_tasks):
    if task_modes is None:
        task_modes = [label_mode for _ in range(num_tasks)]
    elif isinstance(task_modes, str):
        task_modes = [task_modes for _ in range(num_tasks)]
    elif isinstance(task_modes, (list, tuple)):
        task_modes = [str(mode) for mode in task_modes]
    else:
        raise ValueError(f"Expected task_modes to be a string or list, got {type(task_modes).__name__}")

    if len(task_modes) != num_tasks:
        raise ValueError(f"Expected {num_tasks} task modes, got {len(task_modes)}")
    unsupported_modes = sorted({mode for mode in task_modes if mode not in ("multiclass", "multilabel")})
    if unsupported_modes:
        raise ValueError(f"Unsupported task mode(s): {unsupported_modes}")
    return task_modes


def normalize_task_loss_names(label_mode, loss_name, task_loss_names, task_modes):
    if task_loss_names is None:
        if all(mode == label_mode for mode in task_modes):
            return [loss_name for _ in task_modes]
        return [
            loss_name if mode == label_mode else ("weighted_bce" if mode == "multilabel" else "ce")
            for mode in task_modes
        ]
    if isinstance(task_loss_names, str):
        task_loss_names = [task_loss_names for _ in task_modes]
    elif isinstance(task_loss_names, (list, tuple)):
        task_loss_names = [str(task_loss_name) for task_loss_name in task_loss_names]
    else:
        raise ValueError(
            f"Expected task_loss_names to be a string or list, got {type(task_loss_names).__name__}"
        )
    if len(task_loss_names) != len(task_modes):
        raise ValueError(f"Expected {len(task_modes)} task loss names, got {len(task_loss_names)}")
    return task_loss_names


def normalize_multilabel_thresholds(thresholds, num_classes_per_task, task_modes=None):
    if num_classes_per_task is None:
        return thresholds

    if task_modes is None:
        task_modes = ["multilabel" for _ in num_classes_per_task]
    multilabel_task_indices = [
        task_idx for task_idx, task_mode in enumerate(task_modes) if task_mode == "multilabel"
    ]

    if isinstance(thresholds, (int, float)):
        return [
            [float(thresholds) for _ in range(num_classes)] if task_idx in multilabel_task_indices else None
            for task_idx, num_classes in enumerate(num_classes_per_task)
        ]

    if not isinstance(thresholds, (list, tuple)):
        raise ValueError(f"Expected multilabel thresholds to be a scalar or list, got {type(thresholds).__name__}")

    is_nested = any(isinstance(value, (list, tuple)) for value in thresholds)
    if len(multilabel_task_indices) == 1 and not is_nested:
        task_idx = multilabel_task_indices[0]
        if len(thresholds) != num_classes_per_task[task_idx]:
            raise ValueError(
                f"Expected {num_classes_per_task[task_idx]} multilabel thresholds for task {task_idx}, "
                f"got {len(thresholds)}"
            )
        normalized = [None for _ in num_classes_per_task]
        normalized[task_idx] = [float(value) for value in thresholds]
        return normalized

    if len(thresholds) == len(num_classes_per_task):
        threshold_items = list(enumerate(thresholds))
    elif len(thresholds) == len(multilabel_task_indices):
        threshold_items = list(zip(multilabel_task_indices, thresholds))
    else:
        raise ValueError(
            f"Expected thresholds for {len(num_classes_per_task)} tasks or "
            f"{len(multilabel_task_indices)} multilabel tasks, got {len(thresholds)}"
        )

    normalized = [None for _ in num_classes_per_task]
    for task_idx, task_thresholds in threshold_items:
        if task_modes[task_idx] != "multilabel":
            continue
        num_classes = num_classes_per_task[task_idx]
        if isinstance(task_thresholds, (int, float)):
            normalized[task_idx] = [float(task_thresholds) for _ in range(num_classes)]
            continue
        if not isinstance(task_thresholds, (list, tuple)):
            raise ValueError(
                f"Expected thresholds for task {task_idx} to be a scalar or list, "
                f"got {type(task_thresholds).__name__}"
            )
        if len(task_thresholds) != num_classes:
            raise ValueError(f"Expected {num_classes} thresholds for task {task_idx}, got {len(task_thresholds)}")
        normalized[task_idx] = [float(value) for value in task_thresholds]
    return normalized


def sanitize_name_for_path(name):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()).strip("._-")
    return sanitized or "label"


def save_confusion_matrices(
    folder,
    epoch,
    split_name,
    confusion_matrices,
    task_names,
    label_mode="multiclass",
    class_names_per_task=None,
    task_modes=None,
):
    os.makedirs(folder, exist_ok=True)
    if task_modes is None:
        task_modes = [label_mode for _ in confusion_matrices]
    saved_paths = []
    for task_idx, matrix in enumerate(confusion_matrices):
        task_name = task_names[task_idx]
        task_mode = task_modes[task_idx]
        if task_mode == "multilabel":
            label_confusions = np.asarray(matrix, dtype=np.int64)
            if label_confusions.ndim != 3 or label_confusions.shape[1:] != (2, 2):
                raise ValueError(
                    f"Expected multilabel confusion matrix with shape [num_labels, 2, 2], got {label_confusions.shape}"
                )
            class_names = normalize_class_names(
                None if class_names_per_task is None else class_names_per_task[task_idx],
                label_confusions.shape[0],
            )
            for class_idx, class_name in enumerate(class_names):
                label_slug = sanitize_name_for_path(class_name)
                counts_path = os.path.join(
                    folder,
                    f"epoch_{epoch:03d}_{split_name}_{task_name}_{label_slug}_confusion_counts.csv",
                )
                normalized_path = os.path.join(
                    folder,
                    f"epoch_{epoch:03d}_{split_name}_{task_name}_{label_slug}_confusion_row_norm.csv",
                )

                counts_matrix = label_confusions[class_idx]
                row_totals = counts_matrix.sum(axis=1, keepdims=True)
                normalized_matrix = np.divide(
                    counts_matrix.astype(np.float64),
                    row_totals,
                    out=np.zeros_like(counts_matrix, dtype=np.float64),
                    where=row_totals > 0,
                )

                write_confusion_matrix(
                    counts_path,
                    counts_matrix,
                    class_labels=["absent", "present"],
                )
                write_confusion_matrix(
                    normalized_path,
                    np.round(normalized_matrix, 6),
                    class_labels=["absent", "present"],
                )
                saved_paths.extend([counts_path, normalized_path])
            continue

        counts_path = os.path.join(folder, f"epoch_{epoch:03d}_{split_name}_{task_name}_confusion_counts.csv")
        normalized_path = os.path.join(folder, f"epoch_{epoch:03d}_{split_name}_{task_name}_confusion_row_norm.csv")

        counts_matrix = np.asarray(matrix, dtype=np.int64)
        class_names = normalize_class_names(
            None if class_names_per_task is None else class_names_per_task[task_idx],
            counts_matrix.shape[0],
        )
        row_totals = counts_matrix.sum(axis=1, keepdims=True)
        normalized_matrix = np.divide(
            counts_matrix.astype(np.float64),
            row_totals,
            out=np.zeros_like(counts_matrix, dtype=np.float64),
            where=row_totals > 0,
        )

        write_confusion_matrix(counts_path, counts_matrix, class_labels=class_names)
        write_confusion_matrix(normalized_path, np.round(normalized_matrix, 6), class_labels=class_names)
        saved_paths.extend([counts_path, normalized_path])
    return saved_paths


def format_epoch_metrics(split_name, metrics, task_names):
    msg = f"{split_name}: loss={metrics['loss']:.3f} ExactClip={metrics['exact_triplet_acc']:.2f}%"
    if metrics.get("task_micro_f1s"):
        task_f1_names = metrics.get("task_f1_names", task_names)
        parts = []
        for metric_idx, task_name in enumerate(task_f1_names):
            try:
                task_idx = task_names.index(task_name)
            except ValueError:
                task_idx = metric_idx
            task_accs = metrics.get("task_accs", [])
            exact = task_accs[task_idx] if task_idx < len(task_accs) else float("nan")
            precision = metrics["task_micro_precisions"][metric_idx]
            recall = metrics["task_micro_recalls"][metric_idx]
            f1 = metrics["task_micro_f1s"][metric_idx]
            parts.append(
                f"{task_name}: Exact={exact:.2f}% MicroF1={f1:.2f}% P={precision:.2f}% R={recall:.2f}%"
            )
        msg += " | " + " | ".join(parts)
    elif metrics.get("task_accs"):
        task_summary = " ".join(
            [f"{task_name}={task_acc:.2f}%" for task_name, task_acc in zip(task_names, metrics["task_accs"])]
        )
        msg += f" | Tasks: {task_summary}"
    return msg


def compute_micro_prf(tp, fp, fn):
    precision = 100.0 * tp / max(1e-8, tp + fp)
    recall = 100.0 * tp / max(1e-8, tp + fn)
    f1 = 100.0 * (2.0 * tp) / max(1e-8, 2.0 * tp + fp + fn)
    return precision, recall, f1


def slugify_metric_name(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "metric"


def select_best_micro_f1_metric(metrics):
    f1s = metrics.get("task_micro_f1s") or metrics.get("task_f1s")
    if not f1s:
        return None
    task_names_for_f1 = metrics.get("task_f1_names") or [f"task_{idx}" for idx in range(len(f1s))]
    if len(f1s) == 1:
        task_slug = slugify_metric_name(task_names_for_f1[0])
        return {
            "value": float(f1s[0]),
            "name": f"val_{task_slug}_micro_f1",
            "slug": task_slug,
        }
    return {
        "value": float(np.mean(f1s)),
        "name": "val_mean_micro_f1",
        "slug": "mean",
    }


def main(args_eval, resume_preempt=False):

    """Main evaluation function"""

    val_only = args_eval.get("val_only", False)
    if val_only:
        logger.info("VAL ONLY")

    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    args_pretrain = args_eval.get("model_kwargs")
    checkpoint = args_pretrain.get("checkpoint")
    module_name = args_pretrain.get("module_name")
    args_model = args_pretrain.get("pretrain_kwargs")
    args_wrapper = args_pretrain.get("wrapper_kwargs")

    args_exp = args_eval.get("experiment")

    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)
    use_single_head = args_classifier.get("use_single_head", False)
    head_type = args_classifier.get("head_type", "single_head_pooler")
    feature_pool = args_classifier.get("feature_pool", "mean")
    feature_use_layer_norm = args_classifier.get("feature_use_layer_norm", True)
    classifier_hidden_dims = args_classifier.get("classifier_hidden_dims")
    token_pool = args_classifier.get("token_pool", "max")
    token_pool_topk = args_classifier.get("token_pool_topk")
    token_use_layer_norm = args_classifier.get("token_use_layer_norm", True)
    condition_num_classes = args_classifier.get("condition_num_classes")
    condition_embed_dim = args_classifier.get("condition_embed_dim")

    args_data = args_exp.get("data")
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    
    num_classes = args_data.get("num_classes")
    num_classes_per_task = args_data.get("num_classes_per_task", None)
    label_mode = args_data.get("label_mode", "multiclass")
    if label_mode not in ("multiclass", "multilabel"):
        raise ValueError(f"Unsupported label_mode '{label_mode}'")
    
    is_multitask = num_classes_per_task is not None
    if is_multitask:
        logger.info(f"Multi-task mode enabled with {len(num_classes_per_task)} tasks")
        logger.info(f"Classes per task: {num_classes_per_task}")
        if not use_single_head:
            logger.warning("Multi-task mode is enabled but use_single_head=False. Consider setting use_single_head=True for single-head multi-task architecture.")
    else:
        logger.info(f"Single-task mode with {num_classes} classes")
        num_classes_per_task = [num_classes]
    train_data_path = [args_data.get("dataset_train")]
    val_data_path = [args_data.get("dataset_val")]
    task_names = args_data.get("task_names", get_task_display_names(len(num_classes_per_task)))
    if len(task_names) != len(num_classes_per_task):
        raise ValueError(f"Expected {len(num_classes_per_task)} task names, got {len(task_names)}")
    task_modes = normalize_task_modes(label_mode, args_data.get("task_modes"), len(num_classes_per_task))
    multilabel_task_indices = [
        task_idx for task_idx, task_mode in enumerate(task_modes) if task_mode == "multilabel"
    ]
    class_names_per_task = args_data.get("class_names_per_task")
    multilabel_lookup_train = args_data.get("multilabel_lookup_train")
    multilabel_lookup_val = args_data.get("multilabel_lookup_val")
    conditioning_mode = args_data.get("conditioning_mode")
    conditioning_task_indices = args_data.get("conditioning_task_indices")
    if conditioning_task_indices is None:
        conditioning_task_indices = args_data.get("conditioning_task_idx", 0)
    target_task_idx = args_data.get("target_task_idx", 1)
    multilabel_label_fields = args_data.get(
        "multilabel_label_fields",
        get_multilabel_field_names(len(num_classes_per_task)),
    )
    if label_mode == "multilabel" and len(multilabel_label_fields) != len(multilabel_task_indices):
        raise ValueError(
            "multilabel_label_fields should contain one CSV field for each multilabel task. "
            f"Got {len(multilabel_label_fields)} fields for {len(multilabel_task_indices)} multilabel task(s)."
        )
    multilabel_condition_field = args_data.get(
        "multilabel_condition_fields",
        args_data.get("multilabel_condition_field", "conditioning_label"),
    )
    multilabel_count_source_fields = args_data.get("multilabel_count_source_fields", [])
    multilabel_threshold = args_data.get(
        "multilabel_thresholds",
        args_data.get("multilabel_threshold", 0.5),
    )
    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)
    duration = args_data.get("clip_duration", None)
    num_views_per_segment = args_data.get("num_views_per_segment", 1)
    normalization = args_data.get("normalization", None)
    num_workers = args_data.get("num_workers", args_eval.get("num_workers", 12))
    pin_memory = bool(args_data.get("pin_memory", True))
    persistent_workers = bool(args_data.get("persistent_workers", num_workers > 0))
    dataloader_timeout = int(args_data.get("dataloader_timeout", 0))

    args_opt = args_exp.get("optimization")
    batch_size = args_opt.get("batch_size")
    num_epochs = args_opt.get("num_epochs")
    use_bfloat16 = args_opt.get("use_bfloat16")
    default_loss_name = "weighted_bce" if label_mode == "multilabel" else ("weighted_ce" if is_multitask else "ce")
    loss_name = args_opt.get("loss_name", default_loss_name)
    task_loss_names = normalize_task_loss_names(
        label_mode,
        loss_name,
        args_opt.get("task_loss_names"),
        task_modes,
    )
    focal_gamma = args_opt.get("focal_gamma", 2.0)
    class_weight_power = args_opt.get("class_weight_power", 0.5)
    max_class_weight = args_opt.get("max_class_weight", 10.0)
    pos_weight_power = args_opt.get("pos_weight_power", 1.0)
    max_pos_weight = args_opt.get("max_pos_weight", 25.0)
    save_val_confusion_matrices = args_opt.get("save_confusion_matrices", label_mode == "multiclass")
    opt_kwargs = [
        dict(
            ref_wd=kwargs.get("weight_decay"),
            final_wd=kwargs.get("final_weight_decay"),
            start_lr=kwargs.get("start_lr"),
            ref_lr=kwargs.get("lr"),
            final_lr=kwargs.get("final_lr"),
            warmup=kwargs.get("warmup"),
        )
        for kwargs in args_opt.get("multihead_kwargs")
    ]
    
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        logger.info(f"Distributed is initialized - Using existing group")
    else:
        world_size = 1
        rank = 0
        logger.info(f"Distributed is NOT initialized - Running in single-process mode")
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    folder = os.path.join(pretrain_folder, "video_classification_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    confusion_dir = os.path.join(folder, "confusion_matrices")

    if rank == 0:
        csv_columns = [
            ("%d", "epoch"),
            ("%.5f", "train_loss"),
            ("%.5f", "train_acc"),
            ("%.5f", "train_exact_triplet_acc"),
        ]
        if is_multitask:
            csv_columns.extend([("%.5f", f"train_{task_name}_acc") for task_name in task_names])
            csv_columns.extend([
                ("%.5f", f"train_{task_names[task_idx]}_f1")
                for task_idx in multilabel_task_indices
            ])
        csv_columns.extend(
            [
                ("%.5f", "val_loss"),
                ("%.5f", "val_acc"),
                ("%.5f", "val_exact_triplet_acc"),
            ]
        )
        if is_multitask:
            csv_columns.extend([("%.5f", f"val_{task_name}_acc") for task_name in task_names])
            csv_columns.extend([
                ("%.5f", f"val_{task_names[task_idx]}_f1")
                for task_idx in multilabel_task_indices
            ])
        csv_logger = CSVLogger(
            log_file,
            *csv_columns,
            mode="+a" if resume_checkpoint and os.path.exists(log_file) else "w",
        )

    class_counts = None
    class_weights = None
    if label_mode == "multilabel":
        if multilabel_lookup_train is None or multilabel_lookup_val is None:
            raise ValueError(
                "Multi-label mode requires 'multilabel_lookup_train' and 'multilabel_lookup_val' in the config."
            )
        class_weights = [None for _ in num_classes_per_task]
        if any(task_loss_names[task_idx] == "weighted_bce" for task_idx in multilabel_task_indices):
            class_counts, class_weights, num_multilabel_samples = compute_multilabel_pos_weights(
                multilabel_lookup_train,
                multilabel_label_fields,
                power=pos_weight_power,
                max_pos_weight=max_pos_weight,
            )
            class_weights = [
                task_weights.to(device) for task_weights in class_weights
            ]
            task_class_weights = [None for _ in num_classes_per_task]
            for task_idx, task_weights in zip(multilabel_task_indices, class_weights):
                task_class_weights[task_idx] = task_weights
            class_weights = task_class_weights
            if rank == 0:
                logger.info(
                    "Using '%s' with pos_weight_power=%.3f and max_pos_weight=%.3f on %d grouped clips",
                    "weighted_bce",
                    pos_weight_power,
                    max_pos_weight,
                    num_multilabel_samples,
                )
                log_multilabel_pos_weights(
                    [task_names[task_idx] for task_idx in multilabel_task_indices],
                    class_counts,
                    [class_weights[task_idx].cpu() for task_idx in multilabel_task_indices],
                )
        else:
            logger.info("Using task losses in multi-label mode: %s", task_loss_names)
    else:
        if loss_name in ("weighted_ce", "weighted_focal"):
            class_counts, class_weights = compute_multitask_class_weights(
                dataset_paths=train_data_path,
                num_classes_per_task=num_classes_per_task,
                power=class_weight_power,
                max_weight=max_class_weight,
            )
            class_weights = [task_weights.to(device) for task_weights in class_weights]
            if rank == 0:
                logger.info(
                    "Using '%s' with class_weight_power=%.3f and max_class_weight=%.3f",
                    loss_name,
                    class_weight_power,
                    max_class_weight,
                )
                log_multitask_class_weights(task_names, class_counts, [weights.cpu() for weights in class_weights])
        else:
            logger.info("Using '%s' loss", loss_name)

    """Initialize model"""

    encoder = init_module(
        module_name=module_name,
        frames_per_clip=frames_per_clip,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=args_model,
        wrapper_kwargs=args_wrapper,
        device=device,
    )
    if use_single_head and is_multitask:
        if head_type == "single_head_pooler":
            logger.info("Using SingleHeadMultiTaskClassifier")
            classifiers = [
                SingleHeadMultiTaskClassifier(
                    num_classes_per_task=num_classes_per_task,
                    embed_dim=encoder.embed_dim,
                    num_heads=num_heads,
                    depth=num_probe_blocks,
                    use_activation_checkpointing=True,
                ).to(device)
                for _ in opt_kwargs
            ]
        elif head_type == "pooled_feature":
            logger.info(
                "Using PooledFeatureMultiTaskClassifier with feature_pool=%s "
                "classifier_hidden_dims=%s layer_norm=%s",
                feature_pool,
                classifier_hidden_dims if classifier_hidden_dims is not None else [],
                feature_use_layer_norm,
            )
            classifiers = [
                PooledFeatureMultiTaskClassifier(
                    num_classes_per_task=num_classes_per_task,
                    embed_dim=encoder.embed_dim,
                    feature_pool=feature_pool,
                    num_heads=num_heads,
                    depth=num_probe_blocks,
                    classifier_hidden_dims=classifier_hidden_dims,
                    use_layer_norm=feature_use_layer_norm,
                    use_activation_checkpointing=True,
                ).to(device)
                for _ in opt_kwargs
            ]
        elif head_type == "token_aggregation":
            logger.info(
                "Using TokenAggregationMultiTaskClassifier with token_pool=%s token_pool_topk=%s "
                "layer_norm=%s",
                token_pool,
                token_pool_topk,
                token_use_layer_norm,
            )
            classifiers = [
                TokenAggregationMultiTaskClassifier(
                    num_classes_per_task=num_classes_per_task,
                    embed_dim=encoder.embed_dim,
                    token_pool=token_pool,
                    token_pool_topk=token_pool_topk,
                    use_layer_norm=token_use_layer_norm,
                ).to(device)
                for _ in opt_kwargs
            ]
        elif head_type == "conditioned_token_aggregation":
            if condition_num_classes is None:
                raise ValueError(
                    "classifier.condition_num_classes must be set when using "
                    "head_type='conditioned_token_aggregation'"
                )
            logger.info(
                "Using ConditionedTokenAggregationMultiTaskClassifier with token_pool=%s "
                "token_pool_topk=%s layer_norm=%s condition_num_classes=%s "
                "condition_embed_dim=%s",
                token_pool,
                token_pool_topk,
                token_use_layer_norm,
                condition_num_classes,
                condition_embed_dim if condition_embed_dim is not None else encoder.embed_dim,
            )
            classifiers = [
                ConditionedTokenAggregationMultiTaskClassifier(
                    num_classes_per_task=num_classes_per_task,
                    embed_dim=encoder.embed_dim,
                    num_condition_classes=condition_num_classes,
                    condition_embed_dim=condition_embed_dim,
                    token_pool=token_pool,
                    token_pool_topk=token_pool_topk,
                    use_layer_norm=token_use_layer_norm,
                ).to(device)
                for _ in opt_kwargs
            ]
        else:
            raise ValueError(
                f"Unsupported classifier.head_type '{head_type}'. "
                "Expected one of: single_head_pooler, pooled_feature, token_aggregation, conditioned_token_aggregation"
            )
    else:
        if head_type == "token_aggregation":
            logger.info(
                "Using TokenAggregationClassifier with token_pool=%s token_pool_topk=%s layer_norm=%s",
                token_pool,
                token_pool_topk,
                token_use_layer_norm,
            )
            classifiers = [
                TokenAggregationClassifier(
                    num_classes=num_classes if not is_multitask else num_classes_per_task[0],
                    embed_dim=encoder.embed_dim,
                    token_pool=token_pool,
                    token_pool_topk=token_pool_topk,
                    use_layer_norm=token_use_layer_norm,
                ).to(device)
                for _ in opt_kwargs
            ]
        else:
            logger.info("Using standard AttentiveClassifier")
            classifiers = [
                AttentiveClassifier(
                    embed_dim=encoder.embed_dim,
                    num_heads=num_heads,
                    depth=num_probe_blocks,
                    num_classes=num_classes if not is_multitask else num_classes_per_task[0],
                    use_activation_checkpointing=True,
                ).to(device)
                for _ in opt_kwargs
            ]
    
    if world_size > 1:
        classifiers = [DistributedDataParallel(c, static_graph=True) for c in classifiers]
    
    print(classifiers[0])

    train_loader, train_sampler = make_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        eval_duration=duration,
        num_segments=num_segments,
        num_views_per_segment=1,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        dataloader_timeout=dataloader_timeout,
        normalization=normalization,
        use_multitask_wrapper=is_multitask,
        label_mode=label_mode,
        conditioning_mode=conditioning_mode,
        conditioning_task_idx=conditioning_task_indices,
        target_task_idx=target_task_idx,
        multilabel_lookup_path=multilabel_lookup_train,
        multilabel_label_fields=multilabel_label_fields,
        multilabel_condition_field=multilabel_condition_field,
        multilabel_count_source_fields=multilabel_count_source_fields,
    )
    val_loader, _ = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        eval_duration=duration,
        num_views_per_segment=num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        dataloader_timeout=dataloader_timeout,
        normalization=normalization,
        use_multitask_wrapper=is_multitask,
        label_mode=label_mode,
        conditioning_mode=conditioning_mode,
        conditioning_task_idx=conditioning_task_indices,
        target_task_idx=target_task_idx,
        multilabel_lookup_path=multilabel_lookup_val,
        multilabel_label_fields=multilabel_label_fields,
        multilabel_condition_field=multilabel_condition_field,
        multilabel_count_source_fields=multilabel_count_source_fields,
    )
    ipe = len(train_loader)
    logger.info(f"Dataloader created... iterations per epoch: {ipe}")
    logger.info(
        "Dataloader settings: num_workers=%d pin_memory=%s persistent_workers=%s timeout=%ds",
        num_workers,
        pin_memory,
        persistent_workers and num_workers > 0,
        dataloader_timeout,
    )

    """Debug: Check data format from dataloader"""
    logger.info("=" * 80)
    logger.info("Checking dataloader output format...")
    try:
        batch = next(iter(train_loader))
        logger.info(f"Batch contains {len(batch)} elements")
        
        if len(batch) == 5:
            clips, labels, clip_indices, conditioning_labels, conditioning_masks = batch
            logger.info(f"Clips type: {type(clips)}")
            logger.info(
                "Conditioning labels shape: %s dtype: %s sample values: %s",
                conditioning_labels.shape,
                conditioning_labels.dtype,
                conditioning_labels[:min(3, len(conditioning_labels))],
            )
            logger.info(
                "Conditioning masks shape: %s dtype: %s sample values: %s",
                conditioning_masks.shape,
                conditioning_masks.dtype,
                conditioning_masks[:min(3, len(conditioning_masks))],
            )
            if isinstance(clips, list):
                logger.info(f"  Number of temporal segments: {len(clips)}")
                if len(clips) > 0 and isinstance(clips[0], list):
                    logger.info(f"  Number of spatial views: {len(clips[0])}")
                    if len(clips[0]) > 0:
                        logger.info(f"  Clip shape: {clips[0][0].shape}")
            else:
                logger.info(f"  Clips shape: {clips.shape}")
        elif len(batch) == 4:
            clips, labels, clip_indices, conditioning_labels = batch
            logger.info(f"Clips type: {type(clips)}")
            logger.info(
                "Conditioning labels shape: %s dtype: %s sample values: %s",
                conditioning_labels.shape,
                conditioning_labels.dtype,
                conditioning_labels[:min(3, len(conditioning_labels))],
            )
            if isinstance(clips, list):
                logger.info(f"  Number of temporal segments: {len(clips)}")
                if len(clips) > 0 and isinstance(clips[0], list):
                    logger.info(f"  Number of spatial views: {len(clips[0])}")
                    if len(clips[0]) > 0:
                        logger.info(f"  Clip shape: {clips[0][0].shape}")
            else:
                logger.info(f"  Clips shape: {clips.shape}")
        elif len(batch) == 3:
            clips, labels, clip_indices = batch
            logger.info(f"Clips type: {type(clips)}")
            if isinstance(clips, list):
                logger.info(f"  Number of temporal segments: {len(clips)}")
                if len(clips) > 0 and isinstance(clips[0], list):
                    logger.info(f"  Number of spatial views: {len(clips[0])}")
                    if len(clips[0]) > 0:
                        logger.info(f"  Clip shape: {clips[0][0].shape}")
            else:
                logger.info(f"  Clips shape: {clips.shape}")
        elif len(batch) == 2:
            clips, labels = batch
            logger.info(f"Clips shape/type: {clips.shape if hasattr(clips, 'shape') else type(clips)}")
        else:
            logger.info(f"Unexpected batch format with {len(batch)} elements")
            labels = batch[1]
        
        logger.info(f"\nLabels type: {type(labels)}")
        
        if isinstance(labels, (list, tuple)):
            logger.info(f"✓ Multi-task labels detected!")
            logger.info(f"  Number of tasks: {len(labels)}")
            for i, label in enumerate(labels):
                logger.info(f"  Task {i} - shape: {label.shape}, dtype: {label.dtype}")
                logger.info(f"  Task {i} - sample values: {label[:min(3, len(label))]}")
        else:
            logger.info(f"✗ Single-task labels detected")
            logger.info(f"  Labels shape: {labels.shape}")
            logger.info(f"  Labels dtype: {labels.dtype}")
            logger.info(f"  Sample label values: {labels[:min(3, len(labels))]}")
            logger.info(f"\n⚠️  WARNING: Your dataset returns single-task labels but config expects multi-task!")
            logger.info(f"  You need to modify your dataset to return: (video, [label1, label2, label3])")
        
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Error checking dataloader format: {e}")
        import traceback
        logger.error(traceback.format_exc())
        logger.info("=" * 80)
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        classifiers=classifiers,
        opt_kwargs=opt_kwargs,
        iterations_per_epoch=ipe,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16,
    )

    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        classifiers, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifiers=classifiers,
            opt=optimizer,
            scaler=scaler,
            val_only=val_only,
        )
        for _ in range(start_epoch * ipe):
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

    best_val_micro_f1 = -float("inf")

    def save_checkpoint(epoch, checkpoint_path=latest_path, extra_state=None):
        all_classifier_dicts = [c.state_dict() for c in classifiers]
        all_opt_dicts = [o.state_dict() for o in optimizer]

        save_dict = {
            "classifiers": all_classifier_dicts,
            "opt": all_opt_dicts,
            "scaler": None if scaler is None else [s.state_dict() for s in scaler],
            "epoch": epoch,
            "batch_size": batch_size,
            "world_size": world_size,
        }
        if extra_state:
            save_dict.update(extra_state)
        if rank == 0:
            torch.save(save_dict, checkpoint_path)

    """TRAIN LOOP"""
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        train_sampler.set_epoch(epoch)
        if val_only:
            train_metrics = {
                "avg_task_acc": -1.0,
                "loss": -1.0,
                "exact_triplet_acc": -1.0,
                "task_accs": [-1.0 for _ in task_names],
            }
            if multilabel_task_indices:
                train_metrics["task_f1s"] = [-1.0 for _ in multilabel_task_indices]
                train_metrics["task_f1_names"] = [task_names[task_idx] for task_idx in multilabel_task_indices]
        else:
            train_metrics = run_one_epoch(
                device=device,
                training=True,
                encoder=encoder,
                classifiers=classifiers,
                scaler=scaler,
                optimizer=optimizer,
                scheduler=scheduler,
                wd_scheduler=wd_scheduler,
                data_loader=train_loader,
                use_bfloat16=use_bfloat16,
                is_multitask=is_multitask,
                num_classes_per_task=num_classes_per_task,
                use_single_head=use_single_head,
                loss_name=loss_name,
                class_weights=class_weights,
                focal_gamma=focal_gamma,
                task_names=task_names,
                collect_confusion_matrices=False,
                label_mode=label_mode,
                task_modes=task_modes,
                task_loss_names=task_loss_names,
                multilabel_threshold=multilabel_threshold,
            )

        val_metrics = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            classifiers=classifiers,
            scaler=scaler,
            optimizer=optimizer,
            scheduler=scheduler,
            wd_scheduler=wd_scheduler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16,
            is_multitask=is_multitask,
            num_classes_per_task=num_classes_per_task,
            use_single_head=use_single_head,
            loss_name=loss_name,
            class_weights=class_weights,
            focal_gamma=focal_gamma,
            task_names=task_names,
            collect_confusion_matrices=save_val_confusion_matrices and use_single_head and is_multitask,
            label_mode=label_mode,
            task_modes=task_modes,
            task_loss_names=task_loss_names,
            multilabel_threshold=multilabel_threshold,
        )

        logger.info(
            "[%5d] %s | %s",
            epoch + 1,
            format_epoch_metrics("train", train_metrics, task_names),
            format_epoch_metrics("val", val_metrics, task_names),
        )
        if rank == 0:
            log_values = [
                epoch + 1,
                train_metrics["loss"],
                train_metrics["avg_task_acc"],
                train_metrics["exact_triplet_acc"],
            ]
            if is_multitask:
                log_values.extend(train_metrics["task_accs"])
                if multilabel_task_indices:
                    log_values.extend(train_metrics["task_f1s"])
            log_values.extend(
                [
                    val_metrics["loss"],
                    val_metrics["avg_task_acc"],
                    val_metrics["exact_triplet_acc"],
                ]
            )
            if is_multitask:
                log_values.extend(val_metrics["task_accs"])
                if multilabel_task_indices:
                    log_values.extend(val_metrics["task_f1s"])
            csv_logger.log(*log_values)

            if save_val_confusion_matrices and val_metrics.get("confusion_matrices") is not None:
                saved_paths = save_confusion_matrices(
                    confusion_dir,
                    epoch + 1,
                    "val",
                    val_metrics["confusion_matrices"],
                    task_names,
                    label_mode=label_mode,
                    class_names_per_task=class_names_per_task,
                    task_modes=task_modes,
                )
                logger.info("Saved val confusion matrices: %s", ", ".join(saved_paths))

        if val_only:
            return

        save_checkpoint(epoch + 1)
        best_metric = select_best_micro_f1_metric(val_metrics)
        if best_metric is not None and best_metric["value"] > best_val_micro_f1:
            best_val_micro_f1 = best_metric["value"]
            best_extra_state = {
                "best_metric_name": best_metric["name"],
                "best_metric_value": best_metric["value"],
                "best_metric_epoch": epoch + 1,
            }
            best_stable_path = os.path.join(folder, f"best_val_{best_metric['slug']}_micro_f1.pt")
            best_epoch_path = os.path.join(
                folder,
                f"best_epoch_{epoch + 1:04d}_val_{best_metric['slug']}_micro_f1.pt",
            )
            save_checkpoint(epoch + 1, checkpoint_path=best_stable_path, extra_state=best_extra_state)
            save_checkpoint(epoch + 1, checkpoint_path=best_epoch_path, extra_state=best_extra_state)
            if rank == 0:
                logger.info(
                    "Saved best validation micro-F1 checkpoint: %s=%.3f%% at epoch %d (%s)",
                    best_metric["name"],
                    best_metric["value"],
                    epoch + 1,
                    best_stable_path,
                )


def run_one_epoch(
    device,
    training,
    encoder,
    classifiers,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
    is_multitask=False,
    num_classes_per_task=None,
    use_single_head=False,
    loss_name="ce",
    class_weights=None,
    focal_gamma=2.0,
    task_names=None,
    collect_confusion_matrices=False,
    label_mode="multiclass",
    task_modes=None,
    task_loss_names=None,
    multilabel_threshold=0.5,
):

    for c in classifiers:
        c.train(mode=training)

    if num_classes_per_task is not None:
        task_modes = normalize_task_modes(label_mode, task_modes, len(num_classes_per_task))
        task_loss_names = normalize_task_loss_names(label_mode, loss_name, task_loss_names, task_modes)
    else:
        task_modes = []
        task_loss_names = []
    multilabel_task_indices = [
        task_idx for task_idx, task_mode in enumerate(task_modes) if task_mode == "multilabel"
    ]

    multilabel_thresholds = None
    if multilabel_task_indices and num_classes_per_task is not None:
        multilabel_thresholds = [
            (
                torch.as_tensor(task_thresholds, device=device, dtype=torch.float32)
                if task_thresholds is not None
                else None
            )
            for task_thresholds in normalize_multilabel_thresholds(
                multilabel_threshold,
                num_classes_per_task,
                task_modes=task_modes,
            )
        ]

    top1_meters = [AverageMeter() for _ in classifiers]
    exact_triplet_meters = [AverageMeter() for _ in classifiers]
    loss_meters = [AverageMeter() for _ in classifiers]
    
    if is_multitask and num_classes_per_task is not None:
        num_tasks = len(num_classes_per_task)
        task_top1_meters = [[AverageMeter() for _ in range(num_tasks)] for _ in classifiers]
        if multilabel_task_indices:
            task_f1_stats = [
                [{"tp": 0.0, "fp": 0.0, "fn": 0.0} for _ in range(num_tasks)]
                for _ in classifiers
            ]
        else:
            task_f1_stats = None
        if collect_confusion_matrices:
            confusion_matrices = [
                [
                    (
                        torch.zeros((num_classes, 2, 2), device=device, dtype=torch.int64)
                        if task_modes[task_idx] == "multilabel"
                        else torch.zeros((num_classes, num_classes), device=device, dtype=torch.int64)
                    )
                    for task_idx, num_classes in enumerate(num_classes_per_task)
                ]
                for _ in classifiers
            ]
        else:
            confusion_matrices = None
    else:
        confusion_matrices = None
    
    for itr, data in enumerate(data_loader):
        if training:
            [s.step() for s in scheduler]
            [wds.step() for wds in wd_scheduler]

        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bfloat16 and device.type == "cuda"):
            clips = [
                [dij.to(device, non_blocking=True) for dij in di]
                for di in data[0]
            ]
            clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
            conditioning_labels = None
            if len(data) > 3 and data[3] is not None:
                conditioning_labels = data[3].to(device, non_blocking=True)
            conditioning_masks = None
            if len(data) > 4 and data[4] is not None:
                conditioning_masks = data[4].to(device, non_blocking=True, dtype=torch.bool)
            
            labels_list = []
            try:
                batch_size = clips[0][0].shape[0]
            except Exception:
                batch_size = None

            if isinstance(data[1], (list, tuple)):
                for label in data[1]:
                    if label is None:
                        continue
                    if isinstance(label, (list, tuple)) and len(label) == 0:
                        continue
                    if torch.is_tensor(label):
                        labels_list.append(label.to(device, non_blocking=True))
                    else:
                        try:
                            labels_list.append(torch.tensor(label).to(device))
                        except Exception:
                            continue
                if len(labels_list) > 0 and batch_size is None:
                    batch_size = labels_list[0].shape[0]
            else:
                label = data[1]
                if label is None or (isinstance(label, (list, tuple)) and len(label) == 0):
                    labels_list = []
                else:
                    labels = label.to(device, non_blocking=True)
                    labels_list = [labels]
                    batch_size = labels.shape[0]
            
            if use_single_head and is_multitask:
                expected_num_tasks = len(num_classes_per_task)
                if len(labels_list) > 0 and len(labels_list) != expected_num_tasks:
                    raise ValueError(
                        f"Expected {expected_num_tasks} label tensors for multi-task learning, "
                        f"but got {len(labels_list)}. Check your dataset and collate function."
                    )

            with torch.no_grad():
                outputs = encoder(clips, clip_indices)
                if not training:
                    outputs = [
                        [forward_classifier(c, o, conditioning_labels) for o in outputs]
                        for c in classifiers
                    ]
            if training:
                outputs = [
                    [forward_classifier(c, o, conditioning_labels) for o in outputs]
                    for c in classifiers
                ]

        has_labels = len(labels_list) > 0
        if batch_size is None:
            batch_size = 0
        global_batch = float(AllReduceSum.apply(torch.tensor(float(batch_size), device=device)))
        global_batch = max(global_batch, 1.0)

        if use_single_head and is_multitask:
            losses = []
            for coutputs in outputs:
                classifier_losses = []
                for o in coutputs:
                    task_losses = []
                    for task_idx in range(len(num_classes_per_task)):
                        task_key = f"task_{task_idx}"
                        if not has_labels or task_idx >= len(labels_list):
                            continue
                        task_loss = compute_task_loss(
                            o[task_key],
                            labels_list[task_idx],
                            loss_name=task_loss_names[task_idx],
                            class_weights=None if class_weights is None else class_weights[task_idx],
                            focal_gamma=focal_gamma,
                            sample_mask=(
                                conditioning_masks
                                if task_modes[task_idx] == "multilabel"
                                and conditioning_masks is not None
                                and o[task_key].ndim == labels_list[task_idx].ndim == 3
                                else None
                            ),
                        )
                        task_losses.append(task_loss)
                    if len(task_losses) > 0:
                        classifier_losses.append(sum(task_losses))
                    else:
                        classifier_losses.append(torch.tensor(0.0, device=device))
                losses.append(classifier_losses)
        else:
            if has_labels:
                losses = [
                    [
                        compute_task_loss(
                            o,
                            labels_list[0],
                            loss_name=loss_name,
                            class_weights=None if class_weights is None else class_weights[0],
                            focal_gamma=focal_gamma,
                        )
                        for o in coutputs
                    ]
                    for coutputs in outputs
                ]
            else:
                losses = [[torch.tensor(0.0, device=device) for _ in coutputs] for coutputs in outputs]

        with torch.no_grad():
            if use_single_head and is_multitask:
                for c_idx, coutputs in enumerate(outputs):
                    if not has_labels:
                        continue
                    batch_task_accs = []
                    batch_task_weights = []
                    exact_clip_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
                    exact_clip_valid_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

                    for task_idx in range(len(num_classes_per_task)):
                        task_key = f"task_{task_idx}"
                        if task_idx >= len(labels_list):
                            continue

                        if task_modes[task_idx] == "multilabel":
                            task_preds = [torch.sigmoid(o[task_key]) for o in coutputs]
                            task_prediction = sum(task_preds) / len(task_preds)
                            target_mask = labels_list[task_idx] > 0.5
                            task_threshold = multilabel_thresholds[task_idx]
                            pred_mask = task_prediction >= task_threshold

                            if target_mask.ndim == 3:
                                if conditioning_masks is None:
                                    valid_pair_mask = torch.ones(
                                        target_mask.shape[:2],
                                        dtype=torch.bool,
                                        device=device,
                                    )
                                else:
                                    valid_pair_mask = conditioning_masks
                                sample_exact = pred_mask.eq(target_mask).all(dim=-1)
                                task_correct = float(
                                    AllReduceSum.apply((sample_exact & valid_pair_mask).sum(dtype=torch.float32))
                                )
                                task_weight = float(
                                    AllReduceSum.apply(valid_pair_mask.sum(dtype=torch.float32))
                                )
                                task_weight = max(task_weight, 1.0)
                                valid_clip_mask = valid_pair_mask.any(dim=1)
                                exact_clip_mask &= (sample_exact | ~valid_pair_mask).all(dim=1)
                                exact_clip_valid_mask &= valid_clip_mask

                                valid_label_mask = valid_pair_mask.unsqueeze(-1)
                                tp = float(
                                    AllReduceSum.apply(
                                        (pred_mask & target_mask & valid_label_mask).sum(dtype=torch.float32)
                                    )
                                )
                                fp = float(
                                    AllReduceSum.apply(
                                        (pred_mask & ~target_mask & valid_label_mask).sum(dtype=torch.float32)
                                    )
                                )
                                fn = float(
                                    AllReduceSum.apply(
                                        (~pred_mask & target_mask & valid_label_mask).sum(dtype=torch.float32)
                                    )
                                )
                            else:
                                sample_exact = pred_mask.eq(target_mask).all(dim=1)
                                task_correct = float(AllReduceSum.apply(sample_exact.sum(dtype=torch.float32)))
                                task_weight = global_batch
                                exact_clip_mask &= sample_exact

                                tp = float(AllReduceSum.apply((pred_mask & target_mask).sum(dtype=torch.float32)))
                                fp = float(AllReduceSum.apply((pred_mask & ~target_mask).sum(dtype=torch.float32)))
                                fn = float(AllReduceSum.apply((~pred_mask & target_mask).sum(dtype=torch.float32)))

                            task_acc = 100.0 * task_correct / task_weight
                            task_top1_meters[c_idx][task_idx].update(task_acc, n=task_weight)
                            batch_task_accs.append(task_acc)
                            batch_task_weights.append(task_weight)
                            task_f1_stats[c_idx][task_idx]["tp"] += tp
                            task_f1_stats[c_idx][task_idx]["fp"] += fp
                            task_f1_stats[c_idx][task_idx]["fn"] += fn

                            if confusion_matrices is not None:
                                if target_mask.ndim == 3:
                                    confusion_target_mask = target_mask[valid_pair_mask]
                                    confusion_pred_mask = pred_mask[valid_pair_mask]
                                else:
                                    confusion_target_mask = target_mask
                                    confusion_pred_mask = pred_mask
                                confusion_matrices[c_idx][task_idx] += make_multilabel_confusion_matrices(
                                    confusion_target_mask,
                                    confusion_pred_mask,
                                )
                            continue

                        task_preds = [F.softmax(o[task_key], dim=1) for o in coutputs]
                        task_prediction = sum(task_preds) / len(task_preds)
                        target_labels = labels_list[task_idx].to(dtype=torch.long)
                        prediction_indices = task_prediction.max(dim=1).indices
                        task_correct_tensor = prediction_indices.eq(target_labels)
                        task_correct = float(AllReduceSum.apply(task_correct_tensor.sum(dtype=torch.float32)))
                        task_acc = 100.0 * task_correct / global_batch
                        task_top1_meters[c_idx][task_idx].update(task_acc, n=global_batch)
                        batch_task_accs.append(task_acc)
                        batch_task_weights.append(global_batch)
                        exact_clip_mask &= task_correct_tensor

                        if confusion_matrices is not None:
                            confusion_matrices[c_idx][task_idx] += make_confusion_matrix(
                                target_labels,
                                prediction_indices,
                                num_classes_per_task[task_idx],
                            )

                    if len(batch_task_accs) > 0:
                        avg_task_acc = sum(batch_task_accs) / len(batch_task_accs)
                        avg_task_weight = sum(batch_task_weights) / len(batch_task_weights)
                        top1_meters[c_idx].update(avg_task_acc, n=avg_task_weight)
                        exact_clip_valid_count = float(
                            AllReduceSum.apply(exact_clip_valid_mask.sum(dtype=torch.float32))
                        )
                        exact_clip_valid_count = max(exact_clip_valid_count, 1.0)
                        exact_match_acc = 100.0 * float(
                            AllReduceSum.apply(
                                (exact_clip_mask & exact_clip_valid_mask).sum(dtype=torch.float32)
                            )
                        ) / exact_clip_valid_count
                        exact_triplet_meters[c_idx].update(exact_match_acc, n=exact_clip_valid_count)
            else:
                if has_labels:
                    outputs_agg = [sum([F.softmax(o, dim=1) for o in coutputs]) / len(coutputs) for coutputs in outputs]
                    top1_accs = [100.0 * float(AllReduceSum.apply(coutputs.max(dim=1).indices.eq(labels_list[0]).sum(dtype=torch.float32))) / global_batch for coutputs in outputs_agg]
                    for t1m, t1a in zip(top1_meters, top1_accs):
                        t1m.update(t1a, n=global_batch)
                else:
                    pass
            
            loss_vals = [sum([float(AllReduce.apply(lij)) for lij in li]) / len(li) for li in losses]
            for lm, lv in zip(loss_meters, loss_vals):
                lm.update(lv, n=global_batch)

        if training:
            if has_labels:
                if scaler is not None:
                    [[s.scale(lij).backward() for lij in li] for s, li in zip(scaler, losses)]
                    [s.step(o) for s, o in zip(scaler, optimizer)]
                    [s.update() for s in scaler]
                else:
                    [[lij.backward() for lij in li] for li in losses]
                    [o.step() for o in optimizer]
                [o.zero_grad() for o in optimizer]
            else:
                pass

        _agg_top1 = np.array([t1m.avg for t1m in top1_meters])
        _agg_loss = np.array([lm.avg for lm in loss_meters])
        log_msg = (
            "[%5d] loss: %.3f [mem: %.2e]"
            % (
                itr,
                _agg_loss.max(),
                torch.cuda.max_memory_allocated() / 1024.0**2,
            )
        )

        if is_multitask and num_classes_per_task is not None:
            task_accs = []
            for task_idx in range(len(num_classes_per_task)):
                task_acc_avg = np.mean([task_top1_meters[c_idx][task_idx].avg for c_idx in range(len(classifiers))])
                task_accs.append(task_acc_avg)
            exact_triplet_acc = np.mean([exact_triplet_meters[c_idx].avg for c_idx in range(len(classifiers))])
            if multilabel_task_indices:
                task_precisions = []
                task_recalls = []
                task_f1s = []
                for task_idx in multilabel_task_indices:
                    classifier_precisions = []
                    classifier_recalls = []
                    classifier_f1s = []
                    for c_idx in range(len(classifiers)):
                        tp = task_f1_stats[c_idx][task_idx]["tp"]
                        fp = task_f1_stats[c_idx][task_idx]["fp"]
                        fn = task_f1_stats[c_idx][task_idx]["fn"]
                        precision, recall, f1 = compute_micro_prf(tp, fp, fn)
                        classifier_precisions.append(precision)
                        classifier_recalls.append(recall)
                        classifier_f1s.append(f1)
                    task_precisions.append(float(np.mean(classifier_precisions)))
                    task_recalls.append(float(np.mean(classifier_recalls)))
                    task_f1s.append(float(np.mean(classifier_f1s)))
                log_msg += f" | ExactClip={exact_triplet_acc:.1f}%"
                metric_parts = []
                for metric_idx, task_idx in enumerate(multilabel_task_indices):
                    metric_parts.append(
                        f"{task_names[task_idx]}: Exact={task_accs[task_idx]:.1f}% "
                        f"MicroF1={task_f1s[metric_idx]:.1f}% "
                        f"P={task_precisions[metric_idx]:.1f}% "
                        f"R={task_recalls[metric_idx]:.1f}%"
                    )
                log_msg += " | " + " | ".join(metric_parts)
            else:
                log_msg += f" | Exact={exact_triplet_acc:.1f}%"
                log_msg += " | Tasks: " + " ".join(
                    [f"{task_name}={acc:.1f}%" for task_name, acc in zip(task_names, task_accs)]
                )

        logger.info(log_msg)

    best_classifier_idx = int(np.argmax([top1_meter.avg for top1_meter in top1_meters]))
    metrics = {
        "avg_task_acc": float(top1_meters[best_classifier_idx].avg),
        "loss": float(loss_meters[best_classifier_idx].avg),
        "exact_triplet_acc": float(exact_triplet_meters[best_classifier_idx].avg),
        "task_accs": [],
    }

    if is_multitask and num_classes_per_task is not None:
        metrics["task_accs"] = [
            float(task_top1_meters[best_classifier_idx][task_idx].avg)
            for task_idx in range(len(num_classes_per_task))
        ]
        if multilabel_task_indices:
            metrics["task_f1s"] = []
            metrics["task_micro_f1s"] = []
            metrics["task_micro_precisions"] = []
            metrics["task_micro_recalls"] = []
            metrics["task_f1_names"] = [task_names[task_idx] for task_idx in multilabel_task_indices]
            for task_idx in multilabel_task_indices:
                tp = task_f1_stats[best_classifier_idx][task_idx]["tp"]
                fp = task_f1_stats[best_classifier_idx][task_idx]["fp"]
                fn = task_f1_stats[best_classifier_idx][task_idx]["fn"]
                precision, recall, f1 = compute_micro_prf(tp, fp, fn)
                metrics["task_f1s"].append(f1)
                metrics["task_micro_f1s"].append(f1)
                metrics["task_micro_precisions"].append(precision)
                metrics["task_micro_recalls"].append(recall)
        if confusion_matrices is not None:
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                for confusion_matrix in confusion_matrices[best_classifier_idx]:
                    dist.all_reduce(confusion_matrix, op=dist.ReduceOp.SUM)
            metrics["confusion_matrices"] = [
                confusion_matrix.detach().cpu().numpy()
                for confusion_matrix in confusion_matrices[best_classifier_idx]
            ]
    else:
        metrics["exact_triplet_acc"] = metrics["avg_task_acc"]

    return metrics


def load_checkpoint(device, r_path, classifiers, opt, scaler, val_only=False):
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    logger.info(f"read-path: {r_path}")

    pretrained_dict = checkpoint["classifiers"]
    msg = [c.load_state_dict(pd) for c, pd in zip(classifiers, pretrained_dict)]

    if val_only:
        logger.info(f"loaded pretrained classifier from epoch with msg: {msg}")
        return classifiers, opt, scaler, 0

    epoch = checkpoint["epoch"]
    logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")

    [o.load_state_dict(pd) for o, pd in zip(opt, checkpoint["opt"])]

    if scaler is not None:
        [s.load_state_dict(pd) for s, pd in zip(scaler, checkpoint["scaler"])]

    logger.info(f"loaded optimizers from epoch {epoch}")

    return classifiers, opt, scaler, epoch


def load_pretrained(encoder, pretrained, checkpoint_key="target_encoder"):
    logger.info(f"Loading pretrained model from {pretrained}")
    checkpoint = robust_checkpoint_loader(pretrained, map_location="cpu")
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint["encoder"]

    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f"key '{k}' could not be found in loaded state dict")
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f"{pretrained_dict[k].shape} | {v.shape}")
            logger.info(f"key '{k}' is of different shape in model and loaded state dict")
            exit(1)
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    print(encoder)
    logger.info(f"loaded pretrained model with msg: {msg}")
    logger.info(f"loaded pretrained encoder from epoch: {checkpoint['epoch']}\n path: {pretrained}")
    del checkpoint
    return encoder


DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type="VideoDataset",
    img_size=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    pin_memory=True,
    persistent_workers=True,
    dataloader_timeout=0,
    subset_file=None,
    normalization=None,
    use_multitask_wrapper=False,
    label_mode="multiclass",
    conditioning_mode=None,
    conditioning_task_idx=0,
    target_task_idx=1,
    multilabel_lookup_path=None,
    multilabel_label_fields=None,
    multilabel_condition_field="conditioning_label",
    multilabel_count_source_fields=None,
):
    persistent_workers = bool(persistent_workers and num_workers > 0)
    dataloader_timeout = int(dataloader_timeout)
    if normalization is None:
        normalization = DEFAULT_NORMALIZATION

    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=img_size,
        normalize=normalization,
    )

    if label_mode == "multilabel" and dataset_type.lower() == "videodataset":
        dataset, _, data_sampler = make_videodataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            duration=eval_duration,
            num_clips=num_segments,
            allow_clip_overlap=allow_segment_overlap,
            shared_transform=None,
            transform=transform,
            collator=None,
            num_workers=num_workers,
            pin_mem=pin_memory,
            persistent_workers=persistent_workers,
            world_size=world_size,
            rank=rank,
            deterministic=True,
            drop_last=False,
            return_sample_path=True,
        )

        if conditioning_mode == "multilabel_multi_condition_task":
            wrapped_dataset = MultiConditionMultiLabelTaskLabelWrapper(
                dataset,
                multilabel_lookup_path,
                multilabel_label_fields,
                multilabel_condition_field,
            )
        elif conditioning_mode == "multilabel_conditioned_task":
            wrapped_dataset = ConditionedMultiLabelTaskLabelWrapper(
                dataset,
                multilabel_lookup_path,
                multilabel_label_fields,
                multilabel_condition_field,
            )
        else:
            wrapped_dataset = MultiLabelTaskLabelWrapper(
                dataset,
                multilabel_lookup_path,
                multilabel_label_fields,
                count_source_fields=multilabel_count_source_fields,
            )

        data_loader = torch.utils.data.DataLoader(
            wrapped_dataset,
            sampler=data_sampler,
            batch_size=batch_size,
            collate_fn=multitask_collate_fn,
            drop_last=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            timeout=dataloader_timeout,
        )

        logger.info(
            "Applied %s to dataset with custom collate function",
            type(wrapped_dataset).__name__,
        )
        return data_loader, data_sampler

    if conditioning_mode == "triplet_conditioned_task" and dataset_type.lower() == "videodataset":
        dataset, _, data_sampler = make_videodataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            duration=eval_duration,
            num_clips=num_segments,
            allow_clip_overlap=allow_segment_overlap,
            shared_transform=None,
            transform=transform,
            collator=None,
            num_workers=num_workers,
            pin_mem=pin_memory,
            persistent_workers=persistent_workers,
            world_size=world_size,
            rank=rank,
            deterministic=True,
            drop_last=False,
        )

        wrapped_dataset = ConditionedTaskFromTripletWrapper(
            dataset,
            conditioning_task_idx=conditioning_task_idx,
            target_task_idx=target_task_idx,
        )

        data_loader = torch.utils.data.DataLoader(
            wrapped_dataset,
            sampler=data_sampler,
            batch_size=batch_size,
            collate_fn=multitask_collate_fn,
            drop_last=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            timeout=dataloader_timeout,
        )

        logger.info(
            "Applied ConditionedTaskFromTripletWrapper with conditioning_task_idx=%s target_task_idx=%s",
            conditioning_task_idx,
            target_task_idx,
        )
        return data_loader, data_sampler

    if use_multitask_wrapper and dataset_type.lower() == "videodataset":
        dataset, data_loader, data_sampler = make_videodataset(
            data_paths=root_path,
            batch_size=batch_size,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            duration=eval_duration,
            num_clips=num_segments,
            allow_clip_overlap=allow_segment_overlap,
            shared_transform=None,
            transform=transform,
            collator=None,
            num_workers=num_workers,
            pin_mem=pin_memory,
            persistent_workers=persistent_workers,
            world_size=world_size,
            rank=rank,
            deterministic=True,
            drop_last=False,
        )
        
        wrapped_dataset = MultiTaskLabelWrapper(dataset)
        
        data_loader = torch.utils.data.DataLoader(
            wrapped_dataset,
            sampler=data_sampler,
            batch_size=batch_size,
            collate_fn=multitask_collate_fn,
            drop_last=False,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            timeout=dataloader_timeout,
        )
        
        logger.info("Applied MultiTaskLabelWrapper to dataset with custom collate function")
        return data_loader, data_sampler
    else:
        data_loader, data_sampler = init_data(
            data=dataset_type,
            root_path=root_path,
            transform=transform,
            batch_size=batch_size,
            world_size=world_size,
            rank=rank,
            clip_len=frames_per_clip,
            frame_sample_rate=frame_step,
            duration=eval_duration,
            num_clips=num_segments,
            allow_clip_overlap=allow_segment_overlap,
            num_workers=num_workers,
            drop_last=False,
            subset_file=subset_file,
        )
        return data_loader, data_sampler


def init_opt(classifiers, iterations_per_epoch, opt_kwargs, num_epochs, use_bfloat16=False):
    optimizers, schedulers, wd_schedulers = [], [], []
    scalers = None
    for c, kwargs in zip(classifiers, opt_kwargs):
        param_groups = [
            {
                "params": (p for n, p in c.named_parameters()),
                "mc_warmup_steps": int(kwargs.get("warmup") * iterations_per_epoch),
                "mc_start_lr": kwargs.get("start_lr"),
                "mc_ref_lr": kwargs.get("ref_lr"),
                "mc_final_lr": kwargs.get("final_lr"),
                "mc_ref_wd": kwargs.get("ref_wd"),
                "mc_final_wd": kwargs.get("final_wd"),
            }
        ]
        logger.info("Using AdamW")
        optimizers += [torch.optim.AdamW(param_groups)]
        schedulers += [WarmupCosineLRSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        wd_schedulers += [CosineWDSchedule(optimizers[-1], T_max=int(num_epochs * iterations_per_epoch))]
        if scalers is not None:
            scalers.append(torch.cuda.amp.GradScaler())
    return optimizers, scalers, schedulers, wd_schedulers


class WarmupCosineLRSchedule(object):

    def __init__(self, optimizer, T_max, last_epoch=-1):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        for group in self.optimizer.param_groups:
            ref_lr = group.get("mc_ref_lr")
            final_lr = group.get("mc_final_lr")
            start_lr = group.get("mc_start_lr")
            warmup_steps = group.get("mc_warmup_steps")
            T_max = self.T_max - warmup_steps
            if self._step < warmup_steps:
                progress = float(self._step) / float(max(1, warmup_steps))
                new_lr = start_lr + progress * (ref_lr - start_lr)
            else:
                progress = float(self._step - warmup_steps) / float(max(1, T_max))
                new_lr = max(
                    final_lr,
                    final_lr + (ref_lr - final_lr) * 0.5 * (1.0 + math.cos(math.pi * progress)),
                )
            group["lr"] = new_lr


class CosineWDSchedule(object):

    def __init__(self, optimizer, T_max):
        self.optimizer = optimizer
        self.T_max = T_max
        self._step = 0.0

    def step(self):
        self._step += 1
        progress = self._step / self.T_max

        for group in self.optimizer.param_groups:
            ref_wd = group.get("mc_ref_wd")
            final_wd = group.get("mc_final_wd")
            new_wd = final_wd + (ref_wd - final_wd) * 0.5 * (1.0 + math.cos(math.pi * progress))
            if final_wd <= ref_wd:
                new_wd = max(final_wd, new_wd)
            else:
                new_wd = min(final_wd, new_wd)
            group["weight_decay"] = new_wd
