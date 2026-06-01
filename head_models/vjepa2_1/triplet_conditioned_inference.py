#!/path/to/data_root/vjepa2_1/.venv/bin/python
from __future__ import annotations

import argparse
import contextlib
import math
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="Importing from timm.models.layers is deprecated.*",
    category=FutureWarning,
)

import cv2
import numpy as np
import torch
import yaml

from evals.triplet_recog_frozen.models import (
    ConditionedTokenAggregationMultiTaskClassifier,
    PooledFeatureMultiTaskClassifier,
    SingleHeadMultiTaskClassifier,
    TokenAggregationMultiTaskClassifier,
    init_module as init_encoder_module,
)
from evals.video_classification_frozen.utils import make_transforms
from src.utils.checkpoint_loader import robust_checkpoint_loader


DEFAULT_INPUT_VIDEO = Path(
    "/path/to/data_root/globus/surgenet_robotic/cholestectomy/segment_033.mp4"
)
DEFAULT_TOOL_CONFIG = Path(
    "/path/to/data_root/vjepa2_1/configs/heads/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool4.yaml"
)
DEFAULT_VERB_CONFIG = Path(
    "/path/to/data_root/vjepa2_1/configs/heads/verb_multilabel_conditioned_on_tool_train_vitl_latest_token_aggregation_tool4.yaml"
)
DEFAULT_TARGET_CONFIG = Path(
    "/path/to/data_root/vjepa2_1/configs/heads/target_multilabel_conditioned_on_tool_verb_train_vitl_latest_token_aggregation_tool4.yaml"
)
DEFAULT_NORMALIZATION = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
DEFAULT_CLIP_SECONDS = 4.0
DEFAULT_STRIDE_SECONDS = 1.0
DEFAULT_CLIP_FPS = 4.0
FALLBACK_TOOL_NAMES = ["grasper", "hook", "irrigator", "scissors"]
FALLBACK_VERB_NAMES = ["coagulation", "grasp/retract", "null", "cut", "dissect", "clean"]
FALLBACK_TARGET_NAMES = [
    "connective tissue",
    "cystic duct",
    "adhesion",
    "cystic pedicle",
    "gallbladder",
    "gallbladder wall",
    "liver",
    "null",
    "peritoneum",
    "suture",
    "cystic artery",
    "fallciform ligament",
    "gut",
    "omentum",
    "specimen bag",
    "fluid",
]
DISPLAY_TOOL_MAX_COUNTS = {
    "grasper": 2,
    "hook": 1,
    "scissors": 1,
}
PREFERRED_BEST_HEAD_FILENAMES = {
    "tool": ("best_val_tool_micro_f1.pt", "best_epoch_0006_val_tool_f1.pt"),
    "verb": ("best_val_verb_micro_f1.pt", "best_epoch_0006_val_verb_f1.pt"),
    "action": (
        "best_val_action_micro_f1.pt",
        "best_val_verb_micro_f1.pt",
        "best_epoch_0006_val_verb_f1.pt",
    ),
    "target": ("best_val_target_micro_f1.pt", "best_epoch_0005_val_target_f1.pt"),
    "triplet": ("best_val_triplet_micro_f1.pt",),
    "mean": ("best_val_mean_micro_f1.pt",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chained Tool4 multi-label tool -> verb/action -> target inference on a single video."
    )
    parser.add_argument("--input-video", type=Path, default=DEFAULT_INPUT_VIDEO)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument("--tool-config", type=Path, default=DEFAULT_TOOL_CONFIG)
    parser.add_argument("--verb-config", type=Path, default=DEFAULT_VERB_CONFIG)
    parser.add_argument("--target-config", type=Path, default=DEFAULT_TARGET_CONFIG)
    parser.add_argument("--tool-checkpoint", type=Path, default=None)
    parser.add_argument("--verb-checkpoint", type=Path, default=None)
    parser.add_argument("--target-checkpoint", type=Path, default=None)
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
        help="With --head-checkpoint-policy best, fail instead of falling back to latest.pt when no best checkpoint exists.",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        "--encoder-checkpoint",
        dest="backbone_checkpoint",
        type=Path,
        default=None,
        help="Override only the frozen encoder/backbone checkpoint; the three head checkpoints are unchanged.",
    )
    parser.add_argument(
        "--backbone-checkpoint-key",
        "--encoder-checkpoint-key",
        dest="backbone_checkpoint_key",
        type=str,
        default=None,
        help=(
            "Override the encoder key inside --backbone-checkpoint. "
            "If omitted, the script keeps the config key when present, otherwise tries target_encoder then encoder."
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
    parser.add_argument("--verbose", action="store_true", help="Print setup and per-window progress.")
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.load(handle, Loader=yaml.FullLoader)


def resolve_output_path(input_video: Path, output_video: Path | None) -> Path:
    if output_video is not None:
        return output_video
    return input_video.with_name(f"{input_video.stem}_tool4_multilabel_inference.mp4")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_arg)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
    return device


@contextlib.contextmanager
def maybe_suppress_stdout(verbose: bool):
    if verbose:
        yield
        return
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def get_checkpoint_run_dir(config: dict[str, Any]) -> Path:
    folder = Path(config["folder"])
    tag = str(config.get("tag", "")).strip()
    if not tag:
        raise ValueError("Could not derive checkpoint path because config.tag is empty")
    return folder / "video_classification_frozen" / tag


def get_head_task_name(config: dict[str, Any], fallback_name: str) -> str:
    task_names = config.get("experiment", {}).get("data", {}).get("task_names") or []
    if task_names:
        return str(task_names[0]).strip().lower()
    return fallback_name


def find_best_head_checkpoint(run_dir: Path, task_name: str) -> Path | None:
    for filename in PREFERRED_BEST_HEAD_FILENAMES.get(task_name, ()):
        candidate = run_dir / filename
        if candidate.exists():
            return candidate

    generic_candidates = sorted(
        (path for path in run_dir.glob("best*.pt") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if generic_candidates:
        return generic_candidates[0]
    return None


def derive_checkpoint_path(
    config: dict[str, Any],
    explicit_path: Path | None,
    fallback_name: str,
    checkpoint_policy: str = "best",
    strict_best: bool = False,
) -> Path:
    if explicit_path is not None:
        return explicit_path

    run_dir = get_checkpoint_run_dir(config)
    latest_path = run_dir / "latest.pt"
    if checkpoint_policy == "latest":
        return latest_path

    if checkpoint_policy != "best":
        raise ValueError(f"Unsupported head checkpoint policy: {checkpoint_policy}")

    task_name = get_head_task_name(config, fallback_name)
    best_path = find_best_head_checkpoint(run_dir, task_name)
    if best_path is not None:
        return best_path

    if strict_best:
        preferred = ", ".join(PREFERRED_BEST_HEAD_FILENAMES.get(task_name, ())) or "best*.pt"
        raise FileNotFoundError(
            f"No best checkpoint found in {run_dir}. Expected one of: {preferred}. "
            "Run without --strict-best-head-checkpoints to fall back to latest.pt."
        )
    return latest_path


def resolve_multilabel_threshold(config: dict[str, Any], explicit_value: float | None) -> float:
    if explicit_value is not None:
        return float(explicit_value)
    return float(config["experiment"]["data"].get("multilabel_threshold", 0.5))


def class_names_from_config(
    config: dict[str, Any],
    fallback_names: list[str],
    fallback_prefix: str,
) -> list[str]:
    data_cfg = config["experiment"]["data"]
    num_classes_per_task = data_cfg.get("num_classes_per_task")
    if num_classes_per_task is None:
        expected_count = int(data_cfg.get("num_classes", len(fallback_names)))
    else:
        expected_count = int(num_classes_per_task[0])

    class_names_per_task = data_cfg.get("class_names_per_task") or []
    if class_names_per_task and class_names_per_task[0]:
        names = [str(name) for name in class_names_per_task[0]]
    else:
        names = list(fallback_names)

    if len(names) < expected_count:
        names.extend(f"{fallback_prefix}_{idx}" for idx in range(len(names), expected_count))
    return names[:expected_count]


def compare_encoder_settings(reference_cfg: dict[str, Any], candidate_cfg: dict[str, Any], name: str) -> None:
    reference_model = reference_cfg["model_kwargs"]
    candidate_model = candidate_cfg["model_kwargs"]
    reference_data = reference_cfg["experiment"]["data"]
    candidate_data = candidate_cfg["experiment"]["data"]
    mismatches = []
    if reference_model["module_name"] != candidate_model["module_name"]:
        mismatches.append("module_name")
    if reference_model["checkpoint"] != candidate_model["checkpoint"]:
        mismatches.append("checkpoint")
    if reference_model["pretrain_kwargs"] != candidate_model["pretrain_kwargs"]:
        mismatches.append("pretrain_kwargs")
    if reference_model["wrapper_kwargs"] != candidate_model["wrapper_kwargs"]:
        mismatches.append("wrapper_kwargs")
    if reference_data.get("frames_per_clip", 16) != candidate_data.get("frames_per_clip", 16):
        mismatches.append("frames_per_clip")
    if reference_data.get("resolution", 224) != candidate_data.get("resolution", 224):
        mismatches.append("resolution")
    if mismatches:
        raise ValueError(f"Config '{name}' does not match the shared backbone setup: {mismatches}")


def resolve_backbone_checkpoint_key(
    checkpoint_path: Path,
    configured_key: str | None,
    explicit_key: str | None,
) -> str | None:
    if explicit_key:
        return explicit_key

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return configured_key

    if configured_key and configured_key in checkpoint:
        return configured_key

    for candidate_key in ("ema_encoder", "target_encoder", "encoder", "model", "state_dict"):
        if candidate_key in checkpoint:
            return candidate_key

    return configured_key


def override_backbone_checkpoint(
    configs: list[dict[str, Any]],
    checkpoint_path: Path | None,
    checkpoint_key: str | None = None,
) -> None:
    if checkpoint_path is None:
        return
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Backbone checkpoint does not exist: {checkpoint_path}")

    checkpoint_value = str(checkpoint_path)
    configured_key = configs[0]["model_kwargs"]["pretrain_kwargs"]["encoder"].get("checkpoint_key")
    resolved_checkpoint_key = resolve_backbone_checkpoint_key(
        checkpoint_path=checkpoint_path,
        configured_key=configured_key,
        explicit_key=checkpoint_key,
    )
    for config in configs:
        config["model_kwargs"]["checkpoint"] = checkpoint_value
        if resolved_checkpoint_key:
            config["model_kwargs"]["pretrain_kwargs"]["encoder"]["checkpoint_key"] = resolved_checkpoint_key
        if "pretrain" in config and isinstance(config["pretrain"], dict):
            config["pretrain"]["checkpoint"] = checkpoint_value


def build_classifier_from_config(config: dict[str, Any], embed_dim: int) -> torch.nn.Module:
    classifier_cfg = config["experiment"]["classifier"]
    data_cfg = config["experiment"]["data"]
    num_classes_per_task = data_cfg.get("num_classes_per_task")
    if num_classes_per_task is None:
        num_classes_per_task = [int(data_cfg["num_classes"])]

    head_type = classifier_cfg.get("head_type", "single_head_pooler")
    if head_type == "single_head_pooler":
        return SingleHeadMultiTaskClassifier(
            num_classes_per_task=num_classes_per_task,
            embed_dim=embed_dim,
            num_heads=int(classifier_cfg.get("num_heads", 16)),
            depth=int(classifier_cfg.get("num_probe_blocks", 1)),
            use_activation_checkpointing=False,
        )
    if head_type == "pooled_feature":
        return PooledFeatureMultiTaskClassifier(
            num_classes_per_task=num_classes_per_task,
            embed_dim=embed_dim,
            feature_pool=classifier_cfg.get("feature_pool", "mean"),
            num_heads=int(classifier_cfg.get("num_heads", 16)),
            depth=int(classifier_cfg.get("num_probe_blocks", 1)),
            classifier_hidden_dims=classifier_cfg.get("classifier_hidden_dims"),
            use_layer_norm=bool(classifier_cfg.get("feature_use_layer_norm", True)),
            use_activation_checkpointing=False,
        )
    if head_type == "token_aggregation":
        return TokenAggregationMultiTaskClassifier(
            num_classes_per_task=num_classes_per_task,
            embed_dim=embed_dim,
            token_pool=classifier_cfg.get("token_pool", "max"),
            token_pool_topk=classifier_cfg.get("token_pool_topk"),
            use_layer_norm=bool(classifier_cfg.get("token_use_layer_norm", True)),
        )
    if head_type == "conditioned_token_aggregation":
        return ConditionedTokenAggregationMultiTaskClassifier(
            num_classes_per_task=num_classes_per_task,
            embed_dim=embed_dim,
            num_condition_classes=classifier_cfg["condition_num_classes"],
            condition_embed_dim=classifier_cfg.get("condition_embed_dim"),
            token_pool=classifier_cfg.get("token_pool", "max"),
            token_pool_topk=classifier_cfg.get("token_pool_topk"),
            use_layer_norm=bool(classifier_cfg.get("token_use_layer_norm", True)),
        )
    raise ValueError(f"Unsupported head_type '{head_type}' for conditioned inference")


def normalize_classifier_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = {}
    for key, value in state_dict.items():
        normalized_key = key[len("module.") :] if key.startswith("module.") else key
        normalized[normalized_key] = value

    if "condition_embedding.weight" in normalized and "condition_embeddings.0.weight" not in normalized:
        normalized["condition_embeddings.0.weight"] = normalized.pop("condition_embedding.weight")
    if "condition_projection.weight" in normalized and "condition_projections.0.weight" not in normalized:
        normalized["condition_projections.0.weight"] = normalized.pop("condition_projection.weight")
    if "condition_projection.bias" in normalized and "condition_projections.0.bias" not in normalized:
        normalized["condition_projections.0.bias"] = normalized.pop("condition_projection.bias")
    return normalized


def load_classifier_weights(classifier: torch.nn.Module, checkpoint_path: Path) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint does not exist: {checkpoint_path}")
    checkpoint = robust_checkpoint_loader(str(checkpoint_path), map_location=torch.device("cpu"))
    if "classifiers" not in checkpoint or not checkpoint["classifiers"]:
        raise ValueError(f"Checkpoint does not contain classifier weights: {checkpoint_path}")
    state_dict = normalize_classifier_state_dict(checkpoint["classifiers"][0])
    classifier.load_state_dict(state_dict, strict=True)


def build_transform(config: dict[str, Any]):
    data_cfg = config["experiment"]["data"]
    normalization = data_cfg.get("normalization") or DEFAULT_NORMALIZATION
    resolution = int(data_cfg.get("resolution", 224))
    return make_transforms(
        training=False,
        num_views_per_clip=1,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=resolution,
        normalize=normalization,
    )


def build_window_starts(
    num_frames: int,
    source_fps: float,
    clip_seconds: float,
    stride_seconds: float,
    clip_fps: float,
) -> list[float]:
    if num_frames <= 0:
        return [0.0]

    last_sample_offset = max(0.0, clip_seconds - (1.0 / clip_fps))
    last_possible_start = max(0.0, ((num_frames - 1) / source_fps) - last_sample_offset)

    starts = [0.0]
    while starts[-1] + stride_seconds <= last_possible_start + 1e-6:
        starts.append(round(starts[-1] + stride_seconds, 6))

    if abs(starts[-1] - last_possible_start) > 1e-4:
        starts.append(round(last_possible_start, 6))

    deduped = []
    for start in starts:
        if not deduped or abs(start - deduped[-1]) > 1e-4:
            deduped.append(start)
    return deduped


class SequentialVideoSampler:
    """
    Decode source frames in increasing order and cache only sampled frames.

    Decord random seeking can be noisy on long concatenated MP4s. Inference
    windows are processed in chronological order, so sequential decoding avoids
    those seek warnings while preserving the same sampled frame indices.
    """

    def __init__(self, video_path: Path):
        self.video_path = Path(video_path)
        self.capture = cv2.VideoCapture(str(self.video_path))
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open input video: {self.video_path}")

        self.source_fps = float(self.capture.get(cv2.CAP_PROP_FPS))
        if self.source_fps <= 0:
            raise RuntimeError(f"Could not read a positive FPS from: {self.video_path}")
        self.num_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.next_frame_idx = 0
        self.cache: dict[int, np.ndarray] = {}

    def close(self) -> None:
        self.capture.release()
        self.cache.clear()

    def _read_until_cached(self, frame_idx: int) -> None:
        frame_idx = int(frame_idx)
        if frame_idx in self.cache:
            return
        if frame_idx < self.next_frame_idx:
            raise RuntimeError(
                f"Frame {frame_idx} was already decoded and is no longer cached. "
                "Inference windows must be processed in increasing time order."
            )

        while self.next_frame_idx <= frame_idx:
            ok, frame_bgr = self.capture.read()
            if not ok:
                raise RuntimeError(
                    f"Could not decode frame {self.next_frame_idx} from {self.video_path}"
                )
            self.cache[self.next_frame_idx] = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            self.next_frame_idx += 1

    def sample_window(
        self,
        start_time: float,
        frames_per_clip: int,
        clip_fps: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        sample_times = start_time + (np.arange(frames_per_clip, dtype=np.float32) / float(clip_fps))
        frame_indices = np.rint(sample_times * self.source_fps).astype(np.int64)
        frame_indices = np.clip(frame_indices, 0, max(0, self.num_frames - 1))

        min_needed = int(frame_indices.min()) if frame_indices.size else 0
        self.cache = {
            cached_idx: frame
            for cached_idx, frame in self.cache.items()
            if cached_idx >= min_needed
        }

        for frame_idx in sorted(set(int(idx) for idx in frame_indices.tolist())):
            self._read_until_cached(frame_idx)

        frames = np.stack([self.cache[int(idx)] for idx in frame_indices], axis=0)
        return frames, frame_indices


def validate_temporal_sampling(frames_per_clip: int, clip_fps: float, clip_seconds: float) -> None:
    if clip_fps <= 0:
        raise ValueError(f"--clip-fps must be positive, got {clip_fps}")
    expected_clip_seconds = float(frames_per_clip) / float(clip_fps)
    if not math.isclose(float(clip_seconds), expected_clip_seconds, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError(
            "Temporal sampling mismatch: "
            f"frames_per_clip={frames_per_clip} at clip_fps={clip_fps:.4f} samples "
            f"{expected_clip_seconds:.4f}s, but --clip-seconds is {clip_seconds:.4f}s. "
            "Use --clip-fps 4 --clip-seconds 4 for 16 frames sampled at 4 fps."
        )


def prepare_encoder_inputs(
    frames_rgb: np.ndarray,
    frame_indices: np.ndarray,
    transform,
    device: torch.device,
) -> tuple[list[list[torch.Tensor]], list[torch.Tensor]]:
    transformed_views = transform(frames_rgb)
    if len(transformed_views) != 1:
        raise ValueError(f"Expected exactly one spatial view, got {len(transformed_views)}")

    clip_tensor = transformed_views[0].unsqueeze(0).to(device, non_blocking=True)
    clip_indices = torch.from_numpy(frame_indices.astype(np.int64)).unsqueeze(0).to(device, non_blocking=True)
    return [[clip_tensor]], [clip_indices]


def select_multilabel_indices(
    probabilities: torch.Tensor,
    threshold: float,
    max_count: int | None = None,
    fallback_top1: bool = True,
) -> list[int]:
    probabilities = probabilities.detach().float()
    selected = torch.nonzero(probabilities >= threshold, as_tuple=False).flatten().tolist()
    if not selected and fallback_top1 and probabilities.numel() > 0:
        selected = [int(torch.argmax(probabilities).item())]
    selected = sorted(selected, key=lambda idx: float(probabilities[idx].item()), reverse=True)
    if max_count is not None:
        selected = selected[: max(0, int(max_count))]
    return [int(idx) for idx in selected]


def lookup_name(names: list[str], class_id: int, prefix: str) -> str:
    if 0 <= class_id < len(names):
        return names[class_id]
    return f"{prefix}_{class_id}"


def run_chained_prediction(
    encoder: torch.nn.Module,
    tool_head: torch.nn.Module,
    verb_head: torch.nn.Module,
    target_head: torch.nn.Module,
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
) -> dict[str, Any]:
    with torch.no_grad():
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            encoded_outputs = encoder(clips, clip_indices)
            encoded_clip = encoded_outputs[0]

            tool_logits = tool_head(encoded_clip)["task_0"][0]
            tool_probs = torch.sigmoid(tool_logits.float())
            tool_ids = select_multilabel_indices(
                tool_probs,
                threshold=tool_threshold,
                max_count=max_tools,
                fallback_top1=True,
            )

            tools = [
                {
                    "tool_id": tool_id,
                    "tool_name": lookup_name(tool_names, tool_id, "tool"),
                    "tool_probability": float(tool_probs[tool_id].item()),
                }
                for tool_id in tool_ids
            ]

            condition_pairs = []
            if tool_ids:
                tool_conditions = torch.tensor(
                    [[[tool_id] for tool_id in tool_ids]],
                    device=device,
                    dtype=torch.long,
                )
                verb_logits = verb_head(encoded_clip, tool_conditions)["task_0"][0]
                verb_probs_by_tool = torch.sigmoid(verb_logits.float())

                for tool_row, tool_id in enumerate(tool_ids):
                    verb_probs = verb_probs_by_tool[tool_row]
                    verb_ids = select_multilabel_indices(
                        verb_probs,
                        threshold=verb_threshold,
                        max_count=max_verbs_per_tool,
                        fallback_top1=True,
                    )
                    for verb_id in verb_ids:
                        condition_pairs.append(
                            {
                                "tool_id": tool_id,
                                "tool_name": lookup_name(tool_names, tool_id, "tool"),
                                "tool_probability": float(tool_probs[tool_id].item()),
                                "verb_id": verb_id,
                                "verb_name": lookup_name(verb_names, verb_id, "verb"),
                                "verb_probability": float(verb_probs[verb_id].item()),
                            }
                        )

            triplets = []
            if condition_pairs:
                target_conditions = torch.tensor(
                    [[[pair["tool_id"], pair["verb_id"]] for pair in condition_pairs]],
                    device=device,
                    dtype=torch.long,
                )
                target_logits = target_head(encoded_clip, target_conditions)["task_0"][0]
                target_probs_by_pair = torch.sigmoid(target_logits.float())

                for pair_idx, pair in enumerate(condition_pairs):
                    target_probs = target_probs_by_pair[pair_idx]
                    target_ids = select_multilabel_indices(
                        target_probs,
                        threshold=target_threshold,
                        max_count=max_targets_per_pair,
                        fallback_top1=True,
                    )
                    for target_id in target_ids:
                        target_probability = float(target_probs[target_id].item())
                        score = (
                            pair["tool_probability"]
                            * pair["verb_probability"]
                            * target_probability
                        )
                        triplet_name = (
                            f"{pair['tool_name']} | {pair['verb_name']} | "
                            f"{lookup_name(target_names, target_id, 'target')}"
                        )
                        triplets.append(
                            {
                                **pair,
                                "target_id": target_id,
                                "target_name": lookup_name(target_names, target_id, "target"),
                                "target_probability": target_probability,
                                "score": float(score),
                                "triplet_name": triplet_name,
                            }
                        )

            triplets.sort(key=lambda item: item["score"], reverse=True)

    return {
        "start_time": float(start_time),
        "end_time": float(start_time + clip_seconds),
        "tools": tools,
        "triplets": triplets,
    }


def fit_text_to_width(
    text: str,
    max_width: int,
    font,
    base_scale: float,
    min_scale: float,
    thickness: int,
) -> tuple[str, float]:
    scale = float(base_scale)
    while scale > min_scale:
        text_width = cv2.getTextSize(text, font, scale, thickness)[0][0]
        if text_width <= max_width:
            return text, scale
        scale -= 0.03

    if cv2.getTextSize(text, font, min_scale, thickness)[0][0] <= max_width:
        return text, min_scale

    suffix = "..."
    trimmed = text
    while trimmed:
        candidate = trimmed.rstrip() + suffix
        if cv2.getTextSize(candidate, font, min_scale, thickness)[0][0] <= max_width:
            return candidate, min_scale
        trimmed = trimmed[:-1]
    return suffix, min_scale


def build_prediction_cards(
    prediction: dict[str, Any],
    max_triplets_display: int,
) -> list[tuple[str, str]]:
    triplets = prediction.get("triplets", [])
    display_count = min(3, max(1, int(max_triplets_display)))
    if not triplets:
        return [("No chained prediction", "")]

    cards = []
    display_triplets = select_display_triplets(triplets, display_count)
    for triplet_idx, triplet in enumerate(display_triplets, start=1):
        chain = (
            f"{triplet_idx}. {triplet['tool_name']} -> "
            f"{triplet['verb_name']} -> {triplet['target_name']}"
        )
        confidence = (
            f"tool {triplet['tool_probability'] * 100.0:.0f}%   "
            f"action {triplet['verb_probability'] * 100.0:.0f}%   "
            f"target {triplet['target_probability'] * 100.0:.0f}%   "
            f"score {triplet['score'] * 100.0:.1f}%"
        )
        cards.append((chain, confidence))
    return cards


def select_display_triplets(
    triplets: list[dict[str, Any]],
    display_count: int,
) -> list[dict[str, Any]]:
    selected = []
    tool_counts: Counter[str] = Counter()
    display_count = max(0, int(display_count))

    for triplet in sorted(triplets, key=lambda item: item["score"], reverse=True):
        tool_name = str(triplet.get("tool_name", "")).strip().lower()
        max_for_tool = DISPLAY_TOOL_MAX_COUNTS.get(tool_name)
        if max_for_tool is not None and tool_counts[tool_name] >= max_for_tool:
            continue

        selected.append(triplet)
        tool_counts[tool_name] += 1
        if len(selected) >= display_count:
            break

    return selected


def draw_overlay(
    frame_bgr: np.ndarray,
    prediction: dict[str, Any],
    frame_time: float,
    max_triplets_display: int,
) -> np.ndarray:
    annotated = frame_bgr.copy()
    height, width = annotated.shape[:2]
    if width <= 0 or height <= 0:
        return annotated

    cards = build_prediction_cards(prediction, max_triplets_display)
    cards = cards[:3]
    if not cards:
        return annotated

    margin = max(8, min(width, height) // 70)
    panel_left = margin
    panel_top = margin
    panel_width = min(width - (2 * margin), max(280, min(620, int(width * 0.42))))
    if panel_width <= 0:
        return annotated

    panel_right = panel_left + panel_width
    inner_pad = max(7, min(10, panel_width // 70))
    card_gap = max(4, min(7, height // 180))
    card_height = 39
    max_panel_height = height - (2 * margin)
    panel_height = inner_pad * 2 + len(cards) * card_height + (len(cards) - 1) * card_gap
    if panel_height > max_panel_height:
        shrink = max_panel_height / max(panel_height, 1)
        card_height = max(31, int(card_height * shrink))
        card_gap = max(3, int(card_gap * shrink))
        inner_pad = max(5, int(inner_pad * shrink))
        panel_height = inner_pad * 2 + len(cards) * card_height + (len(cards) - 1) * card_gap

    panel_bottom = min(height - margin, panel_top + panel_height)
    panel_height = panel_bottom - panel_top
    if panel_height <= 0:
        return annotated

    overlay = annotated.copy()
    cv2.rectangle(
        overlay,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (12, 18, 22),
        thickness=-1,
    )
    card_colors = [
        (74, 204, 255),
        (126, 226, 143),
        (255, 191, 94),
    ]
    card_fill = (24, 31, 36)
    card_left = panel_left + inner_pad
    card_right = panel_right - inner_pad
    card_width = max(1, card_right - card_left)
    y = panel_top + inner_pad
    for card_idx, _ in enumerate(cards):
        cv2.rectangle(
            overlay,
            (card_left, y),
            (card_right, min(panel_bottom - inner_pad, y + card_height)),
            card_fill,
            thickness=-1,
        )
        cv2.rectangle(
            overlay,
            (card_left, y),
            (card_left + 4, min(panel_bottom - inner_pad, y + card_height)),
            card_colors[card_idx % len(card_colors)],
            thickness=-1,
        )
        y += card_height + card_gap

    annotated = cv2.addWeighted(overlay, 0.48, annotated, 0.52, 0.0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    line1_scale = 0.40
    line2_scale = 0.31
    line1_thickness = 1
    line2_thickness = 1
    text_left = card_left + 10
    text_width = max(1, card_width - 18)
    y = panel_top + inner_pad
    for card_idx, (chain, confidence) in enumerate(cards):
        accent = card_colors[card_idx % len(card_colors)]
        chain, scale1 = fit_text_to_width(
            chain,
            text_width,
            font,
            base_scale=line1_scale,
            min_scale=0.28,
            thickness=line1_thickness,
        )
        confidence, scale2 = fit_text_to_width(
            confidence,
            text_width,
            font,
            base_scale=line2_scale,
            min_scale=0.24,
            thickness=line2_thickness,
        )
        first_baseline = min(y + 15, panel_bottom - 5)
        second_baseline = min(y + 31, panel_bottom - 5)
        cv2.putText(
            annotated,
            chain,
            (text_left, first_baseline),
            font,
            scale1,
            (245, 249, 250),
            line1_thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            confidence,
            (text_left, second_baseline),
            font,
            scale2,
            accent,
            line2_thickness,
            cv2.LINE_AA,
        )
        y += card_height + card_gap
    return annotated


def print_counter_distribution(
    title: str,
    counter: Counter[str],
    num_windows: int,
    top_k: int,
) -> None:
    print(title)
    if not counter:
        print("  none")
        return

    limit = max(1, int(top_k))
    for label, count in counter.most_common(limit):
        window_percent = 100.0 * float(count) / max(float(num_windows), 1.0)
        print(f"  {label}: {count} ({window_percent:.1f}% of windows)")


def print_prediction_distribution(predictions: list[dict[str, Any]], top_k: int = 10) -> None:
    num_windows = len(predictions)
    tool_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    tool_action_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    triplet_counts: Counter[str] = Counter()

    for prediction in predictions:
        for tool in prediction.get("tools", []):
            tool_counts[str(tool["tool_name"])] += 1

        action_conditions_seen = set()
        tool_action_conditions_seen = set()
        for triplet in prediction.get("triplets", []):
            action_key = (int(triplet["tool_id"]), int(triplet["verb_id"]))
            if action_key not in action_conditions_seen:
                action_counts[str(triplet["verb_name"])] += 1
                action_conditions_seen.add(action_key)

            tool_action_name = f"{triplet['tool_name']} -> {triplet['verb_name']}"
            if tool_action_name not in tool_action_conditions_seen:
                tool_action_counts[tool_action_name] += 1
                tool_action_conditions_seen.add(tool_action_name)

            target_counts[str(triplet["target_name"])] += 1
            triplet_counts[
                f"{triplet['tool_name']} -> {triplet['verb_name']} -> {triplet['target_name']}"
            ] += 1

    print("")
    print(f"Prediction distribution across {num_windows} windows")
    print("(percentages are count / windows; multi-label predictions can sum above 100%)")
    print_counter_distribution("Tools", tool_counts, num_windows, top_k)
    print_counter_distribution("Actions conditioned on predicted tools", action_counts, num_windows, top_k)
    print_counter_distribution("Tool-action pairs", tool_action_counts, num_windows, top_k)
    print_counter_distribution("Targets", target_counts, num_windows, top_k)
    print_counter_distribution("Chained triplets", triplet_counts, num_windows, top_k)
    print("")


def open_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for fourcc_name in ("mp4v", "avc1"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
    raise RuntimeError(f"Could not open a video writer for {output_path}")


def annotate_video(
    input_video: Path,
    output_video: Path,
    predictions: list[dict[str, Any]],
    source_fps: float,
    width: int,
    height: int,
    max_triplets_display: int,
    verbose: bool = False,
) -> None:
    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video for writing overlay: {input_video}")

    writer = open_video_writer(output_video, source_fps, width, height)
    prediction_idx = 0
    frame_idx = 0

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            frame_time = frame_idx / source_fps if source_fps > 0 else 0.0
            while (
                prediction_idx + 1 < len(predictions)
                and frame_time >= predictions[prediction_idx + 1]["start_time"] - 1e-6
            ):
                prediction_idx += 1

            annotated = draw_overlay(
                frame_bgr,
                predictions[prediction_idx],
                frame_time,
                max_triplets_display=max_triplets_display,
            )
            writer.write(annotated)

            frame_idx += 1
            if verbose and frame_idx % 64 == 0:
                print(f"[overlay] wrote {frame_idx} frames")
    finally:
        capture.release()
        writer.release()


def main() -> int:
    args = parse_args()
    input_video = args.input_video.resolve()
    output_video = resolve_output_path(input_video, args.output_video).resolve()

    if not input_video.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_video}")
    if input_video == output_video:
        raise ValueError("Refusing to overwrite the input video; choose a different output path")

    device = resolve_device(args.device)
    if args.verbose:
        print(f"Using device: {device}")

    tool_cfg = load_yaml_config(args.tool_config)
    verb_cfg = load_yaml_config(args.verb_config)
    target_cfg = load_yaml_config(args.target_config)
    override_backbone_checkpoint(
        [tool_cfg, verb_cfg, target_cfg],
        args.backbone_checkpoint,
        checkpoint_key=args.backbone_checkpoint_key,
    )
    compare_encoder_settings(tool_cfg, verb_cfg, "verb")
    compare_encoder_settings(tool_cfg, target_cfg, "target")

    tool_threshold = resolve_multilabel_threshold(tool_cfg, args.tool_threshold)
    verb_threshold = resolve_multilabel_threshold(verb_cfg, args.verb_threshold)
    target_threshold = resolve_multilabel_threshold(target_cfg, args.target_threshold)
    tool_names = class_names_from_config(tool_cfg, FALLBACK_TOOL_NAMES, "tool")
    verb_names = class_names_from_config(verb_cfg, FALLBACK_VERB_NAMES, "verb")
    target_names = class_names_from_config(target_cfg, FALLBACK_TARGET_NAMES, "target")

    frames_per_clip = int(tool_cfg["experiment"]["data"].get("frames_per_clip", 16))
    validate_temporal_sampling(
        frames_per_clip=frames_per_clip,
        clip_fps=args.clip_fps,
        clip_seconds=args.clip_seconds,
    )
    resolution = int(tool_cfg["experiment"]["data"].get("resolution", 224))
    transform = build_transform(tool_cfg)

    tool_checkpoint = derive_checkpoint_path(
        tool_cfg,
        args.tool_checkpoint,
        fallback_name="tool",
        checkpoint_policy=args.head_checkpoint_policy,
        strict_best=args.strict_best_head_checkpoints,
    )
    verb_checkpoint = derive_checkpoint_path(
        verb_cfg,
        args.verb_checkpoint,
        fallback_name="verb",
        checkpoint_policy=args.head_checkpoint_policy,
        strict_best=args.strict_best_head_checkpoints,
    )
    target_checkpoint = derive_checkpoint_path(
        target_cfg,
        args.target_checkpoint,
        fallback_name="target",
        checkpoint_policy=args.head_checkpoint_policy,
        strict_best=args.strict_best_head_checkpoints,
    )

    if args.verbose:
        print(f"Loading encoder from {tool_cfg['model_kwargs']['checkpoint']}")
        print(f"Head checkpoint policy: {args.head_checkpoint_policy}")
    with maybe_suppress_stdout(args.verbose):
        encoder = init_encoder_module(
            module_name=tool_cfg["model_kwargs"]["module_name"],
            frames_per_clip=frames_per_clip,
            resolution=resolution,
            checkpoint=tool_cfg["model_kwargs"]["checkpoint"],
            model_kwargs=tool_cfg["model_kwargs"]["pretrain_kwargs"],
            wrapper_kwargs=tool_cfg["model_kwargs"]["wrapper_kwargs"],
            device=device,
        )
    encoder.eval()

    tool_head = build_classifier_from_config(tool_cfg, encoder.embed_dim).to(device).eval()
    verb_head = build_classifier_from_config(verb_cfg, encoder.embed_dim).to(device).eval()
    target_head = build_classifier_from_config(target_cfg, encoder.embed_dim).to(device).eval()

    load_classifier_weights(tool_head, tool_checkpoint)
    load_classifier_weights(verb_head, verb_checkpoint)
    load_classifier_weights(target_head, target_checkpoint)
    if args.verbose:
        print(f"Loaded tool head from {tool_checkpoint}")
        print(f"Loaded verb head from {verb_checkpoint}")
        print(f"Loaded target head from {target_checkpoint}")
        print(
            "Using multi-label thresholds: "
            f"tool={tool_threshold:.3f} verb={verb_threshold:.3f} target={target_threshold:.3f}"
        )

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
        print(f"Running chained inference on {len(window_starts)} windows")

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
            prediction = run_chained_prediction(
                encoder=encoder,
                tool_head=tool_head,
                verb_head=verb_head,
                target_head=target_head,
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
