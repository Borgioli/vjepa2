"""Eval-only target-axis relabeling for the soft triplet refinement pipeline.

This module implements an in-memory, evaluation-only relabeling of the target
axis on the Tool5 clip-level validation split. It does not modify any CSV,
metadata JSON, or config file on disk, and the model weights / routing /
calibration / decoding behavior are unchanged. It only mutates the four
boolean masks that flow into the metric accumulators (predicted target,
ground-truth target, predicted triplet, ground-truth triplet) and provides
"active class" masks so that dropped classes are excluded from Macro F1 and
from Micro TP/FP/FN at summary time.

The relabel is opt-in via the ``--target-relabel-mode`` flag in
``triplet_soft_refinement_eval.py`` and
``triplet_soft_refinement_fit_temperatures.py``. When the flag is omitted, the
helpers in this module are not called and pipeline behavior is byte-for-byte
identical to the un-relabeled run.

Currently supported mode:

- ``tool5_chole_v1``: drop ``suture`` (idx 9), ``gut`` (idx 12),
  ``specimen bag`` (idx 14); merge ``gallbladder wall`` (idx 5) into
  ``gallbladder`` (idx 4). Triplet ids whose target is dropped are removed
  from evaluation; triplet ids whose target is merged are remapped to the
  triplet id with ``target_id=4`` if such a triplet id exists in the
  metadata, otherwise dropped. Dropped triplet ids are excluded from Macro
  F1 averaging and from Micro TP/FP/FN.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


SUPPORTED_RELABEL_MODES: tuple[str, ...] = ("tool5_chole_v1",)


@dataclass(frozen=True)
class TargetRelabelPlan:
    """Immutable description of an applied target-axis relabel.

    Attributes
    ----------
    mode:
        Name of the relabel mode (e.g. ``"tool5_chole_v1"``).
    num_target_classes:
        Number of target classes in the original label space (e.g. 16).
    num_triplet_classes:
        Number of triplet classes in the original label space (e.g. 75).
    dropped_target_ids:
        Sorted tuple of original target ids that should be excluded from the
        target axis evaluation entirely.
    target_merges:
        Dict ``{src_target_id: dst_target_id}`` describing target merges.
        Both ids must lie in ``[0, num_target_classes)`` and ``src != dst``.
    triplet_remap:
        For each original triplet id, the new triplet id it maps to under
        the relabel. ``None`` means the triplet id is dropped (e.g. its
        target is dropped, or its merged target produces an unsupported
        ``(tool, verb, target)`` combination).
    target_active_mask:
        Boolean mask of length ``num_target_classes`` marking which target
        ids participate in metric averaging.
    triplet_active_mask:
        Boolean mask of length ``num_triplet_classes`` marking which
        triplet ids participate in metric averaging.
    diagnostics:
        Free-form dict of integer/string diagnostics for the JSON report
        (counts of remapped triplet ids, dropped-after-merge ids, etc.).
    """

    mode: str
    num_target_classes: int
    num_triplet_classes: int
    dropped_target_ids: tuple[int, ...]
    target_merges: dict[int, int]
    triplet_remap: tuple[int | None, ...]
    target_active_mask: torch.Tensor
    triplet_active_mask: torch.Tensor
    diagnostics: dict[str, Any]

    def to_serializable(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "num_target_classes": int(self.num_target_classes),
            "num_triplet_classes": int(self.num_triplet_classes),
            "dropped_target_ids": list(self.dropped_target_ids),
            "target_merges": {int(k): int(v) for k, v in self.target_merges.items()},
            "active_target_classes": int(self.target_active_mask.sum().item()),
            "active_triplet_classes": int(self.triplet_active_mask.sum().item()),
            "triplet_remap_summary": {
                "kept": int(sum(1 for x in self.triplet_remap if x is not None)),
                "dropped": int(sum(1 for x in self.triplet_remap if x is None)),
            },
            "diagnostics": dict(self.diagnostics),
        }


def _resolve_target_id_by_name(target_names: list[str], name: str) -> int | None:
    lowered = name.strip().lower()
    for idx, candidate in enumerate(target_names):
        if str(candidate).strip().lower() == lowered:
            return idx
    return None


def build_tool5_chole_v1_plan(
    target_names: list[str],
    triplet_to_id: dict[tuple[int, int, int], int],
    num_triplet_classes: int,
) -> TargetRelabelPlan:
    """Build the relabel plan for the ``tool5_chole_v1`` mode.

    The mode drops three target classes and merges ``gallbladder wall`` into
    ``gallbladder``. If the canonical class names are not all present in
    ``target_names``, the corresponding drop or merge is silently skipped
    and recorded in ``diagnostics`` so the JSON output makes the situation
    explicit.
    """
    num_target_classes = int(len(target_names))
    drop_candidates = ("suture", "gut", "specimen bag")
    merge_pairs = (("gallbladder wall", "gallbladder"),)

    diagnostics: dict[str, Any] = {
        "drop_lookup": {},
        "merge_lookup": {},
    }

    dropped: set[int] = set()
    for candidate_name in drop_candidates:
        resolved = _resolve_target_id_by_name(target_names, candidate_name)
        diagnostics["drop_lookup"][candidate_name] = resolved
        if resolved is not None:
            dropped.add(int(resolved))

    target_merges: dict[int, int] = {}
    for src_name, dst_name in merge_pairs:
        src_id = _resolve_target_id_by_name(target_names, src_name)
        dst_id = _resolve_target_id_by_name(target_names, dst_name)
        diagnostics["merge_lookup"][f"{src_name}->{dst_name}"] = {
            "src_id": src_id,
            "dst_id": dst_id,
        }
        if src_id is None or dst_id is None or int(src_id) == int(dst_id):
            continue
        if int(src_id) in dropped or int(dst_id) in dropped:
            continue
        target_merges[int(src_id)] = int(dst_id)

    target_active = torch.ones(num_target_classes, dtype=torch.bool)
    for idx in dropped:
        target_active[idx] = False
    for src in target_merges:
        target_active[src] = False

    triplet_remap: list[int | None] = [None] * int(num_triplet_classes)
    triplet_dropped_for_target_drop = 0
    triplet_remap_target_merge = 0
    triplet_dropped_for_unsupported_merge = 0

    for (tool_id, verb_id, target_id), triplet_id in triplet_to_id.items():
        if int(triplet_id) < 0 or int(triplet_id) >= int(num_triplet_classes):
            continue
        new_target = int(target_id)
        if new_target in dropped:
            triplet_remap[int(triplet_id)] = None
            triplet_dropped_for_target_drop += 1
            continue
        if new_target in target_merges:
            new_target = int(target_merges[new_target])
            mapped_triplet = triplet_to_id.get((int(tool_id), int(verb_id), new_target))
            if mapped_triplet is None:
                triplet_remap[int(triplet_id)] = None
                triplet_dropped_for_unsupported_merge += 1
                continue
            triplet_remap[int(triplet_id)] = int(mapped_triplet)
            triplet_remap_target_merge += 1
            continue
        triplet_remap[int(triplet_id)] = int(triplet_id)

    triplet_active = torch.zeros(int(num_triplet_classes), dtype=torch.bool)
    for new_id in triplet_remap:
        if new_id is None:
            continue
        triplet_active[int(new_id)] = True

    diagnostics.update(
        {
            "dropped_target_ids": sorted(dropped),
            "target_merges": {int(src): int(dst) for src, dst in target_merges.items()},
            "triplet_dropped_for_target_drop": int(triplet_dropped_for_target_drop),
            "triplet_remap_target_merge": int(triplet_remap_target_merge),
            "triplet_dropped_for_unsupported_merge": int(triplet_dropped_for_unsupported_merge),
        }
    )

    return TargetRelabelPlan(
        mode="tool5_chole_v1",
        num_target_classes=num_target_classes,
        num_triplet_classes=int(num_triplet_classes),
        dropped_target_ids=tuple(sorted(dropped)),
        target_merges=dict(target_merges),
        triplet_remap=tuple(triplet_remap),
        target_active_mask=target_active,
        triplet_active_mask=triplet_active,
        diagnostics=diagnostics,
    )


def build_relabel_plan(
    mode: str | None,
    target_names: list[str],
    triplet_to_id: dict[tuple[int, int, int], int],
    num_triplet_classes: int,
) -> TargetRelabelPlan | None:
    if mode is None:
        return None
    if mode == "tool5_chole_v1":
        return build_tool5_chole_v1_plan(
            target_names=target_names,
            triplet_to_id=triplet_to_id,
            num_triplet_classes=int(num_triplet_classes),
        )
    raise ValueError(
        f"Unknown --target-relabel-mode '{mode}'. Supported: {SUPPORTED_RELABEL_MODES}"
    )


def apply_relabel_to_target_mask(
    mask: torch.Tensor,
    plan: TargetRelabelPlan,
) -> torch.Tensor:
    if mask.numel() != plan.num_target_classes:
        raise ValueError(
            f"Target mask length {mask.numel()} != num_target_classes {plan.num_target_classes}"
        )
    out = mask.detach().cpu().bool().reshape(-1).clone()
    for src, dst in plan.target_merges.items():
        if bool(out[int(src)].item()):
            out[int(dst)] = True
        out[int(src)] = False
    for idx in plan.dropped_target_ids:
        out[int(idx)] = False
    return out


def apply_relabel_to_triplet_mask(
    mask: torch.Tensor,
    plan: TargetRelabelPlan,
) -> torch.Tensor:
    if mask.numel() != plan.num_triplet_classes:
        raise ValueError(
            f"Triplet mask length {mask.numel()} != num_triplet_classes {plan.num_triplet_classes}"
        )
    flat = mask.detach().cpu().bool().reshape(-1)
    out = torch.zeros(plan.num_triplet_classes, dtype=torch.bool)
    for old_id in flat.nonzero(as_tuple=False).reshape(-1).tolist():
        new_id = plan.triplet_remap[int(old_id)]
        if new_id is None:
            continue
        out[int(new_id)] = True
    return out


def relabel_label_lookup(
    label_lookup: dict[str, dict[str, torch.Tensor]],
    plan: TargetRelabelPlan,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return a new lookup dict with the val ground-truth target/triplet
    multi-hot vectors rewritten under the relabel plan.

    The ``tool`` and ``verb`` entries are passed through unchanged.
    """
    rebuilt: dict[str, dict[str, torch.Tensor]] = {}
    for clip_path, axes in label_lookup.items():
        new_target = apply_relabel_to_target_mask(axes["target"], plan).to(dtype=axes["target"].dtype)
        new_triplet = apply_relabel_to_triplet_mask(axes["triplet"], plan).to(dtype=axes["triplet"].dtype)
        rebuilt[clip_path] = {
            "tool": axes["tool"],
            "verb": axes["verb"],
            "target": new_target,
            "triplet": new_triplet,
        }
    return rebuilt


def summarize_multilabel_counts_with_active_mask(
    summarize_fn,
    tp: torch.Tensor,
    fp: torch.Tensor,
    fn: torch.Tensor,
    exact_correct: int,
    total: int,
    active_mask: torch.Tensor | None,
) -> dict[str, float]:
    """Wrap an existing ``summarize_multilabel_counts`` callable to honor an
    optional per-class active mask. Counts on inactive classes are zeroed
    before averaging.
    """
    if active_mask is None:
        return summarize_fn(tp=tp, fp=fp, fn=fn, exact_correct=exact_correct, total=total)
    keep = active_mask.detach().cpu().bool().reshape(-1).to(dtype=torch.float64)
    if keep.numel() != tp.numel():
        raise ValueError(
            f"Active mask length {keep.numel()} != counts length {tp.numel()}"
        )
    return summarize_fn(
        tp=(tp.detach().cpu().to(torch.float64).reshape(-1) * keep),
        fp=(fp.detach().cpu().to(torch.float64).reshape(-1) * keep),
        fn=(fn.detach().cpu().to(torch.float64).reshape(-1) * keep),
        exact_correct=exact_correct,
        total=total,
    )
