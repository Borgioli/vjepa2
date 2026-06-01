# Existing in-place documentation

These files were already in the source tree and are preserved at their original paths inside the bundle. **They are the authoritative usage docs** — the per-section files in this README folder only index and summarize them.

Read them in this rough order if you're new to the project:

## Phase recognition

1. `globus/surgenet_phases/yt_robotic_chole_native11_phase_summary.md`
   Dataset summary for the Surgenet native-11 phases.
2. `vjepa2_1/head_phases_ovr/README.md`
   One-vs-rest phase heads — label spaces (`reduced6` / `native11`), default CSV sources, full training workflow.
3. `vjepa2_1/phase_rag/README.md`
   Phase top-k RAG — build/eval/inference commands, default checkpoints, known data caveats.
4. `vjepa2_1/head_phases/AGENTS_SARRARP50_ACTION_BACKBONES.md`
   SARRARP50 action backbones — branched workflow for the SARRARP50 action-recognition task.
5. `vjepa2_1/head_phases/SARRARP50_ACTION_BACKBONE_TESTS.md`
   Test logbook for the SARRARP50 backbones.

## Triplet recognition

6. `globus/surgenet_triplets/yt_robotic_chole_triplet_summary.md`
   Dataset summary for Surgenet triplets.
7. `globus/surgenet_triplets/AGENTS_TRIPLET.md`
   Triplet head — hard-label training and eval.
8. `globus/surgenet_triplets/AGENTS_TRIPLET_soft.md`
   Triplet head — soft-label variant.
9. `globus/surgenet_triplets/AGENTS_SINGLE_TOOL.md`
   Per-tool single-head training (hard labels).
10. `globus/surgenet_triplets/AGENTS_SINGLE_TOOL_SOFT.md`
    Per-tool single-head training (soft labels).
11. `globus/surgenet_triplets/AGENTS_THRESHOLD_TUNING.md`
    How to tune decision thresholds for the multilabel triplet predictions.
12. `globus/surgenet_triplets/AGENTS_INFERENCE.md`
    Inference workflow for triplet heads.
13. `globus/surgenet_triplets/tool_csv/README.md`
    Layout of the per-tool CSV splits.
14. `vjepa2_1/evals/triplet_recog_frozen/README_MULTITASK.md`
    Frozen-backbone multitask triplet eval — config schema and entrypoints.

## One-slide overviews (HTML)

These are PDF-substitute one-slide diagrams useful for orienting newcomers:

- `globus/surgenet_triplets/triplet_multilabel_pipeline_one_slide.html`
- `globus/surgenet_triplets/phase_recognition_ovr_transformer_one_slide.html`

## When the doc and the code disagree

The code wins. The AGENTS_*.md files were written at points in time and may name a checkpoint or CSV that has since moved or been replaced. Before acting on an instruction from them, `ls` the path or `grep` for the symbol — see also the "Before recommending from memory" pattern.
