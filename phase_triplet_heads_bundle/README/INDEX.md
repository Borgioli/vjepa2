# Phase + Triplet Heads Bundle — Index

Bundle generated 2026-05-22 from `/path/to/data_root`.

## What this bundle is

Everything needed to reproduce, evaluate, or extend the **phase-recognition heads** and **triplet-recognition heads** built on top of V-JEPA2 surgical backbones. Self-contained: scripts, labeled clip datasets, trained checkpoints, RAG embedding indices, and the existing per-component documentation.

## How to use this README folder

This folder is meant to give a future agent or collaborator a fast map of the bundle without having to grep through 100 GB of payload.

| File | Read it when you want to … |
|---|---|
| `INDEX.md` (this file) | Get oriented. |
| `01_scripts.md` | Know which Python/shell file does what, and where it lives. |
| `02_datasets.md` | Understand CSV schemas, clip directory layouts, label spaces. |
| `03_checkpoints.md` | Pick the right backbone or head checkpoint for a job. |
| `04_existing_docs.md` | Find the original AGENTS_*.md / README.md files (preserved in-place inside the zip). Authoritative for usage. |
| `05_quickstart.md` | Recreate an embedding index or train a head end-to-end from this bundle. |

**The existing AGENTS_*.md files inside the payload are the source of truth for usage.** Files here only index and summarize them.

## Bundle layout (top of zip)

```
README/                                       ← this folder
vjepa2_1/head_phases/                         ← phase-head scripts (multiclass + sarrarp50)
vjepa2_1/head_phases_ovr/                     ← one-vs-rest phase heads + audits
vjepa2_1/head_triplets/                       ← triplet-head audit/analysis scripts
vjepa2_1/phase_rag/                           ← top-k embedding RAG scripts
vjepa2_1/phase_rag_embeddings/                ← prebuilt RAG indices
vjepa2_1/evals/triplet_recog_frozen/          ← frozen-backbone triplet eval/training pipeline
vjepa2_1/*.pt, *.pth.tar                      ← V-JEPA2 backbones + phase3 head ckpts
vjepa2_1/chekpoints_vjepar_surg_finetuned/    ← surgically fine-tuned phase1/2/3 ckpts
globus/surgenet_phases/                       ← Surgenet native-11 phase clips + CSVs
globus/surgenet_triplets/                     ← Surgenet triplet/tool windows clips + CSVs + AGENTS docs
globus/sitl_phases/                           ← SITL phase clips (native)
globus/sitl_phases_4s_4fps/                   ← SITL phase clips (4s @ 4fps resampled)
globus/cholec80_phases/                       ← Cholec80 minority-phase clips
globus/triplet/                               ← Raw triplet videos + labels + per-clip splits
```

## Size at a glance

| Section | Size |
|---|---|
| Scripts (head_phases + ovr + triplets + phase_rag + triplet eval) | ~1 MB |
| Phase RAG prebuilt indices | ~75 MB |
| Checkpoints (5 head/finetuned + 2 backbones) | ~34.8 GB |
| Datasets (surgenet_phases, surgenet_triplets, sitl_phases, sitl_phases_4s_4fps, cholec80_phases, triplet) | ~80.5 GB |
| **Total** | **~115 GB** |

## What is *not* included

- `globus/surgenet_triplets_replaced_backup_20260416_upload/` — historical backup; deliberately excluded.
- Anything under `vjepa2_*` that isn't directly a head/backbone/script for these tasks (e.g. unrelated experiments, pose/colmap runs).
- The V-JEPA2 source tree itself (the heads load weights into model code that lives elsewhere in the repo). If the target machine doesn't already have the V-JEPA2 codebase, you also need it — clone the project repo separately.
