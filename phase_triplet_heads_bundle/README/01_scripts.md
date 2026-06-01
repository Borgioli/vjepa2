# Scripts

All scripts assume the Python env at `/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python` on the original machine. Paths inside the scripts use absolute paths rooted at `/path/to/phase_triplet_heads_bundle/` — you will need to either reproduce that layout on the target machine or run a `sed`-style path rewrite (see `globus/surgenet_triplets/fix_paths.py` for an existing example).

## `vjepa2_1/head_phases/` — phase heads (multiclass)

| File | Purpose |
|---|---|
| `dataset_creation.py` | Build phase-clip CSV datasets from raw annotations. |
| `video_dataset_creation.py` | Variant: build datasets directly from full videos. |
| `build_sitl_4s_4fps_dataset.py` | Resample SITL phases to 4-second / 4-fps clips → `globus/sitl_phases_4s_4fps/`. |
| `build_sarrarp50_action_dataset.py` | Build SARRARP50 action-recognition dataset. |
| `prepare_sarrarp50_dataset.py` | Prep step for the SARRARP50 build. |
| `convert_sitl_phase_probe_csv.py` | Convert legacy SITL phase probes into the standard CSV format. |
| `convert_temporal_csv.py` | Temporal-CSV conversion helper. |
| `map_sitl_to_cholec80.py` | Map SITL phase labels onto the Cholec80 label space. |
| `analyze_phase_clip_mosaics.py` | Mosaic visualization of phase clips. |
| `analyze_surgenet_phase_clip_mosaics.py` | Same, for Surgenet phases. |
| `AGENTS_SARRARP50_ACTION_BACKBONES.md` | (in-place) SARRARP50 backbone notes. |
| `SARRARP50_ACTION_BACKBONE_TESTS.md` | (in-place) SARRARP50 test logbook. |

## `vjepa2_1/head_phases_ovr/` — one-vs-rest phase heads

| File | Purpose |
|---|---|
| `build_phase_ovr_heads.py` | Emit one binary `train.csv`/`val.csv` + config per phase. |
| `label_spaces.py` | Defines `reduced6` and `native11` label collapses. |
| `analyze_ovr_phase_clip_mosaics.py` | Per-class mosaic viz. |
| `audits/reduced6/` | Per-class audit scripts/data for the reduced-6 label space. |
| `audits_surgenet_only/reduced6/` | Same audits restricted to the Surgenet half of the data. |
| `README.md` | **Authoritative usage doc** — supported label spaces, default CSV sources, training workflow. |

## `vjepa2_1/head_triplets/` — triplet audit scripts

| File | Purpose |
|---|---|
| `analyze_triplet_clip_mosaics.py` | Mosaic viz over triplet clips. |
| `analyze_tool_presence_clip_mosaics.py` | Mosaic viz over tool-presence windows. |
| `audit_utils.py` | Shared audit helpers. |
| `run_all_tool_presence_audits.sh` | Batch-run tool-presence audits. |
| `run_all_tool_presence_audits_tool4.sh` | Variant restricted to tool 4. |
| `run_triplet_audit_top12.sh` | Audit the top-12 triplets. |

## `vjepa2_1/phase_rag/` — top-k embedding RAG for phases

| File | Purpose |
|---|---|
| `phase_topk_embedding_index.py` | Build a top-k token-embedding index from a labeled CSV. |
| `phase_topk_rag_eval.py` | Score a held-out CSV split against an index. |
| `phase_topk_rag_inference.py` | One-clip phase prediction against an index. |
| `README.md` | **Authoritative usage doc** — default checkpoint, build/eval/inference commands, known data caveats (e.g. missing `Batch7` clips). |

Prebuilt indices live alongside in `vjepa2_1/phase_rag_embeddings/` — see `03_checkpoints.md`.

## `vjepa2_1/evals/triplet_recog_frozen/` — frozen-backbone triplet eval

| File | Purpose |
|---|---|
| `eval.py` | Main eval entrypoint. |
| `models.py` | Head model definitions. |
| `dataset_wrapper.py` / `test_dataset_wrapper.py` | Triplet dataset wrappers + tests. |
| `test_multitask.py` / `test_missing_labels.py` | Multitask + missing-label test cases. |
| `example_config_multitask.yaml` | Reference multitask config. |
| `modelcustom/vit_encoder_multiclip.py` | Multi-clip ViT encoder wrapper. |
| `modelcustom/vit_encoder_multiclip_multilevel.py` | Multi-level variant. |
| `README_MULTITASK.md` | **Authoritative usage doc** for the multitask setup. |

## Auxiliary scripts inside the dataset dirs

| File | Purpose |
|---|---|
| `globus/surgenet_triplets/fix_paths.py` | Rewrite legacy CSV paths to the local Desktop root. Good template if you move the bundle to a new machine. |
| `globus/surgenet_triplets/recover_short_clips.py` | Recover/regenerate truncated triplet clips. |
| `globus/triplet/add_index_columns.py` | Add index columns to `triplet_dataset.csv`. |
| `globus/triplet/analysis_class.py` | Class-distribution analysis. |
| `globus/triplet/check_label_overlap.py` | Detect overlapping multilabel rows. |
| `globus/triplet/clip_division.py` / `clip_division_multihead.py` | Train/val clip-level splits (single-head / multi-head variants). |
