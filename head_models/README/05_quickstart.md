# Quickstart

Minimal, copy-pasteable commands. Assumes the bundle was unpacked so that `vjepa2_1/` and `globus/` end up under `/path/to/phase_triplet_heads_bundle/` (or you've symlinked an equivalent root). If you unpack elsewhere, run a path rewrite first — `globus/surgenet_triplets/fix_paths.py` is a working template.

## 0. Python env

The original scripts use:

```
/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python
```

The bundle does **not** include the venv. On the target machine, recreate it from the V-JEPA2 repo's requirements (clone the V-JEPA2 source tree separately — only weights and head code are included here).

## 1. Build a phase RAG index

Per `vjepa2_1/phase_rag/README.md` (the authoritative doc):

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag/phase_topk_embedding_index.py \
  --source-splits train \
  --top-k 8 \
  --batch-size 8 \
  --num-workers 4 \
  --device cuda \
  --skip-missing-clips \
  --output-dir /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag_embeddings/native11_phase3_topk8_train75_NEW \
  --overwrite
```

(A prebuilt version of this exact index ships in the bundle — see `03_checkpoints.md`.)

## 2. Evaluate against an existing index

```bash
/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag/phase_topk_rag_eval.py \
  --index-dir /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag_embeddings/native11_phase3_topk8_train75 \
  --eval-csv /path/to/phase_triplet_heads_bundle/globus/surgenet_phases/yt_robotic_chole_native11_phases_val.csv
```

## 3. Build one-vs-rest phase head datasets + configs

Per `vjepa2_1/head_phases_ovr/README.md`:

```bash
/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/head_phases_ovr/build_phase_ovr_heads.py \
  --label-space reduced6 \
  --sources sitl surgenet
```

Output:

- CSVs → `/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/ovr_phase_heads/reduced6/`
- configs → `/path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads_ovr/reduced6/`

## 4. Triplet head training/eval

Read `globus/surgenet_triplets/AGENTS_TRIPLET.md` (hard labels) or `AGENTS_TRIPLET_soft.md` (soft labels) — these have the current invocations. For frozen-backbone multitask eval, read `vjepa2_1/evals/triplet_recog_frozen/README_MULTITASK.md`.

## 5. Sanity-checks before running anything heavy

```bash
# CSV paths resolve to actual files?
head -1 /path/to/phase_triplet_heads_bundle/globus/surgenet_phases/yt_robotic_chole_native11_phases_val.csv | awk '{print $1}' | xargs ls -lh

# Index metadata says which backbone it expects?
cat /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag_embeddings/native11_phase3_topk8_train75/metadata.json | jq '.encoder, .resolution, .checkpoint'

# Backbone checkpoint exists at the path the index expects?
ls -lh /path/to/phase_triplet_heads_bundle/vjepa2_1/chekpoints_vjepar_surg_finetuned/single4_all_datasets_phase3_higher_res_g8_latest.pth.tar
```

If any of these fail, fix paths *before* launching training — the heads will otherwise burn an hour loading data and then crash.
