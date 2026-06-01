# Phase RAG Scripts

This folder contains the V-JEPA2 top-k embedding RAG utilities for SurgNet
native-11 phase recognition.

The workflow is:

1. build a phase embedding index from labeled clips
2. evaluate held-out clips against that index
3. optionally run one-clip inference

The scripts store encoder settings in the index metadata, so evaluation and
inference normally only need the index path.

## Files

```text
phase_topk_embedding_index.py  build a top-k token embedding index
phase_topk_rag_eval.py         evaluate phase prediction on a CSV split
phase_topk_rag_inference.py    predict phase for one input clip
```

Default data:

```text
/path/to/phase_triplet_heads_bundle/globus/surgenet_phases/yt_robotic_chole_native11_phases_train.csv
/path/to/phase_triplet_heads_bundle/globus/surgenet_phases/yt_robotic_chole_native11_phases_val.csv
```

The scripts remap old CSV paths under:

```text
/path/to/home/Desktop/PhdLeo/annotator_fellow/phase_recog/surgenet_phases
```

to the local root:

```text
/path/to/phase_triplet_heads_bundle/globus/surgenet_phases
```

## Build The Default Surgical Phase3 Index

This uses the same surgical V-JEPA2 checkpoint family as the single-tool heads:

```text
checkpoint: /path/to/phase_triplet_heads_bundle/vjepa2_1/chekpoints_vjepar_surg_finetuned/single4_all_datasets_phase3_higher_res_g8_latest.pth.tar
checkpoint_key: target_encoder
resolution: 512
```

Command:

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
  --output-dir /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag_embeddings/native11_phase3_topk8_train75 \
  --overwrite
```

Current local note: the train CSV references `yt_robotic_chole_Batch7`, but that
clip directory is not present locally. Use `--skip-missing-clips`; the skipped
clips and missing video ids are recorded in `metadata.json`.

## Build With The Distilled 384px Checkpoint

For:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt
```

the scripts automatically use:

```text
checkpoint_key: ema_encoder
resolution: 384
```

Command:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag/phase_topk_embedding_index.py \
  --encoder-checkpoint /path/to/phase_triplet_heads_bundle/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt \
  --source-splits train \
  --top-k 8 \
  --batch-size 8 \
  --num-workers 4 \
  --device cuda \
  --skip-missing-clips \
  --output-dir /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag_embeddings/native11_vitl_dist_vitg384_topk8_train75 \
  --overwrite
```

## Evaluate Held-Out Clips

Evaluate local validation clips from `Batch4` against a built index:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag/phase_topk_rag_eval.py \
  --index /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag_embeddings/native11_phase3_topk8_train75/embeddings.pt \
  --eval-csv /path/to/phase_triplet_heads_bundle/globus/surgenet_phases/yt_robotic_chole_native11_phases_val.csv \
  --video-id yt_robotic_chole_Batch4 \
  --device cuda \
  --batch-size 8 \
  --num-workers 4 \
  --retrieval-k 50 \
  --aggregation unique_clip_max \
  --output-json /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag_embeddings/native11_phase3_topk8_train75/eval_batch4_k50.json \
  --output-csv /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag_embeddings/native11_phase3_topk8_train75/eval_batch4_k50_predictions.csv
```

For the distilled index, only change the `--index` and output paths:

```bash
/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag/phase_topk_rag_eval.py \
  --index /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag_embeddings/native11_vitl_dist_vitg384_topk8_train75/embeddings.pt \
  --video-id yt_robotic_chole_Batch4 \
  --device cuda \
  --retrieval-k 50 \
  --aggregation unique_clip_max
```

You do not need to pass `--encoder-checkpoint` during evaluation when the index
was built with the desired encoder. The evaluator reads `checkpoint`,
`checkpoint_key`, `resolution`, `frames_per_clip`, and `frame_step` from the
index metadata.

## Quick Evaluation Smoke Test

```bash
/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag/phase_topk_rag_eval.py \
  --index /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag_embeddings/native11_phase3_topk8_train75/embeddings.pt \
  --video-id yt_robotic_chole_Batch4 \
  --max-clips 16 \
  --device cuda
```

## One-Clip Inference

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag/phase_topk_rag_inference.py \
  --index /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag_embeddings/native11_phase3_topk8_train75/embeddings.pt \
  --input-video /path/to/phase_triplet_heads_bundle/globus/surgenet_phases/yt_robotic_chole_native11_clips/yt_robotic_chole_Batch4/yt_robotic_chole_Batch4_clip_00001_frames_00006_00021.mp4 \
  --device cuda \
  --retrieval-k 50 \
  --aggregation unique_clip_max \
  --output-json /path/to/phase_triplet_heads_bundle/vjepa2_1/phase_rag_embeddings/native11_phase3_topk8_train75/query_example.json
```

## Useful Arguments

```text
--top-k
  Number of token embeddings stored per indexed clip.

--top-k-query
  Number of token embeddings used from each query/eval clip. If omitted, it
  defaults to the index top-k.

--retrieval-k
  Number of nearest token matches used for voting. Smaller values such as 25 or
  50 can reduce class-prior collapse toward the dominant dissection phase.

--aggregation unique_clip_max
  For each retrieved database clip, use its best token match before voting.

--aggregation all_matches
  Vote with every retrieved token match. This is more sensitive to class
  imbalance.

--encoder-checkpoint
  Overrides the default encoder when building an index. The distilled
  `vjepa2_1_vitl_dist_vitG_384.pt` checkpoint automatically selects
  `ema_encoder` and 384px.
```

## Output

The index file is a PyTorch payload with:

```text
embeddings    [num_clips, top_k, 1024], usually fp16 and L2-normalized
labels        [num_clips]
token_indices [num_clips, top_k]
paths         source clip path for each indexed clip
video_ids     source video id for each indexed clip
metadata      encoder, split, missing clip, and build settings
```

Evaluation writes:

```text
accuracy
macro_f1
macro_recall
per_class
confusion_matrix_rows_true_cols_pred
predictions
```
