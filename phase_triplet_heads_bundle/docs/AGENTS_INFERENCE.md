# Future Codex Guide: Tool4 Chained Multi-Label Inference

This guide is for running and maintaining the current Tool4 triplet inference path. For training details, use:

```text
/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/AGENTS_TRIPLET.md
```

The inference script is:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/triplet_conditioned_inference.py
```

It runs the latest three-head chained Tool4 pipeline:

```text
P(tool | clip)
P(action | clip, predicted tool)
P(target | clip, predicted tool, predicted action)
```

The visualization should show only the first three chained predictions per displayed frame:

```text
tool -> action -> target
tool %   action %   target %   score %
```

The chained score is:

```text
P(tool) * P(action | tool) * P(target | tool, action)
```

## Current Behavior

The script derives the three head checkpoints from their configs. By default it uses:

```bash
--head-checkpoint-policy best
```

That means it first looks for exported best-checkpoint files, then falls back to config-derived `latest.pt` if no best file exists. To force the old behavior:

```bash
--head-checkpoint-policy latest
```

To fail instead of falling back when best files are missing:

```bash
--strict-best-head-checkpoints
```

Default configs:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool4.yaml
/path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/verb_multilabel_conditioned_on_tool_train_vitl_latest_token_aggregation_tool4.yaml
/path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/target_multilabel_conditioned_on_tool_verb_train_vitl_latest_token_aggregation_tool4.yaml
```

Config-derived fallback head checkpoints:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/trained_heads/triplet_multilabel_tool_only_tool4_native/video_classification_frozen/triplet_multilabel_tool_only_head_training_vitl_latest_token_aggregation_tool4_native/latest.pt
/path/to/phase_triplet_heads_bundle/vjepa2_1/trained_heads/verb_multilabel_conditioned_on_tool_tool4_native/video_classification_frozen/verb_multilabel_conditioned_on_tool_head_training_vitl_latest_token_aggregation_tool4_native/latest.pt
/path/to/phase_triplet_heads_bundle/vjepa2_1/trained_heads/target_multilabel_conditioned_on_tool_verb_tool4_native/video_classification_frozen/target_multilabel_conditioned_on_tool_verb_head_training_vitl_latest_token_aggregation_tool4_native/latest.pt
```

## Preferred Best Head Checkpoints

For quality comparisons, prefer the default best-checkpoint policy, or pass explicit best-checkpoint overrides, instead of intentionally forcing config-derived `latest.pt`.

Current logs indicate these best validation epochs:

```text
tool head:   epoch 6 by val_tool_f1 / val_tool_acc
action head: epoch 6 by val_verb_f1 / val_verb_acc
target head: epoch 5 by val_target_f1 / val_target_acc
```

Only `latest.pt` is currently present in these folders. If the best epoch checkpoints are exported, copied, or promoted, use this canonical path pattern:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/trained_heads/triplet_multilabel_tool_only_tool4_native/video_classification_frozen/triplet_multilabel_tool_only_head_training_vitl_latest_token_aggregation_tool4_native/best_epoch_0006_val_tool_f1.pt
/path/to/phase_triplet_heads_bundle/vjepa2_1/trained_heads/verb_multilabel_conditioned_on_tool_tool4_native/video_classification_frozen/verb_multilabel_conditioned_on_tool_head_training_vitl_latest_token_aggregation_tool4_native/best_epoch_0006_val_verb_f1.pt
/path/to/phase_triplet_heads_bundle/vjepa2_1/trained_heads/target_multilabel_conditioned_on_tool_verb_tool4_native/video_classification_frozen/target_multilabel_conditioned_on_tool_verb_head_training_vitl_latest_token_aggregation_tool4_native/best_epoch_0005_val_target_f1.pt
```

The inference script now discovers these best filenames automatically when they exist. You can also pass them explicitly:

```bash
--tool-checkpoint /path/to/phase_triplet_heads_bundle/vjepa2_1/trained_heads/triplet_multilabel_tool_only_tool4_native/video_classification_frozen/triplet_multilabel_tool_only_head_training_vitl_latest_token_aggregation_tool4_native/best_epoch_0006_val_tool_f1.pt \
--verb-checkpoint /path/to/phase_triplet_heads_bundle/vjepa2_1/trained_heads/verb_multilabel_conditioned_on_tool_tool4_native/video_classification_frozen/verb_multilabel_conditioned_on_tool_head_training_vitl_latest_token_aggregation_tool4_native/best_epoch_0006_val_verb_f1.pt \
--target-checkpoint /path/to/phase_triplet_heads_bundle/vjepa2_1/trained_heads/target_multilabel_conditioned_on_tool_verb_tool4_native/video_classification_frozen/target_multilabel_conditioned_on_tool_verb_head_training_vitl_latest_token_aggregation_tool4_native/best_epoch_0005_val_target_f1.pt
```

Before using that command, verify the files exist. If they do not exist, either retrain/export with best-checkpoint saving enabled or use the fallback `latest.pt` paths above.

The default backbone is the original one used by the head configs:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt
checkpoint_key: ema_encoder
```

If `--backbone-checkpoint` is not passed, this original backbone continues to work exactly as before.

## Temporal Sampling

Inference must use the same temporal shape expected by the heads:

```text
frames_per_clip: 16
clip_fps:        4
clip_seconds:    4
```

For high-FPS videos, this does not feed all video frames to the model. It samples:

```text
16 frames over 4 seconds
1 sampled frame every 0.25 seconds
```

For a 29 fps source, example frame indices for the first window are approximately:

```text
0, 7, 14, 22, ..., 109
```

The script has a guard:

```text
clip_seconds must equal frames_per_clip / clip_fps
```

So for the current heads, use:

```bash
--clip-fps 4 \
--clip-seconds 4
```

Use `--stride-seconds` to control how often predictions are made. For example:

```text
--stride-seconds 2  => one prediction every 2 seconds
--stride-seconds 1  => one prediction every 1 second
```

## Frame Reader

The inference script uses a sequential OpenCV sampler for model input frames.

Why: Decord random seeking can print many warnings on long concatenated MP4s:

```text
Failed to skip frames effectively...
```

Those warnings usually mean random seeking is inefficient around keyframes. They are common with joined MP4s, but they are noisy and can make sampling less trustworthy. The current script avoids this by decoding forward in time and caching only sampled frames.

Do not reintroduce Decord random `get_batch()` sampling for this script unless there is a specific reason and the joined-video warnings are handled.

## Run Original Backbone

Use this for the normal current inference path:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/triplet_conditioned_inference.py \
  --input-video /path/to/phase_triplet_heads_bundle/globus/SITL_reduced/video_38_all_clips_clean.mp4 \
  --output-video /path/to/phase_triplet_heads_bundle/globus/SITL_reduced/video_38_all_clips_tool4_multilabel_inference_original_backbone.mp4 \
  --device cuda \
  --clip-fps 4 \
  --clip-seconds 4 \
  --stride-seconds 1 \
  --distribution-top-k 20
```

This uses:

```text
original V-JEPA backbone + current Tool4 tool head + current Tool4 action head + current Tool4 target head
```

## Run Alternate Backbone With Same Heads

The script can run a different frozen encoder while keeping the same three trained heads:

```bash
--backbone-checkpoint /path/to/backbone.pt
```

Equivalent alias:

```bash
--encoder-checkpoint /path/to/backbone.pt
```

For the current alternate checkpoint:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/all_datasets_plus_surgvu_phase3_higher_res_g8_latest.pt
```

The checkpoint has:

```text
encoder
target_encoder
```

It does not have:

```text
ema_encoder
```

The script auto-detects this and uses:

```text
checkpoint_key: target_encoder
```

This is the closest equivalent to the original `ema_encoder` behavior. To force a key manually:

```bash
--backbone-checkpoint-key target_encoder
```

or:

```bash
--backbone-checkpoint-key encoder
```

Run alternate backbone:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/triplet_conditioned_inference.py \
  --input-video /path/to/phase_triplet_heads_bundle/globus/SITL_reduced/video_38_all_clips_clean.mp4 \
  --output-video /path/to/phase_triplet_heads_bundle/globus/SITL_reduced/video_38_all_clips_tool4_multilabel_inference_new_backbone.mp4 \
  --backbone-checkpoint /path/to/phase_triplet_heads_bundle/vjepa2_1/all_datasets_plus_surgvu_phase3_higher_res_g8_latest.pt \
  --device cuda \
  --clip-fps 4 \
  --clip-seconds 4 \
  --stride-seconds 1 \
  --distribution-top-k 20
```

Important caveat: the heads were trained on features from the original backbone. A different backbone may load and run, but quality depends on feature compatibility.

## Single-Video Inference

For any single video:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/triplet_conditioned_inference.py \
  --input-video /path/to/input.mp4 \
  --output-video /path/to/output_tool4_multilabel_inference.mp4 \
  --device cuda \
  --clip-fps 4 \
  --clip-seconds 4 \
  --stride-seconds 1 \
  --distribution-top-k 20
```

For a fast sanity test:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/triplet_conditioned_inference.py \
  --input-video /path/to/input.mp4 \
  --output-video /path/to/output_test.mp4 \
  --device cuda \
  --clip-fps 4 \
  --clip-seconds 4 \
  --stride-seconds 1 \
  --max-windows 5 \
  --distribution-top-k 20
```

Use `--verbose` only when debugging. By default the script is intentionally quiet and prints the distribution once at the end.

## SITL Reduced Clips To One MP4

Example used for `video_38`:

```text
/path/to/phase_triplet_heads_bundle/globus/SITL_reduced/clips/video_38_clip_0.mp4
...
/path/to/phase_triplet_heads_bundle/globus/SITL_reduced/clips/video_38_clip_14.mp4
```

Create a concat list in numeric order:

```bash
python3 -c "from pathlib import Path
base = Path('/path/to/phase_triplet_heads_bundle/globus/SITL_reduced/clips')
out = Path('/tmp/video_38_concat_list.txt')
out.write_text(''.join(f\"file {str(base / f'video_38_clip_{i}.mp4')!r}\\n\" for i in range(15)))"
```

First, a stream-copy concat can be made quickly:

```bash
ffmpeg -y \
  -f concat \
  -safe 0 \
  -i /tmp/video_38_concat_list.txt \
  -c copy \
  /path/to/phase_triplet_heads_bundle/globus/SITL_reduced/video_38_all_clips.mp4
```

For inference, prefer a clean re-encoded joined MP4:

```bash
ffmpeg -y \
  -f concat \
  -safe 0 \
  -i /tmp/video_38_concat_list.txt \
  -c:v libx264 \
  -preset veryfast \
  -crf 18 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  /path/to/phase_triplet_heads_bundle/globus/SITL_reduced/video_38_all_clips_clean.mp4
```

Then run inference:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/triplet_conditioned_inference.py \
  --input-video /path/to/phase_triplet_heads_bundle/globus/SITL_reduced/video_38_all_clips_clean.mp4 \
  --output-video /path/to/phase_triplet_heads_bundle/globus/SITL_reduced/video_38_all_clips_tool4_multilabel_inference.mp4 \
  --device cuda \
  --clip-fps 4 \
  --clip-seconds 4 \
  --stride-seconds 1 \
  --distribution-top-k 20
```

Known output from the previous `video_38` original-backbone run:

```text
/path/to/phase_triplet_heads_bundle/globus/SITL_reduced/video_38_all_clips_clean.mp4
/path/to/phase_triplet_heads_bundle/globus/SITL_reduced/video_38_all_clips_tool4_multilabel_inference.mp4
```

## Printed Distribution

The script prints a distribution summary after all windows are predicted:

```text
Prediction distribution across N windows
Tools
Actions conditioned on predicted tools
Tool-action pairs
Targets
Chained triplets
```

Percentages are:

```text
count / number of inference windows
```

Because this is multi-label, percentages can sum above 100%.

Use:

```bash
--distribution-top-k 20
```

to print more labels.

Example distribution from the previous `video_38` original-backbone run:

```text
Prediction distribution across 439 windows

Tools
  grasper: 439 (100.0% of windows)
  irrigator: 32 (7.3% of windows)
  hook: 15 (3.4% of windows)

Actions conditioned on predicted tools
  grasp/retract: 439 (100.0% of windows)
  clean: 32 (7.3% of windows)
  coagulation: 15 (3.4% of windows)
  dissect: 15 (3.4% of windows)

Chained triplets
  grasper -> grasp/retract -> gallbladder: 439 (100.0% of windows)
  irrigator -> clean -> fluid: 32 (7.3% of windows)
  hook -> coagulation -> connective tissue: 15 (3.4% of windows)
  hook -> dissect -> cystic pedicle: 15 (3.4% of windows)
  hook -> dissect -> cystic duct: 15 (3.4% of windows)
  hook -> dissect -> cystic artery: 14 (3.2% of windows)
  hook -> coagulation -> peritoneum: 7 (1.6% of windows)
```

## Visualization Details

The overlay is intentionally minimal:

```text
top 3 chained predictions only
no time line
no separate tools summary
no "more above thresholds" line
```

The top 3 cards are selected from score-sorted chained triplets with per-tool clip caps:

```text
hook:     max 1 triplet per clip
scissors: max 1 triplet per clip
grasper:  max 2 triplets per clip
```

When more triplets for one of those tools are available, keep the highest scoring ones.

The overlay uses small compact cards inside a semi-transparent top-left panel. Long labels should shrink or truncate instead of going outside the frame.

Relevant CLI flags:

```bash
--max-triplets-display 3
--tool-threshold 0.5
--verb-threshold 0.5
--target-threshold 0.5
--max-tools 4
--max-verbs-per-tool 3
--max-targets-per-pair 3
```

Thresholds default to the values in the configs, currently `0.5`.

## Troubleshooting

If a backbone override fails with:

```text
KeyError: 'ema_encoder'
```

inspect the checkpoint keys. The current script should auto-detect `target_encoder` or `encoder`, but a manual key can be passed:

```bash
--backbone-checkpoint-key target_encoder
```

If an output MP4 is missing or tiny, check whether the run finished and whether OpenCV could open the input video.

If output text is too crowded, reduce the number of displayed targets or conditions:

```bash
--max-tools 2
--max-verbs-per-tool 2
--max-targets-per-pair 2
--max-triplets-display 3
```

If you need progress logs:

```bash
--verbose
```

If using old commands from earlier sessions, remove any `2>/tmp/...` stderr redirection unless you specifically want a log file. The current OpenCV sequential sampler should not emit the Decord skip warnings.
