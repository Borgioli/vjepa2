# Future Codex Guide: Single-Tool Soft Refinement Eval

This document is the working reference for evaluating the single-tool six-head
soft-refinement models described in:

```text
/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/AGENTS_SINGLE_TOOL.md
```

It mirrors the Tool5 all-tools soft-refinement evaluator documented in:

```text
/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/AGENTS_TRIPLET_soft.md
```

but runs one tool family at a time:

```text
hook
clipper
grasper
```

## Main Idea

The evaluator uses the same soft architecture as the full Tool5 refinement
pipeline:

1. run the encoder once on the validation clip
2. compute stage-0 probabilities
   - `tool`
   - `action`
   - `target`
3. route soft probabilities over the pair-conditioned heads
   - `tool | action,target`
   - `action | tool,target`
   - `target | tool,action`
4. blend the previous-stage and refined probabilities
5. optionally run a second refinement pass

With the default:

```text
--refinement-steps 2
```

the prediction is:

1. stage-0 unconditioned heads: `tool0`, `action0`, `target0`
2. first conditioned pass through `tool|action,target`,
   `action|tool,target`, and `target|tool,action`
3. blend with `--blend-alpha`
4. second conditioned pass from the updated probabilities
5. blend with `--blend-alpha-step2`

The implementation reuses the generic soft-refinement code in
`triplet_soft_refinement_inference.py`. Internally that code calls the middle
axis `verb`; in this single-tool evaluator it is exposed as `action`.

For the local triplet metric, the evaluator also runs the final
`target|tool,action` conditioned scores over the final routed tool/action pairs
and maps only supported `(tool, action, target)` combinations back into that
tool's local triplet id space.

## Script

Use:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval.py
```

It evaluates one tool's validation split and reports:

- tool metrics
- action metrics
- target metrics
- local triplet metrics
- joint exact tool/action/target accuracy
- unsupported predicted local triplets
- optional bootstrap confidence intervals
- optional JSON output with packed per-sample masks

It shows a `tqdm` progress bar by default, including elapsed time, throughput,
and estimated time remaining. Use `--no-progress` only for logs or environments
where a progress bar is undesirable.

The local triplet axis is the tool-specific mapping from:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/single_tool/yt_robotic_chole_tool_<tool>_single_tool_metadata.json
```

## Calibration

The evaluator supports logit-temperature calibration with:

```text
--calibration-json
```

but it does not invent temperatures automatically. If no calibration JSON is
passed, the output JSON will report:

```text
head_temperatures_is_identity: true
```

Fit temperatures for one tool with:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_fit_temperatures.py \
  --tool hook \
  --device cuda \
  --output-json /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_temperatures_hook.json
```

Then evaluate with:

```bash
/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval.py \
  --tool hook \
  --device cuda \
  --calibration-json /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_temperatures_hook.json \
  --output-json /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval_hook_calibrated.json
```

For clipper or grasper, replace `hook` in both filenames and `--tool`.

The calibration JSON uses the generic soft-refinement key names:

```text
tool0, verb0, target0, tool_refine, verb_refine, target_refine
```

Here `verb` means the single-tool `action` axis.

Temperature calibration changes sigmoid probabilities and soft routing. It does
not replace threshold tuning. With a final `0.5` threshold, rare classes trained
with weighted BCE may still need lower per-axis or per-class thresholds.

## Threshold Sweeps

Use this helper to tune `multilabel_thresholds` from validation probabilities:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_threshold_sweep.py
```

It accepts any single-tool head config, including conditioned configs. For a
stage-0 clipper action sweep over the `clip` class:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_threshold_sweep.py \
  --config /path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/single_tool/clipper_action_unconditioned.yaml \
  --class-name clip \
  --device cuda \
  --output-json /path/to/phase_triplet_heads_bundle/vjepa2_1/clipper_action_unconditioned_clip_threshold_sweep.json \
  --output-csv /path/to/phase_triplet_heads_bundle/vjepa2_1/clipper_action_unconditioned_clip_threshold_sweep.csv
```

The script runs the head once on validation, sweeps thresholds, and reports the
best threshold by F1. Use the resulting threshold as the second value in:

```text
multilabel_thresholds: [0.5, <clip_threshold>]
```

For broader or finer sweeps, pass:

```text
--threshold-start 0.01 --threshold-end 0.99 --threshold-step 0.005
```

## Single-Head Quality Checks

Use this helper when the threshold is already chosen and you only want to
evaluate one trained head with the thresholds stored in its YAML config:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_head_eval.py
```

This is useful for quick checks such as "does the clipper action head still
predict everything as null?" It prints exact row match, micro precision/recall/F1
over all classes and non-null classes, per-class metrics, and the most common
predicted label combinations.

Example: evaluate the stage-0 clipper action head on about 10% of the validation
split. The clipper validation runtime CSV currently has `7041` clips, so `704`
is roughly 10%.

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_head_eval.py \
  --config /path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/single_tool/clipper_action_unconditioned.yaml \
  --device cuda \
  --max-samples 704 \
  --batch-size 4 \
  --num-workers 8 \
  --output-json /path/to/phase_triplet_heads_bundle/vjepa2_1/clipper_action_unconditioned_eval_10pct.json \
  --output-csv /path/to/phase_triplet_heads_bundle/vjepa2_1/clipper_action_unconditioned_eval_10pct.csv
```

The script uses the config's `multilabel_thresholds` directly. For the clipper
action head, this means:

```yaml
multilabel_thresholds: [0.5, 0.27]
```

If the model is still predicting almost everything as null, the printed
`Top predicted label combinations` will be dominated by `null`, and the `clip`
class will have `predicted_positive` near zero. If it recognizes clip, the
`clip` or `null+clip` combinations will appear, and the `clip` class will have
nonzero true positives.

For target heads, prefer the target-specific helper:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_target_threshold_sweep.py
```

It sweeps every non-null target class and prints a ready-to-paste
`multilabel_thresholds` line. By default, the `null` target threshold stays
fixed at `0.5`.

Example for the clipper stage-0 target head:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_target_threshold_sweep.py \
  --config /path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/single_tool/clipper_target_unconditioned.yaml \
  --device cuda \
  --output-json /path/to/phase_triplet_heads_bundle/vjepa2_1/clipper_target_unconditioned_threshold_sweep.json \
  --output-csv /path/to/phase_triplet_heads_bundle/vjepa2_1/clipper_target_unconditioned_threshold_sweep.csv
```

To tune the conditioned target head instead, replace the config with:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/single_tool/clipper_target_conditioned.yaml
```

The same pattern works for `hook_target_unconditioned.yaml`,
`hook_target_conditioned.yaml`, `grasper_target_unconditioned.yaml`, and
`grasper_target_conditioned.yaml`.

## Hard Rules

- Do not use `triplet_soft_refinement_eval.py` for the single-tool heads. That
  script expects the all-tools Tool5 CSV schema with `verb_multihot` and
  `triplet_multihot`.
- Use `single_tool_soft_refinement_eval.py` for hook, clipper, and grasper.
- Do not edit the per-tool validation CSVs on disk for evaluation.
- Evaluate a tool only after all six trained head checkpoints exist for that
  tool under:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/trained_heads/single_tool/<tool>/
```

- The default eval files are the same per-instrument validation sidecars used
  by the single-tool training configs. For hook, the source file is headerless
  and has `7041` rows:

```text
/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/tool_csv/yt_robotic_chole_tool_hook_val.csv
```

  The evaluator reads the generated hook val sidecars:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/single_tool/yt_robotic_chole_tool_hook_multilabel_val_runtime.csv
/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/single_tool/yt_robotic_chole_tool_hook_multilabel_val.csv
```

At the time this guide was written:

```text
hook:    6 / 6 latest.pt checkpoints present
clipper: 6 / 6 latest.pt checkpoints present
grasper: 0 / 6 latest.pt checkpoints present
```

So grasper evaluation will fail until the grasper six-head training finishes.

## Current Defaults

Recommended defaults for single-tool eval:

```text
--refinement-steps 2
--blend-alpha 0.75
--blend-alpha-step2 0.9
--routing-probability-source current
--support-constrained-routing-mode hard
```

The `hard` support-constrained routing uses each tool's local supported triplets
from the generated metadata JSON to remove impossible conditioning pairs during
soft routing. It does not alter the trained weights.

The default thresholds come from each head config's `multilabel_thresholds`.
Override them with:

```text
--tool-threshold
--action-threshold
--target-threshold
```

Optional route truncation:

```text
--route-topk-tool
--route-topk-action
--route-topk-target
```

For these small single-tool label spaces, leaving top-k unset is usually fine.

Progress display:

```text
default:       tqdm progress bar with ETA
--no-progress: disable tqdm
```

Stage-0-only sanity checks:

```text
--stage0-only
```

This evaluates only the unconditioned `tool0`, `action0`, and `target0` heads.
It does not load or run the conditioned heads, soft routing, or refinement
passes. Local triplets are decoded by taking the thresholded component masks and
keeping only supported `(tool, action, target)` combinations from the tool
metadata.

Evaluate a fraction of the validation split:

```text
--eval-fraction 0.1
```

For hook, `0.1` selects the first `705` clips from the `7041`-clip per-hook
validation split before sharding.

## Single-Machine Hook Eval

Full val:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval.py \
  --tool hook \
  --device cuda \
  --output-json /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval_hook.json
```

Smoke test:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval.py \
  --tool hook \
  --device cuda \
  --max-samples 10 \
  --verbose
```

Stage-0-only 10% check:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval.py \
  --tool hook \
  --device cuda \
  --stage0-only \
  --eval-fraction 0.1 \
  --bootstrap-iterations 0 \
  --output-json /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_stage0_eval_hook_10pct.json
```

## Single-Machine Clipper Eval

Full val:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval.py \
  --tool clipper \
  --device cuda \
  --output-json /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval_clipper.json
```

Smoke test:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval.py \
  --tool clipper \
  --device cuda \
  --max-samples 10 \
  --verbose
```

## Single-Machine Grasper Eval

Run this after all six grasper checkpoints exist:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval.py \
  --tool grasper \
  --device cuda \
  --output-json /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval_grasper.json
```

Smoke test:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval.py \
  --tool grasper \
  --device cuda \
  --max-samples 10 \
  --verbose
```

## Two-Spark Sharded Eval

The script supports the same simple sharding style as the all-tools evaluator:

```text
--num-shards 2
--shard-index 0 or 1
```

Hook shard 0:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval.py \
  --tool hook \
  --device cuda \
  --num-shards 2 \
  --shard-index 0 \
  --output-json /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval_hook_shard0.json
```

Hook shard 1:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval.py \
  --tool hook \
  --device cuda \
  --num-shards 2 \
  --shard-index 1 \
  --output-json /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval_hook_shard1.json
```

For clipper or grasper, change `--tool` and the output filename:

```text
--tool clipper
--output-json /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval_clipper_shard0.json

--tool grasper
--output-json /path/to/phase_triplet_heads_bundle/vjepa2_1/single_tool_soft_refinement_eval_grasper_shard0.json
```

## Important Difference From The All-Tools Evaluator

`triplet_soft_refinement_eval.py` evaluates the shared Tool5 label space:

```text
tool / verb / target / Tool5 triplet
```

`single_tool_soft_refinement_eval.py` evaluates a tool-local label space:

```text
tool-or-null / action / target / local per-tool triplet
```

For example, grasper target class id `14` is `suture`, while hook target class
id `12` is `specimen bag`. Do not compare target ids across tools without using
the generated metadata JSON.

## Validation Checks

Before relying on results:

- confirm all six configs parse for the tool
- confirm all six `latest.pt` or selected best checkpoints exist
- confirm the generated metadata JSON matches the tool slug
- run a `--max-samples 10 --verbose` smoke test
- run full val with `--output-json`
- for sharded runs, merge or compare shard JSONs carefully; this evaluator
  writes per-sample masks, but the all-tools merge utility is not yet generalized
  for the `action` key
