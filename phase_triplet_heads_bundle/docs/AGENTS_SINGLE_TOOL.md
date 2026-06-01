# Future Codex Guide: Single-Tool Heads

This document is the working reference for the per-tool frozen-head setup in
`/path/to/phase_triplet_heads_bundle/vjepa2_1`, built from the new 4-second / stride-1-second
per-tool CSVs.

For evaluating the trained single-tool soft-refinement heads on validation data,
use:

```text
/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/AGENTS_SINGLE_TOOL_SOFT.md
```

The current implemented tools are:

```text
hook
clipper
grasper
```

The goal is to train a separate six-head set per tool family, using that tool's
own local label space from the mapping JSON.

## Hard Rules

- Do not modify the shared frozen-head training code unless the user explicitly
  asks for it.
- Do not regenerate or reinterpret the source per-tool CSVs by hand. Treat these
  as the source of truth:
  - `/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/tool_csv/yt_robotic_chole_tool_hook_train.csv`
  - `/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/tool_csv/yt_robotic_chole_tool_hook_val.csv`
  - `/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/tool_csv/yt_robotic_chole_tool_clipper_train.csv`
  - `/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/tool_csv/yt_robotic_chole_tool_clipper_val.csv`
  - `/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/tool_csv/yt_robotic_chole_tool_grasper_train.csv`
  - `/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/tool_csv/yt_robotic_chole_tool_grasper_val.csv`
- Use the mapping JSON for every label-space decision:
  - `/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/tool_csv/yt_robotic_chole_per_tool_4s_stride1s_mapping.json`
- The source windowing is 4s with stride 1s. Each row represents a 4-second clip
  split into four 1-second slots for null detection. This means `null` and
  the active tool can both be positive in the same 4-second row.
- The media clips live under:
  - `/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/yt_robotic_chole_tool_windows_clips`

## Main Idea

For a single tool, the local per-tool triplet labels are expanded into three
component label spaces:

1. tool/null
2. action
3. target

Then six heads are trained, matching the soft-refinement style:

1. Stage-0 clip heads
   - `tool`
   - `action`
   - `target`
2. Pair-conditioned refinement heads
   - `tool | action,target`
   - `action | tool,target`
   - `target | tool,action`

All six heads freeze the V-JEPA encoder and train only classifier heads.

## Backbone And Head Architecture

All single-tool configs intentionally mirror the surg-finetuned phase3
conditioning-study heads:

```text
checkpoint: /path/to/phase_triplet_heads_bundle/vjepa2_1/chekpoints_vjepar_surg_finetuned/single4_all_datasets_phase3_higher_res_g8_latest.pth.tar
checkpoint_key: target_encoder
model_name: vit_large
head_type: token_aggregation or conditioned_token_aggregation
token_pool: mean
loss_name: weighted_bce
resolution: 512
frames_per_clip: 16
frame_step: 1
```

The architecture should stay exactly aligned with the existing working heads.
Only data paths, label fields, class counts, and output folders change.

The backbone checkpoint is shared. Trained head outputs are separated by tool and
by head under:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/trained_heads/single_tool/<tool>/
```

## Builder

The isolated sidecar builder is:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/build_single_tool_multilabel_datasets.py
```

Rebuild sidecars with:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/build_single_tool_multilabel_datasets.py \
  --tool hook

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/build_single_tool_multilabel_datasets.py \
  --tool clipper

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/build_single_tool_multilabel_datasets.py \
  --tool grasper
```

This writes headered multi-label lookup CSVs plus runtime CSVs under:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/single_tool/
```

The runtime CSVs keep the existing training dataloader shape:

```text
<clip_path> <dummy_label>
```

The lookup CSVs provide the actual multi-hot labels needed by the existing
multi-label wrappers.

## Label Spaces

Component ids are derived directly from the mapping JSON in numeric local-label
order.

### Hook

Tool classes:

```text
0 = null
1 = hook
```

Action classes:

```text
0 = null
1 = coagulation
2 = cut
3 = dissect
```

Target classes:

```text
0  = null
1  = adhesion
2  = connective tissue
3  = cystic artery
4  = cystic pedicle
5  = gallbladder wall
6  = peritoneum
7  = cystic duct
8  = gallbladder
9  = gut
10 = liver
11 = omentum
12 = specimen bag
```

The generated metadata file records the complete local-label mapping:

```text
app/csv_head_models/single_tool/yt_robotic_chole_tool_hook_single_tool_metadata.json
```

### Clipper

Tool classes:

```text
0 = null
1 = clipper
```

Action classes:

```text
0 = null
1 = clip
```

Target classes:

```text
0 = null
1 = cystic artery
2 = cystic duct
3 = liver
```

Metadata:

```text
app/csv_head_models/single_tool/yt_robotic_chole_tool_clipper_single_tool_metadata.json
```

### Grasper

Tool classes:

```text
0 = null
1 = grasper
```

Action classes:

```text
0 = null
1 = grasp/retract
```

Target classes:

```text
0  = null
1  = adhesion
2  = connective tissue
3  = cystic artery
4  = cystic duct
5  = cystic pedicle
6  = fallciform ligament
7  = gallbladder
8  = gallbladder wall
9  = gut
10 = liver
11 = omentum
12 = peritoneum
13 = specimen bag
14 = suture
```

Metadata:

```text
app/csv_head_models/single_tool/yt_robotic_chole_tool_grasper_single_tool_metadata.json
```

## Generated Data Pattern

For each `<tool>` in `hook`, `clipper`, `grasper`, the unconditioned
lookup/runtime files are:

```text
app/csv_head_models/single_tool/yt_robotic_chole_tool_<tool>_multilabel_train.csv
app/csv_head_models/single_tool/yt_robotic_chole_tool_<tool>_multilabel_val.csv
app/csv_head_models/single_tool/yt_robotic_chole_tool_<tool>_multilabel_train_runtime.csv
app/csv_head_models/single_tool/yt_robotic_chole_tool_<tool>_multilabel_val_runtime.csv
```

The conditioned lookup/runtime files are:

```text
app/csv_head_models/single_tool/yt_robotic_chole_tool_<tool>_tool_multilabel_conditioned_on_action_target_train.csv
app/csv_head_models/single_tool/yt_robotic_chole_tool_<tool>_tool_multilabel_conditioned_on_action_target_val.csv
app/csv_head_models/single_tool/yt_robotic_chole_tool_<tool>_tool_multilabel_conditioned_on_action_target_train_runtime.csv
app/csv_head_models/single_tool/yt_robotic_chole_tool_<tool>_tool_multilabel_conditioned_on_action_target_val_runtime.csv

app/csv_head_models/single_tool/yt_robotic_chole_tool_<tool>_action_multilabel_conditioned_on_tool_target_train.csv
app/csv_head_models/single_tool/yt_robotic_chole_tool_<tool>_action_multilabel_conditioned_on_tool_target_val.csv
app/csv_head_models/single_tool/yt_robotic_chole_tool_<tool>_action_multilabel_conditioned_on_tool_target_train_runtime.csv
app/csv_head_models/single_tool/yt_robotic_chole_tool_<tool>_action_multilabel_conditioned_on_tool_target_val_runtime.csv

app/csv_head_models/single_tool/yt_robotic_chole_tool_<tool>_target_multilabel_conditioned_on_tool_action_train.csv
app/csv_head_models/single_tool/yt_robotic_chole_tool_<tool>_target_multilabel_conditioned_on_tool_action_val.csv
app/csv_head_models/single_tool/yt_robotic_chole_tool_<tool>_target_multilabel_conditioned_on_tool_action_train_runtime.csv
app/csv_head_models/single_tool/yt_robotic_chole_tool_<tool>_target_multilabel_conditioned_on_tool_action_val_runtime.csv
```

## Training Config Pattern

All configs live under:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/single_tool/
```

For each `<tool>` in `hook`, `clipper`, `grasper`, the six configs are:

```text
<tool>_tool_unconditioned.yaml
<tool>_action_unconditioned.yaml
<tool>_target_unconditioned.yaml
<tool>_tool_conditioned.yaml
<tool>_action_conditioned.yaml
<tool>_target_conditioned.yaml
```

## Running Hook

Single-machine hook runner:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1
./configs/heads/single_tool/run_hook_single_tool_six_heads.sh
```

Two-Spark hook runners:

```bash
# Spark 0 / node rank 0
cd /path/to/phase_triplet_heads_bundle/vjepa2_1
./configs/heads/single_tool/run_hook_single_tool_six_heads_spark0.sh
```

```bash
# Spark 1 / node rank 1
cd /path/to/phase_triplet_heads_bundle/vjepa2_1
./configs/heads/single_tool/run_hook_single_tool_six_heads_spark1.sh
```

Hook ports:

```text
29711, 29712, 29713, 29714, 29715, 29716
```

## Running Clipper

Two-Spark clipper runners:

```bash
# Spark 0 / node rank 0
cd /path/to/phase_triplet_heads_bundle/vjepa2_1
./configs/heads/single_tool/run_clipper_single_tool_six_heads_spark0.sh
```

```bash
# Spark 1 / node rank 1
cd /path/to/phase_triplet_heads_bundle/vjepa2_1
./configs/heads/single_tool/run_clipper_single_tool_six_heads_spark1.sh
```

Clipper ports:

```text
29721, 29722, 29723, 29724, 29725, 29726
```

## Running Grasper

Two-Spark grasper runners:

```bash
# Spark 0 / node rank 0
cd /path/to/phase_triplet_heads_bundle/vjepa2_1
./configs/heads/single_tool/run_grasper_single_tool_six_heads_spark0.sh
```

```bash
# Spark 1 / node rank 1
cd /path/to/phase_triplet_heads_bundle/vjepa2_1
./configs/heads/single_tool/run_grasper_single_tool_six_heads_spark1.sh
```

Grasper ports:

```text
29731, 29732, 29733, 29734, 29735, 29736
```

For all two-Spark runs, start the spark0 and spark1 scripts at roughly the same
time. The scripts train the six heads in the same order and rendezvous one head
at a time.

## Current Hook Tool/Null Balance

Because labels are multi-label, these are positive-label counts rather than
exclusive class counts.

Train split:

```text
rows:       14238
null:       8428  (59.19%)
hook:       6992  (49.11%)
both:       1182  (8.30%)
only null:  7246  (50.89%)
only hook:  5810  (40.81%)
```

Val split:

```text
rows:       7041
null:       3597  (51.09%)
hook:       4279  (60.77%)
both:        835  (11.86%)
only null:  2762  (39.23%)
only hook:  3444  (48.91%)
```

## Validation Checks Already Used

Useful sanity checks before training:

- `py_compile` the builder.
- `bash -n` all run scripts.
- Parse all six YAML configs.
- Check each config's runtime and lookup CSV paths exist.
- Check `class_names_per_task` lengths match `num_classes_per_task`.
- Check conditioned lookup rows have aligned condition-list and label-sequence
  lengths.
- Check conditioned ids are in range for `condition_num_classes`.
- Check all generated runtime clip paths exist.

At the time this guide was updated, all `21,279` train/val runtime clip paths
existed for hook, clipper, and grasper.

## Extension To Other Tools

The builder accepts other `--tool` slugs from the same mapping JSON, but future
agents should not blindly create configs for additional tools until the user
asks. Each tool can have a different action and target label space, and each
config must be updated to match that tool's generated metadata.
