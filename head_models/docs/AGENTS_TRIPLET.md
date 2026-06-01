# Future Codex Guide: Latest Tool5 Multi-Label Heads

This repo has many old head configs. For the current Tool5 multi-label triplet work, focus on the current three-stage pipeline heads unless the user explicitly asks for older experiments. There is also one classic Tool5 tool-only baseline kept beside the main tool-count head for clean comparisons. Note: Tool5 introduces the `clipper` tool, drops `clipper | null` resting variations, merges the `cut` and `dissect` actions into `cut/dissect`, and merges functionally overlapping tools: `bipolar` → `grasper`, `harmonic shears` → `scissors`.

Terminology note: in this repo, **verb** and **action** are often used for the same triplet component. The latest pipeline is:

1. Predict tools present in the clip.
2. Predict verbs/actions conditioned on each tool.
3. Predict targets conditioned on each `(tool, verb/action)` pair.

All current heads freeze the V-JEPA encoder and train only small classifier heads on top of:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt
checkpoint_key: ema_encoder
model_name: vit_large
head_type: token_aggregation or conditioned_token_aggregation
token_pool: mean
loss_name: weighted_bce
```

Representation note: the current wrapper flattens temporal and spatial tokens together into one sequence in `/path/to/phase_triplet_heads_bundle/vjepa2_1/evals/triplet_recog_frozen/modelcustom/vit_encoder_multiclip.py` around line 134, where the per-clip outputs are reshaped to `[B, T, S, D]`, concatenated across time, and then flattened across `(T, S)`.

## Use These Current Configs

### 1. Tool Multi-Label + Count Head

Use this config:

```text
configs/heads/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5.yaml
```

For a classic no-count Tool5 token-aggregation baseline, use:

```text
configs/heads/original_backbone/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5_classic.yaml
```

Run the count-augmented tool head with:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python evals/main.py \
  --fname configs/heads/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5.yaml \
  --devices cuda:0 \
  --debugmode True
```

Run the classic no-count Tool5 baseline with:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python evals/main.py \
  --fname configs/heads/original_backbone/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5_classic.yaml \
  --devices cuda:0 \
  --debugmode True
```

Two-Spark launch commands for the count-augmented tool head:

Spark 0 / node rank 0:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

NCCL_SOCKET_IFNAME=enp1s0f0np0 \
GLOO_SOCKET_IFNAME=enp1s0f0np0 \
TORCH_DIST_TIMEOUT_SECONDS=300 \
/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/torchrun \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank=0 \
  --master-addr=10.200.1.1 \
  --master-port=29610 \
  -m evals.main \
  --fname /path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5.yaml \
  --devices cuda:0 \
  --debugmode 1 \
  --use_fsdp
```

Spark 1 / node rank 1:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

NCCL_SOCKET_IFNAME=enp1s0f0np0 \
GLOO_SOCKET_IFNAME=enp1s0f0np0 \
TORCH_DIST_TIMEOUT_SECONDS=300 \
/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/torchrun \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank=1 \
  --master-addr=10.200.1.1 \
  --master-port=29610 \
  -m evals.main \
  --fname /path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5.yaml \
  --devices cuda:0 \
  --debugmode 1 \
  --use_fsdp
```

#### Tool Head With Surg-Finetuned Phase2 Backbone

Use this separate config when training the Tool5 tool-count head on the surg-finetuned backbone:

```text
configs/heads/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5_surg_finetuned_phase2.yaml
```

This config loads:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/chekpoints_vjepar_surg_finetuned/single4_all_datasets_phase2_main_g8_latest.pth.tar
checkpoint_key: target_encoder
```

The checkpoint contains `encoder` and `target_encoder`, not `ema_encoder`. Use `target_encoder` for the frozen representation, matching the older finetuned-head configs in this repo.

This config intentionally writes to a different output folder/tag so it does not overwrite the latest Tool5 head trained with the other backbone:

```text
folder: /path/to/phase_triplet_heads_bundle/vjepa2_1/trained_heads/triplet_multilabel_tool_count_tool5_native_surg_finetuned_phase2
tag: triplet_multilabel_tool_count_head_training_vitl_latest_token_aggregation_tool5_native_surg_finetuned_phase2
```

Two-Spark launch commands for the surg-finetuned Phase2 backbone tool head:

Spark 0 / node rank 0:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

NCCL_SOCKET_IFNAME=enp1s0f0np0 \
GLOO_SOCKET_IFNAME=enp1s0f0np0 \
TORCH_DIST_TIMEOUT_SECONDS=300 \
/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/torchrun \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank=0 \
  --master-addr=10.200.1.1 \
  --master-port=29620 \
  -m evals.main \
  --fname /path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5_surg_finetuned_phase2.yaml \
  --devices cuda:0 \
  --debugmode 1 \
  --use_fsdp
```

Spark 1 / node rank 1:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

NCCL_SOCKET_IFNAME=enp1s0f0np0 \
GLOO_SOCKET_IFNAME=enp1s0f0np0 \
TORCH_DIST_TIMEOUT_SECONDS=300 \
/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/torchrun \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank=1 \
  --master-addr=10.200.1.1 \
  --master-port=29620 \
  -m evals.main \
  --fname /path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5_surg_finetuned_phase2.yaml \
  --devices cuda:0 \
  --debugmode 1 \
  --use_fsdp
```

Two-Spark launch commands for the classic no-count Tool5 baseline:

Spark 0 / node rank 0:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

NCCL_SOCKET_IFNAME=enp1s0f0np0 \
GLOO_SOCKET_IFNAME=enp1s0f0np0 \
TORCH_DIST_TIMEOUT_SECONDS=300 \
/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/torchrun \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank=0 \
  --master-addr=10.200.1.1 \
  --master-port=29611 \
  -m evals.main \
  --fname /path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/original_backbone/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5_classic.yaml \
  --devices cuda:0 \
  --debugmode 1 \
  --use_fsdp
```

Spark 1 / node rank 1:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

NCCL_SOCKET_IFNAME=enp1s0f0np0 \
GLOO_SOCKET_IFNAME=enp1s0f0np0 \
TORCH_DIST_TIMEOUT_SECONDS=300 \
/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/torchrun \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank=1 \
  --master-addr=10.200.1.1 \
  --master-port=29611 \
  -m evals.main \
  --fname /path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/original_backbone/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5_classic.yaml \
  --devices cuda:0 \
  --debugmode 1 \
  --use_fsdp
```

#### Sequential Three-Head Original-Backbone Pipeline

The three original-backbone pipeline configs live under:

```text
configs/heads/original_backbone/
```

They use:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt
checkpoint_key: ema_encoder
```

They run in this order:

```text
1. tool-only head
2. verb/action conditioned on tool
3. target conditioned on tool and verb/action
```

Use the paired scripts to run all three heads sequentially across both Spark machines:

Spark 0 / node rank 0:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1
./configs/heads/original_backbone/run_tool5_original_backbone_pipeline_spark0.sh
```

Spark 1 / node rank 1:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1
./configs/heads/original_backbone/run_tool5_original_backbone_pipeline_spark1.sh
```

The scripts use ports `29611`, `29612`, and `29613` for the three sequential distributed launches.

#### Classic Tool-Only Head With Surg-Finetuned Phase2 Backbone

Use this config when training only the 5-tool multi-label head on the surg-finetuned backbone, without the auxiliary `tool_count` task:

```text
configs/heads/surg_finetuned_phase2/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5_classic_surg_finetuned_phase2.yaml
```

This config trains only:

```yaml
task_names: ['tool']
task_modes: ['multilabel']
num_classes_per_task: [5]
```

It loads the surg-finetuned Phase2 checkpoint with `target_encoder`:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/chekpoints_vjepar_surg_finetuned/single4_all_datasets_phase2_main_g8_latest.pth.tar
checkpoint_key: target_encoder
```

It writes to a separate output folder/tag:

```text
folder: /path/to/phase_triplet_heads_bundle/vjepa2_1/trained_heads/triplet_multilabel_tool_only_tool5_native_classic_surg_finetuned_phase2
tag: triplet_multilabel_tool_only_head_training_vitl_latest_token_aggregation_tool5_native_classic_surg_finetuned_phase2
```

Two-Spark launch commands:

Spark 0 / node rank 0:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

NCCL_SOCKET_IFNAME=enp1s0f0np0 \
GLOO_SOCKET_IFNAME=enp1s0f0np0 \
TORCH_DIST_TIMEOUT_SECONDS=300 \
/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/torchrun \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank=0 \
  --master-addr=10.200.1.1 \
  --master-port=29621 \
  -m evals.main \
  --fname /path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/surg_finetuned_phase2/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5_classic_surg_finetuned_phase2.yaml \
  --devices cuda:0 \
  --debugmode 1 \
  --use_fsdp
```

Spark 1 / node rank 1:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

NCCL_SOCKET_IFNAME=enp1s0f0np0 \
GLOO_SOCKET_IFNAME=enp1s0f0np0 \
TORCH_DIST_TIMEOUT_SECONDS=300 \
/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/torchrun \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank=1 \
  --master-addr=10.200.1.1 \
  --master-port=29621 \
  -m evals.main \
  --fname /path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/surg_finetuned_phase2/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5_classic_surg_finetuned_phase2.yaml \
  --devices cuda:0 \
  --debugmode 1 \
  --use_fsdp
```

#### Sequential Three-Head Surg-Finetuned Phase2 Pipeline

The three Phase2 surg-finetuned pipeline configs live under:

```text
configs/heads/surg_finetuned_phase2/
```

They run in this order:

```text
1. tool-only head
2. verb/action conditioned on tool
3. target conditioned on tool and verb/action
```

Use the paired scripts to run all three heads sequentially across both Spark machines. Start the Spark 0 script on node rank 0 and the Spark 1 script on node rank 1:

Spark 0 / node rank 0:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1
./configs/heads/surg_finetuned_phase2/run_tool5_surg_finetuned_phase2_pipeline_spark0.sh
```

Spark 1 / node rank 1:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1
./configs/heads/surg_finetuned_phase2/run_tool5_surg_finetuned_phase2_pipeline_spark1.sh
```

The scripts use ports `29621`, `29622`, and `29623` for the three sequential distributed launches.

Purpose:

```text
P(tools | video)
P(num_tools | video)
```

One clip in, two outputs:

```text
tool:       5-class multi-hot vector
tool_count: 6-way class for how many Tool5 tools are present, 0 through 5
```

Tool classes (after merge: bipolar→grasper, harmonic shears→scissors):

```text
0 = clipper
1 = grasper  (includes bipolar)
2 = hook
3 = irrigator
4 = scissors (includes harmonic shears)
```

Main data files:

```text
app/csv_head_models/triplet_multilabel_train_native_tool5_runtime.csv
app/csv_head_models/triplet_multilabel_val_native_tool5_runtime.csv
app/csv_head_models/triplet_multilabel_train_native_tool5.csv
app/csv_head_models/triplet_multilabel_val_native_tool5.csv
app/csv_head_models/triplet_multilabel_native_tool5_metadata.json
```

Important config fields:

```yaml
label_mode: 'multilabel'
head_type: 'token_aggregation'
multilabel_label_fields: ['tool_multihot']
multilabel_count_source_fields: ['tool_multihot']
task_names: ['tool', 'tool_count']
task_modes: ['multilabel', 'multiclass']
num_classes_per_task: [5, 6]
class_names_per_task:
  - ['clipper', 'grasper', 'hook', 'irrigator', 'scissors']
  - ['0_tools', '1_tool', '2_tools', '3_tools', '4_tools', '5_tools']
multilabel_thresholds: [0.75, 0.5, 0.5, 0.5, 0.5]
task_loss_names: ['weighted_bce', 'ce']
```

Classic baseline fields:

```yaml
multilabel_label_fields: ['tool_multihot']
task_names: ['tool']
task_modes: ['multilabel']
num_classes_per_task: [5]
task_loss_names: ['weighted_bce']
multilabel_thresholds: [0.75, 0.5, 0.5, 0.5, 0.5]
```

`multilabel_thresholds` is read from `experiment.data` and follows the class order above:

```text
clipper, grasper, hook, irrigator, scissors
```

The scalar `multilabel_threshold` remains a fallback for older configs, but the current Tool5 tool head should use the per-class list. Do not put threshold fields under `multihead_kwargs`; those kwargs are optimizer settings.

`multilabel_count_source_fields` derives the `tool_count` label from the same `tool_multihot` column at dataloader time, so this auxiliary head does not require regenerating the Tool5 CSVs.

Output folder:

```text
trained_heads/triplet_multilabel_tool_count_tool5_native/video_classification_frozen/triplet_multilabel_tool_count_head_training_vitl_latest_token_aggregation_tool5_native/
```

Classic no-count output folder:

```text
trained_heads/triplet_multilabel_tool_only_tool5_native_classic/video_classification_frozen/triplet_multilabel_tool_only_head_training_vitl_latest_token_aggregation_tool5_native_classic/
```

### 2. Verb/Action Multi-Label Head Conditioned On Tools

Use this config:

```text
configs/heads/original_backbone/verb_multilabel_conditioned_on_tool_train_vitl_latest_token_aggregation_tool5.yaml
```

For the surg-finetuned Phase2 backbone, use this separate config:

```text
configs/heads/surg_finetuned_phase2/verb_multilabel_conditioned_on_tool_train_vitl_latest_token_aggregation_tool5_surg_finetuned_phase2.yaml
```

It loads:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/chekpoints_vjepar_surg_finetuned/single4_all_datasets_phase2_main_g8_latest.pth.tar
checkpoint_key: target_encoder
```

and writes to:

```text
trained_heads/verb_multilabel_conditioned_on_tool_tool5_native_surg_finetuned_phase2/video_classification_frozen/verb_multilabel_conditioned_on_tool_head_training_vitl_latest_token_aggregation_tool5_native_surg_finetuned_phase2/
```

Purpose:

```text
P(verbs/actions | video, each tool condition)
```

One clip row in, all active tool conditions for that clip in, one 6-class verb/action multi-hot vector out for each valid tool condition. Do **not** duplicate the clip once per tool. This head predicts only verbs/actions; tool ids are conditioning inputs.

Input/output shape intent:

```text
condition labels: [batch, num_tools, 1]
verb labels:      [batch, num_tools, 6]
verb logits:      [batch, num_tools, 6]
condition mask:   [batch, num_tools]
```

Condition classes:

```text
tool condition:
0 = clipper
1 = grasper  (includes bipolar)
2 = hook
3 = irrigator
4 = scissors (includes harmonic shears)
```

Verb/action classes:

```text
0 = coagulation
1 = grasp/retract
2 = null
3 = cut/dissect
4 = clean
5 = clip
```

Main data files:

```text
app/csv_head_models/conditioned_action_multilabel_train_native_tool5_runtime.csv
app/csv_head_models/conditioned_action_multilabel_val_native_tool5_runtime.csv
app/csv_head_models/conditioned_action_multilabel_train_native_tool5.csv
app/csv_head_models/conditioned_action_multilabel_val_native_tool5.csv
app/csv_head_models/conditioned_action_multilabel_native_tool5_metadata.json
```

Builder:

```text
app/csv_head_models/build_conditioned_action_multilabel_dataset_tool5.py
```

Important config fields:

```yaml
label_mode: 'multilabel'
conditioning_mode: 'multilabel_multi_condition_task'
head_type: 'conditioned_token_aggregation'
condition_num_classes: 5
multilabel_label_fields: ['verb_multihots']
multilabel_condition_fields: ['conditioning_tool_ids']
task_names: ['verb']
num_classes_per_task: [6]
```

Output folder:

```text
trained_heads/verb_multilabel_conditioned_on_tool_tool5_native/video_classification_frozen/verb_multilabel_conditioned_on_tool_head_training_vitl_latest_token_aggregation_tool5_native/
```

Current generated dataset size (with rare-only recovered clips + tool merges):

```text
train clips:           4005
train tool conditions: 7287
val clips:             2083
val tool conditions:   3814
```

Iterations per epoch should be based on the clip rows, not the conditioned-tool count.

Do not confuse this with:

```text
configs/heads/action_multilabel_only_train_vitl_latest_token_aggregation_tool4.yaml
```

That older/current artifact predicts actions from the clip after filtering to Tool4 annotations, but it is not the focused conditioned middle head. Use it only if the user explicitly asks for an unconditioned action-only head.

### 3. Target Multi-Label Head Conditioned On Tool And Verb/Action

Use this config:

```text
configs/heads/original_backbone/target_multilabel_conditioned_on_tool_verb_train_vitl_latest_token_aggregation_tool5.yaml
```

For the surg-finetuned Phase2 backbone, use this separate config:

```text
configs/heads/surg_finetuned_phase2/target_multilabel_conditioned_on_tool_verb_train_vitl_latest_token_aggregation_tool5_surg_finetuned_phase2.yaml
```

It loads:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/chekpoints_vjepar_surg_finetuned/single4_all_datasets_phase2_main_g8_latest.pth.tar
checkpoint_key: target_encoder
```

and writes to:

```text
trained_heads/target_multilabel_conditioned_on_tool_verb_tool5_native_surg_finetuned_phase2/video_classification_frozen/target_multilabel_conditioned_on_tool_verb_head_training_vitl_latest_token_aggregation_tool5_native_surg_finetuned_phase2/
```

Purpose:

```text
P(targets | video, each(tool, verb/action) pair)
```

This one is different from the middle head: a single clip can contain multiple `(tool, verb/action)` conditioning pairs. Do **not** duplicate the clip once per pair. The current design encodes the clip once, pads the condition pairs in the batch, and predicts a 16-class target multi-hot vector for each valid pair.

Input/output shape intent:

```text
condition labels: [batch, num_pairs, 2]
target labels:    [batch, num_pairs, 16]
target logits:    [batch, num_pairs, 16]
condition mask:   [batch, num_pairs]
```

Condition classes:

```text
tool condition:
0 = clipper
1 = grasper  (includes bipolar)
2 = hook
3 = irrigator
4 = scissors (includes harmonic shears)

verb/action condition:
0 = coagulation
1 = grasp/retract
2 = null
3 = cut/dissect
4 = clean
5 = clip
```

Target classes:

```text
0  = connective tissue
1  = cystic duct
2  = adhesion
3  = cystic pedicle
4  = gallbladder
5  = gallbladder wall
6  = liver
7  = null
8  = peritoneum
9  = suture
10 = cystic artery
11 = fallciform ligament
12 = gut
13 = omentum
14 = specimen bag
15 = fluid
```

Main data files:

```text
app/csv_head_models/conditioned_target_multilabel_train_native_tool5_runtime.csv
app/csv_head_models/conditioned_target_multilabel_val_native_tool5_runtime.csv
app/csv_head_models/conditioned_target_multilabel_train_native_tool5.csv
app/csv_head_models/conditioned_target_multilabel_val_native_tool5.csv
app/csv_head_models/conditioned_target_multilabel_native_tool5_metadata.json
```

Builder:

```text
app/csv_head_models/build_conditioned_target_multilabel_dataset_tool5.py
```

Important config fields:

```yaml
label_mode: 'multilabel'
conditioning_mode: 'multilabel_multi_condition_task'
head_type: 'conditioned_token_aggregation'
token_pool: 'mean'
condition_num_classes: [5, 6]
multilabel_label_fields: ['target_multihots']
multilabel_condition_fields: ['conditioning_tool_ids', 'conditioning_verb_ids']
task_names: ['target']
num_classes_per_task: [16]
```

Output folder:

```text
trained_heads/target_multilabel_conditioned_on_tool_verb_tool5_native/video_classification_frozen/target_multilabel_conditioned_on_tool_verb_head_training_vitl_latest_token_aggregation_tool5_native/
```

Current generated dataset size (with rare-only recovered clips + tool merges):

```text
train clips:           4005
train condition pairs: 7702
val clips:             2083
val condition pairs:   3962
```

Iterations per epoch should be based on the clip rows, not the condition-pair count.

## Code Paths That Matter

The latest multi-label mechanics live here:

```text
evals/triplet_recog_frozen/dataset_wrapper.py
evals/triplet_recog_frozen/eval.py
evals/triplet_recog_frozen/models.py
evals/triplet_recog_frozen/utils.py
```

Important classes/modes:

```text
MultiLabelTaskLabelWrapper
ConditionedMultiLabelTaskLabelWrapper
MultiConditionMultiLabelTaskLabelWrapper
TokenAggregationMultiTaskClassifier
ConditionedTokenAggregationMultiTaskClassifier
multilabel_count_source_fields
task_modes: ['multilabel', 'multiclass']
task_loss_names
conditioning_mode: multilabel_conditioned_task
conditioning_mode: multilabel_multi_condition_task
```

Multi-label evaluation supports either the legacy scalar threshold:

```yaml
multilabel_threshold: 0.5
```

or per-class thresholds:

```yaml
multilabel_thresholds: [0.75, 0.5, 0.5, 0.5, 0.5]
```

For single-task heads, a flat list is interpreted as class order. For mixed heads with exactly one multi-label task, such as the current `['tool', 'tool_count']` head, a flat list is also interpreted as the class order for the multi-label task. For multiple multi-label tasks, use a nested list with one threshold list per multi-label task.

For multi-condition verb/action and target training, padded condition rows must be masked in loss and metrics. The mask is returned by `MultiConditionMultiLabelTaskLabelWrapper`, collated in `multitask_collate_fn`, and used by `compute_task_loss(..., sample_mask=conditioning_masks)`.

The conditioned head must pool over the token dimension, not the condition dimension. If a future run errors with:

```text
Target size [B, P, 16] must be same as input size [B, 2048, 16]
```

then `ConditionedTokenAggregationMultiTaskClassifier._aggregate_tokens()` is pooling the wrong dimension. The correct target-conditioned output shape is `[B, P, 16]`.

### Multi-Condition Performance Note

The verb/action and target heads both use multiple conditioning rows per clip. The naive conditioned-token implementation materializes:

```text
[batch, conditions, tokens, dim]
```

The target head is still the heavier case because it has up to 6 `(tool, verb/action)` pairs per clip, two conditioning inputs, and 16 target classes. The middle verb/action head has up to 4 tool conditions and 6 verb/action classes.

For the current multi-condition configs, `token_pool: 'mean'` and the classifier is linear, so `ConditionedTokenAggregationMultiTaskClassifier.forward()` should use the optimized algebraic path:

```text
mean_tokens(Linear(x + condition)) == mean_tokens(Linear(x)) + Linear_without_bias(condition)
```

This avoids materializing `[batch, conditions, tokens, dim]` and keeps logits equivalent to the expanded computation up to floating-point noise. Do not remove this optimization unless the head architecture or pooling mode changes. For non-mean token pooling, the expanded path is still used.

The target config may set:

```yaml
persistent_workers: False
dataloader_timeout: 120
```

Those fields only affect dataloader worker lifecycle and stall reporting. They do not change labels, model outputs, losses, metrics, sampler order, or F1 calculation.

## Ignore These Unless Asked

Avoid getting pulled into older configs unless the user explicitly asks:

```text
configs/heads/verb_conditioned_on_tool_train_vitl_latest_token_aggregation.yaml
configs/heads/target_conditioned_on_tool_verb_train_vitl_latest_token_aggregation.yaml
configs/heads/triplet_train*.yaml
configs/heads/triplet_multilabel_train_vitl_latest*.yaml
configs/heads/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5.yaml
configs/heads/*phase*.yaml
configs/heads/action_multilabel_only_train_vitl_latest_token_aggregation_tool4.yaml
```

The old `verb_conditioned_on_tool...` and `target_conditioned_on_tool_verb...` configs are multiclass triplet-conditioned heads, not the latest Tool4 multi-label heads.

## Tool Merges

The original dataset has 9 tool classes. Some are functionally equivalent for the actions they perform:

- **bipolar** (old ID 0) → merged into **grasper** (old ID 2): both perform grasp/retract and coagulation on the same targets
- **harmonic shears** (old ID 3) → merged into **scissors** (old ID 7): both perform cut/dissect on the same targets

The merge is applied in `build_triplet_multilabel_dataset_from_native_multilabel.py` via the `--merge-tools` flag (default: `0:2 3:7`). It replaces the source tool ID with the destination tool ID before any exclusion logic runs. After merging, 4 tools are still excluded (bipolar, harmonic shears, needle driver, stapler) but bipolar and harmonic shears have zero data remaining (all remapped), so only needle driver and stapler assignments are actually dropped.

Impact of the merge on the recovered dataset:

```text
Before merge: 253 train / 174 val clips with no supported triplets (all-zero tool labels)
After merge:   18 train /   9 val clips with no supported triplets
Dropped label assignments: 1118+515 → 190+39  (recovered ~83-92% of previously dropped labels)
Unique triplet classes: 66 → 74  (8 new verb-target combos from merged tools)
```

To disable merging and revert to the pre-merge behavior:

```bash
python3 app/csv_head_models/build_triplet_multilabel_dataset_from_native_multilabel.py --merge-tools
```

## Short-Clip Recovery

The original exporter drops annotation segments shorter than 16 frames and tail fragments after chunking. A recovery script replays the exporter logic against the frame-level annotations, finds every dropped segment, and maps it to an existing sliding-window clip that covers the same frame range.

**Selective recovery (--max-original-count):** Indiscriminate recovery of all dropped segments was found to degrade performance by flooding already dominant classes (e.g. grasper|grasp/retract|gallbladder) with noisy, partially-annotated clips. The script now filters by label frequency: only native labels that appear at most N times in the original split are recovered (default N=50). This targets underrepresented tail labels while leaving well-represented classes untouched.

Key files:

```text
/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/recover_short_clips.py
/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/triplet_labels/               (frame-level annotation JSONs, Batch0-9)
/path/to/phase_triplet_heads_bundle/globus/triplet/clips/yt_robotic_chole_Batch*_4.0fps/  (sliding-window clips, all 9 batches)
```

Output (augmented multilabel CSVs that include both original and recovered rows):

```text
/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/yt_robotic_chole_triplets_multilabel_train_recovered.csv  (4015 rows, +19% vs original 3385)
/path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/yt_robotic_chole_triplets_multilabel_val_recovered.csv    (2096 rows, +23% vs original 1700)
```

To re-run recovery (only needed if annotations or sliding-window clips change):

```bash
cd /path/to/phase_triplet_heads_bundle/globus/surgenet_triplets
python3 recover_short_clips.py                             # rare labels only, threshold=50 (recommended)
python3 recover_short_clips.py --max-original-count 100    # relax threshold to 100
python3 recover_short_clips.py --max-original-count 0      # recover ALL labels (no filtering, not recommended)
python3 recover_short_clips.py --recover-tails              # also recover tail fragments
python3 recover_short_clips.py --dry-run                    # stats only, no CSV output
```

The recovered clips come from `globus/triplet/clips/` (sliding-window clips at 1-second stride). These are the same format as the per-triplet clips (h264, 4 fps, 16 frames) but may differ in resolution per batch. The V-JEPA dataloader handles resizing. Recovered clips contain the annotated triplet for only a fraction of the 16 frames (the annotation was <16 frames); this is a noisier signal but fine for multi-label BCE training.

## Regenerating Data

Run these only if the grouped Tool5 source CSVs changed. To use the recovered (augmented) dataset with tool merges (the default), pass the `_recovered` CSVs as inputs to the first builder. The `--merge-tools` flag defaults to `0:2 3:7` (bipolar→grasper, harmonic shears→scissors):

```bash
python3 app/csv_head_models/build_triplet_multilabel_dataset_from_native_multilabel.py \
  --train-input /path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/yt_robotic_chole_triplets_multilabel_train_recovered.csv \
  --val-input /path/to/phase_triplet_heads_bundle/globus/surgenet_triplets/yt_robotic_chole_triplets_multilabel_val_recovered.csv

python3 app/csv_head_models/build_conditioned_action_multilabel_dataset_tool5.py
python3 app/csv_head_models/build_conditioned_target_multilabel_dataset_tool5.py
```

To revert to the original (non-recovered) dataset, omit the `--train-input` / `--val-input` flags (defaults point to the originals). To disable tool merges, pass an empty `--merge-tools`.

The key grouped Tool5 source files are:

```text
app/csv_head_models/triplet_multilabel_train_native_tool5.csv
app/csv_head_models/triplet_multilabel_val_native_tool5.csv
```

## Two-Node Torchrun Template

Before two-node training, make sure both Spark machines have the same code, configs, CSVs, and video clips. A prudent sync from node 0 to node 1 that preserves remote trained-head checkpoints is:

```bash
rsync -avh --delete --info=progress2 --partial \
  --exclude 'vjepa2_1/trained_heads/' \
  -e "ssh -T -o Compression=no -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  /path/to/phase_triplet_heads_bundle/ \
  sitldvrk@10.200.1.2:/path/to/phase_triplet_heads_bundle/
```

Run the dry-run form first when unsure:

```bash
rsync -avhn --delete --info=progress2 --partial \
  --exclude 'vjepa2_1/trained_heads/' \
  -e "ssh -T -o Compression=no -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  /path/to/phase_triplet_heads_bundle/ \
  sitldvrk@10.200.1.2:/path/to/phase_triplet_heads_bundle/
```

Do not sync while training is active. If checkpoint consistency is needed, sync `vjepa2_1/trained_heads/` deliberately from the node with the checkpoints to keep.

Node 0:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

NCCL_SOCKET_IFNAME=enp1s0f0np0 \
GLOO_SOCKET_IFNAME=enp1s0f0np0 \
TORCH_DIST_TIMEOUT_SECONDS=300 \
/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/torchrun \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank=0 \
  --master-addr=10.200.1.1 \
  --master-port=29612 \
  -m evals.main \
  --fname /path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/original_backbone/target_multilabel_conditioned_on_tool_verb_train_vitl_latest_token_aggregation_tool5.yaml \
  --devices cuda:0 \
  --debugmode 1 \
  --use_fsdp
```

Node 1:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1

NCCL_SOCKET_IFNAME=enp1s0f0np0 \
GLOO_SOCKET_IFNAME=enp1s0f0np0 \
TORCH_DIST_TIMEOUT_SECONDS=300 \
/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/torchrun \
  --nnodes=2 \
  --nproc-per-node=1 \
  --node-rank=1 \
  --master-addr=10.200.1.1 \
  --master-port=29612 \
  -m evals.main \
  --fname /path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/original_backbone/target_multilabel_conditioned_on_tool_verb_train_vitl_latest_token_aggregation_tool5.yaml \
  --devices cuda:0 \
  --debugmode 1 \
  --use_fsdp
```

Change only `--fname` and, if needed, `--master-port` when training one of the other two focused heads.
