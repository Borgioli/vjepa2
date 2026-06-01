# SAR_RARP50 Action Backbone Tests (Polaris)

This benchmark reuses the `video_classification_frozen` eval path on
SAR_RARP50 action labels. SAR_RARP50 is robotic suturing with 8 per-frame
action classes — these configs measure the V-JEPA2.1 phase backbones as
action-recognition transfer representations.

> Originally authored on two Spark workstations with `torchrun --node-rank` and
> `MASTER_ADDR=10.200.1.1`. Ported to Polaris (PBS + mpiexec + PMI) on
> 2026-05-29. All commands below assume:
>
> - The Polaris repo at `/path/to/vjepa2_polaris/`
> - The Polaris venv at `/path/to/venvs/vjepa_polaris/`
> - The SAR raw tree at
>   `/path/to/phase_triplet_heads_bundle/globus/SAR_RARP50`

## Build clips and CSVs

### Synapse-backed prep (downloads + extracts + builds)

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1
/path/to/venvs/vjepa_polaris/bin/python \
  head_phases/prepare_sarrarp50_dataset.py --force
```

Synapse credentials are required. Use a personal access token:

```bash
export SYNAPSE_AUTH_TOKEN='<token>'
```

or create `~/.synapseConfig`:

```ini
[authentication]
authtoken = <token>
```

If Synapse asks for data-access terms, accept access for `syn31997652` in the
Synapse web UI and rerun.

### Already-downloaded archives

```bash
/path/to/venvs/vjepa_polaris/bin/python \
  head_phases/prepare_sarrarp50_dataset.py \
  --source-root /path/to/downloaded/SAR_RARP50_or_archives \
  --skip-download --force
```

### Already-extracted raw data root

```bash
/path/to/venvs/vjepa_polaris/bin/python \
  head_phases/prepare_sarrarp50_dataset.py --skip-download --force
```

## Build the ASFormer ctx3 sequence-label datasets (helper)

The helper builds both the 384 short-side variant (for the ViT-L distilled
backbone) and the 512 short-side variant (for the surg-finetuned phase3
backbone), then mirrors the CSVs into
`vjepa2_polaris/app/csv_head_models/` so the YAMLs find them:

```bash
cd /path/to/vjepa2_polaris
VJEPA_PYTHON=/path/to/venvs/vjepa_polaris/bin/python \
  bash scripts/build_sarrarp50_asformer_datasets.sh
```

Outputs:

```text
globus/sarrarp50_action_clips_3x16f_4fps/                                         # 384 short side
globus/sarrarp50_action_clips_3x16f_4fps_512/                                     # 512 short side
vjepa2_polaris/app/csv_head_models/sarrarp50_actions_4fps_asformer_ctx3_seq_*.csv
vjepa2_polaris/app/csv_head_models/sarrarp50_actions_4fps_512_asformer_ctx3_seq_*.csv
```

## Build options (direct builder invocation)

The helper above wraps `build_sarrarp50_action_dataset.py`. Useful flags if
you want to build only one variant or change defaults:

```text
--variant 4fps                # temporal sampling rate (4fps or 10hz)
--context-clips 3             # ctx3: 3 V-JEPA clips per sample
--sequence-labels             # one int label per temporal token (24 for ctx3)
--tubelet-size 2              # must match the V-JEPA backbone
--resize-short-side 512       # for the 512 phase3 variant (omit for 384)
--sample-stride 10            # one center sample per second from 10Hz SAR labels
--class-weight-mode balanced  # or sqrt_balanced / proportional / none
--unpack-missing-rgb          # if the dataset has video_left.avi but no rgb/
--jobs 4                      # parallel decoding workers
```

The `--sample-stride 10` default keeps one center frame per second from the
10Hz SAR action labels. For the ctx3 ASFormer setup this slides the 12-second
context window every 1 second while still predicting the center timestamp of
the target clip. Use `--sample-stride 40` only if you want one sample per
4-second V-JEPA clip; `--sample-stride 1` for every SAR action frame.

## Run the backbones

### Single-node Polaris run (1 node × 4 GPU)

```bash
cd /path/to/vjepa2_polaris

VJEPA_NUM_NODES=1 VJEPA_NUM_GPUS=4 VJEPA_STRONG_SCALE=1 \
VJEPA_ACCOUNT=<your-PBS-allocation> \
VJEPA_PYTHON=/path/to/venvs/vjepa_polaris/bin/python \
VJEPA_PARTITION=debug VJEPA_TIME_MIN=60 \
bash run_sarrarp50_asformer_polaris.sh \
  configs/heads/sarrarp50/sarrarp50_actions_4fps_asformer_vitl_dist_vitg_ema_384_frozen.yaml
```

### Multi-node Polaris run (2 nodes × 4 GPU)

```bash
VJEPA_NUM_NODES=2 VJEPA_NUM_GPUS=4 VJEPA_STRONG_SCALE=1 \
VJEPA_ACCOUNT=<your-PBS-allocation> \
VJEPA_PYTHON=/path/to/venvs/vjepa_polaris/bin/python \
VJEPA_PARTITION=preemptable VJEPA_TIME_MIN=180 \
bash run_sarrarp50_asformer_polaris.sh \
  configs/heads/sarrarp50/sarrarp50_actions_4fps_512_asformer_surg_finetuned_phase3_target_512_frozen.yaml
```

### Both selected ASFormer configs (chained via PBS afterok)

```bash
bash run_sarrarp50_asformer_polaris.sh \
  configs/heads/sarrarp50/sarrarp50_actions_4fps_asformer_vitl_dist_vitg_ema_384_frozen.yaml \
  configs/heads/sarrarp50/sarrarp50_actions_4fps_512_asformer_surg_finetuned_phase3_target_512_frozen.yaml
```

### Dry run (build PBS scripts, don't submit)

```bash
VJEPA_DRY_RUN=1 bash run_sarrarp50_asformer_polaris.sh <yaml>
```

Each launcher invocation:

1. Calls `scripts/prepare_runtime_config.py` which writes a runtime-adapted
   copy of the YAML under `.runtime_configs/n{N}g{G}[_strong]/…` (per-rank
   batch rescaling, topology suffix appended to the output folder).
2. Inspects `<folder>/video_classification_frozen[/<tag>]/latest.pt` to decide
   `fresh` / `resume` / `complete` (complete jobs are skipped).
3. Invokes `python -m app.main_dist_polaris` which writes
   `<folder>/_submit.pbs` and `qsub`s it (chaining via `afterok` when called
   with multiple configs).

## ASFormer head

ASFormer is selected through:

```yaml
experiment:
  classifier:
    name: asformer
```

Implementation: `vjepa2_polaris/src/models/asformer_head.py`.
Trainer hook: `vjepa2_polaris/evals/video_classification_frozen/eval.py`.

Current SAR_RARP50 ASFormer configs:

```text
configs/heads/sarrarp50/sarrarp50_actions_4fps_512_asformer_surg_finetuned_phase3_target_512_frozen.yaml
configs/heads/sarrarp50/sarrarp50_actions_4fps_asformer_vitl_dist_vitg_ema_384_frozen.yaml
```

The ASFormer path uses three consecutive 4-second V-JEPA clips as
previous/target/future context. Each sample contains `3 × 16` frames; the
dataloader splits it with `num_segments: 3`, and V-JEPA encodes each 16-frame
clip independently. With `preserve_clip_dim: true`, the wrapper preserves
tokens as `[B, 3, T_clip*S, D]`; ASFormer reshapes to `[B, 3, T_clip, S, D]`,
attention-pools spatial tokens per `(clip, time)` step, runs temporal
self-attention + a stack of dilated temporal convolutions, and emits per-token
logits `[B, 24, num_classes]`.

The configs set:

```yaml
experiment:
  data:
    sequence_labels: true
    num_segments: 3
  classifier:
    name: asformer
    asformer_kwargs:
      num_clips: 3
      tokens_per_clip: 8
      temporal_tokens: 24
      return_sequence: true
model_kwargs:
  wrapper_kwargs:
    preserve_clip_dim: true
```

## Action IDs

```text
0 other
1 picking_up_the_needle
2 positioning_the_needle_tip
3 pushing_the_needle_through_the_tissue
4 pulling_the_needle_out_of_the_tissue
5 tying_a_knot
6 cutting_the_suture
7 returning_dropping_the_needle
```

## Output layout

Per-config training writes to:

```text
<folder>/video_classification_frozen/<tag>/log_r{rank}.csv
<folder>/video_classification_frozen/<tag>/latest.pt
<folder>/_submit.pbs
<folder>/params-pretrain.yaml
<folder>/git-info.txt
```

with `<folder>` taken from the YAML (and suffixed `_n{N}g{G}[_strong]` by the
runtime-config tool).
