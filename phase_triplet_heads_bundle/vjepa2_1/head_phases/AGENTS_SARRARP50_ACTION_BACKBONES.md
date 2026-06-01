# Future Codex Guide: SAR_RARP50 Action Backbone Tests (Polaris)

This workflow tests the current phase/video V-JEPA2.1 backbones on SAR_RARP50
action recognition. SAR_RARP50 is robotic suturing with 8 action classes, not
cholecystectomy phase recognition, so describe results as action-recognition
transfer tests for the phase backbones.

> History: this benchmark was originally authored on the two-Spark workstation
> setup using `torchrun --node-rank` and `MASTER_ADDR=10.200.1.1`. As of
> 2026-05-29 the runner, eval code, and ASFormer head were ported to Polaris
> (PBS + mpiexec + PMI). All commands below assume the Polaris repo at
> `/path/to/vjepa2_polaris` and the Polaris venv at
> `/path/to/venvs/vjepa_polaris`.

## File map

Dataset builder (shipped with this bundle, runs anywhere with the raw data):

```text
head_phases/build_sarrarp50_action_dataset.py
head_phases/prepare_sarrarp50_dataset.py
```

Polaris-side code (lives in `vjepa2_polaris/`):

```text
src/models/asformer_head.py
evals/video_classification_frozen/eval.py
evals/video_classification_frozen/modelcustom/vit_encoder_multiclip.py   # preserve_clip_dim
src/datasets/video_dataset.py                                            # sequence_labels parsing
src/datasets/data_manager.py
configs/heads/sarrarp50/
scripts/build_sarrarp50_asformer_datasets.sh
run_sarrarp50_asformer_polaris.sh
app/main_dist_polaris.py                                                 # eval_name routing
```

User-facing notes:

```text
head_phases/SARRARP50_ACTION_BACKBONE_TESTS.md
```

## Step 1: Build SAR clips and probe CSVs

On a Polaris login node with the raw SAR_RARP50 tree mounted under
`/path/to/phase_triplet_heads_bundle/globus/SAR_RARP50`.

Preferred full prep (downloads from Synapse `syn31997652`, extracts, then runs
the clip/CSV builder; needs `SYNAPSE_AUTH_TOKEN`):

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1
/path/to/venvs/vjepa_polaris/bin/python \
  head_phases/prepare_sarrarp50_dataset.py --force
```

If Synapse blocks the download, accept SAR_RARP50 access/terms for
`syn31997652` in the Synapse web UI and rerun.

If the archives are already downloaded elsewhere:

```bash
/path/to/venvs/vjepa_polaris/bin/python \
  head_phases/prepare_sarrarp50_dataset.py \
  --source-root /path/to/downloaded/SAR_RARP50_or_archives \
  --skip-download --force
```

Once the raw data root exists, build the ASFormer ctx3 sequence-label CSVs
for both the 384 and 512 short-side variants with the helper:

```bash
cd /path/to/vjepa2_polaris
VJEPA_PYTHON=/path/to/venvs/vjepa_polaris/bin/python \
  bash scripts/build_sarrarp50_asformer_datasets.sh
```

The helper writes:

```text
globus/sarrarp50_action_clips_3x16f_4fps/                        # 384 short side
globus/sarrarp50_action_clips_3x16f_4fps_512/                    # 512 short side
vjepa2_polaris/app/csv_head_models/sarrarp50_actions_4fps_asformer_ctx3_seq_{train,val,test}.csv
vjepa2_polaris/app/csv_head_models/sarrarp50_actions_4fps_512_asformer_ctx3_seq_{train,val,test}.csv
```

To call the builder directly (one variant), use the same flag set the helper
uses; the relevant flags are documented in
`SARRARP50_ACTION_BACKBONE_TESTS.md`.

## Step 2: Fine-tune the ASFormer head

The Polaris launcher wraps `prepare_runtime_config.py` (per-rank batch
rescaling, topology suffix on the output folder) and submits one PBS job
per YAML, chained via `afterok`.

### One-node smoke (1×4 GPU, debug queue)

```bash
cd /path/to/vjepa2_polaris

VJEPA_NUM_NODES=1 VJEPA_NUM_GPUS=4 VJEPA_STRONG_SCALE=1 \
VJEPA_ACCOUNT=<your-PBS-allocation> \
VJEPA_PYTHON=/path/to/venvs/vjepa_polaris/bin/python \
VJEPA_PARTITION=debug VJEPA_TIME_MIN=60 \
bash run_sarrarp50_asformer_polaris.sh \
  configs/heads/sarrarp50/sarrarp50_actions_4fps_asformer_vitl_dist_vitg_ema_384_frozen.yaml
```

### Multi-node (2 nodes × 4 GPU, preemptable queue)

```bash
VJEPA_NUM_NODES=2 VJEPA_NUM_GPUS=4 VJEPA_STRONG_SCALE=1 \
VJEPA_ACCOUNT=<your-PBS-allocation> \
VJEPA_PYTHON=/path/to/venvs/vjepa_polaris/bin/python \
VJEPA_PARTITION=preemptable VJEPA_TIME_MIN=180 \
bash run_sarrarp50_asformer_polaris.sh \
  configs/heads/sarrarp50/sarrarp50_actions_4fps_512_asformer_surg_finetuned_phase3_target_512_frozen.yaml
```

### Both configs sequentially (chained via PBS afterok)

```bash
bash run_sarrarp50_asformer_polaris.sh \
  configs/heads/sarrarp50/sarrarp50_actions_4fps_asformer_vitl_dist_vitg_ema_384_frozen.yaml \
  configs/heads/sarrarp50/sarrarp50_actions_4fps_512_asformer_surg_finetuned_phase3_target_512_frozen.yaml
```

### Dry run (no qsub)

```bash
VJEPA_DRY_RUN=1 bash run_sarrarp50_asformer_polaris.sh <yaml>
```

Builds the PBS script under `<folder>/_submit.pbs` and prints its path
without submitting.

### Env-var reference (same as `run_three_phases_polaris.sh`)

| Env var | Default | Purpose |
|---|---|---|
| `VJEPA_NUM_NODES` | required | PBS `-l select=N` |
| `VJEPA_NUM_GPUS` | `4` | GPUs per node (2/4/8 supported; 4 is standard on Polaris) |
| `VJEPA_STRONG_SCALE` | `0` | `1` keeps per-rank batch fixed; `0` shrinks per-rank to preserve YAML global |
| `VJEPA_PARTITION` | `prod` | `debug` (≤2n,1h), `debug-scaling` (≤10n,1h), `preemptable` (≤10n,72h), `prod` (≥10n,24h) |
| `VJEPA_ACCOUNT` | required | PBS `-A` |
| `VJEPA_PYTHON` | required | Path to the venv `python` (used inside PBS and by the login-side prep tool) |
| `VJEPA_TIME_MIN` | `720` | Walltime in minutes per phase |
| `VJEPA_FILESYSTEMS` | `home:eagle` | PBS `-l filesystems=` |
| `VJEPA_FOLDER_BASE` | unset | If set, replaces YAML `folder:` with `<base>/<basename(folder)>` |
| `VJEPA_DRY_RUN` | `0` | `1` builds PBS scripts but doesn't qsub |
| `VJEPA_DISABLE_AWS_OFI` | `0` | `1` to disable the AWS OFI NCCL plugin (only on rare NCCL hangs) |

## ASFormer head — how the pieces fit

The head lives in `vjepa2_polaris/src/models/asformer_head.py`. It is selected
through the YAML:

```yaml
experiment:
  classifier:
    name: asformer
    asformer_kwargs:
      num_clips: 3
      tokens_per_clip: 8       # frames_per_clip(16) / tubelet_size(2)
      temporal_tokens: 24
      num_layers: 10
      num_heads: 8
      return_sequence: true
```

It expects per-clip tokens from the encoder wrapper, which requires:

```yaml
model_kwargs:
  wrapper_kwargs:
    preserve_clip_dim: true
```

and frame-wise supervision, which requires:

```yaml
experiment:
  data:
    sequence_labels: true
    num_segments: 3            # ctx3 — one V-JEPA clip per segment
```

The builder emits matching multi-token labels when called with
`--sequence-labels --context-clips 3 --tubelet-size 2`; the dataset loader
decodes the space-delimited integer string into a `LongTensor([24])` per
sample, and the trainer flattens to `[B*24, num_classes]` before CE.

The encoder produces `[B, num_clips, T_clip*S, D]` (when `preserve_clip_dim`
is set); ASFormer reshapes to `[B, num_clips, T_clip, S, D]`, attention-pools
spatial tokens per `(clip, time)` step, runs one temporal self-attention block
plus a stack of dilated 1-D temporal convolutions (dilations
`1, 2, 4, …, 2^(num_layers−1)`), and emits one set of logits per temporal
token: `[B, 24, num_classes]`.

## Config set

```text
configs/heads/sarrarp50/sarrarp50_actions_4fps_asformer_vitl_dist_vitg_ema_384_frozen.yaml
configs/heads/sarrarp50/sarrarp50_actions_4fps_512_asformer_surg_finetuned_phase3_target_512_frozen.yaml
```

Both freeze the backbone and only train the ASFormer head.

## Action labels

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

## Common failure points

- **Builder fails with "Split/video path does not exist"** — SAR_RARP50 is not
  mounted at `--data-root`; check the path or rerun `prepare_sarrarp50_dataset.py`.
- **Training fails with missing CSVs** — run
  `scripts/build_sarrarp50_asformer_datasets.sh` first (or point each YAML at
  the CSV names you actually generated).
- **Trainer raises "classifier.name='asformer' requires preserve_clip_dim: true"** —
  the YAML is missing `model_kwargs.wrapper_kwargs.preserve_clip_dim: true`.
- **Label shape mismatch in CE loss** — `experiment.data.sequence_labels: true`
  must match the CSV labels actually emitted by the builder
  (`--sequence-labels` flag); without it, the dataset returns scalar labels and
  the ASFormer head's per-token CE blows up.
- **OOM on the 512px phase3 config** — drop `experiment.optimization.batch_size`
  to 1 and/or rerun with `VJEPA_NUM_NODES=2 VJEPA_NUM_GPUS=4
  VJEPA_STRONG_SCALE=1`.
- **PBS submission rejected** — `VJEPA_PARTITION=debug` is capped at 2 nodes and
  1 hour. Use `debug-scaling` (≤10n, 1h) or `preemptable` (≤10n, 72h) for
  larger runs.
