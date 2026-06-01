# One-vs-Rest Phase Heads

This folder is a clean sidecar for experimenting with **binary one-vs-rest phase heads** without disturbing the existing multiclass setup.

## What it builds

`build_phase_ovr_heads.py` creates:

- one binary `train.csv` and `val.csv` per target phase
- one matching config per phase under `configs/heads_ovr/`
- one metadata summary JSON with class counts and class weights

Each binary CSV uses the same probe format as the rest of the repo:

```text
/absolute/path/to/clip.mp4 <binary_label>
```

where `binary_label` is:

- `1` if the sample belongs to the target phase
- `0` otherwise

## Default sources

By default the builder uses:

- SITL:
  - `/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/sitl_phases_train.csv`
  - `/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/sitl_phases_val.csv`
- Surgenet native:
  - `/path/to/phase_triplet_heads_bundle/globus/surgenet_phases/yt_robotic_chole_native11_phases_train.csv`
  - `/path/to/phase_triplet_heads_bundle/globus/surgenet_phases/yt_robotic_chole_native11_phases_val.csv`

So the Surgenet side is read from the **native CSVs**, not from repo-side remapped copies.

## Supported label spaces

### `reduced6`

This uses the current 6-class merge:

- `0, 1 -> 0`
- `2 -> 1`
- `3 -> 2`
- `4 -> 3`
- `5, 6 -> 4`
- `7, 8 -> 5`
- `9, 10 -> dropped`

### `native11`

This keeps the full native 11-way phase taxonomy as-is.

## Typical workflow

### 1. Build the one-vs-rest datasets and configs

For the current SITL + native-Surgenet reduced-6 setup:

```bash
/path/to/phase_triplet_heads_bundle/vjepa2_1/.venv/bin/python \
  /path/to/phase_triplet_heads_bundle/vjepa2_1/head_phases_ovr/build_phase_ovr_heads.py \
  --label-space reduced6 \
  --sources sitl surgenet
```

This writes:

- CSVs under:
  `/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/ovr_phase_heads/reduced6/`
- configs under:
  `/path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads_ovr/reduced6/`

### 2. Pick one phase and train that binary head

Example for class `0` after generation:

```bash
cd /path/to/phase_triplet_heads_bundle/vjepa2_1
python3 -m evals.main \
  --fname /path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads_ovr/reduced6/phase_ovr_reduced6_00_clipping-of-the-cystic-duct-artery.yaml \
  --checkpoint /path/to/phase_triplet_heads_bundle/vjepa2_1/latest.pt \
  --model_name vit_large \
  --devices cuda:0
```

## What needs to be prepared in the data

To make one-vs-rest training meaningful, the data prep should be:

1. Decide the label space first.
   For your current experiments that probably means `reduced6`.

2. Make sure every source is mapped into that same label space.
   The builder already does this:
   - SITL uses the repo-side `0..10` phase CSVs.
   - Surgenet uses the native CSVs and remaps them in the builder.

3. Keep separate train and val splits before binarizing.
   The builder preserves the existing split files and only converts labels.

4. Generate one binary dataset per target phase.
   This is the actual one-vs-rest step.

5. Check positive/negative counts per phase.
   Some heads may still be extremely imbalanced. The generated `metadata.json`
   records those counts and writes binary class weights into each config.

6. Train and evaluate each phase head independently.
   At inference time, you can compare all per-phase scores and choose:
   - the max-scoring phase
   - or apply thresholds if you want a more multi-label interpretation

## Notes

- The generated configs use the same probe backbone/training recipe as the current phase heads, just with `num_classes: 2`.
- The builder uses inverse-frequency binary class weights by default.
- This setup does **not** modify the existing multiclass configs.
