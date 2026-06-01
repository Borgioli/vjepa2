# Polaris / Sophia port notes

This bundle was generated on the Spark workstation, where all data lived under `/path/to/data_root/`. On 2026-05-28 every script and config was rewritten in place to run on the Polaris and Sophia clusters at ALCF.

## What changed

| Category | Spark | Now |
|---|---|---|
| Path prefix | `/path/to/data_root/` | `/path/to/phase_triplet_heads_bundle/` |
| Python shebang | `#!/path/to/data_root/vjepa2_1/.venv/bin/python` | `#!/usr/bin/env python3` |
| Shell `PYTHON_BIN` | hardcoded venv | `"${VJEPA_PYTHON_BIN:-/lus/eagle/projects/tpc/leonardo_borgioli/venvs/vjepa/bin/python}"` |

Rewritten file kinds: `.py`, `.sh`, `.yaml`, `metadata.json`, dataset `.csv`, `.md` docs (incl. INDEX/scripts/datasets/quickstart), `generation_summary.txt`.

## Running scripts

The shell launchers (`run_all_tool_presence_audits.sh`, `run_all_phase_ovr_audits_reduced6.sh`, etc.) pick up the right Python automatically on **Sophia** via the default `${VJEPA_PYTHON_BIN:-...}`. On **Polaris**, export the cluster venv first:

```bash
export VJEPA_PYTHON_BIN=/lus/eagle/projects/tpc/leonardo_borgioli/venvs/vjepa_polaris/bin/python
./vjepa2_1/head_triplets/run_all_tool_presence_audits.sh
```

Direct `.py` invocations use `#!/usr/bin/env python3`, so make sure the right venv is activated (or call the script with the venv binary explicitly):

```bash
# Polaris
/lus/eagle/projects/tpc/leonardo_borgioli/venvs/vjepa_polaris/bin/python \
  vjepa2_1/head_phases/analyze_phase_clip_mosaics.py --help

# Sophia
/lus/eagle/projects/tpc/leonardo_borgioli/venvs/vjepa/bin/python \
  vjepa2_1/head_phases/analyze_phase_clip_mosaics.py --help
```

## What was intentionally NOT rewritten

Frozen run records and historical artifacts — they describe past runs on Spark and rewriting would corrupt that history:

- `vjepa2_1/phase_rag_embeddings/<run>/eval_*.json`
- `vjepa2_1/phase_rag_embeddings/<run>/query_*.json`
- `vjepa2_1/phase_rag_embeddings/<run>/*_predictions.csv`
- `globus/surgenet_triplets/tool_csv/*_mapping.json`
- `README/INDEX.md` line 3: `Bundle generated 2026-05-22 from /path/to/data_root` (provenance note)
- `phase_triplet_heads_bundle.html` (release manifest)

Scripts do not read these as inputs.

## Inputs the bundle does NOT provide

The scripts reference these locations, which are expected to exist on the cluster but are NOT part of the bundle:

- `vjepa2_1/app/csv_head_models/*.csv` — built by the dataset-creation scripts; or copy from the main `/path/to/vjepa2_1/app/csv_head_models/` tree
- `vjepa2_1/configs/heads_ovr/*.yaml` — emitted by `head_phases_ovr/build_phase_ovr_heads.py`
- Any audit `--output-dir` paths under `vjepa2_1/app/` — created on first run

## Re-running the port

If the bundle gets moved or new Spark paths sneak in, re-run the rewriter at `/tmp/port_bundle.py` (or save it somewhere durable):

```bash
python3 /tmp/port_bundle.py           # dry run
python3 /tmp/port_bundle.py --apply   # apply
```

Rules and skip patterns live in the script.
