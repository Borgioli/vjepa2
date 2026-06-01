# V-JEPA2.1 Ablation Builder

A single-file browser tool to design ablation studies for the V-JEPA2.1
three-phase pretraining pipeline (pretrain → video-segments → cooldown) and
emit a ZIP containing all the YAML configs plus one launcher `.sh` that runs
them sequentially with resume/skip support.

It is the **build-once, run-many** interface for everything that today lives in
`configs/train_2_1/vitl_Neil/vitl16_ablation_*` and the
`run_three_phases_*_single4.sh` scripts.

```
tools/ablation_builder/
├── ablation_builder.html   ← the app (open in a browser)
└── README.md               ← this file
```

---

## 1. Serve the app

The HTML has to be loaded over `http://` (not `file://`) for the "Load .sh /
YAMLs" feature to be able to fetch the referenced YAMLs back from disk.

### On Sophia or Polaris

```bash
cd /path/to/vjepa2
python -m http.server 8765
```

`8765` is arbitrary — pick any free port. Leave that terminal alone.

> **Important:** start the server from the **project root**
> (`vjepa2_1/`), not from inside `tools/ablation_builder/`. The "Load .sh"
> feature resolves YAML paths like `configs/train_2_1/...` relative to the
> server root.

### On your laptop — open an SSH tunnel

In a new local terminal:

```bash
ssh -N -L 8765:localhost:8765 <user>@<host>
```

`-N` means "no remote shell, just hold the tunnel" — the terminal will look
frozen, that's correct. Leave it running.

For Polaris swap the host:

```bash
ssh -N -L 8765:localhost:8765 <user>@polaris.alcf.anl.gov
```

### Open the browser

```
http://localhost:8765/tools/ablation_builder/ablation_builder.html
```

You should see the white "V-JEPA2.1 Ablation Builder" UI with one starter
experiment.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `Error 404: File not found` | The server isn't serving from the project root. Stop it (`Ctrl-C`), `cd` to `/path/to/vjepa2`, restart `python -m http.server 8765`. |
| ssh terminal looks frozen | That's what `-N` does. As long as it hasn't returned to a prompt, the tunnel is up — just open the browser. |
| `Address already in use` on Sophia | Pick a different port (`python -m http.server 9123`) and match it on the laptop side. |
| `Address already in use` on the laptop | Pick a different *local* port: `ssh -N -L 9123:localhost:8765 ...` and browse to `http://localhost:9123/...`. |
| Browser shows `connection refused` | The tunnel exited (often network drop). Re-run the ssh command. |

---

## 2. Use the UI

### Header

| Button | What it does |
|---|---|
| **Load .sh / YAMLs** | Recover an existing experiment from an `.sh` launcher (this tool's output, or one of the existing `run_three_phases_*_single4.sh`). |
| **Export config (JSON)** | Save the in-progress experiment list to a JSON file. Use this if you'll keep iterating across sessions. |
| **Import config (JSON)** | Restore a previously exported JSON. |
| **+ Add experiment** | Append a new experiment (3 phases, seeded with the all-datasets baseline). |
| **Download ZIP** | Emit the final ZIP with all YAMLs + the launcher `.sh`. |

### Toolbar (settings shared by every experiment)

- **Project root** — where the YAMLs and `.sh` will live on disk. Default
  `/path/to/vjepa2`. Used only as
  documentation in the README inside the ZIP.
- **Checkpoint root** — base folder for the `folder:` field in every YAML.
  Each phase becomes `<checkpoint root>/<exp_name>/<phaseN_*>`.
- **Configs subdir** — where the YAMLs land inside the ZIP, relative to the
  project root. Default `configs/train_2_1/vitl_Neil` so dropping the ZIP at
  the project root produces YAMLs alongside the existing
  `vitl16_ablation_all-datasets_single4/` family.
- **Launcher .sh filename** — name of the launcher file inside the ZIP.
- **Cluster** — picks the venv and default GPU count:
  - **Sophia**: `/path/to/venvs/vjepa/bin/python`, 8 GPUs.
  - **Polaris**: `/path/to/venvs/vjepa_polaris/bin/python`, 4 GPUs.
- **Override num GPUs** — leave blank to inherit cluster default, or set
  `1`/`2`/`4`/`8`.
- **Num nodes** — `1` (default) keeps the original single-node launch
  (`python -m app.main --devices cuda:0..`). Set to `>1` to emit the Polaris
  multi-node launch path (`mpiexec -n WORLD -ppn NUM_GPUS python -m
  app.main_dist_polaris …`). When `>1` the launcher must be invoked from inside
  a PBS allocation that has already exported `MASTER_ADDR`, `MASTER_PORT`, the
  NCCL/libfabric env, and the venv — see the multi-node example in §4.

### Per-experiment

Each experiment is a card with:
- A name (becomes the folder name under `configs/` and the
  `<exp_name>` segment in every checkpoint folder).
- Buttons: **Duplicate**, **↑/↓** (reorder), **Delete**.
- Three collapsible phase cards: phase 1 pretrain, phase 2 video-segments,
  phase 3 cooldown.

### Per-phase

Every field that appears in the baseline YAMLs is editable, grouped into
sections:

- **top-level** — `app`, `nodes`, `tasks_per_node`, `cpus_per_task`,
  `mem_per_gpu`. `folder` is auto-set.
- **data.datasets** — checkboxes for each of the 11 CSVs in
  `app/csv_vjepa_reduced/`. Each checked row contributes one entry to
  `datasets`, `datasets_weights`, `dataset_fpcs` (kept aligned automatically).
  Use the "+ Add CSV" row to register a CSV not in the default list.
- **data** — `batch_size`, `crop_size`, `patch_size`, `tubelet_size`, `fps`,
  `num_workers`, `num_clips`, `persistent_workers`, `pin_mem`,
  `dataset_type`.
- **data_aug** — `auto_augment`, `motion_shift`,
  `random_resize_aspect_ratio`, `random_resize_scale`, `reprob`.
- **loss** — `loss_exp`, `predict_all`, `shift_by_n`,
  `weight_distance_loss`.
- **mask** — one card per mask block (add/remove blocks freely): aspect
  ratio, spatial/temporal scale, num_blocks, max_keep, etc.
- **meta** — `dtype`, `eval_freq`, `save_every_freq`, `seed`, `read_checkpoint`,
  `load_checkpoint`, `load_predictor`, `use_sdpa`. For phase 1 also exposes
  `pretrain_checkpoint` (the Meta init) and the encoder-key fields. Phase 2
  and 3 chain automatically from the previous phase's `latest.pth.tar`.
- **model** — full model block (vit_large defaults).
- **optimization** — `lr`, `start_lr`, `final_lr`, `ema`, `epochs`, `ipe`,
  `ipe_scale`, `warmup`, `weight_decay`, `final_weight_decay`.

### Live preview (right panel)

Tabs across the top let you click into:
- Each phase YAML of each experiment (`exp_name: p1 p2 p3`)
- The generated `launcher.sh`

Edits in the form update the preview in real time. The caret position and the
open/closed state of phase cards survive re-renders, so typing in a field
won't collapse the surrounding card.

---

## 3. Worked example

Goal: ablate three configurations sequentially.
1. All datasets, baseline hyperparams.
2. All datasets, SITL down-weighted to 0.5.
3. All datasets, SITL=0.5 *and* a higher cooldown LR.

Steps:

1. Click **+ Add experiment** twice (you now have 3 cards).
2. Rename them `all_datasets`, `sitl_50pct`, `sitl_50pct_high_cooldown_lr`.
3. In `sitl_50pct`, expand each of the 3 phases and change the SITL row's
   weight from `1` to `0.5`. (Or: leave phase 1 the same and only change phase
   2/3 — your call.)
4. In `sitl_50pct_high_cooldown_lr`, click **Duplicate** to start from
   `sitl_50pct`, rename, then open phase 3 → `optimization` and bump `lr`
   from `0.000525` to, say, `0.001`.
5. Pick your cluster in the toolbar.
6. Click **Download ZIP**. A file like
   `vjepa2_1_ablation_2026-05-28T17-32-04.zip` lands on your laptop.

---

## 4. Run the result

The ZIP layout matches the project root:

```
README.txt
run_ablation_batch.sh
configs/train_2_1/vitl_Neil/
    all_datasets/
        pretrain-256px-16f.yaml
        video-segments-256px-16f.yaml
        cooldown-512px-16f.yaml
    sitl_50pct/
        ...
    sitl_50pct_high_cooldown_lr/
        ...
```

Drop it on the cluster and unzip in place:

```bash
scp ~/Downloads/vjepa2_1_ablation_*.zip \
    <user>@<host>:/path/to/vjepa2/

# on Sophia
cd /path/to/vjepa2
unzip vjepa2_1_ablation_*.zip
chmod +x run_ablation_batch.sh
./run_ablation_batch.sh
```

The launcher is self-contained — it inlines the resume/skip logic from
`run_three_phases_family_single4.sh`, so it does **not** require that script
to be present. It only needs `scripts/prepare_runtime_config.py` at the
project root (which is already there).

### Runtime overrides

Environment variables you can set on the command line to retarget without
regenerating the ZIP:

| Var | Effect |
|---|---|
| `VJEPA_CLUSTER` | Label printed in logs (cosmetic). |
| `VJEPA_PYTHON_BIN` | Path to a different python venv. |
| `VJEPA_NUM_GPUS` | `1`, `2`, `4`, or `8`. |
| `VJEPA_NUM_NODES` | `1` → single-node `python -m app.main`. `>1` → multi-node `mpiexec … app.main_dist_polaris` (must be launched from inside a PBS allocation). |
| `VJEPA_DEVICES` | Explicit space-separated list, e.g. `"cuda:0 cuda:1"`. Must match `VJEPA_NUM_GPUS`. Only used in single-node mode. |

Example — flip a Sophia launcher to Polaris on the fly:

```bash
VJEPA_PYTHON_BIN=/path/to/venvs/vjepa_polaris/bin/python \
VJEPA_NUM_GPUS=4 \
./run_ablation_batch.sh
```

### Multi-node run on Polaris (Num nodes > 1)

The inner `run_ablation_batch.sh` is **not** a PBS script — it's the
inner loop. For multi-node you wrap it in a PBS submission script that does
the env init and then invokes it. Template (matches the pattern of
`~/start_vjepa_ablation_2026-05-28.sh`):

```bash
#!/bin/bash -l
#PBS -N my_ablation
#PBS -l select=8
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug-scaling
#PBS -A ModCon

set -eo pipefail

# --- Polaris framework init ---
ulimit -c unlimited
module use /soft/modulefiles
module load conda
conda activate base
cd /path/to/
source venvs/vjepa_polaris/bin/activate

# --- NCCL / libfabric env (required for app.main_dist_polaris) ---
export MPICH_GPU_SUPPORT_ENABLED=1
export FI_PROVIDER=cxi
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_LAUNCH_MODE=GROUP
export PYTHONFAULTHANDLER=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:${LD_LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:${LD_LIBRARY_PATH:-}
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
export MPICH_OFI_NIC_POLICY=NUMA

export MASTER_ADDR=$(head -n1 "$PBS_NODEFILE")
export MASTER_PORT=29500

# --- Hand off to the generated launcher ---
cd /path/to/vjepa2   # or wherever you unzipped
VJEPA_NUM_NODES=8 VJEPA_NUM_GPUS=4 \
    ./run_ablation_batch.sh
```

Submit it:

```bash
qsub start_vjepa_ablation_<name>.sh
qstat -u $USER    # track
```

A few rules of thumb:

- `select=N` in PBS **must equal** `VJEPA_NUM_NODES` you pass to the launcher
  (otherwise nodes sit idle or `mpiexec` over-subscribes).
- `debug-scaling` is capped at 10 nodes and 1h walltime
  ([Polaris docs](https://docs.alcf.anl.gov/polaris/running-jobs/)); for longer
  runs use `prod` and set `walltime` accordingly.
- The launcher's resume/skip logic still works across submissions, so if the
  1h debug-scaling walltime kills it mid-phase just `qsub` the same wrapper
  again — completed phases will skip, the live one resumes.

### Resume / skip behaviour

For each phase the launcher checks
`<folder>/latest.pth.tar`:

- File missing → **fresh** run.
- File present, recorded epoch < `optimization.epochs` → **resume**.
- File present, recorded epoch ≥ `optimization.epochs` → **skip**.

Safe to re-run the launcher after a job dies.

---

## 5. Round-trip an existing run

To tweak an ablation you already have on disk:

1. Click **Load .sh / YAMLs**.
2. Pick the launcher `.sh`.
   - If you're serving from the project root via `python -m http.server` the
     YAMLs are pulled from the same origin automatically.
   - Otherwise multi-select the `.sh` together with the referenced `.yaml`
     files in the same dialog — they're matched by basename.
3. The UI reloads each experiment with all fields populated. Edit, add
   experiments, re-download a fresh ZIP.

This works on both the generated `run_ablation_batch.sh` files and the
existing `run_three_phases_*_single4.sh` family scripts (the latter are
chunked every 3 YAMLs since they don't carry per-experiment comment markers).

---

## 6. Saving in-progress state

If you'll come back tomorrow:

- **Export config (JSON)** → save the file. The JSON includes the toolbar
  options and every experiment's full phase state.
- **Import config (JSON)** → restores everything including cluster choice
  and the GPU override.

JSON exports are independent of the YAMLs on disk; they round-trip the exact
in-UI state.
