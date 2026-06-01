# Datasets

All datasets follow the same per-row CSV convention used throughout the heads:

```
/absolute/path/to/clip.mp4 <int_label>
```

…with the absolute paths rooted at `/path/to/phase_triplet_heads_bundle/`. To use this bundle on a different host, either symlink `/path/to/phase_triplet_heads_bundle/` to the unpack location, or run a path-rewrite (template: `globus/surgenet_triplets/fix_paths.py`).

## `globus/surgenet_phases/` — ~2.2 GB

YouTube robotic cholecystectomy phase clips, native-11 label space.

```
yt_robotic_chole_native11_clips/                ← clip .mp4s
yt_robotic_chole_native11_phases_train.csv      ← train split
yt_robotic_chole_native11_phases_val.csv        ← val split
yt_robotic_chole_native11_phase_summary.md      ← summary doc (see 04_existing_docs.md)
```

## `globus/surgenet_triplets/` — ~38 GB

Surgenet triplets + per-tool windows. The largest single section.

```
yt_robotic_chole_triplets_clips/                ← per-triplet clips
yt_robotic_chole_tool_windows_clips/            ← per-tool window clips
yt_robotic_chole_tool_windows_clips.zip        ← packed copy of the above
tool_csv/                                       ← per-tool CSV splits (has its own README.md)
triplet_labels/                                 ← label artifacts
yt_robotic_chole_triplets_train.csv             ← single-label triplet split
yt_robotic_chole_triplets_multilabel_train.csv  ← multilabel triplet split
yt_robotic_chole_triplets_multilabel_val.csv
yt_robotic_chole_triplet_mapping.json           ← id ↔ name mapping
yt_robotic_chole_triplet_summary.md             ← dataset summary
triplet_multilabel_pipeline_one_slide.html      ← one-slide pipeline overview
phase_recognition_ovr_transformer_one_slide.html
AGENTS_TRIPLET.md / AGENTS_TRIPLET_soft.md      ← triplet head usage (hard / soft labels)
AGENTS_SINGLE_TOOL.md / AGENTS_SINGLE_TOOL_SOFT.md ← single-tool variants
AGENTS_INFERENCE.md                             ← inference workflow
AGENTS_THRESHOLD_TUNING.md                      ← threshold-tuning notes
```

The `.zip` is redundant with the unpacked `yt_robotic_chole_tool_windows_clips/` directory — keep one if you're tight on space.

## `globus/sitl_phases/` — ~15 GB

In-house SITL phase clips at native fps.

```
clips/                  ← clip .mp4s
annotations.csv         ← per-clip annotations
gif_phases/             ← preview gifs per phase
generation_summary.txt  ← how the clips were generated
```

## `globus/sitl_phases_4s_4fps/` — ~2.4 GB

Same SITL phases resampled to 4-second / 4-fps clips (the format consumed by the V-JEPA2 heads). Built by `vjepa2_1/head_phases/build_sitl_4s_4fps_dataset.py`.

```
clips/
annotations.csv
generation_summary.txt
```

## `globus/cholec80_phases/` — ~2.9 GB

Cholec80 minority phases, 4s@4fps, 1-fps duplication.

```
cholec80_4minority_4s4fps_dup1fps_clips/
cholec80_4minority_4s4fps_dup1fps_train.csv
cholec80_4minority_4s4fps_dup1fps_val.csv
cholec80_4minority_4s4fps_dup1fps_mapping.json
```

## `globus/triplet/` — ~20 GB

Raw triplet videos + labels + per-clip splits.

```
videos/                              ← source full videos
clips/                               ← extracted clips
labels/                              ← per-video label files
plots/                               ← analysis plots
triplet_dataset.csv                  ← canonical triplet rows
triplet_dataset_with_indices.csv     ← rows + index columns (from add_index_columns.py)
triplet_analysis.csv                 ← class/co-occurrence analysis
wetransfer_six-videos_2025-12-04_2129.zip ← packed transfer copy of source videos
```

## Label spaces (phase)

- **`native11`** — the original 11-way Surgenet/SITL phase taxonomy.
- **`reduced6`** — 6-way collapse used by the OvR heads. Mapping:
  - `0, 1 → 0`
  - `2 → 1`
  - `3 → 2`
  - `4 → 3`
  - `5, 6 → 4`
  - `7, 8 → 5`
  - `9, 10 → dropped`

Defined in `vjepa2_1/head_phases_ovr/label_spaces.py`.
