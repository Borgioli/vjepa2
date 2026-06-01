# Future Codex Guide: Tool And Verb Threshold Tuning

Use this guide when the user asks to inspect the latest validation results, make tables like the previous discussion, or update thresholds for the Tool5 pipeline.

## Core Rule

Tune thresholds from the saved validation confusion matrices, not from train loss alone.

For each label, read the epoch-005 validation binary confusion matrix:

```text
true/pred,absent,present
absent,TN,FP
present,FN,TP
```

Compute:

```text
precision   = TP / (TP + FP)
recall      = TP / (TP + FN)
F1          = 2 * precision * recall / (precision + recall)
pred/actual = (TP + FP) / (TP + FN)
```

Interpret `pred/actual`:

```text
~1.00  balanced
>1.10  over-predicted, threshold likely too low
<0.90  under-predicted, threshold likely too high
```

Use small threshold moves once labels are close to balanced. Avoid large swings after `pred/actual` is near 1.0.

## Tool Head

Config files:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5.yaml
/path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/triplet_multilabel_tool_only_train_vitl_latest_token_aggregation_tool5_classic.yaml
```

The user often runs the classic no-count config, but update both configs unless they explicitly say otherwise.

Threshold order:

```text
clipper, grasper, hook, irrigator, scissors
```

Current converged-ish threshold set from the discussion:

```yaml
multilabel_thresholds: [0.70, 0.39, 0.45, 0.76, 0.54]
```

Latest good table style:

```text
| tool | Precision | Recall | F1 | pred/actual | Delta F1 | TP | FP | FN | read |
```

Read examples:

```text
clipper pred/actual 1.06 -> balanced
grasper pred/actual 0.94 -> slightly under
scissors pred/actual 1.01 -> balanced
```

Suggested adjustment logic:

```text
Over-predicted: raise threshold by 0.02 to 0.05.
Under-predicted: lower threshold by 0.01 to 0.05.
Near balanced: leave it alone or nudge by only 0.01.
```

The tool confusion matrices live under folders like:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/trained_heads/triplet_multilabel_tool_only_tool5_native_classic/video_classification_frozen/triplet_multilabel_tool_only_head_training_vitl_latest_token_aggregation_tool5_native_classic/confusion_matrices/
/path/to/phase_triplet_heads_bundle/vjepa2_1/trained_heads/triplet_multilabel_tool_count_tool5_native/video_classification_frozen/triplet_multilabel_tool_count_head_training_vitl_latest_token_aggregation_tool5_native/confusion_matrices/
```

The latest run is usually the folder with the newest `epoch_005_val_tool_clipper_confusion_counts.csv` timestamp:

```bash
find /path/to/phase_triplet_heads_bundle/vjepa2_1/trained_heads \
  -type f \
  -name 'epoch_005_val_tool_clipper_confusion_counts.csv' \
  -printf '%T@ %p\n' | sort -n | tail -n 12
```

## Verb/Action Head Conditioned On Tool

Config file:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/configs/heads/verb_multilabel_conditioned_on_tool_train_vitl_latest_token_aggregation_tool5.yaml
```

This head trains with ground-truth tool conditions:

```text
P(verb/action | video, GT tool condition)
```

At inference time, predicted tools from the tool head become the conditions.

Threshold order:

```text
coagulation, grasp/retract, null, cut, dissect, clean
```

Current threshold set from the discussion:

```yaml
multilabel_thresholds: [0.58, 0.50, 0.65, 0.50, 0.65, 0.55]
```

Known behavior from the first run:

```text
grasp/retract, cut, clean were already excellent.
coagulation was useful but over-predicted at threshold 0.50.
dissect was heavily over-predicted at threshold 0.50.
null was weak/noisy and should not be trusted without special handling.
```

A later threshold `[0.65, 0.50, 0.80, 0.50, 0.80, 0.55]` overcorrected:

```text
dissect and null became never predicted.
coagulation became balanced but lost too much F1.
```

So prefer moderate moves for action thresholds.

The verb/action confusion matrices live here:

```text
/path/to/phase_triplet_heads_bundle/vjepa2_1/trained_heads/verb_multilabel_conditioned_on_tool_tool5_native/video_classification_frozen/verb_multilabel_conditioned_on_tool_head_training_vitl_latest_token_aggregation_tool5_native/confusion_matrices/
```

File names:

```text
epoch_005_val_verb_coagulation_confusion_counts.csv
epoch_005_val_verb_grasp_retract_confusion_counts.csv
epoch_005_val_verb_null_confusion_counts.csv
epoch_005_val_verb_cut_confusion_counts.csv
epoch_005_val_verb_dissect_confusion_counts.csv
epoch_005_val_verb_clean_confusion_counts.csv
```

## Table Command

Use this pattern to create the same table from confusion matrix CSVs.

Tool table:

```bash
python3 - <<'PY'
import csv
from pathlib import Path

base = Path("/path/to/confusion_matrices")
labels = ["clipper", "grasper", "hook", "irrigator", "scissors"]
prefix = "epoch_005_val_tool"

print("| label | Precision | Recall | F1 | pred/actual | TP | FP | FN | read |")
print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
for label in labels:
    rows = list(csv.reader((base / f"{prefix}_{label}_confusion_counts.csv").open()))
    tn, fp = int(rows[1][1]), int(rows[1][2])
    fn, tp = int(rows[2][1]), int(rows[2][2])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    pred_actual = (tp + fp) / (tp + fn) if tp + fn else 0.0
    if pred_actual > 1.10:
        read = "over-predicted"
    elif pred_actual < 0.90:
        read = "under-predicted"
    else:
        read = "balanced"
    print(f"| {label} | {precision:.3f} | {recall:.3f} | {f1:.3f} | {pred_actual:.2f} | {tp} | {fp} | {fn} | {read} |")
PY
```

Verb/action table: use the same script but set:

```python
labels = ["coagulation", "grasp_retract", "null", "cut", "dissect", "clean"]
prefix = "epoch_005_val_verb"
```

## Visualization

The saved confusion matrices already include raw counts and row-normalized CSVs. To visualize them quickly:

1. Use raw counts to understand absolute FP/FN burden.
2. Use row-normalized matrices to see error rates independent of class support.
3. Prefer a small heatmap per label, or a compact table with `precision`, `recall`, `F1`, and `pred/actual`.

Quick matplotlib heatmap command for one matrix:

```bash
python3 - <<'PY'
import csv
from pathlib import Path
import matplotlib.pyplot as plt

csv_path = Path("/path/to/epoch_005_val_tool_clipper_confusion_counts.csv")
out_path = csv_path.with_suffix(".png")

rows = list(csv.reader(csv_path.open()))
matrix = [
    [int(rows[1][1]), int(rows[1][2])],
    [int(rows[2][1]), int(rows[2][2])],
]

fig, ax = plt.subplots(figsize=(4, 3))
im = ax.imshow(matrix)
ax.set_xticks([0, 1], ["pred absent", "pred present"])
ax.set_yticks([0, 1], ["true absent", "true present"])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(matrix[i][j]), ha="center", va="center")
ax.set_title(csv_path.stem)
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(out_path, dpi=160)
print(out_path)
PY
```

For final reporting to the user, prefer the table over images unless they ask for plots.

## What To Say To The User

Keep responses short. Show:

```text
1. Threshold list used, if known.
2. Table.
3. One-sentence interpretation.
4. New threshold list if an update is warranted.
```

If a run did not change results, say the output folder appears unchanged or the wrong config was used.

