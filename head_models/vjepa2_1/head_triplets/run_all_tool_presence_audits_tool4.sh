#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${VJEPA_PYTHON_BIN:-/lus/eagle/projects/tpc/leonardo_borgioli/venvs/vjepa/bin/python}"
TRAIN_CSV="/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/triplet_multilabel_train_native_tool4.csv"
VAL_CSV="/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/triplet_multilabel_val_native_tool4.csv"
METADATA_JSON="/path/to/phase_triplet_heads_bundle/vjepa2_1/app/csv_head_models/triplet_multilabel_native_tool4_metadata.json"
OUTPUT_ROOT="/path/to/phase_triplet_heads_bundle/vjepa2_1/app/tool_clip_audit/tool4"

TOOL_IDS=(0 1 2 3)
TOOL_NAMES=(grasper hook irrigator scissors)

for index in "${!TOOL_IDS[@]}"; do
  tool_id="${TOOL_IDS[$index]}"
  tool_name="${TOOL_NAMES[$index]}"
  output_dir="${OUTPUT_ROOT}/${tool_name}"

  printf '\n=== Running tool4 audit [%s] %s ===\n' "${tool_id}" "${tool_name}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_tool_presence_clip_mosaics.py" \
    --train-csv "${TRAIN_CSV}" \
    --val-csv "${VAL_CSV}" \
    --metadata "${METADATA_JSON}" \
    --tool-id "${tool_id}" \
    --tool-name "${tool_name}" \
    --output-dir "${output_dir}" \
    "$@"
done

printf '\nCompleted %d tool4 audits.\n' "${#TOOL_IDS[@]}"
