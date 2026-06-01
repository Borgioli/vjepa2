#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${VJEPA_PYTHON_BIN:-/lus/eagle/projects/tpc/leonardo_borgioli/venvs/vjepa/bin/python}"
OUTPUT_ROOT="/path/to/phase_triplet_heads_bundle/vjepa2_1/app/tool_clip_audit"

TOOL_IDS=(0 1 2 3 4 5)
TOOL_NAMES=(grasper hook bipolar clipper scissors irrigator)

for index in "${!TOOL_IDS[@]}"; do
  tool_id="${TOOL_IDS[$index]}"
  tool_name="${TOOL_NAMES[$index]}"
  output_dir="${OUTPUT_ROOT}/${tool_name}"

  printf '\n=== Running tool audit [%s] %s ===\n' "${tool_id}" "${tool_name}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_tool_presence_clip_mosaics.py" \
    --tool-id "${tool_id}" \
    --tool-name "${tool_name}" \
    --output-dir "${output_dir}" \
    "$@"
done

printf '\nCompleted %d tool audits.\n' "${#TOOL_IDS[@]}"
