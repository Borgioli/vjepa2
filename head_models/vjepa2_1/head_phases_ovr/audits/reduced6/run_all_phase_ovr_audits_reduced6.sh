#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${VJEPA_PYTHON_BIN:-/lus/eagle/projects/tpc/leonardo_borgioli/venvs/vjepa/bin/python}"

shopt -s nullglob
AUDIT_SCRIPTS=("${SCRIPT_DIR}"/analyze_phase_ovr_reduced6_*.py)
shopt -u nullglob

if [[ ${#AUDIT_SCRIPTS[@]} -eq 0 ]]; then
  echo "No reduced6 OVR audit scripts found in ${SCRIPT_DIR}" >&2
  exit 1
fi

for audit_script in "${AUDIT_SCRIPTS[@]}"; do
  audit_name="$(basename "${audit_script}")"
  printf '\n=== Running %s ===\n' "${audit_name}"
  "${PYTHON_BIN}" "${audit_script}" "$@"
done

printf '\nCompleted %d reduced6 OVR audits.\n' "${#AUDIT_SCRIPTS[@]}"
