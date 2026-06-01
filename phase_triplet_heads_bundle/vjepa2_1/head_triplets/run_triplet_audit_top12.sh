#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${VJEPA_PYTHON_BIN:-/lus/eagle/projects/tpc/leonardo_borgioli/venvs/vjepa/bin/python}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/analyze_triplet_clip_mosaics.py" --top-k-classes 12 "$@"
