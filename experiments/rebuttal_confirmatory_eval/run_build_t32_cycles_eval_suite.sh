#!/usr/bin/env bash
set -euo pipefail

# Commands run in the active Python environment (Conda or uv venv). To use a named
# Conda environment without activating it: AGRI_RUN="conda run --no-capture-output -n agrimanager".
AGRI_RUN="${AGRI_RUN:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$SCRIPT_DIR/config/e4_t32_cycles_profit.yaml"
DATASET_DIR="$SCRIPT_DIR/data/e4_t32_cycles_profit"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

$AGRI_RUN \
    bash entrypoints/dataset/build.sh \
    --config "$CONFIG" \
    --output-dir "$DATASET_DIR" \
    --num-workers 1

$AGRI_RUN \
    python experiments/rebuttal_confirmatory_eval/normalize_eval_suite_paths.py --apply

$AGRI_RUN \
    python experiments/rebuttal_confirmatory_eval/audit_t32_cycles_eval_suite.py

