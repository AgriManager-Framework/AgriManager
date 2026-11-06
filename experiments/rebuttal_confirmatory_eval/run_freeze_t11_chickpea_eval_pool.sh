#!/usr/bin/env bash
set -euo pipefail

# Commands run in the active Python environment (Conda or uv venv). To use a named
# Conda environment without activating it: AGRI_RUN="conda run --no-capture-output -n agrimanager".
AGRI_RUN="${AGRI_RUN:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

$AGRI_RUN \
    python experiments/rebuttal_confirmatory_eval/prepare_t11_eval_pool.py --crop chickpea

