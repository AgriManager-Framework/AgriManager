#!/usr/bin/env bash
set -euo pipefail

# Commands run in the active Python environment (Conda or uv venv). To use a named
# Conda environment without activating it: AGRI_RUN="conda run --no-capture-output -n agrimanager".
AGRI_RUN="${AGRI_RUN:-}"

# T3.1 has one target cell per simulator, so the three are built together:
# a partial build would leave the cross-simulator comparison undefined.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

$AGRI_RUN \
    python experiments/rebuttal_confirmatory_eval/prepare_maize_eval_pool.py

for SUITE in e4_t31_cross_sim_wofost e4_t31_cross_sim_dssat e4_t31_cross_sim_cycles; do
    $AGRI_RUN \
        bash entrypoints/dataset/build.sh \
        --config "$SCRIPT_DIR/config/${SUITE}.yaml" \
        --output-dir "$SCRIPT_DIR/data/${SUITE}" \
        --num-workers 1
done

$AGRI_RUN \
    python experiments/rebuttal_confirmatory_eval/normalize_eval_suite_paths.py --apply

$AGRI_RUN \
    python experiments/rebuttal_confirmatory_eval/audit_t31_eval_suite.py
