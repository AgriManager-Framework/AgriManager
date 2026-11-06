#!/usr/bin/env bash
# Build the three v2 CycleGym E4 suites (T3.1 target, T3.2 profit, T3.2 yield)
# on the unused weather years 2004-2016, then normalize paths and audit them on
# the physical scenario key. Why v2 exists: config/e4_t31_cross_sim_cycles_v2.yaml.
set -euo pipefail

# Commands run in the active Python environment (Conda or uv venv). To use a named
# Conda environment without activating it: AGRI_RUN="conda run --no-capture-output -n agrimanager".
AGRI_RUN="${AGRI_RUN:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for SUITE in e4_t31_cross_sim_cycles_v2 e4_t32_cycles_profit_v2 e4_t32_cycles_yield_v2; do
    $AGRI_RUN \
        bash entrypoints/dataset/build.sh \
        --config "$SCRIPT_DIR/config/$SUITE.yaml" \
        --output-dir "$SCRIPT_DIR/data/$SUITE" \
        --num-workers 1
done

$AGRI_RUN \
    python experiments/rebuttal_confirmatory_eval/normalize_eval_suite_paths.py --apply

$AGRI_RUN \
    python experiments/rebuttal_confirmatory_eval/audit_cycles_v2_eval_suites.py
