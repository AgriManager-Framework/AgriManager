#!/usr/bin/env bash
# Build the frozen T1.1 Chickpea confirmatory test suite, LLM and numerical.
#
# Both variants render the same five weather-regime groups from the same frozen
# pool, so the LLM and numerical policies are scored on identical scenarios.
# The audit asserts that pairing, and asserts that no scenario appears in the
# T1.1 training dataset, which is what makes this a test set rather than the
# reused validation set the meta-review objected to.
#
# The source pool is not regenerated here. It is already frozen by
# `run_freeze_t11_chickpea_eval_pool.sh`; rebuilding it from this script would
# risk moving the ground under a suite that is already final.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

LLM_CONFIG="$SCRIPT_DIR/config/e4_t11_chickpea_weather_regime.yaml"
NN_CONFIG="$SCRIPT_DIR/config/e4_t11_chickpea_weather_regime_nn.yaml"
LLM_DIR="$SCRIPT_DIR/data/e4_t11_chickpea_weather_regime"
NN_DIR="$SCRIPT_DIR/data/e4_t11_chickpea_weather_regime_nn"
POOL_DIR="$SCRIPT_DIR/data/source_pools/t11_chickpea_e4"

cd "$PROJECT_ROOT"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for regime in id drought wet hot cold; do
    if [[ ! -f "$POOL_DIR/test_${regime}/chickpea.parquet" ]]; then
        echo "Missing frozen pool group: $POOL_DIR/test_${regime}/chickpea.parquet" >&2
        echo "Run experiments/rebuttal_confirmatory_eval/run_freeze_t11_chickpea_eval_pool.sh first." >&2
        exit 1
    fi
done

conda run -n agrimanager \
    bash entrypoints/dataset/build.sh \
    --config "$LLM_CONFIG" \
    --output-dir "$LLM_DIR" \
    --num-workers 1

conda run -n agrimanager \
    bash entrypoints/dataset/build.sh \
    --config "$NN_CONFIG" \
    --output-dir "$NN_DIR" \
    --num-workers 1

conda run -n agrimanager \
    python experiments/rebuttal_confirmatory_eval/normalize_eval_suite_paths.py --apply

conda run -n agrimanager \
    python experiments/rebuttal_confirmatory_eval/audit_t11_eval_suite.py --crop chickpea \
    --dataset-dir "$LLM_DIR"

conda run -n agrimanager \
    python experiments/rebuttal_confirmatory_eval/audit_t11_eval_suite.py --crop chickpea \
    --dataset-dir "$NN_DIR" \
    --paired-with "$LLM_DIR"
