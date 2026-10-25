#!/usr/bin/env bash
set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODEL_CONFIG="agrimanager/model_interface/configs/openrouter/hy3-free_thinking.yaml"

cd "$PROJECT_ROOT"
mkdir -p \
    "$SCRIPT_DIR/results/openrouter_eval" \
    "$SCRIPT_DIR/logs"

# The three fixed runs use at most 962 normal requests in total:
#   13 * 24 WOFOST turns + 13 * 24 DSSAT turns + 13 * 26 CycleGym turns.
# This leaves headroom below OpenRouter's 1,000-request daily free-model quota
# for today's format preflights and occasional invalid-action retries.
bash entrypoints/eval/eval.sh \
    --config-name openrouter \
    "data.inference_file=$SCRIPT_DIR/data/t32_unified_wofost/val_wofost_lnw_profit.parquet" \
    "data.validation_axis=schema_tuple" \
    "data.max_samples=13" \
    "model.config=${MODEL_CONFIG}" \
    "runtime.temperature=0" \
    "runtime.max_tokens=4096" \
    "runtime.max_retries=2" \
    "output.dir=$SCRIPT_DIR/results/openrouter_eval/t32_openrouter_hy3_free_heldout1_quota_eval" \
    --log-file "$SCRIPT_DIR/logs/t32_openrouter_hy3_free_heldout1_quota_eval.log"

bash entrypoints/eval/eval.sh \
    --config-name openrouter \
    "data.inference_file=$SCRIPT_DIR/data/t32_unified_gym_dssat_profit/val.parquet" \
    "data.validation_axis=schema_tuple" \
    "data.max_samples=13" \
    "model.config=${MODEL_CONFIG}" \
    "runtime.temperature=0" \
    "runtime.max_tokens=4096" \
    "runtime.max_retries=2" \
    "output.dir=$SCRIPT_DIR/results/openrouter_eval/t32_openrouter_hy3_free_heldout3_quota_eval" \
    --log-file "$SCRIPT_DIR/logs/t32_openrouter_hy3_free_heldout3_quota_eval.log"

bash entrypoints/eval/eval.sh \
    --config-name openrouter \
    "data.inference_file=$SCRIPT_DIR/data/t32_unified_cycles_profit/val.parquet" \
    "data.validation_axis=schema_tuple" \
    "data.max_samples=13" \
    "model.config=${MODEL_CONFIG}" \
    "runtime.temperature=0" \
    "runtime.max_tokens=4096" \
    "runtime.max_retries=2" \
    "output.dir=$SCRIPT_DIR/results/openrouter_eval/t32_openrouter_hy3_free_heldout4_quota_eval" \
    --log-file "$SCRIPT_DIR/logs/t32_openrouter_hy3_free_heldout4_quota_eval.log"
