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

# Resume after the first 13-episode DSSAT attempt was lost to two truncated
# HTTP-200 JSON bodies. Eleven complete episodes per remaining subset use at
# most 550 normal requests, leaving retry headroom under today's free quota.
bash entrypoints/eval/eval.sh \
    --config-name openrouter \
    "data.inference_file=$SCRIPT_DIR/data/t32_unified_gym_dssat_profit/val.parquet" \
    "data.validation_axis=schema_tuple" \
    "data.max_samples=11" \
    "model.config=${MODEL_CONFIG}" \
    "+data.env_config_overrides.action_parser=legacy" \
    "runtime.temperature=0" \
    "runtime.max_tokens=4096" \
    "runtime.max_retries=2" \
    "output.dir=$SCRIPT_DIR/results/openrouter_eval/t32_openrouter_hy3_free_heldout3_quota_resume" \
    --log-file "$SCRIPT_DIR/logs/t32_openrouter_hy3_free_heldout3_quota_resume.log"

bash entrypoints/eval/eval.sh \
    --config-name openrouter \
    "data.inference_file=$SCRIPT_DIR/data/t32_unified_cycles_profit/val.parquet" \
    "data.validation_axis=schema_tuple" \
    "data.max_samples=11" \
    "model.config=${MODEL_CONFIG}" \
    "+data.env_config_overrides.action_parser=legacy" \
    "runtime.temperature=0" \
    "runtime.max_tokens=4096" \
    "runtime.max_retries=2" \
    "output.dir=$SCRIPT_DIR/results/openrouter_eval/t32_openrouter_hy3_free_heldout4_quota_resume" \
    --log-file "$SCRIPT_DIR/logs/t32_openrouter_hy3_free_heldout4_quota_resume.log"
