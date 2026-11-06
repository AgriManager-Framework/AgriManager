#!/usr/bin/env bash
set -euo pipefail

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_NAME="t32_openrouter_glm52_thinking_heldout134_eval"
MODEL_CONFIG="agrimanager/model_interface/configs/openrouter/glm-5.2_thinking.yaml"
WOFOST_VAL="$SCRIPT_DIR/data/t32_unified_wofost/val_wofost_lnw_profit.parquet"
DSSAT_VAL="$SCRIPT_DIR/data/t32_unified_gym_dssat_profit/val.parquet"
CYCLES_VAL="$SCRIPT_DIR/data/t32_unified_cycles_profit/val.parquet"
OUTPUT_DIR="$SCRIPT_DIR/results/openrouter_eval/$RUN_NAME"
LOG_FILE="$SCRIPT_DIR/logs/$RUN_NAME.log"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR" "$SCRIPT_DIR/logs"

bash entrypoints/eval/eval.sh \
    --config-name openrouter \
    "data.inference_file=[${WOFOST_VAL},${DSSAT_VAL},${CYCLES_VAL}]" \
    "data.validation_axis=schema_tuple" \
    "data.max_samples=null" \
    "model.config=${MODEL_CONFIG}" \
    "+data.env_config_overrides.action_parser=legacy" \
    "runtime.temperature=0" \
    "runtime.max_tokens=4096" \
    "runtime.max_retries=2" \
    "output.dir=${OUTPUT_DIR}" \
    --log-file "$LOG_FILE"
