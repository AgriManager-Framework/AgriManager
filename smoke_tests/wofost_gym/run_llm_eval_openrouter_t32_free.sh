#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="t32_heldout_wofost_lnw_profit_openrouter_free_smoke"
INFERENCE_FILE="experiments/t3_2_unified_prompt_conditioned_policy/data/t32_unified_wofost/val_wofost_lnw_profit.parquet"
MODEL_CONFIG="agrimanager/model_interface/configs/openrouter/nemotron-3-nano-30b-a3b-free_no_reasoning.yaml"
OUTPUT_DIR="$SCRIPT_DIR/results/llm_eval/$RUN_NAME"
LOG_FILE="$SCRIPT_DIR/logs/$RUN_NAME.log"

cd "$PROJECT_ROOT"

if [[ ! -f "$INFERENCE_FILE" ]]; then
    echo "Missing checked-in T3.2 validation dataset: $INFERENCE_FILE" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$SCRIPT_DIR/logs"

bash entrypoints/eval/eval.sh \
    --config-name openrouter \
    "data.inference_file=${INFERENCE_FILE}" \
    "data.validation_axis=schema_tuple" \
    "data.max_samples=1" \
    "model.config=${MODEL_CONFIG}" \
    "output.dir=${OUTPUT_DIR}" \
    "runtime.temperature=0" \
    "runtime.max_tokens=128" \
    "runtime.max_retries=0" \
    --log-file "$LOG_FILE"
