#!/usr/bin/env bash
# Build the T2.2 Family A LLM think dataset once.
#
# The R2 and R3 replicates of this cell both read one dataset directory, so they
# must not each rebuild it: concurrent jobs writing the same parquet paths can
# read each other's half-written files. Run this once, then submit the replicate
# jobs, which only check that the parquets exist.
#
# Generation is deterministic, so rerunning this reproduces the same parquets
# and both replicates train on identical data. That is what makes them a seed
# comparison rather than a data-and-seed comparison.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_CONFIG="$SCRIPT_DIR/config/t22_family_a_think.yaml"
LOG_DIR="$SCRIPT_DIR/logs"
BUILD_LOG="$LOG_DIR/t22_family_a_think_build_dataset.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"

bash entrypoints/dataset/build.sh \
    --config "$DATASET_CONFIG" \
    --num-workers 1 \
    2>&1 | tee "$BUILD_LOG"
