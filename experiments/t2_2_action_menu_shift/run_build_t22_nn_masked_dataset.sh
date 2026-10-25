#!/usr/bin/env bash
# Build the E6 mask-aware numerical-PPO dataset once.
#
# Family A and Family B share this dataset directory. Build it before launching
# either training run so concurrent jobs do not write the same parquet files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_CONFIG="$SCRIPT_DIR/config/t22_nn_masked_transfer.yaml"
LOG_DIR="$SCRIPT_DIR/logs"
BUILD_LOG="$LOG_DIR/t22_nn_masked_transfer_build_dataset.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

bash entrypoints/dataset/build.sh \
    --config "$DATASET_CONFIG" \
    --num-workers 1 \
    2>&1 | tee "$BUILD_LOG"
