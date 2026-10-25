#!/usr/bin/env bash
# Build the T1.2 cross-crop 4ID numerical-PPO dataset once.
#
# The n1 and n8 runs of this cell both read one dataset directory, so they must
# not each rebuild it: concurrent jobs writing the same parquet paths can read
# each other's half-written files. Run this once, then submit the training jobs,
# which only check that the parquets exist.
#
# Generation is deterministic, so rerunning this reproduces the same parquets
# and both runs train on identical data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_CONFIG="$SCRIPT_DIR/config/cross_crop_4id_nn_traits_v1_23d.yaml"
LOG_DIR="$SCRIPT_DIR/logs"
BUILD_LOG="$LOG_DIR/cross_crop_4id_nn_traits_v1_23d_build_dataset.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

bash entrypoints/dataset/build.sh \
    --config "$DATASET_CONFIG" \
    --num-workers 1 \
    2>&1 | tee "$BUILD_LOG"
