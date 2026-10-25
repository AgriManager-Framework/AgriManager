#!/usr/bin/env bash
# Build the E7 target-trained numerical-PPO dataset once.
#
# The twelve E7 training runs share this one dataset directory, so they must not
# each rebuild it: on a shared filesystem twelve concurrent writers to the same
# parquet paths corrupt them. Run this once before submitting those jobs; each
# training script then only checks that its two parquets exist and refuses to
# start otherwise.
#
# Generation is deterministic (`generation_seed: 42`), so rerunning this
# reproduces byte-identical parquets and every seed trains on identical data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_CONFIG="$SCRIPT_DIR/config/t22_nn_target_trained.yaml"
LOG_DIR="$SCRIPT_DIR/logs"
BUILD_LOG="$LOG_DIR/t22_nn_target_trained_build_dataset.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

bash entrypoints/dataset/build.sh \
    --config "$DATASET_CONFIG" \
    --num-workers 1 \
    2>&1 | tee "$BUILD_LOG"
