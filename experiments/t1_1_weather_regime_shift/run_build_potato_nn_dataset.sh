#!/usr/bin/env bash
# Build the T1.1 Potato numerical-PPO dataset once.
#
# The seed replicates of this baseline all read one dataset directory, so they
# must not each rebuild it: concurrent jobs writing the same parquet paths can
# read each other's half-written files. Run this once, then submit the seed
# jobs, which only check that the parquets exist.
#
# Generation is deterministic (`generation_seed: 42`), so every seed trains on
# identical data and rerunning this reproduces the same parquets.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_CONFIG="$SCRIPT_DIR/config/weather_regime_potato_nn_without_traits.yaml"
LOG_DIR="$SCRIPT_DIR/logs"
BUILD_LOG="$LOG_DIR/weather_regime_potato_nn_without_traits_build_dataset.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"

bash entrypoints/dataset/build.sh \
    --config "$DATASET_CONFIG" \
    --num-workers 1 \
    2>&1 | tee "$BUILD_LOG"
