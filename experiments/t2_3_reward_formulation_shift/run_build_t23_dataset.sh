#!/usr/bin/env bash
# Build the T2.3 reward-formulation dataset once.
#
# All eight E10 replicates read this one dataset directory, so they must not
# each rebuild it: concurrent jobs writing the same parquet paths can read each
# other's half-written files. Run this once, then submit the replicate jobs,
# which only check that the parquets exist.
#
# Generation is deterministic -- the config carries a fixed generation seed --
# so rerunning this reproduces the same parquets and every replicate trains on
# identical data. That is what makes them a seed comparison rather than a
# data-and-seed comparison.
#
# One dataset serves both arms here. Unlike T2.2, where think and no-think are
# separate dataset configs, T2.3 sets the arm at training time through
# `+data.env_config_overrides.require_think`, which is how the references ran.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_CONFIG="$SCRIPT_DIR/config/t23_reward_formulation_shift.yaml"
LOG_DIR="$SCRIPT_DIR/logs"
BUILD_LOG="$LOG_DIR/t23_reward_formulation_shift_build_dataset.log"

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export WOFOST_GYM_PATH="${WOFOST_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/WOFOSTGym}"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/verl:$WOFOST_GYM_PATH:$WOFOST_GYM_PATH/pcse_gym:$WOFOST_GYM_PATH/pcse:${PYTHONPATH:-}"

bash entrypoints/dataset/build.sh \
    --config "$DATASET_CONFIG" \
    --num-workers 1 \
    2>&1 | tee "$BUILD_LOG"
