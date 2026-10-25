#!/usr/bin/env bash
# Build both T1.3 price-regime datasets once, think and no-think.
#
# The four E12 replicates read these two directories, so they must not each
# rebuild them: the reference script rebuilds on every launch, and four
# concurrent jobs writing the same parquet paths would read each other's
# half-written files. Generation is deterministic, so building once and reusing
# is what makes the four runs a seed comparison rather than a data-and-seed
# comparison.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

mkdir -p "$SCRIPT_DIR/logs"
cd "$PROJECT_ROOT"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CYCLES_GYM_CYCLESGYM_ROOT="${CYCLES_GYM_CYCLESGYM_ROOT:-${CYCLES_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/CyclesGym}/cyclesgym}"
export CYCLES_GYM_CYCLES_ROOT="${CYCLES_GYM_CYCLES_ROOT:-${CYCLES_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/CyclesGym}/cycles}"
export CYCLES_GYM_WORKING_DIR="/tmp/t13-build-${SLURM_JOB_ID:-$$}"
mkdir -p "$CYCLES_GYM_WORKING_DIR"

for arm in think no_think; do
    bash entrypoints/dataset/build.sh \
        --config "$SCRIPT_DIR/config/price_regime_llm_${arm}.yaml" \
        --num-workers 1 \
        2>&1 | tee "$SCRIPT_DIR/logs/t13_price_regime_llm_${arm}_build.log"
done

# The numerical arm shares this directory across all three seeds for the same
# reason, and its config is price_regime_nn.yaml rather than an arm variant.
bash entrypoints/dataset/build.sh \
    --config "$SCRIPT_DIR/config/price_regime_nn.yaml" \
    --num-workers 1 \
    2>&1 | tee "$SCRIPT_DIR/logs/t13_price_regime_nn_build.log"
