#!/usr/bin/env bash
# Build all five T3.2 unified datasets once.
#
# Every E2 replicate reads these five directories. The fixed run scripts rebuild
# all five on each launch, so two replicates started together race on the same
# paths; the replicate scripts here check instead, and this is what fills the
# directories. Generation is deterministic, so building once and reusing is what
# makes the seeds a seed comparison rather than a data-and-seed comparison.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

mkdir -p "$SCRIPT_DIR/logs"
cd "$PROJECT_ROOT"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export CYCLES_GYM_CYCLESGYM_ROOT="${CYCLES_GYM_CYCLESGYM_ROOT:-${CYCLES_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/CyclesGym}/cyclesgym}"
export CYCLES_GYM_CYCLES_ROOT="${CYCLES_GYM_CYCLES_ROOT:-${CYCLES_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/CyclesGym}/cycles}"
export CYCLES_GYM_WORKING_DIR="/tmp/t32-build-${SLURM_JOB_ID:-$$}"
mkdir -p "$CYCLES_GYM_WORKING_DIR"

for cfg in t32_unified_wofost t32_unified_gym_dssat_yield t32_unified_gym_dssat_profit \
           t32_unified_cycles_yield t32_unified_cycles_profit; do
    bash entrypoints/dataset/build.sh \
        --config "$SCRIPT_DIR/config/${cfg}.yaml" \
        --num-workers 1 \
        2>&1 | tee "$SCRIPT_DIR/logs/${cfg}_build.log"
done
