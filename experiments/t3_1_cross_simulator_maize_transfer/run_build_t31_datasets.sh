#!/usr/bin/env bash
# Build the three T3.1 cross-simulator datasets once, up front.
#
# The reference `run_cross_sim_*_llm_think_train.sh` scripts rebuilt all three
# datasets inside every launch. That was harmless for a single run, but the seed
# replicates run concurrently: six jobs rebuilding the same parquets would write
# over each other and read each other's half-written files, and the replicates
# would no longer differ only in their seed.
#
# So the build moved here. Run this once, confirm the six splits exist, then
# launch the replicates, which verify and fail rather than build.
#
# Each dataset config carries a fixed generation seed, so this build is
# deterministic: R1, R2 and R3 train on identical data and differ only in the
# training seed.
#
# Building DSSAT requires the gym-DSSAT runtime. Building CycleGym requires this
# account's own CyclesGym checkout, because the environment chmods its binary on
# startup and raises PermissionError against another user's copy.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

WOFOST_CONFIG="$SCRIPT_DIR/config/t31_cross_sim_wofost.yaml"
DSSAT_CONFIG="$SCRIPT_DIR/config/t31_cross_sim_gym_dssat.yaml"
CYCLES_CONFIG="$SCRIPT_DIR/config/t31_cross_sim_cycles_gym.yaml"

LOG_DIR="$SCRIPT_DIR/logs"

prepare_dssat_runtime() {
    if [[ -f "$PROJECT_ROOT/smoke_tests/gym_dssat/_activate_spack.sh" ]]; then
        # shellcheck disable=SC1091
        source "$PROJECT_ROOT/smoke_tests/gym_dssat/_activate_spack.sh"
    fi

    local candidate
    for candidate in \
        "$PROJECT_ROOT/AgriManagerExternal/gym_dssat_pdi/gym-dssat-pdi" \
        "$PROJECT_ROOT/../AgriManagerExternal/gym_dssat_pdi/gym-dssat-pdi"
    do
        if [[ -d "$candidate/gym_dssat_pdi" ]]; then
            case ":${PYTHONPATH:-}:" in
                *":$candidate:"*) ;;
                *) export PYTHONPATH="$candidate:${PYTHONPATH:-}" ;;
            esac
            break
        fi
    done
}

prepare_cycles_runtime() {
    export CYCLES_GYM_CYCLESGYM_ROOT="${CYCLES_GYM_CYCLESGYM_ROOT:-${CYCLES_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/CyclesGym}/cyclesgym}"
    export CYCLES_GYM_CYCLES_ROOT="${CYCLES_GYM_CYCLES_ROOT:-${CYCLES_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/CyclesGym}/cycles}"
    export CYCLES_GYM_WORKING_DIR="/tmp/cycles-gym-build-$$"
    mkdir -p "$CYCLES_GYM_WORKING_DIR"
}

build_dataset() {
    local config="$1"
    local dataset_id
    dataset_id="$(basename "$config" .yaml)"

    echo "[dataset] building ${dataset_id}"
    bash entrypoints/dataset/build.sh \
        --config "$config" \
        --num-workers 1 \
        2>&1 | tee "$LOG_DIR/build_${dataset_id}.log"
}

mkdir -p "$LOG_DIR"
cd "$PROJECT_ROOT"
prepare_dssat_runtime
prepare_cycles_runtime

build_dataset "$WOFOST_CONFIG"
build_dataset "$DSSAT_CONFIG"
build_dataset "$CYCLES_CONFIG"

echo
echo "[dataset] splits now on disk:"
for dataset_id in t31_cross_sim_wofost t31_cross_sim_gym_dssat t31_cross_sim_cycles_gym; do
    for split in train val; do
        split_file="$SCRIPT_DIR/data/${dataset_id}/${split}.parquet"
        if [[ -f "$split_file" ]]; then
            printf '  OK   %s\n' "$split_file"
        else
            printf '  MISS %s\n' "$split_file"
        fi
    done
done
