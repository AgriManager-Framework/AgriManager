#!/usr/bin/env bash
# Seed replication of the T1.1 Potato DQN run, seed 2.
#
# The PPO side of this comparison has three seeds, so a single DQN run would
# leave the algorithm-breadth result as a point estimate against an interval.
# This script and its seed-3 sibling put the same interval on the DQN side.
#
# Identical to run_wofost_weather_regime_ood_potato_dqn_train_n1.sh except for
# agent.seed and the run name. Same dataset config, same interaction budget,
# same validation groups, so the three seeds are pooled legitimately.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="wofost_weather_regime_ood_potato_dqn_train_8ep_n1_s2"
DATASET_CONFIG="$SCRIPT_DIR/config/weather_regime_potato_nn_without_traits.yaml"
DATASET_ID="$(basename "$DATASET_CONFIG" .yaml)"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"
TRAIN_FILE="$DATASET_DIR/train.parquet"
VAL_ID_FILE="$DATASET_DIR/val_id.parquet"
VAL_DROUGHT_FILE="$DATASET_DIR/val_drought.parquet"
VAL_WET_FILE="$DATASET_DIR/val_wet.parquet"
VAL_HOT_FILE="$DATASET_DIR/val_hot.parquet"
VAL_COLD_FILE="$DATASET_DIR/val_cold.parquet"
VAL_SET_OVERRIDES=(
    "+data.val_sets.id=${VAL_ID_FILE}"
    "+data.val_sets.drought=${VAL_DROUGHT_FILE}"
    "+data.val_sets.wet=${VAL_WET_FILE}"
    "+data.val_sets.hot=${VAL_HOT_FILE}"
    "+data.val_sets.cold=${VAL_COLD_FILE}"
)
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
SAVE_ROOT="$SCRIPT_DIR/results/nn_train/${RUN_NAME}"

mkdir -p "$SCRIPT_DIR/logs" "$SAVE_ROOT"

cd "$PROJECT_ROOT"

DATASET_FILES=(
    "$TRAIN_FILE"
    "$VAL_ID_FILE"
    "$VAL_DROUGHT_FILE"
    "$VAL_WET_FILE"
    "$VAL_HOT_FILE"
    "$VAL_COLD_FILE"
)

# The dataset is built once by run_build_potato_nn_dataset.sh and shared with
# the PPO baseline and the other DQN seeds. Rebuilding it here would let
# concurrent seed jobs read each other's half-written parquets.
for required in "${DATASET_FILES[@]}"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing $required" >&2
        echo "Run experiments/t1_1_weather_regime_shift/run_build_potato_nn_dataset.sh first." >&2
        exit 1
    fi
done

bash entrypoints/train/nn_train.sh \
    --config-name nn \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=null" \
    "data.validation_axis=weather_regime" \
    "${VAL_SET_OVERRIDES[@]}" \
    "agent.type=DQN" \
    "agent.track=true" \
    "agent.wandb_project_name=agrimanager_t1_1_weather_regime_shift" \
    "agent.num_envs=32" \
    "agent.num_steps=72" \
    "agent.checkpoint_mode=epoch" \
    "agent.learning_rate=0.0001" \
    "agent.gamma=1" \
    "agent.buffer_size=100000" \
    "agent.learning_starts=2304" \
    "agent.batch_size=64" \
    "agent.train_freq=4" \
    "agent.gradient_steps=1" \
    "agent.target_update_interval=1000" \
    "agent.exploration_fraction=0.1" \
    "agent.exploration_final_eps=0.05" \
    "agent.seed=2" \
    "agent.exp_name=${RUN_NAME}" \
    "runtime.vec_env=subproc" \
    "runtime.subproc_start_method=fork" \
    "runtime.sample_with_replacement=false" \
    "runtime.shard_train_scenarios_across_envs=true" \
    "runtime.train_epochs=8" \
    "runtime.resume.mode=auto_latest" \
    "runtime.validation.enabled=true" \
    "runtime.validation.val_before_train=true" \
    "runtime.validation.evals_per_epoch=4" \
    "output.save_folder=${SAVE_ROOT}" \
    2>&1 | tee "$LOG_FILE"
