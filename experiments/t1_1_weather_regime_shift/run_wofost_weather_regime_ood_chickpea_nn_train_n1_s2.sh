#!/usr/bin/env bash
# Seed replication of the T1.1 Chickpea numerical PPO
# baseline, seed 2.
#
# The submitted paper compares LLM-RL against this NN-PPO baseline within a
# fixed schema, but both sides rest on a single training run. This script and
# its sibling add the two runs needed to put an interval on the neural half of
# that comparison; the existing run is seed 1.
#
# n1 and not n8: the paper's LM-versus-NN comparison is built from
# analysis/weather_lm_vs_nn_n1_summary.csv, which covers chickpea and potato at
# n1. The n8 variant is a separate configuration and is not the cell being
# replicated.
#
# Identical to run_wofost_weather_regime_ood_chickpea_nn_train_n1.sh except for
# agent.seed and the run name. Same dataset config, same interaction budget,
# same validation groups, so the three seeds pool legitimately.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="wofost_weather_regime_ood_chickpea_nn_train_8ep_n1_s2"
DATASET_CONFIG="$SCRIPT_DIR/config/weather_regime_chickpea_nn_without_traits.yaml"
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

# The dataset is built once by run_build_chickpea_nn_dataset.sh and shared by every
# run of this baseline. Rebuilding it here would let concurrent jobs read
# each other's half-written parquets.
for required in "${DATASET_FILES[@]}"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing $required" >&2
        echo "Run experiments/t1_1_weather_regime_shift/run_build_chickpea_nn_dataset.sh first." >&2
        exit 1
    fi
done

bash entrypoints/train/nn_train.sh \
    --config-name nn \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=null" \
    "data.validation_axis=weather_regime" \
    "${VAL_SET_OVERRIDES[@]}" \
    "agent.track=true" \
    "agent.wandb_project_name=agrimanager_t1_1_weather_regime_shift" \
    "agent.num_envs=32" \
    "agent.num_steps=72" \
    "agent.num_minibatches=4" \
    "agent.n_epochs=1" \
    "agent.checkpoint_mode=epoch" \
    "agent.learning_rate=0.00025" \
    "agent.gamma=1" \
    "agent.gae_lambda=0.99" \
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
