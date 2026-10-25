#!/usr/bin/env bash
# Seed replication of the T1.3 CyclesGym numerical
# PPO baseline, seed 2.
#
# The submitted paper compares LLM-RL against this NN baseline within a fixed
# schema, but both sides rest on a single training run. This script and its
# sibling add the two runs needed to put an interval on the neural half; the
# existing run is seed 1.
#
# Not affected by fa1bc68: the doubled-action-scale defect required a config to
# set `fert_amount`/`irrig_amount` on top of `scale_action_amounts_by_interval`,
# and the T1.3 configs set none of the three. That is why the fix touched
# sixteen configs and none of them was this one.
#
# Identical to run_nn_train_price_regime.sh except for agent.seed and the run
# name. Same dataset config, same interaction budget, same validation groups.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="t1_3_cycles_gym_price_regime_nn_train_8ep_n1_s2"
DATASET_CONFIG="$SCRIPT_DIR/config/price_regime_nn.yaml"
DATASET_ID="t1_3_cycles_gym_price_regime_nn"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"
TRAIN_FILE="$DATASET_DIR/train.parquet"
VAL_FILE="$DATASET_DIR/val.parquet"
DATASET_BUILD_LOG="$SCRIPT_DIR/logs/${RUN_NAME}_build_dataset.log"
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
SAVE_ROOT="$SCRIPT_DIR/results/nn_train/${RUN_NAME}"

mkdir -p "$SCRIPT_DIR/logs" "$SAVE_ROOT"

cd "$PROJECT_ROOT"

# The dataset is built once by run_build_t13_datasets.sh and shared by all three
# seeds. The reference script rebuilds on every launch; three concurrent seeds
# doing that raced on one directory on 2026-08-17 and two of them died at 13
# seconds with "No such file or directory" after another job's rebuild removed
# the parquets mid-read. Check, never build.
for required in "$TRAIN_FILE" "$VAL_FILE"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing $required" >&2
        echo "Run experiments/t1_3_cycles_gym_price_regime/run_build_t13_datasets.sh first." >&2
        exit 1
    fi
done

bash entrypoints/train/nn_train.sh \
    --config-name nn \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=${VAL_FILE}" \
    "data.validation_axis=price_regime" \
    "agent.track=true" \
    "agent.wandb_project_name=agrimanager_t1_3_cycles_gym_price_regime" \
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
