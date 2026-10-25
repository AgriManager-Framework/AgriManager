#!/usr/bin/env bash
# Target-trained numerical PPO reference for the T2.2 `ln`
# action menu, seed 3.
#
# Trains and evaluates on `ln` only, so observation and action dimensions are
# fixed and the existing fixed-interface NN-PPO trainer applies unchanged. No
# action masking is involved; that is E6's separate concern. This policy is a
# reachable learned-policy reference for the `ln` target, not an
# optimal-control upper bound, and it is deliberately given target-menu training
# that the transferring LLM policies never receive.
#
# Three seeds (1, 2, 3) are run per target menu. Single-run results are not
# reliable, so the numerical references carry the same uncertainty reporting as
# the language-model arms; numerical-RL is CPU-only and cheap
# enough that this costs little.
#
# `runtime.train_epochs=16` over 800 scenarios gives the same
# environment-interaction budget as the existing T1.1 NN-PPO baseline
# (1600 scenarios x 8 epochs), so the references match that budget.
#
# Confirmatory evaluation is a separate one-shot step against the frozen
# `e4_t22_action_menu_nn` suite, whose scenarios are identical to the LLM T2.2
# test suite. Do not evaluate against it from inside training.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="t22-nn-target-ln-s3"
TRAIN_SEED=3
TARGET_MENU="ln"

DATASET_CONFIG="$SCRIPT_DIR/config/t22_nn_target_trained.yaml"
DATASET_ID="$(basename "$DATASET_CONFIG" .yaml)"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"
TRAIN_FILE="$DATASET_DIR/train_${TARGET_MENU}.parquet"
VAL_FILE="$DATASET_DIR/val_${TARGET_MENU}.parquet"

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
SAVE_ROOT="$SCRIPT_DIR/results/nn_train/${RUN_NAME}"

mkdir -p "$LOG_DIR" "$SAVE_ROOT"
cd "$PROJECT_ROOT"

# The dataset is built once by run_build_t22_nn_target_dataset.sh and shared by
# all twelve E7 runs. Rebuilding it here would let concurrent jobs corrupt each
# other's parquets on the shared filesystem.
for required in "$TRAIN_FILE" "$VAL_FILE"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing $required" >&2
        echo "Run experiments/t2_2_action_menu_shift/run_build_t22_nn_target_dataset.sh first." >&2
        exit 1
    fi
done

bash entrypoints/train/nn_train.sh \
    --config-name nn \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=null" \
    "data.validation_axis=action_menu" \
    "+data.val_sets.${TARGET_MENU}=${VAL_FILE}" \
    "agent.track=true" \
    "agent.wandb_project_name=agrimanager_t2_2_action_menu_shift" \
    "agent.num_envs=32" \
    "agent.num_steps=72" \
    "agent.num_minibatches=4" \
    "agent.n_epochs=1" \
    "agent.checkpoint_mode=epoch" \
    "agent.learning_rate=0.00025" \
    "agent.gamma=1" \
    "agent.gae_lambda=0.99" \
    "agent.seed=${TRAIN_SEED}" \
    "agent.exp_name=${RUN_NAME}" \
    "runtime.vec_env=subproc" \
    "runtime.subproc_start_method=fork" \
    "runtime.sample_with_replacement=false" \
    "runtime.shard_train_scenarios_across_envs=true" \
    "runtime.train_epochs=16" \
    "runtime.resume.mode=auto_latest" \
    "runtime.validation.enabled=true" \
    "runtime.validation.val_before_train=true" \
    "runtime.validation.evals_per_epoch=2" \
    "output.save_folder=${SAVE_ROOT}" \
    2>&1 | tee "$LOG_FILE"
