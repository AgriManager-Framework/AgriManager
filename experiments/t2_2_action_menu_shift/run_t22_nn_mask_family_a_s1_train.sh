#!/usr/bin/env bash
# Mask-aware numerical PPO transfer for T2.2 Family A.
#
# Trains on the full-coverage `lnpkw` and `lnpk` menus under a canonical
# 17-action vocabulary with per-menu legal-action masks. This is the neural
# counterpart to the T2.2 Family A transfer setting, not an LLM replication.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="t22-nn-mask-family-a-s1"
TRAIN_SEED=1

DATASET_CONFIG="$SCRIPT_DIR/config/t22_nn_masked_transfer.yaml"
DATASET_ID="$(basename "$DATASET_CONFIG" .yaml)"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"
TRAIN_LNPKW_FILE="$DATASET_DIR/train_lnpkw.parquet"
TRAIN_LNPK_FILE="$DATASET_DIR/train_lnpk.parquet"
VAL_LNPKW_FILE="$DATASET_DIR/val_lnpkw.parquet"
VAL_LNPK_FILE="$DATASET_DIR/val_lnpk.parquet"
VAL_LNW_FILE="$DATASET_DIR/val_lnw.parquet"
VAL_LN_FILE="$DATASET_DIR/val_ln.parquet"
VAL_LW_FILE="$DATASET_DIR/val_lw.parquet"

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
SAVE_ROOT="$SCRIPT_DIR/results/nn_train/${RUN_NAME}"

mkdir -p "$LOG_DIR" "$SAVE_ROOT"
cd "$PROJECT_ROOT"

DATASET_FILES=(
    "$TRAIN_LNPKW_FILE"
    "$TRAIN_LNPK_FILE"
    "$VAL_LNPKW_FILE"
    "$VAL_LNPK_FILE"
    "$VAL_LNW_FILE"
    "$VAL_LN_FILE"
    "$VAL_LW_FILE"
)

for required in "${DATASET_FILES[@]}"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing $required" >&2
        echo "Run experiments/t2_2_action_menu_shift/run_build_t22_nn_masked_dataset.sh first." >&2
        exit 1
    fi
done

bash entrypoints/train/nn_train.sh \
    --config-name nn \
    "data.train_files=[${TRAIN_LNPKW_FILE},${TRAIN_LNPK_FILE}]" \
    "data.val_files=null" \
    "data.validation_axis=action_menu" \
    "+data.val_sets.lnpkw=${VAL_LNPKW_FILE}" \
    "+data.val_sets.lnpk=${VAL_LNPK_FILE}" \
    "+data.val_sets.lnw=${VAL_LNW_FILE}" \
    "+data.val_sets.ln=${VAL_LN_FILE}" \
    "+data.val_sets.lw=${VAL_LW_FILE}" \
    "agent.type=MaskablePPO" \
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
    "runtime.train_epochs=8" \
    "runtime.resume.mode=auto_latest" \
    "runtime.validation.enabled=true" \
    "runtime.validation.val_before_train=true" \
    "runtime.validation.evals_per_epoch=2" \
    "output.save_folder=${SAVE_ROOT}" \
    2>&1 | tee "$LOG_FILE"
