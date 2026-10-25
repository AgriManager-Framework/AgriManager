#!/usr/bin/env bash
# T1.2 16ID numerical-RL seed replicate S3, n1 arm.
#
# Independent training replicate of the paper reference run
# `run_wofost_generalization_cross_crop_16id_nn_train_n1.sh`, whose W&B run
# is `5ddeuf76`. Everything that defines the experiment is identical to the
# reference except `agent.seed`.
#
# The paper's T1.2 numerical cells are the `*_nn_train_8ep_n1` and
# `*_nn_train_8ep_n8` runs, which is what the figure package reads. They are
# **not** the `*_16ep_matched_train` runs: those trained on 3200 scenarios with
# a single mixed validation file and were resumed from a checkpoint, and the
# figure package never references them. Both variants of a coverage setting are
# separate reported cells, so both need their own seeds.
#
# Reference protocol, read from the saved config of the run itself:
#   num_train_scenarios  1600      (the same budget the LLM arms use)
#   train_epochs         8
#   n_epochs             1         (PPO update epochs; this is the n1/n8 axis)
#   num_envs             32
#   num_steps            72
#   num_minibatches      4
#   learning_rate        0.00025
#   validation           val_files null, two named sets id_crops + heldout_crops
#
# Seed protocol for this family: R1 = agent.seed 1 (the reference), S2 = 2,
# S3 = 3. The numerical agent takes a single seed knob, unlike the LLM arms
# which set data.seed and actor.data_loader_seed together.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="schema-t12-crosscrop-16id-nn-n1-s3"
DATASET_CONFIG="$SCRIPT_DIR/config/cross_crop_16id_nn_traits_v1_23d.yaml"
DATASET_ID="$(basename "$DATASET_CONFIG" .yaml)"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"
TRAIN_FILE="$DATASET_DIR/train.parquet"
VAL_ID_CROPS_FILE="$DATASET_DIR/val_id_crops.parquet"
VAL_HELDOUT_CROPS_FILE="$DATASET_DIR/val_heldout_crops.parquet"
VAL_SET_OVERRIDES=(
    "+data.val_sets.id_crops=${VAL_ID_CROPS_FILE}"
    "+data.val_sets.heldout_crops=${VAL_HELDOUT_CROPS_FILE}"
)
LOG_FILE="$SCRIPT_DIR/logs/${RUN_NAME}.log"
SAVE_ROOT="$SCRIPT_DIR/results/nn_train/${RUN_NAME}"

mkdir -p "$SCRIPT_DIR/logs" "$SAVE_ROOT"

cd "$PROJECT_ROOT"

# The dataset is built once by run_build_cross_crop_16id_nn_dataset.sh and shared
# by the n1 and n8 runs of this cell. Rebuilding it here would let concurrent jobs
# read each other's half-written parquets.
for required in "$TRAIN_FILE" "$VAL_ID_CROPS_FILE" "$VAL_HELDOUT_CROPS_FILE"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing $required" >&2
        echo "Run experiments/t1_2_cross_crop_trait_shift/run_build_cross_crop_16id_nn_dataset.sh first." >&2
        exit 1
    fi
done

bash entrypoints/train/nn_train.sh \
    --config-name nn \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=null" \
    "data.validation_axis=crop" \
    "${VAL_SET_OVERRIDES[@]}" \
    "agent.track=true" \
    "agent.wandb_project_name=agrimanager_wofost_cross_crop_generalization" \
    "agent.num_envs=32" \
    "agent.num_steps=72" \
    "agent.num_minibatches=4" \
    "agent.n_epochs=1" \
    "agent.checkpoint_mode=epoch" \
    "agent.learning_rate=0.00025" \
    "agent.gamma=1" \
    "agent.gae_lambda=0.99" \
    "agent.seed=3" \
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
