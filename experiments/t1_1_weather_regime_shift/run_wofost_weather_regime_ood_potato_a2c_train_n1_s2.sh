#!/usr/bin/env bash
# Seed replication of the T1.1 Potato A2C run, seed 2.
#
# With PPO and DQN this makes three algorithms spanning two design axes rather
# than three unrelated points: PPO and A2C are both on-policy actor-critic and
# differ in the clipped surrogate, while PPO and DQN differ in on- versus
# off-policy learning. Each pair isolates one axis.
#
# The interaction budget is identical by construction: `total_timesteps` comes
# from `train_epochs` and the dataset size, not from the algorithm.
#
# Hyperparameters follow one rule. Knobs that both algorithms expose and that were set
# for this task carry over from the PPO baseline unchanged -- `gamma=1` for the
# short finite episodes that end in a terminal harvest reward, the 72-step
# rollout, the entropy bonus, the value coefficient, the gradient clip, and the
# GAE trace. What stays at Stable-Baselines3's A2C default is the part that
# belongs to A2C itself: the 7e-4 learning rate, which is paired with A2C's
# RMSprop optimizer. Copying PPO's Adam learning rate onto a different optimizer
# would not be "same but for the algorithm", it would be an untuned transplant.
#
# Hyperparameters that PPO exposes and A2C does not (`n_epochs`,
# `num_minibatches`, `clip_coef`) are absent rather than passed and ignored.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="wofost_weather_regime_ood_potato_a2c_train_8ep_n1_s2"
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

# The dataset is built once by run_build_potato_nn_dataset.sh and shared with the
# PPO and DQN runs this one is compared against. Rebuilding it here would break
# the guarantee that all three algorithms saw identical scenarios.
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
    "agent.type=A2C" \
    "agent.track=true" \
    "agent.wandb_project_name=agrimanager_t1_1_weather_regime_shift" \
    "agent.num_envs=32" \
    "agent.num_steps=72" \
    "agent.checkpoint_mode=epoch" \
    "agent.learning_rate=0.0007" \
    "agent.gamma=1" \
    "agent.gae_lambda=0.99" \
    "agent.ent_coef=0.01" \
    "agent.vf_coef=0.5" \
    "agent.max_grad_norm=0.5" \
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
