#!/usr/bin/env bash
# Internal helper for the fixed T2.3 seed replicates in this directory.
# The public run_*.sh wrappers hardcode every experiment-defining value.
#
# Usage: _run_t23_seed_replicate.sh VARIANT ARM REPLICATE SEED
#   VARIANT   objectives | yield_only
#   ARM       think | no_think
#   REPLICATE r2 | r3
#   SEED      91 (R2) | 92 (R3)
#
# T2.3 reward-formulation shift, WOFOST maize, Qwen3-4B-Instruct, GRPO.
# Independent training replicates of the four paper reference runs. Everything
# that defines the experiment is identical to the reference except the seed.
#
# The references (experiment-tracker run ids of the submitted runs):
#
#   objectives  think     8l9c0vxp   4 epochs, 400 steps
#   objectives  no_think  gyzao2k2   4 epochs, 400 steps
#   yield_only  think     h3v8z1uf   4 epochs, 400 steps
#   yield_only  no_think  2kx0h7jn   4 epochs, 400 steps
#
# All four are 4 epochs and 400 steps, so the think and no-think arms of a cell
# are directly comparable. `ndz9sdfg` is *not* the objectives no-think
# reference: it is 2 epochs and 200 steps and appears only in the matched-step
# secondary analysis, whose CSV carries `comparison=step_200`. Replicating it
# would reproduce the secondary comparison and not the headline cell.
#
# The yield-only ablation is not a smaller dataset. It passes
# train_yield_max.parquet three times where the objectives cell passes three
# different objective files, so the row count is matched and only the diversity
# of objectives is removed. That is the ablation, and it must be preserved
# exactly.
#
# The arm is a training-time override here, unlike T2.2 where it is baked into
# the dataset. One dataset serves all eight replicates.
#
# Seed protocol for the E10 replicate family:
#   R1 (paper reference) : data.seed unset (unseeded shuffle, not reproducible)
#                          actor.data_loader_seed = 42 (framework default)
#   R2 (this script)     : 91
#   R3                   : 92
# Both knobs are set together so one integer identifies the replicate. The think
# and no-think arms of a cell share a seed: the cell is reported as a paired
# think-minus-no-think effect, so different seeds would confound that contrast
# with seed variation. objectives and yield_only are separate cells rather than
# paired arms, so they reuse the same pair.
#
# Deliberate differences from the reference scripts, and why:
#   trainer.total_training_steps=400
#       Made explicit. The references left it unset and derived 400 from
#       total_epochs=4. Pinning it means an interrupted replicate is reported
#       against a declared target rather than an inferred one.
#   trainer.max_actor_ckpt_to_keep=2 (and critic)
#       The references kept 1, which deletes the previous checkpoint before
#       writing the next and leaves a window with no complete checkpoint on
#       disk. Keeping 2 never triggers that branch.
#   actor.checkpoint.save_contents=[model,extra]
#       A full Qwen3-4B checkpoint is 42.4 GB; dropping optimizer state gives
#       17.6 GB. The cost is that training cannot resume mid-run, which this
#       protocol never does -- it selects the terminal or last-valid checkpoint
#       and never resumes into a different trajectory.
#   Results under $AGRI_WORK_DIR, outside the repository
#       Retained checkpoints are 17.6 GB each; put AGRI_WORK_DIR on a filesystem
#       with room for them (on Lustre, AGRI_LUSTRE_POOL selects a pool).
#   The dataset is not rebuilt here
#       Eight replicates share one dataset directory. Concurrent rebuilds would
#       let them read each other's half-written parquets, and would stop them
#       being a seed comparison. Build it once with run_build_t23_dataset.sh.
#
# Retained from the references on purpose:
#   trainer.val_before_train=True, trainer.test_freq=10
#       The references validated this way -- 8l9c0vxp carries 41 validation
#       rows at steps 0,10,...,400 -- and that curve is what makes peak-to-final
#       retention computable and comparable with R1 curve for curve. These are
#       training-time splits shipped with the dataset config, never E4 data.
#   trainer.rollout_filter.enable=False
#       The references ran with the filter off. Turning it on for the
#       replicates would make them incomparable. This differs from T1.2 and
#       T2.2, where the references ran with it on.
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 VARIANT ARM REPLICATE SEED" >&2
    exit 2
fi

VARIANT="$1"
ARM="$2"
REPLICATE="$3"
TRAIN_SEED="$4"

case "$VARIANT" in objectives|yield_only) ;; *) echo "bad VARIANT: $VARIANT" >&2; exit 2 ;; esac
case "$ARM" in think|no_think) ;; *) echo "bad ARM: $ARM" >&2; exit 2 ;; esac

if [[ "$ARM" == "think" ]]; then
    REQUIRE_THINK=True
else
    REQUIRE_THINK=False
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="schema-t23-${VARIANT//_/-}-${ARM//_/-}-${REPLICATE}"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"

DATASET_CONFIG="$SCRIPT_DIR/config/t23_reward_formulation_shift.yaml"
DATASET_ID="$(basename "$DATASET_CONFIG" .yaml)"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"

TRAIN_YIELD_FILE="$DATASET_DIR/train_yield_max.parquet"
TRAIN_PROFIT_FILE="$DATASET_DIR/train_profit_max.parquet"
TRAIN_WATER_FILE="$DATASET_DIR/train_water_stewardship.parquet"
VAL_YIELD_FILE="$DATASET_DIR/val_yield_max.parquet"
VAL_PROFIT_FILE="$DATASET_DIR/val_profit_max.parquet"
VAL_WATER_FILE="$DATASET_DIR/val_water_stewardship.parquet"
VAL_NUTRIENT_FILE="$DATASET_DIR/val_nutrient_stewardship.parquet"

# The yield-only ablation repeats the yield file three times so the row count
# matches the objectives cell and only objective diversity is removed.
if [[ "$VARIANT" == "objectives" ]]; then
    TRAIN_FILES_ARG="[${TRAIN_YIELD_FILE},${TRAIN_PROFIT_FILE},${TRAIN_WATER_FILE}]"
else
    TRAIN_FILES_ARG="[${TRAIN_YIELD_FILE},${TRAIN_YIELD_FILE},${TRAIN_YIELD_FILE}]"
fi

VAL_SET_OVERRIDES=(
    "+data.val_sets.yield_max=${VAL_YIELD_FILE}"
    "+data.val_sets.profit_max=${VAL_PROFIT_FILE}"
    "+data.val_sets.water_stewardship=${VAL_WATER_FILE}"
    "+data.val_sets.nutrient_stewardship=${VAL_NUTRIENT_FILE}"
)

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
RESULT_DIR="${AGRI_WORK_DIR:-$HOME/agrimanager_work}/agri_t23_results/t23_llm_train/${RUN_NAME}"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

# Checkpoints go under $AGRI_WORK_DIR (default ~/agrimanager_work), outside the
# repository: each retained checkpoint is 17.6 GB.
ensure_hdd_pool() {
    # Optional: on Lustre, stripe the checkpoint directory onto the pool named by
    # AGRI_LUSTRE_POOL. Does nothing when the variable is unset or lfs is missing.
    local dir="$1" pool="${AGRI_LUSTRE_POOL:-}"
    [[ -n "$pool" ]] || return 0
    command -v lfs >/dev/null 2>&1 || return 0
    lfs getstripe -d "$dir" 2>/dev/null | grep -q "pool:[[:space:]]*$pool" && return 0
    if lfs setstripe --pool "$pool" -c 4 "$dir" 2>/dev/null; then
        echo "striped $dir onto the $pool pool"
    else
        echo "WARNING: could not stripe $dir onto $pool" >&2
    fi
}
ensure_hdd_pool "$RESULT_DIR"
ensure_hdd_pool "$LOG_DIR"

for required in \
    "$TRAIN_YIELD_FILE" "$TRAIN_PROFIT_FILE" "$TRAIN_WATER_FILE" \
    "$VAL_YIELD_FILE" "$VAL_PROFIT_FILE" "$VAL_WATER_FILE" "$VAL_NUTRIENT_FILE"
do
    if [[ ! -f "$required" ]]; then
        echo "Dataset $DATASET_DIR is missing or incomplete: $required" >&2
        echo "Run experiments/t2_3_reward_formulation_shift/run_build_t23_dataset.sh first." >&2
        exit 1
    fi
done

cd "$PROJECT_ROOT"
export WOFOST_GYM_PATH="${WOFOST_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/WOFOSTGym}"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/verl:$WOFOST_GYM_PATH:$WOFOST_GYM_PATH/pcse_gym:$WOFOST_GYM_PATH/pcse:${PYTHONPATH:-}"

exec bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name agri_grpo \
    "data.train_files=${TRAIN_FILES_ARG}" \
    "data.val_files=null" \
    "data.seed=${TRAIN_SEED}" \
    "data.validation_axis=reward_formulation" \
    "${VAL_SET_OVERRIDES[@]}" \
    "+data.env_config_overrides.require_think=${REQUIRE_THINK}" \
    "+data.env_config_overrides.thinking_mode=think" \
    "+data.env_config_overrides.think_tag=tool_call" \
    "data.gen_batch_size=16" \
    "data.max_response_length=512" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "critic.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.actor.data_loader_seed=${TRAIN_SEED}" \
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]" \
    "actor_rollout_ref.rollout.n=4" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64" \
    "trainer.logger=[\"console\", \"wandb\"]" \
    "trainer.project_name=agrimanager_wofost_reward_formulation_shift" \
    "trainer.default_local_dir=${RESULT_DIR}" \
    "trainer.experiment_name=${RUN_NAME}" \
    "trainer.total_epochs=4" \
    "trainer.total_training_steps=400" \
    "trainer.val_before_train=True" \
    "trainer.test_freq=10" \
    "trainer.n_gpus_per_node=2" \
    "trainer.save_freq=100" \
    "trainer.max_actor_ckpt_to_keep=2" \
    "trainer.max_critic_ckpt_to_keep=2" \
    "trainer.rollout_filter.enable=False" \
    "ray_kwargs.ray_init.num_cpus=32" \
    "+ray_kwargs.ray_init.include_dashboard=False" \
    "+ray_kwargs.ray_init._temp_dir=/tmp/ray_${SLURM_JOB_ID:-${SLURM_JOBID:-$$}}"
