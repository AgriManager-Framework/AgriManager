#!/usr/bin/env bash
# Seed replicate E8-R2, gym-DSSAT source, think arm.
#
# T3.1 cross-simulator maize transfer. One source policy is trained on
# gym-DSSAT and validated on all three simulators. Independent training
# replicate of the paper reference run
# `run_cross_sim_gym_dssat_llm_think_train.sh`. Everything that defines the
# experiment is identical to the reference except the random seed and the
# checkpoint-retention settings noted below.
#
# Seed protocol for the E8 replicate family:
#   R1 (paper reference) : data.seed unset (unseeded shuffle, not reproducible)
#                          actor.data_loader_seed = 42 (framework default)
#   R2                   : 71
#   R3                   : 72
# Both seed knobs are set together so one number identifies the replicate.
# `data.seed` fixes the training-data shuffle order; `actor.data_loader_seed`
# fixes minibatch order inside the actor update. They are distinct knobs.
#
# The think and no-think arms of a given replicate index share their seed. T3.1
# reports a paired think-minus-no-think effect, so giving the two arms different
# seeds would confound that contrast with seed variation.
#
# Deliberate differences from the R1 script, and why:
#   trainer.total_training_steps=200
#       Made explicit. The reference left it unset and derived the target from
#       trainer.total_epochs=2: the train split holds 1600 scenarios and
#       data.gen_batch_size=16, so two epochs are 1600 * 2 / 16 = 200 steps.
#       Pinning it means an interrupted replicate is reported against a declared
#       target instead of an inferred one.
#   trainer.max_actor_ckpt_to_keep=2 (and critic)
#       save_freq is 100 over 200 steps, so checkpoints land at 100 and 200. The
#       config default keeps 1, which deletes step 100 before writing step 200
#       and leaves a window with no complete checkpoint on disk.
#   actor.checkpoint.save_contents=[model,extra]  (default also writes optimizer)
#       A full Qwen3-4B checkpoint measures 42.4 GB: 17.6 GB sharded model plus
#       24.8 GB optimizer state. Dropping the optimizer gives 17.6 GB per
#       checkpoint, so both retained checkpoints fit alongside concurrent runs.
#       The cost is that training cannot resume mid-run; that is already the
#       declared protocol, which selects the terminal or last-valid checkpoint
#       and never resumes into a different trajectory. Convert the sharded model
#       to Transformers layout offline with `verl.model_merger`.
#   The dataset is not rebuilt here.
#       The reference rebuilt all three datasets on every launch. Six concurrent
#       replicates would write the same parquets on top of each other and read
#       each other's half-written files, which would stop them being a seed
#       comparison. Build once with `run_build_t31_datasets.sh`; this script
#       verifies the splits exist and fails if they do not.
#
# Retained from the reference on purpose:
#   trainer.test_freq=10, trainer.val_before_train=True
#       Both are the config defaults, so the reference ran with them and this
#       replicate inherits the same values without restating them. The
#       validation curve is what makes peak-to-final retention computable. These
#       are the training-time splits shipped with the dataset config, never E4
#       data, so the rule that E4 must not touch training is unaffected.
#   think_tag=tool_call
#       T3.1 tags reasoning as tool_call, not as think. Changing it would make
#       the replicate incomparable with R1 and with the T3.2 model axis.
#   trainer.n_gpus_per_node=2
#       Matches the reference. Batch sizes are global, so this affects sharding
#       and throughput only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="schema-t31-dssat-think-r2"
TRAIN_SEED=71
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
REQUIRE_THINK=True

WOFOST_DATASET_DIR="$SCRIPT_DIR/data/t31_cross_sim_wofost"
DSSAT_DATASET_DIR="$SCRIPT_DIR/data/t31_cross_sim_gym_dssat"
CYCLES_DATASET_DIR="$SCRIPT_DIR/data/t31_cross_sim_cycles_gym"

TRAIN_FILE="$SCRIPT_DIR/data/t31_cross_sim_gym_dssat/train.parquet"
WOFOST_VAL_FILE="$WOFOST_DATASET_DIR/val.parquet"
DSSAT_VAL_FILE="$DSSAT_DATASET_DIR/val.parquet"
CYCLES_VAL_FILE="$CYCLES_DATASET_DIR/val.parquet"

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
# Checkpoints go under $AGRI_WORK_DIR (default ~/agrimanager_work), outside the
# repository. Six runs retain two checkpoints each at 17.6 GB, about 211 GB.
RESULT_DIR="${AGRI_WORK_DIR:-$HOME/agrimanager_work}/agri_t31_results/t31_llm_train/${RUN_NAME}"

# The reference logs to this W&B project. Keep the replicates beside it.
# To log to your own W&B account, export WANDB_ENTITY=<your-entity>.

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
    # Point CycleGym at this account's own checkout. The environment chmods its
    # binary on startup, which raises PermissionError against another user's
    # copy; that failure already cost the T3.2 diff_models sweep a run.
    export CYCLES_GYM_CYCLESGYM_ROOT="${CYCLES_GYM_CYCLESGYM_ROOT:-${CYCLES_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/CyclesGym}/cyclesgym}"
    export CYCLES_GYM_CYCLES_ROOT="${CYCLES_GYM_CYCLES_ROOT:-${CYCLES_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/CyclesGym}/cycles}"
    export CYCLES_GYM_WORKING_DIR="/tmp/cycles-gym-${SLURM_JOB_ID:-$$}"
    mkdir -p "$CYCLES_GYM_WORKING_DIR"
}

mkdir -p "$LOG_DIR" "$RESULT_DIR"
cd "$PROJECT_ROOT"
prepare_dssat_runtime
prepare_cycles_runtime

for required in "$TRAIN_FILE" "$WOFOST_VAL_FILE" "$DSSAT_VAL_FILE" "$CYCLES_VAL_FILE"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing $required" >&2
        echo "Run experiments/t3_1_cross_simulator_maize_transfer/run_build_t31_datasets.sh first." >&2
        exit 1
    fi
done

VAL_SET_OVERRIDES=(
    "+data.val_sets.wofost=${WOFOST_VAL_FILE}"
    "+data.val_sets.dssat=${DSSAT_VAL_FILE}"
    "+data.val_sets.cycles=${CYCLES_VAL_FILE}"
)

exec bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name agri_grpo \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=null" \
    "data.seed=${TRAIN_SEED}" \
    "data.validation_axis=simulator" \
    "${VAL_SET_OVERRIDES[@]}" \
    "+data.env_config_overrides.require_think=${REQUIRE_THINK}" \
    "+data.env_config_overrides.thinking_mode=think" \
    "+data.env_config_overrides.think_tag=tool_call" \
    "data.max_prompt_length=2048" \
    "data.gen_batch_size=16" \
    "data.max_response_length=512" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.actor.data_loader_seed=${TRAIN_SEED}" \
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]" \
    "actor_rollout_ref.rollout.n=4" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64" \
    "trainer.logger=[\"console\", \"wandb\"]" \
    "trainer.project_name=agrimanager_cross_simulator_maize_transfer" \
    "trainer.default_local_dir=${RESULT_DIR}" \
    "trainer.experiment_name=${RUN_NAME}" \
    "trainer.total_epochs=2" \
    "trainer.total_training_steps=200" \
    "trainer.n_gpus_per_node=2" \
    "trainer.save_freq=100" \
    "trainer.max_actor_ckpt_to_keep=2" \
    "trainer.max_critic_ckpt_to_keep=2" \
    "ray_kwargs.ray_init.num_cpus=32" \
    "+ray_kwargs.ray_init.include_dashboard=False" \
    "+ray_kwargs.ray_init._temp_dir=/tmp/ray_${SLURM_JOB_ID:-${SLURM_JOBID:-$$}}"
