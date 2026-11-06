#!/usr/bin/env bash
# Internal helper for the fixed T1.3 seed replicates in this directory.
# The public run_*.sh wrappers hardcode every experiment-defining value.
#
# Usage: _run_t13_seed_replicate.sh ARM REPLICATE SEED
#   ARM        think | no_think
#   REPLICATE  r2 | r3
#   SEED       111 (R2) | 112 (R3)
#
# T1.3 price-regime shift, CyclesGym, Qwen3-4B-Instruct, GRPO. Independent
# training replicates of the paper reference runs, identical to them except the
# seed:
#
#   think     tct9g5zb   2026-05-03, 200 steps
#   no_think  a4wbykfc   2026-05-03, 132 steps (early stop)
#
# Both postdate the c5fc069 dataset restructure of 2026-05-02, so R1 pools
# legitimately with these replicates. The T1.3 configs also set none of
# `fert_amount`, `irrig_amount` or `scale_action_amounts_by_interval`, which is
# why `fa1bc68` did not touch them: the doubled-action-scale defect needed both
# keys and this environment has neither.
#
# The arm is chosen by the dataset config, as in T2.2 and unlike T2.3.
#
# Seed protocol for the E12 replicate family:
#   R1 (paper reference) : data.seed unset, actor.data_loader_seed = 42
#   R2 (this script)     : 111
#   R3                   : 112
# The think and no-think arms of a replicate share a seed, because the cell is
# reported as a paired think-minus-no-think effect.
#
# Deliberate differences from the reference, and why:
#   trainer.total_training_steps=200
#       Made explicit; the reference derived it from total_epochs=2. An
#       interrupted replicate is then reported against a declared target.
#   trainer.max_actor_ckpt_to_keep=2 (and critic)
#       The reference kept the default, which deletes the previous checkpoint
#       before writing the next and leaves a window with none complete.
#   actor.checkpoint.save_contents=[model,extra]
#       17.6 GB per checkpoint instead of 42.4 GB. Costs mid-run resume, which
#       this protocol never uses.
#   Results under $AGRI_WORK_DIR, outside the repository
#       Retained checkpoints are 17.6 GB each; put AGRI_WORK_DIR on a filesystem
#       with room for them (on Lustre, AGRI_LUSTRE_POOL selects a pool).
#   The dataset is not rebuilt here
#       The reference rebuilds on every launch. Four replicates sharing one
#       dataset directory would read each other's half-written parquets. Build
#       once with run_build_t13_datasets.sh.
#
# Everything else is carried verbatim, including the 2048-token prompt budget
# and 1024-token response budget, which are larger than the other tiers use.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 ARM REPLICATE SEED" >&2
    exit 2
fi

ARM="$1"; REPLICATE="$2"; TRAIN_SEED="$3"
case "$ARM" in think|no_think) ;; *) echo "bad ARM: $ARM" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="schema-t13-price-${ARM//_/-}-${REPLICATE}"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"

DATASET_ID="t1_3_cycles_gym_price_regime_llm_${ARM}"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"
TRAIN_FILE="$DATASET_DIR/train.parquet"
VAL_FILE="$DATASET_DIR/val.parquet"

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
RESULT_DIR="${AGRI_WORK_DIR:-$HOME/agrimanager_work}/agri_t13_results/t13_llm_train/${RUN_NAME}"

mkdir -p "$LOG_DIR" "$RESULT_DIR"

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

for required in "$TRAIN_FILE" "$VAL_FILE"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing $required" >&2
        echo "Run experiments/t1_3_cycles_gym_price_regime/run_build_t13_datasets.sh first." >&2
        exit 1
    fi
done

cd "$PROJECT_ROOT"

# CycleGym rewrites and chmods its own executable at startup, so use the
# project-owned runtime and a job-private working directory rather than another
# user's copy, which fails with PermissionError.
export CYCLES_GYM_CYCLESGYM_ROOT="${CYCLES_GYM_CYCLESGYM_ROOT:-${CYCLES_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/CyclesGym}/cyclesgym}"
export CYCLES_GYM_CYCLES_ROOT="${CYCLES_GYM_CYCLES_ROOT:-${CYCLES_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/CyclesGym}/cycles}"
export CYCLES_GYM_WORKING_DIR="/tmp/t13-cycles-${SLURM_JOB_ID:-$$}"
mkdir -p "$CYCLES_GYM_WORKING_DIR"

exec bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name agri_grpo \
    "+data.env_config_overrides.action_parser=legacy" \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=${VAL_FILE}" \
    "data.seed=${TRAIN_SEED}" \
    "data.validation_axis=price_regime" \
    "data.gen_batch_size=16" \
    "data.max_prompt_length=2048" \
    "data.max_response_length=1024" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.actor.data_loader_seed=${TRAIN_SEED}" \
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]" \
    "actor_rollout_ref.rollout.n=4" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64" \
    "trainer.logger=[\"console\", \"wandb\"]" \
    "trainer.project_name=agrimanager_t1_3_cycles_gym_price_regime" \
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
