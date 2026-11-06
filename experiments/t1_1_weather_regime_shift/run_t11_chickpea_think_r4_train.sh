#!/usr/bin/env bash
# Seed replicate R4, Think arm.
#
# T1.1 Chickpea weather-regime shift, Qwen3-4B-Instruct, GRPO, thinking.
# Independent training replicate of the paper reference run (E1-T-R1,
# `run_wofost_weather_regime_ood_chickpea_llm_think_train.sh`, W&B x8owyftg).
# Everything that defines the experiment is identical to the reference except
# the random seed.
#
# Seed protocol for the E1 replicate family:
#   R1 (paper reference) : data.seed unset (unseeded shuffle, not reproducible)
#                          actor.data_loader_seed = 42 (framework default)
#   R2                   : 51
#   R4 (this script)     : 52
# Both seed knobs are set together so one number identifies the replicate.
# `data.seed` fixes the training-data shuffle order; `actor.data_loader_seed`
# fixes minibatch order inside the actor update. They are distinct knobs.
#
# The Think and No-think arms of a given replicate index share their seed
# (R2 = 51 on both arms, R4 = 52 on both). E1 reports a paired Think-minus-
# No-think effect, so the two arms must differ only in the reasoning format;
# giving them different seeds would confound that contrast with seed variation.
#
# Note that the checked-in R1 script does not reproduce the R1 run: the script
# writes to W&B project `agrimanager_t1_1_weather_regime_shift`, but the actual
# reference run lives in `agrimanager_wofost_weather_regime_ood`. This script
# follows the run, so the replicates land beside the reference.
#
# Deliberate differences from the R1 script, and why:
#   trainer.total_training_steps=200
#       Made explicit. The reference left it unset and derived 200 steps from
#       total_epochs=2. Pinning it means an interrupted replicate is reported
#       against a declared target instead of an inferred one.
#   trainer.max_actor_ckpt_to_keep=2 (and critic)
#       save_freq=100 over 200 steps writes checkpoints at 100 and 200. The
#       reference kept 1, which deletes step 100 before writing step 200 and
#       leaves a window with no complete checkpoint on disk. Keeping 2 retains
#       both and never triggers the deletion branch.
#   actor.checkpoint.save_contents=[model,extra]  (default drops 'optimizer')
#       A full Qwen3-4B checkpoint measures 42.4 GB: 17.6 GB sharded model plus
#       24.8 GB optimizer state. Dropping the optimizer gives 17.6 GB per
#       checkpoint, so both fit alongside other runs on one rented instance.
#       The cost is that training cannot resume mid-run; that is already the
#       declared protocol, which selects the terminal or last-valid checkpoint
#       and never resumes into a different trajectory. Convert the sharded model
#       to Transformers layout offline with `verl.model_merger`.
#
# Retained from the reference on purpose, against the general execution rule:
#   trainer.val_before_train=True, trainer.test_freq=10
#       The T3.2 runs disable in-training validation, but the T1/T2 references
#       enable it and their validation curve is what makes peak-to-final
#       retention computable here. A replicate that dropped the curve could not
#       be compared with R1 curve for curve. These validation sets are the
#       training-time splits shipped with the dataset config, never E4 data, so
#       the "E4 must not touch training" rule is unaffected. Checkpoint
#       selection remains terminal/last-valid, never best-validation.
#   trainer.rollout_filter.enable=True, max_consecutive_all_filtered_steps=5
#       Made explicit rather than inherited. The reference ran with the filter
#       on, and E1B-N-R1 shows it can stop a run early (192/200). Turning it off
#       for the replicates would make them incomparable with R1.
#   trainer.n_gpus_per_node=2
#       Matches the reference. Batch sizes are global, so this affects sharding
#       and throughput only. Two of these runs fit on one 4-GPU host.
#
# Dataset parquets are rebuilt from the committed config on every launch. The
# dataset config carries a fixed generation seed, so the rebuild is
# deterministic and R1/R2/R4 train on identical data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="schema-t11-chickpea-think-r4"
TRAIN_SEED=53
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"

DATASET_CONFIG="$SCRIPT_DIR/config/weather_regime_chickpea_llm_without_traits_think.yaml"
DATASET_ID="$(basename "$DATASET_CONFIG" .yaml)"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"

TRAIN_FILE="$DATASET_DIR/train.parquet"
VAL_ID_FILE="$DATASET_DIR/val_id.parquet"
VAL_DROUGHT_FILE="$DATASET_DIR/val_drought.parquet"
VAL_WET_FILE="$DATASET_DIR/val_wet.parquet"
VAL_HOT_FILE="$DATASET_DIR/val_hot.parquet"
VAL_COLD_FILE="$DATASET_DIR/val_cold.parquet"

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
# Checkpoints go under $AGRI_WORK_DIR (default ~/agrimanager_work), outside the
# repository: each retained checkpoint is 17.6 GB.
RESULT_DIR="${AGRI_WORK_DIR:-$HOME/agrimanager_work}/agri_t11_results/t11_llm_train/${RUN_NAME}"

prepend_pythonpath() {
    local path="$1"
    if [[ -d "$path" ]]; then
        case ":${PYTHONPATH:-}:" in
            *":$path:"*) ;;
            *) export PYTHONPATH="$path${PYTHONPATH:+:$PYTHONPATH}" ;;
        esac
    fi
}

prepare_pythonpath() {
    prepend_pythonpath "$PROJECT_ROOT"
    prepend_pythonpath "$PROJECT_ROOT/verl"
}

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
cd "$PROJECT_ROOT"
prepare_pythonpath

# The dataset is built once by run_build_weather_regime_chickpea_llm_without_traits_think_dataset.sh and shared by the
# R2 and R4 replicates. Rebuilding it here would let concurrent replicates read
# each other's half-written parquets, and would stop them being a seed comparison.
for required in "$TRAIN_FILE" "$VAL_ID_FILE" "$VAL_DROUGHT_FILE" "$VAL_WET_FILE" "$VAL_HOT_FILE" "$VAL_COLD_FILE"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing $required" >&2
        echo "Run experiments/t1_1_weather_regime_shift/run_build_weather_regime_chickpea_llm_without_traits_think_dataset.sh first." >&2
        exit 1
    fi
done

VAL_SET_OVERRIDES=(
    "+data.val_sets.id=${VAL_ID_FILE}"
    "+data.val_sets.drought=${VAL_DROUGHT_FILE}"
    "+data.val_sets.wet=${VAL_WET_FILE}"
    "+data.val_sets.hot=${VAL_HOT_FILE}"
    "+data.val_sets.cold=${VAL_COLD_FILE}"
)

exec bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name agri_grpo \
    "+data.env_config_overrides.action_parser=legacy" \
    "data.train_batch_size=1024" \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=null" \
    "data.seed=${TRAIN_SEED}" \
    "data.validation_axis=weather_regime" \
    "${VAL_SET_OVERRIDES[@]}" \
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
    "trainer.project_name=agrimanager_wofost_weather_regime_ood" \
    "trainer.default_local_dir=${RESULT_DIR}" \
    "trainer.experiment_name=${RUN_NAME}" \
    "trainer.total_epochs=2" \
    "trainer.total_training_steps=200" \
    "trainer.val_before_train=True" \
    "trainer.test_freq=10" \
    "trainer.n_gpus_per_node=2" \
    "trainer.save_freq=100" \
    "trainer.max_actor_ckpt_to_keep=2" \
    "trainer.max_critic_ckpt_to_keep=2" \
    "trainer.rollout_filter.enable=True" \
    "trainer.rollout_filter.max_consecutive_all_filtered_steps=5" \
    "ray_kwargs.ray_init.num_cpus=32" \
    "+ray_kwargs.ray_init.include_dashboard=False" \
    "+ray_kwargs.ray_init._temp_dir=/tmp/ray_${SLURM_JOB_ID:-${SLURM_JOBID:-$$}}"
