#!/usr/bin/env bash
# Seed replicate R3, Family B.
#
# T2.2 action-menu shift, Family B (`lnpk` + `lw`), Qwen3-4B-Instruct, GRPO,
# thinking. Independent training replicate of the paper reference run
# (E1C-B-R1, `run_t22_family_b_llm_think_train.sh`, W&B pkgi7m60). Everything
# that defines the experiment is identical to the reference except the random
# seed.
#
# Seed protocol for the E1C replicate family:
#   R1 (paper reference) : data.seed unset (unseeded shuffle, not reproducible)
#                          actor.data_loader_seed = 42 (framework default)
#   R2                   : 71
#   R3 (this script)     : 72
# Both seed knobs are set together so one number identifies the replicate.
# `data.seed` fixes the training-data shuffle order; `actor.data_loader_seed`
# fixes minibatch order inside the actor update. They are distinct knobs.
#
# Family A and Family B of a given replicate index share their seed (R2 = 71 on
# both families, R3 = 72 on both). E1C reports the same-`lnw` Family A minus
# Family B difference, which is the submitted +0.34 versus -0.37 result. The two
# families must therefore differ only in their training action menus; giving
# them different seeds would confound that contrast with seed variation.
#
# Deliberate differences from the R1 script, and why:
#   trainer.total_training_steps=200
#       Made explicit. The reference left it unset and derived 200 steps from
#       total_epochs=2. Pinning it means an interrupted replicate is reported
#       against a declared target instead of an inferred one, which matters here
#       because the Family B reference crashed at 197/200.
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
#   Self-contained instead of sourcing `_t22_train_setup.sh`
#       The reference wrappers delegate to a shared helper that resolves the run
#       name, dataset paths and result directory at call time. A replicate run
#       definition has to be readable as one file, so everything is inlined.
#   Results under the experiment directory, not a per-user `/tmp` work dir
#       The helper places checkpoints and logs under `/tmp/<user>/...`. That
#       does not survive a rented instance and embeds a username, which the
#       artifact rules forbid. Both now sit beside the other experiments.
#
# Retained from the reference on purpose, against the general execution rule:
#   trainer.val_before_train=True, trainer.test_freq=10
#       The T3.2 runs disable in-training validation, but the T1/T2 references
#       enable it and their validation curve is what makes peak-to-final
#       retention computable here. A replicate that dropped the curve could not
#       be compared with R1 curve for curve. These five validation menus are the
#       training-time splits shipped with the dataset config, never E4 data, so
#       the "E4 must not touch training" rule is unaffected. Checkpoint
#       selection remains terminal/last-valid, never best-validation.
#   trainer.rollout_filter.enable=True, max_consecutive_all_filtered_steps=5
#       Made explicit rather than inherited. The reference ran with the filter
#       on; turning it off for the replicates would make them incomparable.
#   eager attention, use_remove_padding=False, explicit critic model path
#       Carried over verbatim from the reference. These change kernel selection
#       and padding handling, so dropping them would alter numerics.
#   trainer.n_gpus_per_node=2
#       Matches the reference. Batch sizes are global, so this affects sharding
#       and throughput only. Two of these runs fit on one 4-GPU host.
#
# Dataset parquets are rebuilt from the committed config on every launch. The
# dataset config carries a fixed generation seed, so the rebuild is
# deterministic and R1/R2/R3 train on identical data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="schema-t22-family-b-think-r3"
TRAIN_SEED=72
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"

DATASET_CONFIG="$SCRIPT_DIR/config/t22_family_b_think.yaml"
DATASET_ID="$(basename "$DATASET_CONFIG" .yaml)"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"

TRAIN_SPLITS=(lnpk lw)

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
RESULT_DIR="$SCRIPT_DIR/results/llm_train/${RUN_NAME}"

export WOFOST_GYM_PATH="${WOFOST_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/WOFOSTGym}"

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
    prepend_pythonpath "$WOFOST_GYM_PATH/pcse"
    prepend_pythonpath "$WOFOST_GYM_PATH/pcse_gym"
    prepend_pythonpath "$WOFOST_GYM_PATH"
}

join_hydra_list() {
    local first=1 item
    printf '['
    for item in "$@"; do
        if [[ "$first" -eq 0 ]]; then
            printf ','
        fi
        printf '%s' "$item"
        first=0
    done
    printf ']'
}

mkdir -p "$LOG_DIR" "$RESULT_DIR"
cd "$PROJECT_ROOT"
prepare_pythonpath

# The dataset is built once by run_build_t22_family_b_think_dataset.sh and shared by the
# R2 and R3 replicates. Rebuilding it here would let concurrent replicates read
# each other's half-written parquets, and would stop them being a seed comparison.
for required in "$DATASET_DIR"/train_*.parquet "$DATASET_DIR"/val_*.parquet; do
    if [[ ! -f "$required" ]]; then
        echo "Dataset $DATASET_DIR is missing or incomplete" >&2
        echo "Run experiments/t2_2_action_menu_shift/run_build_t22_family_b_think_dataset.sh first." >&2
        exit 1
    fi
done

TRAIN_FILES=()
for split in "${TRAIN_SPLITS[@]}"; do
    TRAIN_FILES+=("$DATASET_DIR/train_${split}.parquet")
done
TRAIN_FILES_ARG="$(join_hydra_list "${TRAIN_FILES[@]}")"

VAL_SET_OVERRIDES=(
    "+data.val_sets.lnpkw=${DATASET_DIR}/val_lnpkw.parquet"
    "+data.val_sets.lnpk=${DATASET_DIR}/val_lnpk.parquet"
    "+data.val_sets.lnw=${DATASET_DIR}/val_lnw.parquet"
    "+data.val_sets.ln=${DATASET_DIR}/val_ln.parquet"
    "+data.val_sets.lw=${DATASET_DIR}/val_lw.parquet"
)

exec bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name agri_grpo \
    "+data.env_config_overrides.action_parser=legacy" \
    "data.train_batch_size=1024" \
    "data.train_files=${TRAIN_FILES_ARG}" \
    "data.val_files=null" \
    "data.seed=${TRAIN_SEED}" \
    "data.validation_axis=action_menu" \
    "${VAL_SET_OVERRIDES[@]}" \
    "data.gen_batch_size=16" \
    "data.max_response_length=512" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "+actor_rollout_ref.model.override_config.attn_implementation=eager" \
    "actor_rollout_ref.model.use_remove_padding=False" \
    "critic.model.path=${MODEL_PATH}" \
    "+critic.model.override_config.attn_implementation=eager" \
    "critic.model.use_remove_padding=False" \
    "actor_rollout_ref.actor.data_loader_seed=${TRAIN_SEED}" \
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]" \
    "actor_rollout_ref.rollout.name=vllm" \
    "actor_rollout_ref.rollout.n=4" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64" \
    "trainer.logger=[\"console\", \"wandb\"]" \
    "trainer.project_name=agrimanager_wofost_action_menu_shift" \
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
