#!/usr/bin/env bash
# Seed replicate R2, Think arm.
#
# T2.1 observation-schema shift, Qwen3-4B-Instruct, GRPO, thinking.
# Independent training replicate of the paper reference run (E1B-T-R1,
# `run_t21_observation_schema_shift_llm_think_train.sh`, W&B evapvorx).
# Everything that defines the experiment is identical to the reference except
# the random seed.
#
# Seed protocol for the E1B replicate family:
#   R1 (paper reference) : data.seed unset (unseeded shuffle, not reproducible)
#                          actor.data_loader_seed = 42 (framework default)
#   R2 (this script)     : 61
#   R3                   : 62
# Both seed knobs are set together so one number identifies the replicate.
# `data.seed` fixes the training-data shuffle order; `actor.data_loader_seed`
# fixes minibatch order inside the actor update. They are distinct knobs.
#
# The Think and No-think arms of a given replicate index share their seed
# (R2 = 61 on both arms, R3 = 62 on both). E1B reports a paired Think-minus-
# No-think effect, so the two arms must differ only in the reasoning format;
# giving them different seeds would confound that contrast with seed variation.
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
#       both and never triggers the deletion branch. This matters more here than
#       elsewhere: the No-think reference stopped at 192/200, so under retention
#       1 its only surviving checkpoint is step 100.
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
#       be compared with R1 curve for curve. These eight validation sets are the
#       training-time schema splits shipped with the dataset config, never E4
#       data, so the "E4 must not touch training" rule is unaffected. Checkpoint
#       selection remains terminal/last-valid, never best-validation.
#   trainer.rollout_filter.enable=True, max_consecutive_all_filtered_steps=5
#       Made explicit rather than inherited. The reference ran with the filter
#       on, and it is what stopped E1B-N-R1 at 192/200. Turning it off for the
#       replicates would make them incomparable with R1, and would also suppress
#       the termination behaviour E1B is meant to report.
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

RUN_NAME="schema-t21-observation-think-r2"
TRAIN_SEED=61
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"

DATASET_CONFIG="$SCRIPT_DIR/config/t21_observation_schema_shift_llm_think.yaml"
DATASET_ID="$(basename "$DATASET_CONFIG" .yaml)"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"

TRAIN_FILE="$DATASET_DIR/train.parquet"
VAL_S1_FILE="$DATASET_DIR/val_s1.parquet"
VAL_S2A_FILE="$DATASET_DIR/val_s2a.parquet"
VAL_S2B_FILE="$DATASET_DIR/val_s2b.parquet"
VAL_S2C_FILE="$DATASET_DIR/val_s2c.parquet"
VAL_S2D_FILE="$DATASET_DIR/val_s2d.parquet"
VAL_S3_FILE="$DATASET_DIR/val_s3.parquet"
VAL_S4_FILE="$DATASET_DIR/val_s4.parquet"
VAL_S5_FILE="$DATASET_DIR/val_s5.parquet"

LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
RESULT_DIR="$SCRIPT_DIR/results/llm_train/${RUN_NAME}"

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
cd "$PROJECT_ROOT"
prepare_pythonpath

# The dataset is built once by run_build_t21_observation_schema_shift_llm_think_dataset.sh and shared by the
# R2 and R3 replicates. Rebuilding it here would let concurrent replicates read
# each other's half-written parquets, and would stop them being a seed comparison.
for required in "$TRAIN_FILE" "$VAL_S1_FILE" "$VAL_S2A_FILE" "$VAL_S2B_FILE" "$VAL_S2C_FILE" "$VAL_S2D_FILE" "$VAL_S3_FILE" "$VAL_S4_FILE" "$VAL_S5_FILE"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing $required" >&2
        echo "Run experiments/t2_1_observation_schema_shift/run_build_t21_observation_schema_shift_llm_think_dataset.sh first." >&2
        exit 1
    fi
done

VAL_SET_OVERRIDES=(
    "+data.val_sets.s1_full_current=${VAL_S1_FILE}"
    "+data.val_sets.s2a_no_stage_time=${VAL_S2A_FILE}"
    "+data.val_sets.s2b_no_resource_state=${VAL_S2B_FILE}"
    "+data.val_sets.s2c_no_management_history=${VAL_S2C_FILE}"
    "+data.val_sets.s2d_no_weather_context=${VAL_S2D_FILE}"
    "+data.val_sets.s3_domain_synonym_rename=${VAL_S3_FILE}"
    "+data.val_sets.s4_compact_growth_superset=${VAL_S4_FILE}"
    "+data.val_sets.s5_anonymous_label_rename=${VAL_S5_FILE}"
)

exec bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name agri_grpo \
    "+data.env_config_overrides.action_parser=legacy" \
    "data.train_batch_size=1024" \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=null" \
    "data.seed=${TRAIN_SEED}" \
    "data.validation_axis=observation_schema" \
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
    "trainer.project_name=agrimanager_wofost_observation_schema_shift" \
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
