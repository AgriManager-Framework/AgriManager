#!/usr/bin/env bash
# Independent-model-family test on the T3.2 benchmark.
# Thinking arm.
#
# Meta-Llama-3.2-3B-Instruct, GRPO, on the same T3.2 unified prompt-conditioned
# policy protocol as the Qwen3-4B-Instruct paper reference (W&B 7rs3aczc). This
# pair answers whether the reasoning-as-stabilizer result holds outside the Qwen
# family, so it must stay comparable to that reference in every respect except
# the backbone.
#
# Both arms of this pair use TRAIN_SEED=142. That is the point: with one run per
# arm, giving the two arms different seeds would confound the think/no-think
# contrast with seed variation and make the comparison uninterpretable. An
# earlier launch used 142 for think and 242 for no-think and had exactly that
# defect. `require_think` is the only variable that differs between
# run_t32_unified_llama32_think_train.sh and
# run_t32_unified_llama32_no_think_train.sh.
#
# The reasoning format is pinned to think_tag=tool_call with
# max_response_length=512, matching the paper reference pair. Do not switch to
# the `think`@1024 format: the existing Qwen2.5 pair varies the tag and the
# token budget alongside require_think and is therefore not a clean paired
# comparison. This pair must not repeat that.
#
# The checked-in R1 script does not reproduce the R1 run: it says
# val_before_train=True and test_freq=10, but the reference run used False and 0.
# Where they disagree, this script follows the run.
#
# Protocol settings that differ from the stale R1 script, and why:
#   trainer.val_before_train=False, test_freq=0, log_val_generations=0
#       Matches the reference run, and is the replicate execution rule: expensive
#       LLM-RL runs do not run the in-training validation suite. Checkpoint
#       selection is terminal/last-valid, never best validation.
#   trainer.max_actor_ckpt_to_keep=4 (and critic)
#       save_freq=200 over 800 steps writes checkpoints at 200/400/600/800.
#       Keeping 4 retains all of them, which peak-to-final retention analysis
#       needs. An earlier launch lowered this to 1, which combined with the
#       disabled validation curve left no way to locate the peak at all.
#   actor.checkpoint.save_contents=[model,extra]  (default drops 'optimizer')
#       Optimizer state roughly doubles checkpoint size and is only needed to
#       resume training, which the terminal/last-valid selection rule never
#       does. Dropping it keeps four Llama-3.2-3B checkpoints near 52 GB against
#       a 155 GB instance disk. Convert the sharded model to Transformers layout
#       offline with `verl.model_merger`.
#   trainer.n_gpus_per_node=4, total_training_steps=800
#       Matches the accelerator count and step budget of the reference run.
#
# Dataset parquets are rebuilt from the committed configs on every launch. The
# dataset configs carry `generation_seed: 42`, so the rebuild is deterministic
# and every T3.2 run trains on identical data.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RUN_NAME="schema-t32-llama32-think-s1"
TRAIN_SEED=142
MODEL_PATH="meta-llama/Llama-3.2-3B-Instruct"
REQUIRE_THINK=True

WOFOST_CONFIG="$SCRIPT_DIR/config/t32_unified_wofost.yaml"
DSSAT_YIELD_CONFIG="$SCRIPT_DIR/config/t32_unified_gym_dssat_yield.yaml"
DSSAT_PROFIT_CONFIG="$SCRIPT_DIR/config/t32_unified_gym_dssat_profit.yaml"
CYCLES_YIELD_CONFIG="$SCRIPT_DIR/config/t32_unified_cycles_yield.yaml"
CYCLES_PROFIT_CONFIG="$SCRIPT_DIR/config/t32_unified_cycles_profit.yaml"
WOFOST_DATASET_DIR="$SCRIPT_DIR/data/$(basename "$WOFOST_CONFIG" .yaml)"
DSSAT_YIELD_DATASET_DIR="$SCRIPT_DIR/data/$(basename "$DSSAT_YIELD_CONFIG" .yaml)"
DSSAT_PROFIT_DATASET_DIR="$SCRIPT_DIR/data/$(basename "$DSSAT_PROFIT_CONFIG" .yaml)"
CYCLES_YIELD_DATASET_DIR="$SCRIPT_DIR/data/$(basename "$CYCLES_YIELD_CONFIG" .yaml)"
CYCLES_PROFIT_DATASET_DIR="$SCRIPT_DIR/data/$(basename "$CYCLES_PROFIT_CONFIG" .yaml)"

WOFOST_TRAIN_LNPKW_YIELD_FILE="$WOFOST_DATASET_DIR/train_wofost_lnpkw_yield.parquet"
WOFOST_TRAIN_LNPKW_PROFIT_FILE="$WOFOST_DATASET_DIR/train_wofost_lnpkw_profit.parquet"
WOFOST_TRAIN_LNPKW_WATER_FILE="$WOFOST_DATASET_DIR/train_wofost_lnpkw_water.parquet"
WOFOST_TRAIN_LNPK_PROFIT_FILE="$WOFOST_DATASET_DIR/train_wofost_lnpk_profit.parquet"
WOFOST_TRAIN_LNPK_NUTRIENT_FILE="$WOFOST_DATASET_DIR/train_wofost_lnpk_nutrient.parquet"
WOFOST_TRAIN_LNW_YIELD_FILE="$WOFOST_DATASET_DIR/train_wofost_lnw_yield.parquet"
DSSAT_YIELD_TRAIN_FILE="$DSSAT_YIELD_DATASET_DIR/train.parquet"
CYCLES_YIELD_TRAIN_FILE="$CYCLES_YIELD_DATASET_DIR/train.parquet"

WOFOST_VAL_LNPKW_PROFIT_FILE="$WOFOST_DATASET_DIR/val_wofost_lnpkw_profit.parquet"
WOFOST_VAL_LNPK_PROFIT_FILE="$WOFOST_DATASET_DIR/val_wofost_lnpk_profit.parquet"
WOFOST_VAL_LNW_YIELD_FILE="$WOFOST_DATASET_DIR/val_wofost_lnw_yield.parquet"
WOFOST_VAL_LNW_PROFIT_FILE="$WOFOST_DATASET_DIR/val_wofost_lnw_profit.parquet"
WOFOST_VAL_LNPKW_NUTRIENT_FILE="$WOFOST_DATASET_DIR/val_wofost_lnpkw_nutrient.parquet"
DSSAT_YIELD_VAL_FILE="$DSSAT_YIELD_DATASET_DIR/val.parquet"
DSSAT_PROFIT_VAL_FILE="$DSSAT_PROFIT_DATASET_DIR/val.parquet"
CYCLES_YIELD_VAL_FILE="$CYCLES_YIELD_DATASET_DIR/val.parquet"
CYCLES_PROFIT_VAL_FILE="$CYCLES_PROFIT_DATASET_DIR/val.parquet"

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

prepare_dssat_runtime() {
    prepare_pythonpath

    if [[ -f "$PROJECT_ROOT/smoke_tests/gym_dssat/_activate_spack.sh" ]]; then
        # shellcheck disable=SC1091
        source "$PROJECT_ROOT/smoke_tests/gym_dssat/_activate_spack.sh"
    fi

    prepare_pythonpath

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

join_hydra_list() {
    local first=1
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

build_dataset() {
    local config="$1"
    local dataset_id
    dataset_id="$(basename "$config" .yaml)"
    bash entrypoints/dataset/build.sh \
        --config "$config" \
        --num-workers 1 \
        2>&1 | tee "$LOG_DIR/${RUN_NAME}_${dataset_id}_build_dataset.log"
}

mkdir -p "$LOG_DIR" "$RESULT_DIR"
cd "$PROJECT_ROOT"
prepare_dssat_runtime

build_dataset "$WOFOST_CONFIG"
build_dataset "$DSSAT_YIELD_CONFIG"
build_dataset "$DSSAT_PROFIT_CONFIG"
build_dataset "$CYCLES_YIELD_CONFIG"
build_dataset "$CYCLES_PROFIT_CONFIG"

TRAIN_FILES="$(
    join_hydra_list \
        "$WOFOST_TRAIN_LNPKW_YIELD_FILE" \
        "$WOFOST_TRAIN_LNPKW_PROFIT_FILE" \
        "$WOFOST_TRAIN_LNPKW_WATER_FILE" \
        "$WOFOST_TRAIN_LNPK_PROFIT_FILE" \
        "$WOFOST_TRAIN_LNPK_NUTRIENT_FILE" \
        "$WOFOST_TRAIN_LNW_YIELD_FILE" \
        "$DSSAT_YIELD_TRAIN_FILE" \
        "$CYCLES_YIELD_TRAIN_FILE"
)"
VAL_SET_OVERRIDES=(
    "+data.val_sets.seen_wofost_lnpkw_profit=${WOFOST_VAL_LNPKW_PROFIT_FILE}"
    "+data.val_sets.seen_wofost_lnpk_profit=${WOFOST_VAL_LNPK_PROFIT_FILE}"
    "+data.val_sets.seen_wofost_lnw_yield=${WOFOST_VAL_LNW_YIELD_FILE}"
    "+data.val_sets.seen_dssat_yield=${DSSAT_YIELD_VAL_FILE}"
    "+data.val_sets.seen_cycles_yield=${CYCLES_YIELD_VAL_FILE}"
    "+data.val_sets.heldout_wofost_lnw_profit=${WOFOST_VAL_LNW_PROFIT_FILE}"
    "+data.val_sets.heldout_wofost_lnpkw_nutrient=${WOFOST_VAL_LNPKW_NUTRIENT_FILE}"
    "+data.val_sets.heldout_dssat_profit=${DSSAT_PROFIT_VAL_FILE}"
    "+data.val_sets.heldout_cycles_profit=${CYCLES_PROFIT_VAL_FILE}"
)

exec bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name agri_grpo \
    "+data.env_config_overrides.action_parser=legacy" \
    "data.train_files=${TRAIN_FILES}" \
    "data.val_files=null" \
    "data.seed=${TRAIN_SEED}" \
    "data.validation_axis=schema_tuple" \
    "${VAL_SET_OVERRIDES[@]}" \
    "+data.env_config_overrides.require_think=${REQUIRE_THINK}" \
    "+data.env_config_overrides.thinking_mode=think" \
    "+data.env_config_overrides.think_tag=tool_call" \
    "data.max_prompt_length=2048" \
    "data.gen_batch_size=16" \
    "data.dataloader_num_workers=8" \
    "data.max_response_length=512" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "actor_rollout_ref.actor.data_loader_seed=${TRAIN_SEED}" \
    "actor_rollout_ref.actor.checkpoint.save_contents=[model,extra]" \
    "actor_rollout_ref.rollout.n=4" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" \
    "actor_rollout_ref.rollout.agent.num_workers=16" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64" \
    "trainer.logger=[\"console\", \"wandb\"]" \
    "trainer.project_name=agrimanager_t32_unified_prompt_conditioned_policy" \
    "trainer.default_local_dir=${RESULT_DIR}" \
    "trainer.experiment_name=${RUN_NAME}" \
    "trainer.total_epochs=4" \
    "trainer.total_training_steps=800" \
    "trainer.val_before_train=False" \
    "trainer.test_freq=0" \
    "trainer.log_val_generations=0" \
    "trainer.n_gpus_per_node=4" \
    "trainer.save_freq=200" \
    "trainer.max_actor_ckpt_to_keep=4" \
    "trainer.max_critic_ckpt_to_keep=4" \
    "trainer.rollout_filter.enable=False" \
    "ray_kwargs.ray_init.num_cpus=32" \
    "+ray_kwargs.ray_init.include_dashboard=False" \
    "+ray_kwargs.ray_init._temp_dir=/tmp/ray_${SLURM_JOB_ID:-${SLURM_JOBID:-$$}}"
