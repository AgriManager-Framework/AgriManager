#!/usr/bin/env bash
# Exploratory T3.2 backbone and optimizer sweep (one training seed per run).
#
# The appendix compares think and no-think pairs on the T3.2 validation set for
# other backbones and optimizers (Qwen3-4B-Instruct-2507 under DrGRPO and PPO,
# Qwen2.5-3B-Instruct under GRPO) and reports a Qwen3-8B pair. Those runs are
# exploratory: one seed per arm, validation set or a single test-suite pass.
# This script is the launcher for that sweep; the confirmatory Qwen3-4B runs use
# run_t32_unified_qwen3_{think,no_think}_r{2,3,4}_train.sh and the Llama-3.2
# pair uses run_t32_unified_llama32_{think,no_think}_train.sh.
#
# Usage:
#   bash run_t32_backbone_sweep_train.sh --algo grpo|drgrpo|ppo \
#       --think think|tool_call|no_think --model <hf-model-id> [--suffix NAME]
#
# The sweep as published (Hugging Face repositories t32-diffmodels-*; values read
# from each checkpoint's run_manifest.json: GPUs = FSDP world size, save = save_freq):
#   repository                  --algo  --think    --model                      GPUs save
#   qwen3-4b-tool-drgrpo        drgrpo  tool_call  Qwen/Qwen3-4B-Instruct-2507  4    25
#   qwen3-4b-no-think-drgrpo    drgrpo  no_think   Qwen/Qwen3-4B-Instruct-2507  4    25
#   qwen3-4b-tool-ppo           ppo     tool_call  Qwen/Qwen3-4B-Instruct-2507  4    25
#   qwen3-4b-no-think-ppo       ppo     no_think   Qwen/Qwen3-4B-Instruct-2507  4    25
#   qwen25-3b-think-grpo        grpo    think      Qwen/Qwen2.5-3B-Instruct     2    25  (MAX_RESPONSE_LENGTH=1024)
#   qwen25-3b-no-think-grpo     grpo    no_think   Qwen/Qwen2.5-3B-Instruct     2    25
#   qwen3-8b-think-grpo         grpo    think      Qwen/Qwen3-8B                4    200
#   qwen3-0p6b-think-grpo       grpo    think      Qwen/Qwen3-0.6B              2    25  (not reported in the paper)
# The Qwen3-8B no-think arm (grpo, no_think) stopped at step 493 and was not
# published. Every run trained 800 steps (4 epochs of the 3,200 training
# scenarios at 16 prompts per step) with the rollout filter off and data-loader
# seed 42 (the Qwen3-0.6B run: 0). data.seed was not set for these runs, so their
# data order is not reproducible; set TRAIN_SEED to fix it for new runs. Set
# N_GPUS and SAVE_FREQ per row to match a published run.
#
# Caveat for comparisons: --think think uses the <think> tag, while the paper's
# confirmatory Qwen3-4B runs and the Llama-3.2 pair use --think tool_call (the
# <tool_call> tag, 512 response tokens). The Qwen2.5-3B think arm was run with
# <think> and 1024 response tokens (see run_t32_unified_llama32_think_train.sh),
# so that pair differs in more than the response format.
#
# Algorithms: grpo is the paper default (agri_grpo.yaml). drgrpo is GRPO without
# per-group std normalization and with sequence-mean token-sum loss aggregation.
# ppo uses agri_ppo.yaml (critic) with 8 rollouts per prompt.
#
# Environment overrides: N_GPUS, TOTAL_EPOCHS, TOTAL_TRAINING_STEPS, SAVE_FREQ,
# TRAIN_SEED, LOADER_SEED, MAX_RESPONSE_LENGTH, GPU_MEMORY_UTILIZATION, TEST_FREQ,
# T32_ENABLE_VALIDATION=0, WANDB_MODE.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ALGO="grpo"
THINK="tool_call"
MODEL_PATH="Qwen/Qwen3-4B-Instruct-2507"
SUFFIX=""
N_GPUS="${N_GPUS:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-4}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-800}"
SAVE_FREQ="${SAVE_FREQ:-25}"
TEST_FREQ="${TEST_FREQ:-10}"
TRAIN_SEED="${TRAIN_SEED:-}"
LOADER_SEED="${LOADER_SEED:-42}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-16}"
ROLLOUT_N="${ROLLOUT_N:-4}"
PPO_ROLLOUT_N="${PPO_ROLLOUT_N:-8}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.4}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-4096}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-4}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
ACTOR_MINI_BATCH_SIZE="${ACTOR_MINI_BATCH_SIZE:-128}"
AGENT_WORKERS="${AGENT_WORKERS:-4}"
T32_ENABLE_VALIDATION="${T32_ENABLE_VALIDATION:-1}"

usage() { sed -n '/^# Usage:/,/^#$/p' "$0"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --algo) ALGO="$2"; shift 2 ;;
        --think) THINK="$2"; shift 2 ;;
        --model) MODEL_PATH="$2"; shift 2 ;;
        --suffix) SUFFIX="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done
case "$ALGO" in grpo|drgrpo|ppo) ;; *) echo "--algo must be grpo, drgrpo or ppo" >&2; exit 2 ;; esac
case "$THINK" in think|tool_call|no_think) ;; *) echo "--think must be think, tool_call or no_think" >&2; exit 2 ;; esac

model_slug="$(printf '%s' "$MODEL_PATH" | tr '/:.' '---' | tr -cs 'A-Za-z0-9_-' '-')"
RUN_NAME="t32_backbone_sweep_${SUFFIX:-${ALGO}_${THINK}_${model_slug}}"

CONFIG_NAME="agri_grpo"
EXTRA_OVERRIDES=()
if [[ "$ALGO" == "ppo" ]]; then
    CONFIG_NAME="agri_ppo"
    ROLLOUT_N="$PPO_ROLLOUT_N"
elif [[ "$ALGO" == "drgrpo" ]]; then
    EXTRA_OVERRIDES+=("algorithm.adv_estimator=grpo"
                      "algorithm.norm_adv_by_std_in_grpo=False"
                      "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum")
fi

REQUIRE_THINK=False
THINK_TAG="tool_call"
[[ "$THINK" != "no_think" ]] && REQUIRE_THINK=True
[[ "$THINK" == "think" ]] && THINK_TAG="think"

SEED_OVERRIDES=("actor_rollout_ref.actor.data_loader_seed=${LOADER_SEED}")
[[ -n "$TRAIN_SEED" ]] && SEED_OVERRIDES+=("data.seed=${TRAIN_SEED}")

WOFOST_CONFIG="$SCRIPT_DIR/config/t32_unified_wofost.yaml"
DSSAT_YIELD_CONFIG="$SCRIPT_DIR/config/t32_unified_gym_dssat_yield.yaml"
DSSAT_PROFIT_CONFIG="$SCRIPT_DIR/config/t32_unified_gym_dssat_profit.yaml"
CYCLES_YIELD_CONFIG="$SCRIPT_DIR/config/t32_unified_cycles_yield.yaml"
CYCLES_PROFIT_CONFIG="$SCRIPT_DIR/config/t32_unified_cycles_profit.yaml"
data_dir() { printf '%s\n' "$SCRIPT_DIR/data/$(basename "$1" .yaml)"; }
W="$(data_dir "$WOFOST_CONFIG")"
DY="$(data_dir "$DSSAT_YIELD_CONFIG")"
DP="$(data_dir "$DSSAT_PROFIT_CONFIG")"
CY="$(data_dir "$CYCLES_YIELD_CONFIG")"
CP="$(data_dir "$CYCLES_PROFIT_CONFIG")"

TRAIN_FILES="[$W/train_wofost_lnpkw_yield.parquet,$W/train_wofost_lnpkw_profit.parquet,$W/train_wofost_lnpkw_water.parquet,$W/train_wofost_lnpk_profit.parquet,$W/train_wofost_lnpk_nutrient.parquet,$W/train_wofost_lnw_yield.parquet,$DY/train.parquet,$CY/train.parquet]"
VAL_OVERRIDES=(
    "+data.val_sets.seen_wofost_lnpkw_profit=$W/val_wofost_lnpkw_profit.parquet"
    "+data.val_sets.seen_wofost_lnpk_profit=$W/val_wofost_lnpk_profit.parquet"
    "+data.val_sets.seen_wofost_lnw_yield=$W/val_wofost_lnw_yield.parquet"
    "+data.val_sets.seen_dssat_yield=$DY/val.parquet"
    "+data.val_sets.seen_cycles_yield=$CY/val.parquet"
    "+data.val_sets.heldout_wofost_lnw_profit=$W/val_wofost_lnw_profit.parquet"
    "+data.val_sets.heldout_wofost_lnpkw_nutrient=$W/val_wofost_lnpkw_nutrient.parquet"
    "+data.val_sets.heldout_dssat_profit=$DP/val.parquet"
    "+data.val_sets.heldout_cycles_profit=$CP/val.parquet"
)
VAL_BEFORE_TRAIN=False
if [[ "$T32_ENABLE_VALIDATION" == "0" || "$T32_ENABLE_VALIDATION" == "false" || "$T32_ENABLE_VALIDATION" == "False" ]]; then
    VAL_OVERRIDES=()
    TEST_FREQ=-1
fi

prepend_pythonpath() {
    [[ -d "$1" ]] || return 0
    case ":${PYTHONPATH:-}:" in *":$1:"*) ;; *) export PYTHONPATH="$1${PYTHONPATH:+:$PYTHONPATH}" ;; esac
}

LOG_DIR="$SCRIPT_DIR/logs"
RESULT_DIR="$SCRIPT_DIR/results/llm_train/${RUN_NAME}"
LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
mkdir -p "$LOG_DIR" "$RESULT_DIR"
cd "$PROJECT_ROOT"

# DSSAT runtime (see docs/getting-started/install_dssat.md), as in the other T3.2 launchers.
prepend_pythonpath "$PROJECT_ROOT"
prepend_pythonpath "$PROJECT_ROOT/verl"
if [[ -f "$PROJECT_ROOT/smoke_tests/gym_dssat/_activate_spack.sh" ]]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/smoke_tests/gym_dssat/_activate_spack.sh"
fi
prepend_pythonpath "$PROJECT_ROOT"
prepend_pythonpath "$PROJECT_ROOT/verl"
for candidate in "$PROJECT_ROOT/AgriManagerExternal/gym_dssat_pdi/gym-dssat-pdi" \
                 "$PROJECT_ROOT/../AgriManagerExternal/gym_dssat_pdi/gym-dssat-pdi"; do
    if [[ -d "$candidate/gym_dssat_pdi" ]]; then prepend_pythonpath "$candidate"; break; fi
done

# Datasets are rebuilt from the committed configs (generation_seed 42, deterministic).
for config in "$WOFOST_CONFIG" "$DSSAT_YIELD_CONFIG" "$DSSAT_PROFIT_CONFIG" "$CYCLES_YIELD_CONFIG" "$CYCLES_PROFIT_CONFIG"; do
    bash entrypoints/dataset/build.sh --config "$config" --num-workers 1 \
        2>&1 | tee "$LOG_DIR/${RUN_NAME}_$(basename "$config" .yaml)_build_dataset.log"
done

echo "T3.2 backbone sweep run: ${RUN_NAME} (model ${MODEL_PATH}, algo ${ALGO}, think ${THINK})"

exec bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name "$CONFIG_NAME" \
    "+data.env_config_overrides.action_parser=legacy" \
    "data.train_files=${TRAIN_FILES}" \
    "data.val_files=null" \
    "data.validation_axis=schema_tuple" \
    "${VAL_OVERRIDES[@]}" \
    "${SEED_OVERRIDES[@]}" \
    "+data.env_config_overrides.require_think=${REQUIRE_THINK}" \
    "+data.env_config_overrides.thinking_mode=think" \
    "+data.env_config_overrides.think_tag=${THINK_TAG}" \
    "data.max_prompt_length=2048" \
    "data.max_response_length=${MAX_RESPONSE_LENGTH}" \
    "data.gen_batch_size=${GEN_BATCH_SIZE}" \
    "data.dataloader_num_workers=4" \
    "actor_rollout_ref.model.path=${MODEL_PATH}" \
    "+actor_rollout_ref.model.override_config.attn_implementation=eager" \
    "actor_rollout_ref.model.use_remove_padding=False" \
    "actor_rollout_ref.rollout.n=${ROLLOUT_N}" \
    "actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}" \
    "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}" \
    "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}" \
    "actor_rollout_ref.rollout.agent.num_workers=${AGENT_WORKERS}" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=${ACTOR_MINI_BATCH_SIZE}" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1" \
    "+critic.model.override_config.attn_implementation=eager" \
    "critic.model.use_remove_padding=False" \
    "critic.ppo_micro_batch_size_per_gpu=1" \
    "trainer.logger=[\"console\",\"wandb\"]" \
    "trainer.project_name=agrimanager_t32_unified_prompt_conditioned_policy" \
    "trainer.experiment_name=${RUN_NAME}" \
    "trainer.default_local_dir=${RESULT_DIR}" \
    "trainer.total_epochs=${TOTAL_EPOCHS}" \
    "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}" \
    "trainer.val_before_train=${VAL_BEFORE_TRAIN}" \
    "trainer.test_freq=${TEST_FREQ}" \
    "trainer.log_val_generations=4" \
    "trainer.n_gpus_per_node=${N_GPUS}" \
    "trainer.save_freq=${SAVE_FREQ}" \
    "trainer.rollout_filter.enable=False" \
    "+ray_kwargs.ray_init.include_dashboard=False" \
    "${EXTRA_OVERRIDES[@]}"
