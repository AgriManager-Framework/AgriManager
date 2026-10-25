#!/usr/bin/env bash
# Internal helper for the fixed T3.1 E4 run definitions in this directory.
# The public run_*.sh wrappers hardcode every experiment-defining value.
set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "Usage: $0 RUN_NAME SOURCE SEED WANDB_RUN_ID RUN_DEFINITION CELL_ID" >&2
    exit 2
fi

RUN_NAME="$1"
SOURCE="$2"
TRAIN_SEED="$3"
WANDB_RUN_ID="$4"
RUN_DEFINITION="$5"
CELL_ID="$6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKPOINT_ROOT="${AGRI_WORK_DIR:-$HOME/agrimanager_work}/agri_t31_results/t31_llm_train"
ARTIFACT_ROOT="${AGRI_WORK_DIR:-$HOME/agrimanager_work}/agri_t31_results/t31_e4"
SOURCE_CHECKPOINT="$CHECKPOINT_ROOT/$RUN_NAME/global_step_200"
MERGE_WORK_DIR="$ARTIFACT_ROOT/merged"
MERGED_CHECKPOINT="$MERGE_WORK_DIR/${RUN_NAME}-gs200"
EVAL_WORK_DIR="$ARTIFACT_ROOT/evaluations/$RUN_NAME"
SUITE_DIR="$PROJECT_ROOT/experiments/rebuttal_confirmatory_eval/data"
DATASET_DIR="$SCRIPT_DIR/data/t31_cross_sim_${SOURCE}"

if [[ ! -f "$SOURCE_CHECKPOINT/actor/fsdp_config.json" ]]; then
    echo "Missing terminal checkpoint: $SOURCE_CHECKPOINT" >&2
    exit 1
fi
if [[ ! -f "$DATASET_DIR/train.parquet" || ! -f "$DATASET_DIR/val.parquet" ]]; then
    echo "Missing frozen training provenance under $DATASET_DIR" >&2
    exit 1
fi
for group in \
    e4_t31_cross_sim_wofost \
    e4_t31_cross_sim_dssat \
    e4_t31_cross_sim_cycles
do
    if [[ ! -f "$SUITE_DIR/$group/manifest.json" ]]; then
        echo "Missing audited E4 suite: $SUITE_DIR/$group" >&2
        exit 1
    fi
done

mkdir -p "$MERGE_WORK_DIR" "$EVAL_WORK_DIR"
cd "$PROJECT_ROOT"

# Source the same DSSAT runtime resolver used by the completed training runs.
# shellcheck disable=SC1091
source "$PROJECT_ROOT/smoke_tests/gym_dssat/_activate_spack.sh"

# CycleGym mutates/chmods its executable at startup, so use the project-owned
# runtime and a job-private working directory.
export CYCLES_GYM_CYCLESGYM_ROOT="${CYCLES_GYM_CYCLESGYM_ROOT:-${CYCLES_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/CyclesGym}/cyclesgym}"
export CYCLES_GYM_CYCLES_ROOT="${CYCLES_GYM_CYCLES_ROOT:-${CYCLES_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/CyclesGym}/cycles}"
export CYCLES_GYM_WORKING_DIR="/tmp/t31-e4-cycles-${SLURM_JOB_ID:-$$}"
mkdir -p "$CYCLES_GYM_WORKING_DIR"

bash entrypoints/tools/publish_checkpoint.sh \
    --checkpoint-dir "$SOURCE_CHECKPOINT" \
    --repo-id "local/$RUN_NAME" \
    --run-name "$RUN_NAME" \
    --experiment-id "$CELL_ID" \
    --seed "$TRAIN_SEED" \
    --base-model "Qwen/Qwen3-4B-Instruct-2507" \
    --checkpoint-step 200 \
    --total-training-steps 200 \
    --save-freq 100 \
    --require-think True \
    --thinking-mode think \
    --think-tag tool_call \
    --run-definition "$RUN_DEFINITION" \
    --wandb-run-id "$WANDB_RUN_ID" \
    --dataset-dir "$DATASET_DIR" \
    --revision step-200 \
    --work-dir "$MERGE_WORK_DIR" \
    --no-upload \
    --no-verify-load

bash entrypoints/eval/e5_evaluate_checkpoint.sh \
    --checkpoint-repo "local/$RUN_NAME" \
    --revision step-200 \
    --checkpoint-dir "$MERGED_CHECKPOINT" \
    --suite-repo "local/schema-shift-confirmatory-eval-v1" \
    --suite-dir "$SUITE_DIR" \
    --results-repo "local/schema-results-t31" \
    --tier t31 \
    --work-dir "$EVAL_WORK_DIR" \
    --no-upload \
    --keep-work-dir

echo "T3.1 E4 evaluation complete: $EVAL_WORK_DIR/bundle"
