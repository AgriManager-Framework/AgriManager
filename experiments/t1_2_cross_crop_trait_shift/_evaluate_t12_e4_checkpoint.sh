#!/usr/bin/env bash
# Internal helper for the fixed T1.2 E4 run definitions in this directory.
# The public run_*.sh wrappers hardcode every experiment-defining value.
#
# Scores one terminal T1.2 Think checkpoint on the frozen e4_t12_cross_crop
# suite: 512 seen-crop and 512 held-out-crop scenarios, disjoint from the
# training and validation data of the run being scored. The held-out split is
# where this setting's result lives, so both are always scored together.
set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "Usage: $0 RUN_NAME COVERAGE SEED WANDB_RUN_ID RUN_DEFINITION CELL_ID" >&2
    exit 2
fi

RUN_NAME="$1"
COVERAGE="$2"
TRAIN_SEED="$3"
WANDB_RUN_ID="$4"
RUN_DEFINITION="$5"
CELL_ID="$6"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CHECKPOINT_ROOT="${AGRI_WORK_DIR:-$HOME/agrimanager_work}/agri_t12_results/t12_llm_train"
ARTIFACT_ROOT="${AGRI_WORK_DIR:-$HOME/agrimanager_work}/agri_t12_results/t12_e4"
SOURCE_CHECKPOINT="$CHECKPOINT_ROOT/$RUN_NAME/global_step_200"
MERGE_WORK_DIR="$ARTIFACT_ROOT/merged"
MERGED_CHECKPOINT="$MERGE_WORK_DIR/${RUN_NAME}-gs200"
EVAL_WORK_DIR="$ARTIFACT_ROOT/evaluations/$RUN_NAME"
SUITE_DIR="$PROJECT_ROOT/experiments/rebuttal_confirmatory_eval/data"
DATASET_DIR="$SCRIPT_DIR/data/cross_crop_${COVERAGE}id_llm_traits_v1_23d_think"

if [[ ! -f "$SOURCE_CHECKPOINT/actor/fsdp_config.json" ]]; then
    echo "Missing terminal checkpoint: $SOURCE_CHECKPOINT" >&2
    exit 1
fi
LATEST="$(cat "$CHECKPOINT_ROOT/$RUN_NAME/latest_checkpointed_iteration.txt" 2>/dev/null || echo "")"
if [[ "$LATEST" != "200" ]]; then
    echo "$RUN_NAME reports latest checkpoint $LATEST, not 200; refusing to score a partial run." >&2
    exit 1
fi
# The training provenance recorded in the manifest must be the paper's split,
# not the 3200-row rebuild that briefly sat at this path.
for f in train.parquet val_id_crops.parquet val_heldout_crops.parquet; do
    if [[ ! -f "$DATASET_DIR/$f" ]]; then
        echo "Missing frozen training provenance: $DATASET_DIR/$f" >&2
        exit 1
    fi
done
ROWS="$(python -c "import pyarrow.parquet as pq,sys;print(pq.ParquetFile(sys.argv[1]).metadata.num_rows)" "$DATASET_DIR/train.parquet")"
if [[ "$ROWS" != "1600" ]]; then
    echo "Training provenance has $ROWS rows, not the 1600 the protocol declares." >&2
    exit 1
fi
if [[ ! -f "$SUITE_DIR/e4_t12_cross_crop/manifest.json" ]]; then
    echo "Missing audited E4 suite: $SUITE_DIR/e4_t12_cross_crop" >&2
    exit 1
fi

mkdir -p "$MERGE_WORK_DIR" "$EVAL_WORK_DIR"

# Merged checkpoints (about 8 GB each) go under $AGRI_WORK_DIR; on Lustre,
# AGRI_LUSTRE_POOL selects a pool for them.
if [[ -n "${AGRI_LUSTRE_POOL:-}" ]] && command -v lfs >/dev/null 2>&1; then
    for d in "$ARTIFACT_ROOT" "$MERGE_WORK_DIR" "$EVAL_WORK_DIR"; do
        lfs getstripe -d "$d" 2>/dev/null | grep -q "pool:[[:space:]]*$AGRI_LUSTRE_POOL" \
            || lfs setstripe --pool "$AGRI_LUSTRE_POOL" -c 4 "$d" 2>/dev/null || true
    done
fi

cd "$PROJECT_ROOT"

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
    --results-repo "local/schema-results-t12" \
    --tier t12 \
    --work-dir "$EVAL_WORK_DIR" \
    --no-upload \
    --keep-work-dir

echo "T1.2 E4 evaluation complete: $EVAL_WORK_DIR/bundle"
