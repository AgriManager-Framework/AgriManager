#!/usr/bin/env bash
# Build the numerical-policy variant of the frozen T2.2 action-menu test suite.
#
# This renders the same frozen 128-scenario maize base as
# `run_build_t22_eval_suite.sh` with `llm_mode: false`, so the target-trained
# PPO policies (E7) are scored on exactly the scenarios the LLM policies are
# scored on. The audit step asserts that pairing rather than assuming it.
#
# The maize source pool is not regenerated here. It is already frozen and the
# LLM suite has consumed it; regenerating it from this script would risk
# rewriting the pool underneath a suite that is already final. The audit checks
# this suite against the frozen pool, which catches any drift.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$SCRIPT_DIR/config/e4_t22_action_menu_nn.yaml"
DATASET_DIR="$SCRIPT_DIR/data/e4_t22_action_menu_nn"
LLM_DATASET_DIR="$SCRIPT_DIR/data/e4_t22_action_menu"
SOURCE_POOL="$SCRIPT_DIR/data/source_pools/maize_e4/test/maize.parquet"

cd "$PROJECT_ROOT"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ ! -f "$SOURCE_POOL" ]]; then
    echo "Missing frozen maize pool: $SOURCE_POOL" >&2
    echo "Run experiments/rebuttal_confirmatory_eval/run_freeze_maize_eval_pool.sh first." >&2
    exit 1
fi

if [[ ! -f "$LLM_DATASET_DIR/test_lnw.parquet" ]]; then
    echo "Missing the LLM T2.2 suite at $LLM_DATASET_DIR" >&2
    echo "Run experiments/rebuttal_confirmatory_eval/run_build_t22_eval_suite.sh first." >&2
    exit 1
fi

conda run -n agrimanager \
    bash entrypoints/dataset/build.sh \
    --config "$CONFIG" \
    --output-dir "$DATASET_DIR" \
    --num-workers 1

conda run -n agrimanager \
    python experiments/rebuttal_confirmatory_eval/normalize_eval_suite_paths.py --apply

conda run -n agrimanager \
    python experiments/rebuttal_confirmatory_eval/audit_t22_eval_suite.py \
    --dataset-dir "$DATASET_DIR" \
    --paired-with "$LLM_DATASET_DIR"
