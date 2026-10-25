#!/usr/bin/env bash
# Evaluate the twelve E7 target-trained PPO policies on the
# frozen T2.2 action-menu test suite.
#
# Each policy is scored on its own menu only. A plain PPO policy has a fixed
# action-space dimension, so an `ln` policy (5 actions) cannot be run on
# `lnpkw` (17 actions) at all. That is the paper's point about fixed-interface
# neural policies under action-schema shift, and it is why this script has no
# cross-menu cells: the missing cells are structural, not omitted. E6's
# mask-aware policies are what fill them.
#
# Checkpoint selection follows the declared rule: the terminal checkpoint, which
# is the `agent.zip` the run writes when it completes. Not the best-validation
# checkpoint, and never chosen by looking at E4 scores. If a run did not finish,
# this script fails rather than silently falling back to a periodic checkpoint;
# substituting a last-valid checkpoint is a documented decision, not a default.
#
# Inference settings match the validation settings the runs used while training
# (`seed: 0`, `num_repeats: 1`, `deterministic: false`), so a test score is
# comparable to the training curve it continues.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SUITE_DIR="$PROJECT_ROOT/experiments/rebuttal_confirmatory_eval/data/e4_t22_action_menu_nn"
TRAIN_ROOT="$SCRIPT_DIR/results/nn_train"
EVAL_ROOT="$SCRIPT_DIR/results/nn_eval_e4"
LOG_DIR="$SCRIPT_DIR/logs"

MENUS=(lnw ln lw lnpkw)
SEEDS=(1 2 3)

cd "$PROJECT_ROOT"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"

mkdir -p "$LOG_DIR" "$EVAL_ROOT"

if [[ ! -f "$SUITE_DIR/audit.json" ]]; then
    echo "Missing audited T2.2 numerical test suite: $SUITE_DIR/audit.json" >&2
    echo "Run experiments/rebuttal_confirmatory_eval/run_build_t22_nn_eval_suite.sh first." >&2
    exit 1
fi

MISSING=0
for menu in "${MENUS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        agent="$TRAIN_ROOT/t22-nn-target-${menu}-s${seed}/agent.zip"
        if [[ ! -f "$agent" ]]; then
            echo "Missing terminal checkpoint: $agent" >&2
            MISSING=1
        fi
    done
done
if [[ "$MISSING" -eq 1 ]]; then
    echo "Every E7 run must finish before the frozen evaluation runs." >&2
    exit 1
fi

for menu in "${MENUS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_name="t22-nn-target-${menu}-s${seed}"
        # Evaluation is deterministic at seed 0, so a policy that already has a
        # result would only produce the same numbers again. Skipping it makes the
        # script resumable: a run that dies partway, or one extended with new
        # policies, costs only the cells that are actually missing. Delete a result
        # directory to force that policy to be scored again.
        if [[ -f "$EVAL_ROOT/$run_name/metrics.json" ]]; then
            echo "=== ${run_name}: already evaluated, skipping ==="
            continue
        fi
        echo "=== ${run_name} on frozen test_${menu} ==="
        bash entrypoints/eval/nn_eval.sh \
            --config-name nn \
            "data.inference_file=${SUITE_DIR}/test_${menu}.parquet" \
            "data.validation_axis=action_menu" \
            "agent.path=${TRAIN_ROOT}/${run_name}/agent.zip" \
            "runtime.seed=0" \
            "runtime.num_repeats=1" \
            "runtime.deterministic=false" \
            "output.dir=${EVAL_ROOT}/${run_name}" \
            --log-file "${LOG_DIR}/eval_e4_${run_name}.log"
    done
done

echo
echo "E5 T2.2 numerical evaluation complete: ${#MENUS[@]} menus x ${#SEEDS[@]} seeds"
echo "Results: $EVAL_ROOT"
