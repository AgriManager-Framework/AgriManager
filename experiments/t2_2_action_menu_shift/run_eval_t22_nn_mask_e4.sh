#!/usr/bin/env bash
# Evaluate the six E6 mask-aware policies on the frozen T2.2
# action-menu test suite.
#
# Unlike E7, every policy is scored on *all four* menus. That is the point: a
# mask-aware policy acts on one canonical 17-action vocabulary and reads a
# per-menu legal-action mask, so the same weights run on menus it never trained
# on. Family A trains on `lnpkw`+`lnpk` and Family B on `lnpk`+`lw`, so which
# cells are held-out transfer and which are seen differs by family and is
# recorded in the results rather than assumed here.
#
# The suite is used unmodified, but it is driven through the masked wrapper.
# `e4_t22_action_menu_nn` was frozen for E7's native-width policies and predates
# E6, so its env configs do not set `nn_action_space`. Building the environment
# without it would give an `ln` policy 5 native actions against a 17-action
# policy: three of the four menus would fail outright, and `lnpkw` would *not*
# fail because its native width also happens to be 17, silently feeding
# canonical action ids in as native ones. The failing three are exactly the
# held-out menus, so the partial success would be worse than a clean error.
#
# Injecting `nn_action_space=canonical_masked` restores the wrapper the training
# data declared. It adds no information: the mask is a function of `env_id`
# alone, and `env_id` is already in the frozen suite. `data.env_overrides`
# refuses scenario-identifying keys and re-checks scenario ids after applying,
# so "only the wrapper changed" is enforced by the loader rather than asserted
# in a comment.
#
# Checkpoint selection follows the declared rule: the terminal `agent.zip`. If a
# run did not finish, this script fails rather than falling back to a periodic
# checkpoint. Inference settings match the training-time validation settings
# (`seed: 0`, `num_repeats: 1`, `deterministic: false`).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SUITE_DIR="$PROJECT_ROOT/experiments/rebuttal_confirmatory_eval/data/e4_t22_action_menu_nn"
TRAIN_ROOT="$SCRIPT_DIR/results/nn_train"
EVAL_ROOT="$SCRIPT_DIR/results/nn_eval_e4"
LOG_DIR="$SCRIPT_DIR/logs"

FAMILIES=(a b)
SEEDS=(1 2 3)
MENUS=(lnw ln lw lnpkw)

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
for family in "${FAMILIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        agent="$TRAIN_ROOT/t22-nn-mask-family-${family}-s${seed}/agent.zip"
        if [[ ! -f "$agent" ]]; then
            echo "Missing terminal checkpoint: $agent" >&2
            MISSING=1
        fi
    done
done
if [[ "$MISSING" -eq 1 ]]; then
    echo "Every E6 run must finish before the frozen evaluation runs." >&2
    exit 1
fi

for family in "${FAMILIES[@]}"; do
    for seed in "${SEEDS[@]}"; do
        run_name="t22-nn-mask-family-${family}-s${seed}"
        for menu in "${MENUS[@]}"; do
            # Evaluation is deterministic at seed 0, so a policy that already has a
            # result would only produce the same numbers again. Skipping it makes the
            # script resumable: a run that dies partway, or one extended with new
            # policies, costs only the cells that are actually missing. Delete a result
            # directory to force that policy to be scored again.
            if [[ -f "$EVAL_ROOT/${run_name}__${menu}/metrics.json" ]]; then
                echo "=== ${run_name} on test_${menu}: already evaluated, skipping ==="
                continue
            fi
            echo "=== ${run_name} on frozen test_${menu} ==="
            bash entrypoints/eval/nn_eval.sh \
                --config-name nn \
                "data.inference_file=${SUITE_DIR}/test_${menu}.parquet" \
                "data.validation_axis=action_menu" \
                "+data.env_overrides.nn_action_space=canonical_masked" \
                "agent.path=${TRAIN_ROOT}/${run_name}/agent.zip" \
                "runtime.seed=0" \
                "runtime.num_repeats=1" \
                "runtime.deterministic=false" \
                "output.dir=${EVAL_ROOT}/${run_name}__${menu}" \
                --log-file "${LOG_DIR}/eval_e4_${run_name}__${menu}.log"
        done
    done
done

echo
echo "E5 T2.2 mask-aware evaluation complete: ${#FAMILIES[@]} families x ${#SEEDS[@]} seeds x ${#MENUS[@]} menus"
echo "Results: $EVAL_ROOT"
