#!/usr/bin/env bash
# Evaluate the T1.1 Chickpea numerical-PPO policies on the
# frozen weather-regime test suite.
#
# All five regime groups are scored in one pass per policy, so a policy sees the
# same scenarios in the same order regardless of which group it is being
# reported under, and the per-group split comes from `validation_set` labels
# rather than from separate invocations.
#
# The suite is disjoint from every T1.1 training and validation split; the
# audit in `experiments/rebuttal_confirmatory_eval/audit_t11_potato_eval_suite.py`
# checks that rather than asserting it. This is what separates these numbers
# from the submitted ones, which reported validation scores as final results.
#
# Checkpoint selection follows the declared rule: the terminal checkpoint, which
# is the `agent.zip` a completed run writes. Not the best-validation checkpoint,
# and never chosen by looking at E4 scores.
#
# Inference settings match the validation settings the runs used while training
# (`seed: 0`, `num_repeats: 1`, `deterministic: false`).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SUITE_DIR="$PROJECT_ROOT/experiments/rebuttal_confirmatory_eval/data/e4_t11_chickpea_weather_regime_nn"
TRAIN_ROOT="$SCRIPT_DIR/results/nn_train"
EVAL_ROOT="$SCRIPT_DIR/results/nn_eval_e4"
LOG_DIR="$SCRIPT_DIR/logs"

RUNS=(
    # Three seeds of the n1 cell, matching the potato script. s2 and s3 were
    # trained after this list was first written and were simply never added, so
    # the chickpea cell would have been scored at one seed while its potato
    # counterpart carried three.
    wofost_weather_regime_ood_chickpea_nn_train_8ep_n1
    wofost_weather_regime_ood_chickpea_nn_train_8ep_n1_s2
    wofost_weather_regime_ood_chickpea_nn_train_8ep_n1_s3
    wofost_weather_regime_ood_chickpea_nn_train_8ep_n8
)

cd "$PROJECT_ROOT"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"

mkdir -p "$LOG_DIR" "$EVAL_ROOT"

if [[ ! -f "$SUITE_DIR/audit.json" ]]; then
    echo "Missing audited T1.1 numerical test suite: $SUITE_DIR/audit.json" >&2
    echo "Run experiments/rebuttal_confirmatory_eval/run_build_t11_chickpea_eval_suite.sh first." >&2
    exit 1
fi

MISSING=0
for run_name in "${RUNS[@]}"; do
    if [[ ! -f "$TRAIN_ROOT/$run_name/agent.zip" ]]; then
        echo "Missing terminal checkpoint: $TRAIN_ROOT/$run_name/agent.zip" >&2
        MISSING=1
    fi
done
if [[ "$MISSING" -eq 1 ]]; then
    echo "Every T1.1 numerical run must finish before the frozen evaluation runs." >&2
    exit 1
fi

for run_name in "${RUNS[@]}"; do
    # Evaluation is deterministic at seed 0, so a policy that already has a
    # result would only produce the same numbers again. Skipping it makes the
    # script resumable: a run that dies partway, or one extended with new
    # policies, costs only the cells that are actually missing. Delete a result
    # directory to force that policy to be scored again.
    if [[ -f "$EVAL_ROOT/$run_name/metrics.json" ]]; then
        echo "=== ${run_name}: already evaluated, skipping ==="
        continue
    fi
    echo "=== ${run_name} on the frozen five-regime suite ==="
    bash entrypoints/eval/nn_eval.sh \
        --config-name nn \
        "data.inference_file=null" \
        "+data.inference_sets.id=${SUITE_DIR}/test_id.parquet" \
        "+data.inference_sets.drought=${SUITE_DIR}/test_drought.parquet" \
        "+data.inference_sets.wet=${SUITE_DIR}/test_wet.parquet" \
        "+data.inference_sets.hot=${SUITE_DIR}/test_hot.parquet" \
        "+data.inference_sets.cold=${SUITE_DIR}/test_cold.parquet" \
        "data.validation_axis=weather_regime" \
        "agent.path=${TRAIN_ROOT}/${run_name}/agent.zip" \
        "runtime.seed=0" \
        "runtime.num_repeats=1" \
        "runtime.deterministic=false" \
        "output.dir=${EVAL_ROOT}/${run_name}" \
        --log-file "${LOG_DIR}/eval_e4_${run_name}.log"
done

echo
echo "E5 T1.1 Chickpea numerical evaluation complete: ${#RUNS[@]} policies x 5 regimes"
echo "Results: $EVAL_ROOT"
