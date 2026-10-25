#!/usr/bin/env bash
# Evaluate the T1.3 price-regime numerical policies on the
# frozen price-regime test suite.
#
# The suite is `e4_t13_price_regime_nn`, the LLM suite `e4_t13_price_regime`
# re-rendered with `llm_mode: false`. All 384 scenarios pair in order, verified
# on `scenario_id` and on the whole env_config apart from `dataset_id`,
# `llm_mode` and the reasoning fields.
#
# Getting that pairing needed one thing the WOFOST cells did not. CyclesGym
# seeds every scenario and every sampled crop price from a hash payload that
# includes the dataset id, so renaming the config alone produced 384 different
# scenarios at different prices with zero overlap in `scenario_id` -- a suite
# that looks like a pair and is not one. The NN config therefore sets
# `scenario_seed_id: e4_t13_price_regime`, which lets it keep its own identity
# while borrowing the LLM suite's draw.
#
# Checkpoint selection follows the declared rule: the terminal checkpoint, the
# `agent.zip` a completed run writes. Not the best-validation checkpoint, and
# never chosen by looking at E4 scores.
#
# Inference settings match the validation settings used during training
# (`seed: 0`, `num_repeats: 1`, `deterministic: false`).
#
# All three seeds are replicates here. T1.3 has no poolable paper reference --
# its configs were created after every T1.3 run on the board -- so s1 is a fresh
# run like s2 and s3 rather than the submitted one.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SUITE_DIR="$PROJECT_ROOT/experiments/rebuttal_confirmatory_eval/data/e4_t13_price_regime_nn"
TRAIN_ROOT="$SCRIPT_DIR/results/nn_train"
EVAL_ROOT="$SCRIPT_DIR/results/nn_eval_e4"
LOG_DIR="$SCRIPT_DIR/logs"

RUNS=(
    t1_3_cycles_gym_price_regime_nn_train_8ep_n1_s1
    t1_3_cycles_gym_price_regime_nn_train_8ep_n1_s2
    t1_3_cycles_gym_price_regime_nn_train_8ep_n1_s3
)

cd "$PROJECT_ROOT"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"

# CycleGym mutates and chmods its executable at startup, so it needs the
# project-owned runtime and a private working directory; a shared one fails with
# PermissionError against another user's binary.
export CYCLES_GYM_CYCLESGYM_ROOT="${CYCLES_GYM_CYCLESGYM_ROOT:-${CYCLES_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/CyclesGym}/cyclesgym}"
export CYCLES_GYM_CYCLES_ROOT="${CYCLES_GYM_CYCLES_ROOT:-${CYCLES_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/CyclesGym}/cycles}"
export CYCLES_GYM_WORKING_DIR="${CYCLES_GYM_WORKING_DIR:-/tmp/t13-nn-e4-cycles-${SLURM_JOB_ID:-$$}}"
mkdir -p "$CYCLES_GYM_WORKING_DIR"

mkdir -p "$LOG_DIR" "$EVAL_ROOT"

if [[ ! -f "$SUITE_DIR/manifest.json" ]]; then
    echo "Missing the numerical T1.3 test suite: $SUITE_DIR" >&2
    echo "Run experiments/rebuttal_confirmatory_eval/run_build_t13_nn_eval_suite.sh first." >&2
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
    echo "Every T1.3 numerical run must finish before the frozen evaluation runs." >&2
    exit 1
fi

for run_name in "${RUNS[@]}"; do
    # Deterministic at seed 0, so re-scoring a policy that already has a result
    # would only reproduce it. Skipping makes this resumable.
    if [[ -f "$EVAL_ROOT/$run_name/metrics.json" ]]; then
        echo "=== ${run_name}: already evaluated, skipping ==="
        continue
    fi
    echo "=== ${run_name} on the frozen price-regime suite ==="
    bash entrypoints/eval/nn_eval.sh \
        --config-name nn \
        "data.inference_file=null" \
        "+data.inference_sets.test=${SUITE_DIR}/test.parquet" \
        "data.validation_axis=price_regime" \
        "agent.path=${TRAIN_ROOT}/${run_name}/agent.zip" \
        "runtime.seed=0" \
        "runtime.num_repeats=1" \
        "runtime.deterministic=false" \
        "output.dir=${EVAL_ROOT}/${run_name}" \
        --log-file "${LOG_DIR}/eval_e4_${run_name}.log"
done

echo
echo "E5 T1.3 numerical evaluation complete: ${#RUNS[@]} policies"
echo "Results: $EVAL_ROOT"
