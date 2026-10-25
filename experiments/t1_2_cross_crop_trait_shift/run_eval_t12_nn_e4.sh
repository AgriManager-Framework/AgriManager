#!/usr/bin/env bash
# Evaluate the T1.2 cross-crop numerical policies on the
# frozen cross-crop test suite.
#
# The suite is `e4_t12_cross_crop_nn`, which is the LLM suite `e4_t12_cross_crop`
# re-rendered with `llm_mode: false`. Same frozen source pool, same generation
# seed, same 512-scenario groups -- verified scenario for scenario, all 1024
# rows, differing only in `dataset_id`, `llm_mode` and the reasoning fields.
# That identity is the point: it makes the LLM-versus-numerical comparison in
# this cell a paired contrast on the same scenarios rather than two samples from
# the same distribution.
#
# Checkpoint selection follows the declared rule: the terminal checkpoint, which
# is the `agent.zip` a completed run writes. Not the best-validation checkpoint,
# and never chosen by looking at E4 scores.
#
# Inference settings match the validation settings used during training
# (`seed: 0`, `num_repeats: 1`, `deterministic: false`).
#
# Which runs, and why not the others. Each coverage has two NN cells, `n1` and
# `n8`, and the figure package builds the T1.2 NN curves from
# `*_nn_train_8ep_n1` and `*_nn_train_8ep_n8`. The `16ep_matched`,
# `without_traits`, `train1600` and bare `_nn_train` directories in the same
# tree are different protocols -- different row counts, different validation
# sets, and in the matched case resumed rather than trained from scratch, with
# all seeds at `agent.seed=1`. Scoring them here would answer a question the
# paper does not ask, so the list below is explicit rather than a glob.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

SUITE_DIR="$PROJECT_ROOT/experiments/rebuttal_confirmatory_eval/data/e4_t12_cross_crop_nn"
TRAIN_ROOT="$SCRIPT_DIR/results/nn_train"
EVAL_ROOT="$SCRIPT_DIR/results/nn_eval_e4"
LOG_DIR="$SCRIPT_DIR/logs"

# Six cells at three seeds. s1 is the paper reference run and keeps its original
# name; s2 and s3 are the replicates, which differ from it on `agent.seed` alone.
RUNS=(
    wofost_generalization_cross_crop_4id_nn_train_8ep_n1
    schema-t12-crosscrop-4id-nn-n1-s2
    schema-t12-crosscrop-4id-nn-n1-s3
    wofost_generalization_cross_crop_4id_nn_train_8ep_n8
    schema-t12-crosscrop-4id-nn-n8-s2
    schema-t12-crosscrop-4id-nn-n8-s3
    wofost_generalization_cross_crop_8id_nn_train_8ep_n1
    schema-t12-crosscrop-8id-nn-n1-s2
    schema-t12-crosscrop-8id-nn-n1-s3
    wofost_generalization_cross_crop_8id_nn_train_8ep_n8
    schema-t12-crosscrop-8id-nn-n8-s2
    schema-t12-crosscrop-8id-nn-n8-s3
    wofost_generalization_cross_crop_16id_nn_train_8ep_n1
    schema-t12-crosscrop-16id-nn-n1-s2
    schema-t12-crosscrop-16id-nn-n1-s3
    wofost_generalization_cross_crop_16id_nn_train_8ep_n8
    schema-t12-crosscrop-16id-nn-n8-s2
    schema-t12-crosscrop-16id-nn-n8-s3
)

# Reading the results: aggregate by crop, and do not use the `all` group.
#
# This evaluator groups output by `validation_axis` value, so the metrics come
# out per crop -- `val-env-crop/rice/...` -- while the LLM side of this cell
# reports one record per suite file, `test_id_crops` and `test_heldout_crops`.
# The two are reconcilable and the numerical output is the finer of the two: the
# splits are crop-disjoint, so each split's mean is the mean of its four crop
# means, every crop carrying 128 scenarios.
#
#   in-distribution : cotton, millet, rapeseed, rice
#   held out        : barley, groundnut, pigeonpea, seed_onion
#
# `val-env-crop/all/...` pools all 1024 scenarios across both splits. For a cell
# whose entire claim is the in-distribution versus held-out gap, that number
# averages the two sides of the comparison together, and it looks like a
# perfectly ordinary aggregate while doing it.

cd "$PROJECT_ROOT"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"

mkdir -p "$LOG_DIR" "$EVAL_ROOT"

if [[ ! -f "$SUITE_DIR/manifest.json" ]]; then
    echo "Missing the numerical T1.2 test suite: $SUITE_DIR" >&2
    echo "Run experiments/rebuttal_confirmatory_eval/run_build_t12_nn_eval_suite.sh first." >&2
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
    echo "Every T1.2 numerical run must finish before the frozen evaluation runs." >&2
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
    echo "=== ${run_name} on the frozen cross-crop suite ==="
    bash entrypoints/eval/nn_eval.sh \
        --config-name nn \
        "data.inference_file=null" \
        "+data.inference_sets.id_crops=${SUITE_DIR}/test_id_crops.parquet" \
        "+data.inference_sets.heldout_crops=${SUITE_DIR}/test_heldout_crops.parquet" \
        "data.validation_axis=crop" \
        "agent.path=${TRAIN_ROOT}/${run_name}/agent.zip" \
        "runtime.seed=0" \
        "runtime.num_repeats=1" \
        "runtime.deterministic=false" \
        "output.dir=${EVAL_ROOT}/${run_name}" \
        --log-file "${LOG_DIR}/eval_e4_${run_name}.log"
done

echo
echo "E5 T1.2 numerical evaluation complete: ${#RUNS[@]} policies x 2 crop groups"
echo "Results: $EVAL_ROOT"
