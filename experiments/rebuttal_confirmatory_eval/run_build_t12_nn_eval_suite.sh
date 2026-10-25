#!/usr/bin/env bash
# Build the numerical-policy rendering of the T12 E4 suite.
#
# This is not a new test set. It re-renders the already frozen and audited
# scenarios of `e4_t12_cross_crop` with `llm_mode: false`, so a numerical policy
# sees vector observations where the LLM saw prompts. Same source pool, same
# generation seed, same groups: the config differs only in `dataset_id`,
# `llm_mode`, and the reasoning fields, which have no meaning off the LLM path.
#
# That identity is the point. It is what lets the LLM and numerical arms of
# T12 be compared as a paired contrast on the same scenarios rather than as
# two samples from the same distribution, which is exactly why the T1.1 and T2.2
# suites were built in pairs this way.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG="$SCRIPT_DIR/config/e4_t12_cross_crop_nn.yaml"
DATASET_DIR="$SCRIPT_DIR/data/e4_t12_cross_crop_nn"
LLM_DATASET_DIR="$SCRIPT_DIR/data/e4_t12_cross_crop"

cd "$PROJECT_ROOT"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# The LLM suite is the reference this one must match scenario for scenario.
if [[ ! -f "$LLM_DATASET_DIR/manifest.json" ]]; then
    echo "Missing the LLM T12 suite at $LLM_DATASET_DIR" >&2
    exit 1
fi

conda run -n agrimanager \
    python experiments/rebuttal_confirmatory_eval/prepare_t12_eval_pool.py

conda run -n agrimanager \
    bash entrypoints/dataset/build.sh \
    --config "$CONFIG" \
    --output-dir "$DATASET_DIR" \
    --num-workers 1

conda run -n agrimanager \
    python experiments/rebuttal_confirmatory_eval/normalize_eval_suite_paths.py --apply
