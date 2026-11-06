#!/usr/bin/env bash
# Build the numerical-policy rendering of the T13 E4 suite.
#
# This is not a new test set. It re-renders the already frozen and audited
# scenarios of `e4_t13_price_regime` with `llm_mode: false`, so a numerical policy
# sees vector observations where the LLM saw prompts. Same source pool, same
# generation seed, same groups: the config differs only in `dataset_id`,
# `llm_mode`, and the reasoning fields, which have no meaning off the LLM path.
#
# That identity is the point. It is what lets the LLM and numerical arms of
# T13 be compared as a paired contrast on the same scenarios rather than as
# two samples from the same distribution, which is exactly why the T1.1 and T2.2
# suites were built in pairs this way.
set -euo pipefail

# Commands run in the active Python environment (Conda or uv venv). To use a named
# Conda environment without activating it: AGRI_RUN="conda run --no-capture-output -n agrimanager".
AGRI_RUN="${AGRI_RUN:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIG="$SCRIPT_DIR/config/e4_t13_price_regime_nn.yaml"
DATASET_DIR="$SCRIPT_DIR/data/e4_t13_price_regime_nn"
LLM_DATASET_DIR="$SCRIPT_DIR/data/e4_t13_price_regime"

cd "$PROJECT_ROOT"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="$PROJECT_ROOT/.cache/huggingface"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# The LLM suite is the reference this one must match scenario for scenario.
if [[ ! -f "$LLM_DATASET_DIR/manifest.json" ]]; then
    echo "Missing the LLM T13 suite at $LLM_DATASET_DIR" >&2
    exit 1
fi

$AGRI_RUN \
    bash entrypoints/dataset/build.sh \
    --config "$CONFIG" \
    --output-dir "$DATASET_DIR" \
    --num-workers 1

$AGRI_RUN \
    python experiments/rebuttal_confirmatory_eval/normalize_eval_suite_paths.py --apply
