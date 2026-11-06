#!/usr/bin/env bash
# E5: evaluate one published checkpoint against the frozen E4 suites.
#
# Usage:
#   bash entrypoints/eval/e5_evaluate_checkpoint.sh \
#       --checkpoint-repo <owner>/schema-t32-qwen-think-r2 \
#       --revision step-800 \
#       --suite-repo AgriManager-Framework/agrimanager-test-suite \
#       --results-repo <owner>/schema-results-t32-qwen \
#       --tier t32 \
#       [--max-samples 8] [--no-upload]
#
# The reasoning format is read from the checkpoint's own run_manifest.json,
# applied as an env-config override and then asserted, so a checkpoint can never
# be scored under a condition it was not trained for. Checkpoints published
# before that field existed need --require-think, and the manifest records that
# the value was supplied by hand rather than read.
#
# Requires HF_TOKEN unless --no-upload. Supply names that must not appear in the
# bundle through PUBLISH_FORBID_TERMS or repeated --forbid flags.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# vLLM 0.11 otherwise JIT-compiles FlashInfer's top-k/top-p sampling kernels on
# first use, which needs a full CUDA toolkit (nvcc, headers, a C++ compiler) on
# the GPU node; the pip-installed stack has none. PyTorch's sampler gives the
# same greedy decoding. Export VLLM_USE_FLASHINFER_SAMPLER=1 to use FlashInfer.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# The evaluator drives the simulators, so their paths have to resolve here the
# same way they do for training.
export WOFOST_GYM_PATH="${WOFOST_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/WOFOSTGym}"
export DSSAT_GYM_PATH="${DSSAT_GYM_PATH:-$PROJECT_ROOT/spack/gym-dssat-pdi}"
export PDI_PLUGIN_PATH="${PDI_PLUGIN_PATH:-$DSSAT_GYM_PATH/lib}"
export LD_LIBRARY_PATH="$DSSAT_GYM_PATH/lib:$DSSAT_GYM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/verl:$WOFOST_GYM_PATH:$WOFOST_GYM_PATH/pcse_gym:$WOFOST_GYM_PATH/pcse:$PROJECT_ROOT/../AgriManagerExternal/gym_dssat_pdi/gym-dssat-pdi:${PYTHONPATH:-}"

NEEDS_TOKEN=1
for arg in "$@"; do
    if [[ "$arg" == "--no-upload" ]]; then
        NEEDS_TOKEN=0
        break
    fi
done
if [[ "$NEEDS_TOKEN" -eq 1 && -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set; export it, or pass --no-upload for a local run." >&2
    exit 2
fi

cd "$PROJECT_ROOT"
exec python -u -m agrimanager.checkpoint.evaluate \
    --project-root "$PROJECT_ROOT" \
    "$@"
