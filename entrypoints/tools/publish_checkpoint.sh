#!/usr/bin/env bash
# Merge one FSDP training checkpoint and publish it as a private HF repository.
#
# Usage:
#   bash entrypoints/tools/publish_checkpoint.sh \
#       --checkpoint-dir <.../global_step_200> \
#       --repo-id <owner>/<neutral-name> \
#       --run-name <neutral-name> \
#       --experiment-id E2-Q-T-R2 \
#       --seed 43 \
#       --base-model Qwen/Qwen3-4B-Instruct-2507 \
#       --checkpoint-step 200 \
#       [--no-upload] [any other flag the module accepts]
#
# Requires HF_TOKEN in the environment. Supply the names that must never appear
# in the artifact through PUBLISH_FORBID_TERMS (comma-separated) or repeated
# --forbid flags; they are deliberately not stored in the repository.
#
# Run `--no-upload` first on any new run: it merges, sanitizes, verifies and
# writes the manifest without touching the Hub, so the artifact can be inspected
# before it leaves the machine.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/verl:${PYTHONPATH:-}"

# Only an actual upload needs credentials. The recommended --no-upload dry run
# must work without them.
NEEDS_TOKEN=1
for arg in "$@"; do
    if [[ "$arg" == "--no-upload" ]]; then
        NEEDS_TOKEN=0
        break
    fi
done
if [[ "$NEEDS_TOKEN" -eq 1 && -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set; export it, or pass --no-upload for a dry run." >&2
    exit 2
fi

cd "$PROJECT_ROOT"
exec python -u -m agrimanager.checkpoint.publish \
    --project-root "$PROJECT_ROOT" \
    "$@"
