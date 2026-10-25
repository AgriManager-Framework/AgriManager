#!/usr/bin/env bash
# Fixed T1.2 8ID Think replicate R4. Paper-reference protocol; seed 83.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_t12_think_seed_replicate.sh" 8 r4 83
