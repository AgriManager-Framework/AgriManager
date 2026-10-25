#!/usr/bin/env bash
# Fixed T1.2 4ID Think replicate R2. Paper-reference protocol; seed 81.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_t12_think_seed_replicate.sh" 4 r2 81
