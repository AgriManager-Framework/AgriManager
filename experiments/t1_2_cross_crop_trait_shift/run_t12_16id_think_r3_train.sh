#!/usr/bin/env bash
# Fixed T1.2 16ID Think replicate R3. Paper-reference protocol; seed 82.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_t12_think_seed_replicate.sh" 16 r3 82
