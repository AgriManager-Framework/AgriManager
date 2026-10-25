#!/usr/bin/env bash
# T1.2 4ID No-think seed replicate R4 (seed 83).
# Paired with run_t12_4id_think_r4_train.sh: same seed, same protocol,
# differing only in require_think, which the dataset carries.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_t12_no_think_seed_replicate.sh" 4 r4 83
