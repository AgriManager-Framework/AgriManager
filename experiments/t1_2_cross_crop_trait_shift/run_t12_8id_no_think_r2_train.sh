#!/usr/bin/env bash
# T1.2 8ID No-think seed replicate R2 (seed 81).
# Paired with run_t12_8id_think_r2_train.sh: same seed, same protocol,
# differing only in require_think, which the dataset carries.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_t12_no_think_seed_replicate.sh" 8 r2 81
