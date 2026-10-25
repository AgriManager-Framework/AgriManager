#!/usr/bin/env bash
# Seed replicate R3, T2.3 yield only cell, no-think arm.
#
# T2.3 reward-formulation shift, WOFOST maize, Qwen3-4B-Instruct, GRPO,
# no thinking. Independent training replicate of the paper reference run
# (W&B 2kx0h7jn, 4 epochs, 400 steps). Identical to that reference
# except the random seed (92).
#
# Trains on train_yield_max.parquet three times, so the row count matches the objectives cell and only objective diversity is removed.
#
# Shares seed 92 with run_t23_yield_only_think_r3_train.sh, its paired arm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_t23_seed_replicate.sh" yield_only no_think r3 92
