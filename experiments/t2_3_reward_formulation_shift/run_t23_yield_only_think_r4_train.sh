#!/usr/bin/env bash
# Seed replicate R4, T2.3 yield only cell, think arm.
#
# T2.3 reward-formulation shift, WOFOST maize, Qwen3-4B-Instruct, GRPO,
# thinking. Independent training replicate of the paper reference run
# (W&B h3v8z1uf, 4 epochs, 400 steps). Identical to that reference
# except the random seed (93).
#
# Trains on train_yield_max.parquet three times, so the row count matches the objectives cell and only objective diversity is removed.
#
# Shares seed 92 with run_t23_yield_only_no_think_r4_train.sh, its paired arm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_t23_seed_replicate.sh" yield_only think r4 93
