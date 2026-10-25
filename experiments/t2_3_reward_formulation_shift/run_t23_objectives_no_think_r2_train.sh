#!/usr/bin/env bash
# Seed replicate R2, T2.3 objectives cell, no-think arm.
#
# T2.3 reward-formulation shift, WOFOST maize, Qwen3-4B-Instruct, GRPO,
# no thinking. Independent training replicate of the paper reference run
# (W&B gyzao2k2, 4 epochs, 400 steps). Identical to that reference
# except the random seed (91).
#
# Trains on three distinct objective files.
#
# Shares seed 91 with run_t23_objectives_think_r2_train.sh, its paired arm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_t23_seed_replicate.sh" objectives no_think r2 91
