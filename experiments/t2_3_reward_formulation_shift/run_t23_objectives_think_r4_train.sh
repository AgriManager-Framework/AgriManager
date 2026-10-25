#!/usr/bin/env bash
# Seed replicate R4, T2.3 objectives cell, think arm.
#
# T2.3 reward-formulation shift, WOFOST maize, Qwen3-4B-Instruct, GRPO,
# thinking. Independent training replicate of the paper reference run
# (W&B 8l9c0vxp, 4 epochs, 400 steps). Identical to that reference
# except the random seed (93).
#
# Trains on three distinct objective files.
#
# Shares seed 92 with run_t23_objectives_no_think_r4_train.sh, its paired arm.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_t23_seed_replicate.sh" objectives think r4 93
