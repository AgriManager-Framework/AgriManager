#!/usr/bin/env bash
# Seed replicate R3, T1.3 price-regime shift, no-think arm.
#
# CyclesGym price-regime shift, Qwen3-4B-Instruct, GRPO, no thinking. Independent
# training replicate of the paper reference run (W&B a4wbykfc, 132 steps,
# 2026-05-03). Identical to that reference except the random seed (112).
#
# Shares seed 112 with run_t13_price_think_r3_train.sh, its paired arm,
# because the cell is reported as a paired think-minus-no-think effect.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_t13_seed_replicate.sh" no_think r3 112
