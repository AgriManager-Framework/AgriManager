#!/usr/bin/env bash
# Frozen E4 evaluation of the terminal CycleGym-source R3 specialist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_evaluate_t31_e4_checkpoint_node_tmp.sh" \
    schema-t31-cycles-think-r3 \
    cycles_gym \
    72 \
    m22ie0ym \
    experiments/t3_1_cross_simulator_maize_transfer/run_t31_cycles_think_r3_train.sh \
    T3.1-CYCLES-T-R3
