#!/usr/bin/env bash
# Frozen E4 evaluation of the terminal DSSAT-source R3 specialist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_evaluate_t31_e4_checkpoint.sh" \
    schema-t31-dssat-think-r3 \
    gym_dssat \
    72 \
    7u2lo744 \
    experiments/t3_1_cross_simulator_maize_transfer/run_t31_dssat_think_r3_train.sh \
    T3.1-DSSAT-T-R3
