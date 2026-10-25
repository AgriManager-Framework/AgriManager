#!/usr/bin/env bash
# Frozen E4 evaluation of the terminal DSSAT-source R2 specialist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_evaluate_t31_e4_checkpoint.sh" \
    schema-t31-dssat-think-r2 \
    gym_dssat \
    71 \
    92shygji \
    experiments/t3_1_cross_simulator_maize_transfer/run_t31_dssat_think_r2_train.sh \
    T3.1-DSSAT-T-R2
