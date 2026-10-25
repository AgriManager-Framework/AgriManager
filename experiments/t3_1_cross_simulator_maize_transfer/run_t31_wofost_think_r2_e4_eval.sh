#!/usr/bin/env bash
# Frozen E4 evaluation of the terminal WOFOST-source R2 specialist.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_evaluate_t31_e4_checkpoint.sh" \
    schema-t31-wofost-think-r2 \
    wofost \
    71 \
    pcddyf5m \
    experiments/t3_1_cross_simulator_maize_transfer/run_t31_wofost_think_r2_train.sh \
    T3.1-WOFOST-T-R2
