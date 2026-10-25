#!/usr/bin/env bash
# Frozen E4 evaluation of the terminal T1.2 8ID Think R3 replicate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_evaluate_t12_e4_checkpoint.sh" \
    schema-t12-crosscrop-8id-think-r3 \
    8 \
    82 \
    tppja5y3 \
    experiments/t1_2_cross_crop_trait_shift/run_t12_8id_think_r3_train.sh \
    T1.2-8ID-T-R3
