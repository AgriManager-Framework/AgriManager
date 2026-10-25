#!/usr/bin/env bash
# Frozen E4 evaluation of the terminal T1.2 4ID Think R2 replicate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_evaluate_t12_e4_checkpoint.sh" \
    schema-t12-crosscrop-4id-think-r2 \
    4 \
    81 \
    elrk7ut7 \
    experiments/t1_2_cross_crop_trait_shift/run_t12_4id_think_r2_train.sh \
    T1.2-4ID-T-R2
