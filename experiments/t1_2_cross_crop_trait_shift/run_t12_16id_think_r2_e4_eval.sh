#!/usr/bin/env bash
# Frozen E4 evaluation of the terminal T1.2 16ID Think R2 replicate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_evaluate_t12_e4_checkpoint.sh" \
    schema-t12-crosscrop-16id-think-r2 \
    16 \
    81 \
    ua22dc36 \
    experiments/t1_2_cross_crop_trait_shift/run_t12_16id_think_r2_train.sh \
    T1.2-16ID-T-R2
