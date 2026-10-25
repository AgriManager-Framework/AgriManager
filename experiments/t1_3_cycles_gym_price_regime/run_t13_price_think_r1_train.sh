#!/usr/bin/env bash
# Seed replicate R1, T1.3 price-regime shift, think arm.
#
# CyclesGym price-regime shift, Qwen3-4B-Instruct, GRPO, thinking.
#
# Unlike every other tier, T1.3 has NO poolable paper reference. Every T1.3 run
# on W&B sits in an earlier project and dates from
# 2026-04-29 to 2026-05-06, while the config this script builds from,
# config/price_regime_llm_think.yaml, was created 2026-05-14 in 9c5721e and did
# not exist when those runs executed. Commit d05152c had also rewritten
# prompt.py (+130 lines) and env.py (+209) on 2026-05-03 23:04, after the
# 02:23 and 06:07 reference runs that day.
#
# So R1 here is a fresh seeded run rather than the unseeded paper reference:
# all three T1.3 seeds are generated on the current data path, and the old runs
# are not pooled with them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_run_t13_seed_replicate.sh" think r1 110
