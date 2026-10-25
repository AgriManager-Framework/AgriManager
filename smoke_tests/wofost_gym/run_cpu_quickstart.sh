#!/usr/bin/env bash
# CPU-only quickstart: build 8 maize scenarios and score two scripted policies
# (no-action and the scripted agronomic reference) on them. No GPU, no model
# download, no experiment tracker. Takes a few minutes on a laptop-class CPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DATASET_CONFIG="$SCRIPT_DIR/config/quickstart_maize_cpu.yaml"
DATA_DIR="$SCRIPT_DIR/data"
RESULTS_DIR="$SCRIPT_DIR/results/quickstart"
VAL_FILE="$DATA_DIR/quickstart_maize_cpu/val.parquet"

mkdir -p "$DATA_DIR" "$RESULTS_DIR"
cd "$PROJECT_ROOT"

bash entrypoints/dataset/build.sh \
    --config "$DATASET_CONFIG" \
    --num-workers 1 \
    --output-dir "$DATA_DIR/quickstart_maize_cpu"

for POLICY in no_action ag_heuristic; do
    bash entrypoints/eval/baseline_eval.sh \
        --config-name "$POLICY" \
        data.inference_file="$VAL_FILE" \
        output.dir="$RESULTS_DIR/$POLICY" \
        wandb.enabled=false
done

python - "$RESULTS_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for policy in ("no_action", "ag_heuristic"):
    metrics_path = root / policy / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    keys = [k for k in metrics if "objective_reward" in k or "profit_ge_kg_ha" in k]
    print(f"{policy}: {metrics_path}")
    for key in sorted(keys)[:6]:
        print(f"  {key} = {metrics[key]}")
PY
