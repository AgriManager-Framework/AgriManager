#!/usr/bin/env bash
# Run the scripted anchors on every frozen test-suite split.
#
#   OUT_ROOT=<output dir> bash experiments/rebuttal_confirmatory_eval/run_e4_anchors.sh
#
# The paper reports every score next to two anchors, no-action and the scripted
# agronomic reference, and gives the raw return, both anchor returns and the
# anchor gap for every headline cell. This script scores both anchors on the
# frozen test-suite splits.
#
# Which scripted reference, per simulator, follows what the submitted figures
# used:
#
#   WOFOST          extension_guideline -- the per-crop policy the paper used,
#                   kept as paper_references/extension_guideline_rollout_paper.py.
#                   The current ag_heuristic module replaced it under the same
#                   label and is a different policy.
#   DSSAT, Cycles   paper_sim_heuristic -- the first implementation, kept as
#                   paper_references/random_rollout_paper_sim.py, which is what
#                   the paper's DSSAT and Cycles anchors ran; see
#                   paper_sim_anchor_rollout.py for what it reproduces exactly
#                   (Cycles) and what it only reconstructs (DSSAT, whose paper
#                   value was set by hand). T1.3 is left out of this mode: on
#                   crop planning the first implementation and the current
#                   module take the same fallback, so it would repeat the
#                   current run.
#
# The current ag_heuristic is run on every split as well. On the Cycles
# fertilization task it degenerates to the zero action, which is why it cannot
# stand in for the paper's anchor there. On the WOFOST splits it disagrees with
# the paper's reference by sign on some crops; running both makes the choice of
# reference a reading of numbers.
#
# The two paper-era policies are copied into the output directory with their
# sha256, so the exact code behind every reference number is recorded next to it
# (paper_references/README.md lists the expected hashes).
#
# Resumable: a split with a metrics.json is skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUITE_DIR="$SCRIPT_DIR/data"
OUT_ROOT="${OUT_ROOT:?set OUT_ROOT to the output directory for the anchor runs}"
JOBS="${JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
PAPER_REF_DIR="$SCRIPT_DIR/paper_references"
# Space-separated subset of modes to run; empty runs all. Lets a second job add
# a mode without touching tasks a running job still owns.
MODES="${MODES:-}"

export WOFOST_GYM_PATH="${WOFOST_GYM_PATH:-$PROJECT_ROOT/../AgriManagerExternal/WOFOSTGym}"
export DSSAT_GYM_PATH="${DSSAT_GYM_PATH:-$PROJECT_ROOT/spack/gym-dssat-pdi}"
export PDI_PLUGIN_PATH="${PDI_PLUGIN_PATH:-$DSSAT_GYM_PATH/lib}"
export LD_LIBRARY_PATH="$DSSAT_GYM_PATH/lib:$DSSAT_GYM_PATH/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/verl:$WOFOST_GYM_PATH:$WOFOST_GYM_PATH/pcse_gym:$WOFOST_GYM_PATH/pcse:$PROJECT_ROOT/../AgriManagerExternal/gym_dssat_pdi/gym-dssat-pdi:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

cd "$PROJECT_ROOT"
mkdir -p "$OUT_ROOT"

paper_policy="$OUT_ROOT/_paper_policy/extension_guideline_rollout.py"
paper_sim="$OUT_ROOT/_paper_policy/random_rollout_paper_sim.py"
mkdir -p "$(dirname "$paper_policy")"
cp "$PAPER_REF_DIR/extension_guideline_rollout_paper.py" "$paper_policy"
cp "$PAPER_REF_DIR/random_rollout_paper_sim.py" "$paper_sim"
{
    echo "source: $PAPER_REF_DIR/extension_guideline_rollout_paper.py"
    echo "sha256: $(sha256sum "$paper_policy" | cut -d' ' -f1)"
    echo "source: $PAPER_REF_DIR/random_rollout_paper_sim.py"
    echo "sha256: $(sha256sum "$paper_sim" | cut -d' ' -f1)"
} > "$OUT_ROOT/_paper_policy/PROVENANCE"

# One line per task: mode <TAB> group <TAB> parquet <TAB> axis <TAB> simulator
# A job restricted by MODES keeps its own task list, so two jobs never share one.
tasks="$OUT_ROOT/_tasks${MODES:+_${MODES// /_}}.tsv"
python3 - "$SUITE_DIR" "$MODES" > "$tasks" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
only = set(sys.argv[2].split())
for g in sorted(root.glob("e4_t*")):
    if g.name.endswith("_nn") or not (g / "manifest.json").is_file():
        continue
    m = json.loads((g / "manifest.json").read_text())
    sim, axis = m["env_name"], m.get("validation_axis") or "none"
    modes = ["no_action", "ag_heuristic"] + (["extension_guideline"] if sim == "wofost_gym" else [])
    if sim in ("gym_dssat", "cycles_gym") and g.name != "e4_t13_price_regime":
        modes.append("paper_sim_heuristic")
    for p in sorted(g.glob("test*.parquet")):
        for mode in modes:
            if only and mode not in only:
                continue
            print("\t".join([mode, g.name, str(p), axis, sim]))
PY
echo "$(wc -l < "$tasks") anchor tasks, $JOBS in parallel -> $OUT_ROOT"

run_one() {
    local mode="$1" group="$2" parquet="$3" axis="$4" sim="$5"
    local split out log
    split="$(basename "$parquet" .parquet)"
    out="$OUT_ROOT/$mode/$group/$split"
    [[ -f "$out/metrics.json" ]] && { echo "skip  $mode $group/$split"; return 0; }
    rm -rf "$out"; mkdir -p "$out"
    log="$out/run.log"
    # CycleGym writes into its working directory per episode; a private one per
    # task keeps parallel tasks from colliding.
    export CYCLES_GYM_WORKING_DIR="${CYCLES_GYM_WORKING_DIR_BASE:-/tmp/e4-anchors-$$}/$mode-$group-$split"
    mkdir -p "$CYCLES_GYM_WORKING_DIR"
    local axis_arg=()
    if [[ "$mode" == "paper_sim_heuristic" ]]; then
        python -u "$PROJECT_ROOT/experiments/rebuttal_confirmatory_eval/paper_sim_anchor_rollout.py" \
            "$paper_sim" "$parquet" "$out" "$axis" > "$log" 2>&1
    elif [[ "$mode" == "extension_guideline" ]]; then
        [[ "$axis" != none ]] && axis_arg=(--validation-axis "$axis")
        python -u "$paper_policy" --data "$parquet" --output-dir "$out" \
            --mode extension_guideline --seed 42 --num-trials 1 "${axis_arg[@]}" > "$log" 2>&1
    else
        [[ "$axis" != none ]] && axis_arg=("data.validation_axis=$axis")
        python -u -m agrimanager.rollout.inference.random_rollout \
            --config-path "$PROJECT_ROOT/entrypoints/eval/config" --config-name "$mode" \
            "data.inference_file=$parquet" "${axis_arg[@]}" \
            runtime.seed=42 runtime.num_trials=1 "output.dir=$out" wandb.enabled=false \
            > "$log" 2>&1
    fi
    local rc=$?
    if [[ $rc -eq 0 && -f "$out/metrics.json" ]]; then
        echo "done  $mode $group/$split"
    else
        echo "FAIL  $mode $group/$split (rc=$rc), see $log"; tail -3 "$log" | sed 's/^/      /'
        rm -f "$out/metrics.json"
        return 1
    fi
}
export -f run_one
export OUT_ROOT PROJECT_ROOT paper_policy paper_sim

set +e
# Slowest simulators first so they do not become the tail.
sort -t$'\t' -k5,5 -r "$tasks" | awk -F'\t' '{print ($5=="gym_dssat"?0:($5=="cycles_gym"?1:2)) "\t" $0}' \
    | sort -s -k1,1n | cut -f2- \
    | xargs -P "$JOBS" -d '\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r m g p a s <<< "{}"; run_one "$m" "$g" "$p" "$a" "$s"'
rc=$?
set -e
n_done=0
while IFS=$'\t' read -r m g p _ _; do
    [[ -f "$OUT_ROOT/$m/$g/$(basename "$p" .parquet)/metrics.json" ]] && n_done=$((n_done+1))
done < "$tasks"
echo "anchors complete: $n_done / $(wc -l < "$tasks") tasks have metrics.json (xargs rc=$rc)"
exit "$rc"
