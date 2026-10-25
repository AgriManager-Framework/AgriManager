#!/usr/bin/env bash
# Re-run the Tier 1 anchor baselines on the corrected action scale.
#
# The submitted Tier 1 anchors were evaluated on the numerical (`_nn`) datasets,
# which carried the duplicated `scale_action_amounts_by_interval` defect fixed in
# fa1bc68. Every anchor that applies fertilizer therefore received 200 kg/ha and
# 50 cm per level instead of 20 kg/ha and 5 cm. The no-action anchor is immune
# because it applies nothing, so the submitted ag-heuristic anchor landed *below*
# the no-action anchor on every Tier 1 cell and the paper's normalization
# denominator (`ag_heuristic - no_action`) came out negative:
#
#   T1.1 potato    no-action +0.10  ag-heuristic -1.04  random -24.02
#   T1.1 chickpea  no-action +0.01  ag-heuristic -1.56  random -19.55
#   T1.2 4ID       no-action +0.55  ag-heuristic -0.43  random -17.04
#   T1.2 8ID       no-action +0.40  ag-heuristic -0.41  random -18.82
#   T1.2 16ID      no-action +0.43  ag-heuristic -0.42  random -17.88
#
# Only Tier 1 is affected. Every Tier 2 and Tier 3 anchor was evaluated on an LLM
# dataset, which never set `fert_amount`/`irrig_amount` and so always ran at the
# intended scale; T1.3 uses CyclesGym, whose action interface is unrelated to the
# WOFOST-Gym scaling and whose config is not among the 16 that fa1bc68 corrected.
#
# The protocol below reproduces the submitted anchor runs exactly -- same
# validation files, same `weather_regime`/`crop` axes, same seed 42, same single
# trial -- so the action scale is the only thing that changed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

RESULT_ROOT="$SCRIPT_DIR/results/baseline_eval_corrected"
LOG_DIR="$SCRIPT_DIR/logs/baseline_eval_corrected"

mkdir -p "$RESULT_ROOT" "$LOG_DIR"

cd "$PROJECT_ROOT"

T11="$SCRIPT_DIR/t1_1_weather_regime_shift/data"
T12="$SCRIPT_DIR/t1_2_cross_crop_trait_shift/data"

# label | validation axis | comma-joined validation parquets
CASES=(
    "t1_1_potato|weather_regime|$T11/weather_regime_potato_nn_without_traits/val_cold.parquet,$T11/weather_regime_potato_nn_without_traits/val_drought.parquet,$T11/weather_regime_potato_nn_without_traits/val_hot.parquet,$T11/weather_regime_potato_nn_without_traits/val_id.parquet,$T11/weather_regime_potato_nn_without_traits/val_wet.parquet"
    "t1_1_chickpea|weather_regime|$T11/weather_regime_chickpea_nn_without_traits/val_cold.parquet,$T11/weather_regime_chickpea_nn_without_traits/val_drought.parquet,$T11/weather_regime_chickpea_nn_without_traits/val_hot.parquet,$T11/weather_regime_chickpea_nn_without_traits/val_id.parquet,$T11/weather_regime_chickpea_nn_without_traits/val_wet.parquet"
    "t1_2_4id|crop|$T12/cross_crop_4id_nn_traits_v1_23d/val_heldout_crops.parquet,$T12/cross_crop_4id_nn_traits_v1_23d/val_id_crops.parquet"
    "t1_2_8id|crop|$T12/cross_crop_8id_nn_traits_v1_23d/val_heldout_crops.parquet,$T12/cross_crop_8id_nn_traits_v1_23d/val_id_crops.parquet"
    "t1_2_16id|crop|$T12/cross_crop_16id_nn_traits_v1_23d/val_heldout_crops.parquet,$T12/cross_crop_16id_nn_traits_v1_23d/val_id_crops.parquet"
)

MODES=(no_action ag_heuristic random)

for case_spec in "${CASES[@]}"; do
    IFS='|' read -r label axis files <<< "$case_spec"

    # A stale parquet here would silently reintroduce the defect the rerun exists
    # to remove, so require every file rather than skipping what is missing.
    IFS=',' read -r -a file_list <<< "$files"
    for required in "${file_list[@]}"; do
        if [[ ! -f "$required" ]]; then
            echo "Missing $required" >&2
            echo "Rebuild the Tier 1 numerical datasets before running the anchors." >&2
            exit 1
        fi
    done

    for mode in "${MODES[@]}"; do
        run_name="${mode}_${label}_corrected_scale"
        output_dir="$RESULT_ROOT/$label/$mode"
        log_file="$LOG_DIR/${run_name}.log"
        mkdir -p "$output_dir"

        echo "[run] $run_name"
        bash entrypoints/eval/baseline_eval.sh \
            --config-name "$mode" \
            --log-file "$log_file" \
            "data.inference_file=[${files}]" \
            "data.validation_axis=${axis}" \
            "runtime.seed=42" \
            "runtime.num_trials=1" \
            "output.dir=${output_dir}" \
            "wandb.enabled=true" \
            "wandb.project_name=agrimanager_anchors" \
            "wandb.name=${run_name}" \
            "wandb.tags=[baseline,${mode},${label},corrected_action_scale]"
    done
done

echo "[done] Corrected Tier 1 anchors complete. Results: $RESULT_ROOT"
