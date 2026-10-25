#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RESULT_ROOT="${AG_HEURISTIC_RESULT_ROOT:-$SCRIPT_DIR/results/baseline_eval/ag_heuristic}"
LOG_DIR="${AG_HEURISTIC_LOG_DIR:-$SCRIPT_DIR/logs/ag_heuristic}"
SEED="${AG_HEURISTIC_SEED:-42}"
NUM_TRIALS="${AG_HEURISTIC_NUM_TRIALS:-1}"
WANDB_ENABLED="${AG_HEURISTIC_WANDB_ENABLED:-false}"
RUN_MISSING="${AG_HEURISTIC_RUN_MISSING:-skip}"

mkdir -p "$RESULT_ROOT" "$LOG_DIR"

prepare_dssat_runtime() {
    if [[ -f "$PROJECT_ROOT/smoke_tests/gym_dssat/_activate_spack.sh" ]]; then
        # shellcheck disable=SC1091
        source "$PROJECT_ROOT/smoke_tests/gym_dssat/_activate_spack.sh"
    fi

    local candidate
    for candidate in \
        "$PROJECT_ROOT/AgriManagerExternal/gym_dssat_pdi/gym-dssat-pdi" \
        "$PROJECT_ROOT/../AgriManagerExternal/gym_dssat_pdi/gym-dssat-pdi"
    do
        if [[ -d "$candidate/gym_dssat_pdi" ]]; then
            case ":${PYTHONPATH:-}:" in
                *":$candidate:"*) ;;
                *) export PYTHONPATH="$candidate:${PYTHONPATH:-}" ;;
            esac
            break
        fi
    done
}

hydra_list() {
    local sep=""
    local result="["
    local value
    for value in "$@"; do
        result+="${sep}${value}"
        sep=","
    done
    result+="]"
    printf '%s\n' "$result"
}

collect_existing() {
    local -n _out="$1"
    shift
    local candidate
    local matched
    for candidate in "$@"; do
        matched=0
        if [[ "$candidate" == *"*"* ]]; then
            local expanded
            shopt -s nullglob
            for expanded in $candidate; do
                _out+=("$expanded")
                matched=1
            done
            shopt -u nullglob
        elif [[ -f "$candidate" ]]; then
            _out+=("$candidate")
            matched=1
        fi
        if [[ "$matched" -eq 0 && "$RUN_MISSING" != "skip" ]]; then
            echo "[missing] $candidate" >&2
            return 1
        fi
    done
}

run_case() {
    local label="$1"
    local validation_axis="$2"
    shift 2

    local files=()
    collect_existing files "$@"
    if [[ "${#files[@]}" -eq 0 ]]; then
        echo "[skip] $label: no parquet files found"
        return 0
    fi

    local inference_file
    if [[ "${#files[@]}" -eq 1 ]]; then
        inference_file="${files[0]}"
    else
        inference_file="$(hydra_list "${files[@]}")"
    fi

    local output_dir="$RESULT_ROOT/$label"
    local log_file="$LOG_DIR/${label}.log"
    mkdir -p "$output_dir"

    echo "[run] $label"
    echo "      files=${files[*]}"
    bash entrypoints/eval/baseline_eval.sh \
        --config-name ag_heuristic \
        --log-file "$log_file" \
        "data.inference_file=${inference_file}" \
        "data.validation_axis=${validation_axis}" \
        "runtime.seed=${SEED}" \
        "runtime.num_trials=${NUM_TRIALS}" \
        "output.dir=${output_dir}" \
        "wandb.enabled=${WANDB_ENABLED}" \
        "wandb.name=ag_heuristic_${label}" \
        "wandb.tags=[baseline,ag_heuristic,${label}]" \
        2>&1 | tee -a "$log_file"
}

prepare_dssat_runtime

# T1.1 Weather-regime shift
run_case "t1_1_chickpea" "weather_regime" \
    "$SCRIPT_DIR/t1_1_weather_regime_shift/data/weather_regime_chickpea_nn_without_traits/val_*.parquet"
run_case "t1_1_potato" "weather_regime" \
    "$SCRIPT_DIR/t1_1_weather_regime_shift/data/weather_regime_potato_nn_without_traits/val_*.parquet"

# T1.2 Cross-crop trait shift
run_case "t1_2_4id" "crop" \
    "$SCRIPT_DIR/t1_2_cross_crop_trait_shift/data/cross_crop_4id_llm_traits_v1_23d_no_think/val_*.parquet"
run_case "t1_2_8id" "crop" \
    "$SCRIPT_DIR/t1_2_cross_crop_trait_shift/data/cross_crop_8id_llm_traits_v1_23d_no_think/val_*.parquet"
run_case "t1_2_16id" "crop" \
    "$SCRIPT_DIR/t1_2_cross_crop_trait_shift/data/cross_crop_16id_llm_traits_v1_23d_no_think/val_*.parquet"

# T1.3 Price-regime shift
run_case "t1_3_cycles_price_regime" "price_regime" \
    "$SCRIPT_DIR/t1_3_cycles_gym_price_regime/data/t1_3_cycles_gym_price_regime_llm_no_think/val_*.parquet" \
    "$SCRIPT_DIR/t1_3_cycles_gym_price_regime/data/t1_3_cycles_gym_price_regime_llm_no_think/test_*.parquet"

# T2.1 Observation-schema shift
run_case "t2_1_observation_schema" "observation_schema" \
    "$SCRIPT_DIR/t2_1_observation_schema_shift/data/t21_observation_schema_shift_llm_no_think/val_*.parquet"

# T2.2 Action-menu shift
run_case "t2_2_family_a" "action_menu" \
    "$SCRIPT_DIR/t2_2_action_menu_shift/data/t22_family_a_no_think/val_*.parquet"
run_case "t2_2_family_b" "action_menu" \
    "$SCRIPT_DIR/t2_2_action_menu_shift/data/t22_family_b_no_think/val_*.parquet"

# T2.3 Reward-formulation shift
run_case "t2_3_reward_formulation" "reward_formulation" \
    "$SCRIPT_DIR/t2_3_reward_formulation_shift/data/t23_reward_formulation_shift/val_*.parquet"

# T3.1 Cross-simulator maize transfer
run_case "t3_1_wofost_target" "simulator" \
    "$SCRIPT_DIR/t3_1_cross_simulator_maize_transfer/data/t31_cross_sim_wofost/val.parquet"
run_case "t3_1_dssat_target" "simulator" \
    "$SCRIPT_DIR/t3_1_cross_simulator_maize_transfer/data/t31_cross_sim_gym_dssat/val.parquet"
run_case "t3_1_cycles_target" "simulator" \
    "$SCRIPT_DIR/t3_1_cross_simulator_maize_transfer/data/t31_cross_sim_cycles_gym/val.parquet"

# T3.2 Unified prompt-conditioned policy
run_case "t3_2_seen_control" "schema_tuple" \
    "$SCRIPT_DIR/t3_2_unified_prompt_conditioned_policy/data/t32_unified_wofost/val_wofost_lnpkw_profit.parquet" \
    "$SCRIPT_DIR/t3_2_unified_prompt_conditioned_policy/data/t32_unified_wofost/val_wofost_lnpk_profit.parquet" \
    "$SCRIPT_DIR/t3_2_unified_prompt_conditioned_policy/data/t32_unified_wofost/val_wofost_lnw_yield.parquet" \
    "$SCRIPT_DIR/t3_2_unified_prompt_conditioned_policy/data/t32_unified_gym_dssat_yield/val.parquet" \
    "$SCRIPT_DIR/t3_2_unified_prompt_conditioned_policy/data/t32_unified_cycles_yield/val.parquet"
run_case "t3_2_heldout" "schema_tuple" \
    "$SCRIPT_DIR/t3_2_unified_prompt_conditioned_policy/data/t32_unified_wofost/val_wofost_lnw_profit.parquet" \
    "$SCRIPT_DIR/t3_2_unified_prompt_conditioned_policy/data/t32_unified_wofost/val_wofost_lnpkw_nutrient.parquet" \
    "$SCRIPT_DIR/t3_2_unified_prompt_conditioned_policy/data/t32_unified_gym_dssat_profit/val.parquet" \
    "$SCRIPT_DIR/t3_2_unified_prompt_conditioned_policy/data/t32_unified_cycles_profit/val.parquet"

echo "[done] Ag-heuristic baseline runs complete. Results: $RESULT_ROOT"
