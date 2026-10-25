"""Source-backed extension-guideline scripted rollout for WOFOST-Gym.

This module is intentionally separate from ``random_rollout.py`` because the
extension-guideline baseline has a different claim boundary from random,
no-action, and PPV0 references. It evaluates a conservative agronomic reference
policy defined in ``research/paper/experiment_cards/overview/baselines.md``.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

from agrimanager.rollout.inference.inference_rollout import (
    _build_summary,
    _dump_json_atomic,
    _extract_trial_metrics,
    _extract_validation_metric_dict,
    load_dataset,
)
from agrimanager.rollout.inference.random_rollout import (
    _executed_action,
    _scalar_action_id,
    _seed_action_spaces,
    _zero_action,
    create_env,
    get_action_space,
)

VALID_MODES = {"no_action", "extension_guideline"}
MODE_ALIASES = {
    "no-action": "no_action",
    "noaction": "no_action",
    "zero": "no_action",
    "zero_action": "no_action",
    "ag_heuristic": "extension_guideline",
    "ag-heuristic": "extension_guideline",
    "agheuristic": "extension_guideline",
    "heuristic": "extension_guideline",
    "expert": "extension_guideline",
    "guideline": "extension_guideline",
    "extension-guideline": "extension_guideline",
    "extension_guideline": "extension_guideline",
}

PULSE_LEGUMES = {
    "chickpea",
    "cowpea",
    "fababean",
    "groundnut",
    "mungbean",
    "pigeonpea",
    "soybean",
}

WOFOST_EXTENSION_GUIDELINES: Dict[str, Dict[str, float]] = {
    "__default__": {
        "n_cap": 60.0,
        "p_cap": 20.0,
        "k_cap": 20.0,
        "irrig_cap_mm": 80.0,
        "irrig_trigger": 0.31,
        "p_threshold": 5.0,
        "k_threshold": 5.0,
        "min_fert_temp": 4.0,
    },
    "barley": {"n_cap": 90.0, "irrig_cap_mm": 80.0},
    "chickpea": {
        "n_cap": 0.0,
        "starter_n_cap": 0.0,
        "p_cap": 20.0,
        "k_cap": 0.0,
        "irrig_cap_mm": 50.0,
        "irrig_trigger": 0.28,
        "min_fert_temp": 6.0,
    },
    "cotton": {"n_cap": 80.0, "p_cap": 20.0, "k_cap": 40.0, "irrig_cap_mm": 120.0},
    "maize": {"n_cap": 120.0, "p_cap": 20.0, "k_cap": 20.0, "irrig_cap_mm": 100.0},
    "millet": {"n_cap": 60.0, "irrig_cap_mm": 60.0},
    "potato": {
        "n_cap": 120.0,
        "p_cap": 40.0,
        "k_cap": 80.0,
        "irrig_cap_mm": 180.0,
        "irrig_trigger": 0.37,
        "p_threshold": 7.0,
        "k_threshold": 8.0,
        "min_fert_temp": 5.0,
    },
    "rapeseed": {"n_cap": 120.0, "p_cap": 20.0, "k_cap": 40.0, "irrig_cap_mm": 80.0},
    "rice": {"n_cap": 100.0, "p_cap": 20.0, "k_cap": 20.0, "irrig_cap_mm": 160.0},
    "sorghum": {"n_cap": 70.0, "irrig_cap_mm": 70.0},
    "sugarbeet": {"n_cap": 100.0, "p_cap": 20.0, "k_cap": 60.0, "irrig_cap_mm": 120.0},
    "sunflower": {"n_cap": 60.0, "p_cap": 20.0, "k_cap": 20.0, "irrig_cap_mm": 80.0},
    "sweetpotato": {"n_cap": 40.0, "p_cap": 20.0, "k_cap": 40.0, "irrig_cap_mm": 80.0},
    "wheat": {"n_cap": 100.0, "p_cap": 20.0, "k_cap": 20.0, "irrig_cap_mm": 80.0},
}

for _pulse_crop in PULSE_LEGUMES:
    WOFOST_EXTENSION_GUIDELINES.setdefault(
        _pulse_crop,
        {
            "n_cap": 0.0,
            "starter_n_cap": 0.0,
            "p_cap": 20.0,
            "k_cap": 20.0,
            "irrig_cap_mm": 60.0,
            "irrig_trigger": 0.29,
            "min_fert_temp": 6.0,
        },
    )


def _normalize_mode(mode: str) -> str:
    normalized = str(mode or "extension_guideline").strip().lower().replace(" ", "_")
    normalized = MODE_ALIASES.get(normalized, normalized)
    if normalized not in VALID_MODES:
        valid = ", ".join(sorted(VALID_MODES | set(MODE_ALIASES)))
        raise ValueError(f"Unsupported extension-guideline mode {mode!r}. Expected one of: {valid}")
    return normalized


def _numeric_obs_value(payload: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        for candidate in (key, key.upper(), key.lower()):
            if candidate not in payload:
                continue
            try:
                value = float(payload[candidate])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    return float(default)


def _config_value(env: Any, key: str, default: Any = None) -> Any:
    for source in (getattr(env, "config", None), getattr(env, "env_config", None)):
        if source is None:
            continue
        if isinstance(source, dict) and key in source:
            return source[key]
        if hasattr(source, key):
            return getattr(source, key)
    return default


def _objective_id(env: Any) -> str:
    value = _config_value(env, "objective_id")
    if value is None:
        reward_wrapper = str(_config_value(env, "env_reward", "") or "").lower()
        if "wso" in reward_wrapper or "yield" in reward_wrapper:
            value = "yield_max"
    return str(value or "profit_max").strip().lower()


def _crop_name(env: Any) -> str:
    value = _config_value(env, "crop_name") or _config_value(env, "cultivar")
    return str(value or "unknown").strip().lower()


def _wofost_guideline_for_crop(crop_name: str, objective: str) -> Dict[str, float]:
    guideline = dict(WOFOST_EXTENSION_GUIDELINES["__default__"])
    guideline.update(WOFOST_EXTENSION_GUIDELINES.get(crop_name, {}))

    if objective == "yield_max":
        for key in ("n_cap", "p_cap", "k_cap", "irrig_cap_mm"):
            guideline[key] = float(guideline.get(key, 0.0)) * 1.15
        guideline["irrig_trigger"] = float(guideline.get("irrig_trigger", 0.31)) + 0.02
    elif objective == "water_stewardship":
        guideline["irrig_cap_mm"] = float(guideline.get("irrig_cap_mm", 0.0)) * 0.60
        guideline["irrig_trigger"] = float(guideline.get("irrig_trigger", 0.31)) - 0.03
    elif objective == "nutrient_stewardship":
        for key in ("n_cap", "p_cap", "k_cap"):
            guideline[key] = float(guideline.get(key, 0.0)) * 0.70
    return guideline


def _wofost_action_id_under_cap(env: Any, kind: str, remaining_amount: float) -> int:
    """Return the largest available WOFOST action that does not exceed a cap."""
    if remaining_amount <= 0.0:
        return 0

    prompt = getattr(env, "prompt_generator", None)
    action_id_fn = getattr(prompt, "_action_id", None)
    if not callable(action_id_fn):
        return 0

    if kind == "irrig":
        step = float(getattr(prompt, "irrig_amount", 0.5))
        max_level = int(getattr(prompt, "num_irrig", 4))
    else:
        step = float(getattr(prompt, "fert_amount", 2.0))
        max_level = int(getattr(prompt, "num_fert", 4))
    if step <= 0.0 or max_level <= 0:
        return 0

    level = min(max_level, int(math.floor((float(remaining_amount) + 1e-9) / step)))
    if level <= 0:
        return 0
    action_id = action_id_fn(kind, level)
    return int(action_id) if action_id is not None else 0


def _wofost_irrig_action_under_cap(env: Any, remaining_mm: float) -> int:
    return _wofost_action_id_under_cap(env, "irrig", float(remaining_mm) / 10.0)


def _wofost_stage_n_target(n_cap: float, dvs: float, objective: str) -> float:
    if dvs < 0.15:
        return 0.0
    if dvs < 0.45:
        return 0.40 * n_cap
    if dvs < 0.90:
        return 0.80 * n_cap
    if objective == "yield_max" and dvs < 1.15:
        return n_cap
    return 0.0


def select_extension_guideline_action(env: Any, action_space: Any, observation: Dict[str, Any]) -> Any:
    env_name = str(_config_value(env, "env_name", "") or "").lower()
    class_name = type(env).__name__.lower()
    if env_name != "wofost_gym" and "wofost" not in class_name:
        return _zero_action(action_space)

    prompt = getattr(env, "prompt_generator", None)
    available = set(getattr(prompt, "available_action_kinds", ()) or ())
    crop = _crop_name(env)
    objective = _objective_id(env)
    guideline = _wofost_guideline_for_crop(crop, objective)

    dvs = _numeric_obs_value(observation, "DVS", default=0.0)
    temp = _numeric_obs_value(observation, "TEMP", default=20.0)
    rain_cm = _numeric_obs_value(observation, "RAIN", default=0.0)
    sm = _numeric_obs_value(observation, "SM", default=1.0)
    navail = _numeric_obs_value(observation, "NAVAIL", default=999.0)
    pavail = _numeric_obs_value(observation, "PAVAIL", default=999.0)
    kavail = _numeric_obs_value(observation, "KAVAIL", default=999.0)
    total_n = _numeric_obs_value(observation, "TOTN", "total_n_kg_ha", default=0.0)
    total_p = _numeric_obs_value(observation, "TOTP", "total_p_kg_ha", default=0.0)
    total_k = _numeric_obs_value(observation, "TOTK", "total_k_kg_ha", default=0.0)
    total_irrig_mm = _numeric_obs_value(observation, "total_irrig_mm", default=float("nan"))
    if not math.isfinite(total_irrig_mm):
        total_irrig_mm = 10.0 * _numeric_obs_value(
            observation,
            "TOTIRRIG",
            "total_irrig_cm",
            default=0.0,
        )

    if dvs < 0.0:
        return 0

    n_cap = float(guideline.get("n_cap", 0.0))
    p_cap = float(guideline.get("p_cap", 0.0))
    k_cap = float(guideline.get("k_cap", 0.0))
    irrig_cap_mm = float(guideline.get("irrig_cap_mm", 0.0))
    irrig_trigger = float(guideline.get("irrig_trigger", 0.31))
    min_fert_temp = float(guideline.get("min_fert_temp", 4.0))
    low_rain = rain_cm < 0.20
    can_irrigate = "irrig" in available and 0.15 <= dvs < 1.35 and total_irrig_mm < irrig_cap_mm

    if can_irrigate and low_rain and sm < irrig_trigger - 0.04:
        return _wofost_irrig_action_under_cap(env, irrig_cap_mm - total_irrig_mm)

    can_fertilize = temp >= min_fert_temp and 0.05 <= dvs < 1.10
    if can_fertilize and 0.05 <= dvs < 0.55:
        if "p" in available and pavail < float(guideline.get("p_threshold", 5.0)) and total_p < p_cap:
            return _wofost_action_id_under_cap(env, "p", p_cap - total_p)
        if "k" in available and kavail < float(guideline.get("k_threshold", 5.0)) and total_k < k_cap:
            return _wofost_action_id_under_cap(env, "k", k_cap - total_k)

    if can_fertilize and "n" in available:
        if crop in PULSE_LEGUMES:
            n_target = float(guideline.get("starter_n_cap", 0.0)) if dvs < 0.35 else 0.0
        else:
            n_target = _wofost_stage_n_target(n_cap, dvs, objective)
            if navail > 25.0:
                n_target = min(n_target, total_n)
        if n_target > total_n:
            return _wofost_action_id_under_cap(env, "n", min(n_cap, n_target) - total_n)

    if can_irrigate and low_rain and sm < irrig_trigger:
        return _wofost_irrig_action_under_cap(env, irrig_cap_mm - total_irrig_mm)

    return 0


def _select_action(mode: str, env: Any, action_space: Any, observation: Dict[str, Any]) -> Any:
    if mode == "no_action":
        return _zero_action(action_space)
    return select_extension_guideline_action(env, action_space, observation or {})


def _run_single_trial(
    *,
    envs: List[Any],
    env_configs: List[Dict[str, Any]],
    action_spaces: List[Any],
    turn_num: int,
    mode: str,
) -> List[Dict[str, Any]]:
    results = [{"env_id": i, "env_config": env_configs[i], "turns": []} for i in range(len(envs))]

    observations = []
    done_flags = []
    for env in tqdm(envs, desc="Resetting"):
        _, info = env.reset()
        info = info or {}
        observations.append(info.get("observation") or info.get("turn_metrics") or {})
        done_flags.append(False)

    for turn in tqdm(range(turn_num), desc="Turns", leave=False):
        active_indices = [i for i, done in enumerate(done_flags) if not done]
        if not active_indices:
            break

        for i in active_indices:
            current_observation = observations[i]
            action = _select_action(mode, envs[i], action_spaces[i], current_observation or {})

            try:
                _, reward, done, info = envs[i].step(action)
                info = info or {}
                executed_action = _executed_action(info, action)
                action_id = _scalar_action_id(executed_action)
                turn_record = {
                    "turn": turn,
                    "turn_prompt": None,
                    "observation": current_observation,
                    "raw_llm_response": None,
                    "llm_reasoning": None,
                    "retries": 0,
                    "executed_action": executed_action,
                    "executed_action_id": action_id,
                    "invalid_action": bool(info.get("invalid_action", False)),
                    "reward": float(reward),
                    "done": bool(done),
                    "post_turn_metrics": info.get("turn_metrics", {}),
                }
                if done:
                    turn_record["trajectory_metrics"] = info.get("trajectory_metrics", {})
                results[i]["turns"].append(turn_record)

                if done:
                    done_flags[i] = True
                    observations[i] = None
                else:
                    observations[i] = info.get("observation") or info.get("turn_metrics") or {}
            except Exception as exc:
                results[i]["turns"].append(
                    {
                        "turn": turn,
                        "turn_prompt": None,
                        "observation": current_observation,
                        "raw_llm_response": None,
                        "llm_reasoning": None,
                        "retries": 0,
                        "executed_action": action,
                        "executed_action_id": _scalar_action_id(action),
                        "invalid_action": None,
                        "reward": None,
                        "done": True,
                        "error": str(exc),
                        "post_turn_metrics": {},
                    }
                )
                done_flags[i] = True
                observations[i] = None

        if all(done_flags):
            break

    return results


def run_extension_guideline_rollout(
    output_dir: str,
    *,
    mode: str = "extension_guideline",
    seed: int = 42,
    num_trials: int = 1,
    validation_axis: str | None = None,
    test_files: str | list[str] | None = None,
    env_configs: List[Dict[str, Any]] | None = None,
    env_name: str = "wofost_gym",
) -> None:
    mode = _normalize_mode(mode)
    num_trials = max(1, int(num_trials))

    if env_configs is None:
        if not test_files:
            raise ValueError("run_extension_guideline_rollout requires test_files or env_configs.")
        env_configs, env_name = load_dataset(test_files)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)

    envs = []
    configs = []
    action_spaces = []
    for env_config in tqdm(env_configs, desc="Creating envs"):
        row_env_name = env_config.get("env_name", env_name)
        env, config = create_env(row_env_name, env_config)
        envs.append(env)
        configs.append(config)
        action_spaces.append(get_action_space(env))

    turn_num = max(int(config.turn_num) for config in configs)
    all_trial_metrics = []
    try:
        for trial in range(num_trials):
            trial_seed = int(seed) + trial
            random.seed(trial_seed)
            np.random.seed(trial_seed)
            _seed_action_spaces(action_spaces, trial_seed)

            results = _run_single_trial(
                envs=envs,
                env_configs=env_configs,
                action_spaces=action_spaces,
                turn_num=turn_num,
                mode=mode,
            )

            trial_dir = output_path if num_trials == 1 else output_path / f"trial_{trial}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            _dump_json_atomic(trial_dir / "results.json", results, "extension guideline results")

            trial_metrics = _extract_trial_metrics(results)
            trial_metrics["baseline_mode"] = mode
            trial_metrics["seed"] = trial_seed
            all_trial_metrics.append(trial_metrics)
            _dump_json_atomic(trial_dir / "metrics.json", trial_metrics, "extension guideline metrics")

            validation_metrics = _extract_validation_metric_dict(results, validation_axis=validation_axis)
            _dump_json_atomic(
                trial_dir / "validation_metrics.json",
                validation_metrics,
                "training-style validation metrics",
            )

        if num_trials > 1:
            summary = _build_summary(all_trial_metrics)
            summary["baseline_mode"] = mode
            summary["seed"] = int(seed)
            _dump_json_atomic(output_path / "summary.json", summary, "cross-trial summary")
    finally:
        for env in envs:
            close = getattr(env, "close", None)
            if callable(close):
                close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run no-action or extension-guideline WOFOST baseline.")
    parser.add_argument("--data", required=True, help="Parquet dataset file.")
    parser.add_argument("--output-dir", required=True, help="Directory for results.")
    parser.add_argument("--mode", default="extension_guideline", help="no_action or extension_guideline.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--validation-axis", default=None)
    args = parser.parse_args()

    run_extension_guideline_rollout(
        output_dir=args.output_dir,
        mode=args.mode,
        seed=args.seed,
        num_trials=args.num_trials,
        validation_axis=args.validation_axis,
        test_files=args.data,
    )


if __name__ == "__main__":
    main()
