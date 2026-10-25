"""CPU smoke rollout for the pollution-penalized CycleGym objective.

Runs short episodes of ``objective_id: pollution_penalized_revenue`` on the
real Cycles simulator and checks that the logged reward components add up:

- crop planning (the T1.3 rotation env, 4 seasons): CycleGym's default action
  (corn, planting week 0; the rotation env has no fertilizer action) and a
  fixed corn/soybean alternation;
- corn crop growth (1 season, weekly N decisions): no fertilizer and a fixed
  schedule of 60 kg N/ha in each of the first four weeks.

Usage (needs a CycleGym checkout with the Cycles binary, see
docs/benchmark/reward_objectives.md):

    python smoke_tests/cycles_gym/pollution_reward_smoke.py \
        --cycles-gym-path ../AgriManagerExternal/CyclesGym
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agrimanager.env.cycles_gym.env import CyclesEnv  # noqa: E402
from agrimanager.env.cycles_gym.env_config import CyclesEnvConfig  # noqa: E402
from agrimanager.env.cycles_gym.n_loss import (  # noqa: E402
    N_LOSS_METRIC_SUFFIX,
    POLLUTION_OBJECTIVE_ID,
    n_loss_totals,
)

TOLERANCE = 1e-6

# Shaping bonuses and the reward scale copied from the T1.3 configs, so the
# smoke exercises the same scalar-reward path as T1.3 training.
T13_REWARD_PARAMS = {
    "crop_planning_same_crop_penalty": 0.05,
    "crop_planning_same_crop_penalty_cap": 0.4,
    "crop_planning_cereal_run_penalty": 0.02,
    "crop_planning_legume_break_bonus": 0.03,
    "crop_planning_cereal_after_legume_bonus": 0.02,
    "crop_planning_change_crop_bonus": 0.01,
}


def _crop_planning_config(cycles_gym_path: str, runtime_path: str) -> CyclesEnvConfig:
    return CyclesEnvConfig(
        env_id="CropPlanningRockSpringsRW-v1",
        cycles_gym_path=cycles_gym_path,
        cycles_runtime_path=runtime_path,
        llm_mode=False,
        objective_id=POLLUTION_OBJECTIVE_ID,
        reward_scale=1000.0,
        reward_params=dict(T13_REWARD_PARAMS),
        turn_num=4,
        seed=300000,
        env_kwargs={
            "start_year": 1986,
            "end_year": 1989,
            "rotation_crops": ["CornRM.100", "SoybeanMG.3"],
            "crop_prices": {"CornRM.100": 250.0, "SoybeanMG.3": 300.0},
        },
    )


def _corn_config(cycles_gym_path: str, runtime_path: str) -> CyclesEnvConfig:
    return CyclesEnvConfig(
        env_id="CornShortRockSpringsFW-v1",
        cycles_gym_path=cycles_gym_path,
        cycles_runtime_path=runtime_path,
        llm_mode=False,
        objective_id=POLLUTION_OBJECTIVE_ID,
        reward_scale=1000.0,
        turn_num=200,
        seed=1,
    )


def _rollout(env: CyclesEnv, policy: Callable[[int], Any]) -> Dict[str, Any]:
    env.reset()
    start_date = getattr(env.env, "date", None)
    steps: List[Dict[str, float]] = []
    done = False
    step_idx = 0
    while not done:
        _, reward, done, info = env.step(policy(step_idx))
        step_idx += 1
        tm = info["turn_metrics"]
        steps.append(
            {
                "reward": float(reward),
                "normalized_env_reward": float(tm["normalized_env_reward"]),
                "format_reward": float(tm["format_reward"]),
                "shaping_reward": float(tm["shaping_reward"]),
                "n_loss_penalty_reward": float(tm["n_loss_penalty_reward"]),
                "n_loss_pen_usd": float(tm["n_loss_pen_usd"]),
                "n_loss_kg_ha": float(tm["n_loss_kg_ha"]),
                "n_loss_terms_kg_ha": float(
                    sum(tm[f"n_loss__{suffix}"] for suffix in N_LOSS_METRIC_SUFFIX.values())
                ),
            }
        )
    end_date = getattr(env.env, "date", None)
    traj = env.get_trajectory_metrics()

    # Independent check: the per-step windows must tile the horizon, so the
    # episode totals equal one direct sum over every simulated day.
    soil_n_df = env.env.soil_n_manager.soil_n_state
    direct = n_loss_totals(
        soil_n_df,
        start_date,
        end_date,
        include_bypass=env._n_loss_include_bypass,
    )
    return {
        "steps": steps,
        "traj": {k: v for k, v in traj.items() if isinstance(v, (int, float))},
        "direct_n_loss_kg_ha": float(sum(direct.values())),
        "start_date": str(start_date),
        "end_date": str(end_date),
    }


def _check(result: Dict[str, Any]) -> Dict[str, float]:
    steps = result["steps"]
    traj = result["traj"]
    sum_reward = sum(s["reward"] for s in steps)
    sum_components = sum(
        s["normalized_env_reward"] + s["format_reward"] + s["shaping_reward"] - s["n_loss_penalty_reward"]
        for s in steps
    )
    checks = {
        "sum_step_reward": sum_reward,
        "sum_components": sum_components,
        "episode_reward": traj["episode_reward"],
        "n_loss_pen_usd": traj["n_loss_pen_usd"],
        "sum_step_pen_usd": sum(s["n_loss_pen_usd"] for s in steps),
        "n_loss_kg_ha": traj["n_loss_kg_ha"],
        "direct_n_loss_kg_ha": result["direct_n_loss_kg_ha"],
    }
    assert abs(sum_reward - sum_components) < TOLERANCE, checks
    assert abs(sum_reward - traj["episode_reward"]) < TOLERANCE, checks
    assert abs(checks["n_loss_pen_usd"] - checks["sum_step_pen_usd"]) < TOLERANCE, checks
    assert abs(checks["n_loss_kg_ha"] - checks["direct_n_loss_kg_ha"]) < TOLERANCE, checks
    for step in steps:
        assert abs(step["n_loss_kg_ha"] - step["n_loss_terms_kg_ha"]) < TOLERANCE, step
        assert abs(step["n_loss_penalty_reward"] - step["n_loss_pen_usd"] / 1000.0) < TOLERANCE, step
    if "pen_net_revenue" in traj:
        expected = traj["cumulative_gross_revenue"] - traj["n_loss_pen_usd"]
        checks["cumulative_gross_revenue"] = traj["cumulative_gross_revenue"]
        checks["pen_net_revenue"] = traj["pen_net_revenue"]
        assert abs(traj["pen_net_revenue"] - expected) < TOLERANCE, checks
    return checks


def run_smoke(cycles_gym_path: str, runtime_path: str | None = None) -> Dict[str, Dict[str, Any]]:
    runtime_path = runtime_path or tempfile.mkdtemp(prefix="agrimanager_cycles_pollution_")
    policies = {
        "crop_planning/default_action": (
            _crop_planning_config,
            lambda i: [0, 0],
        ),
        "crop_planning/corn_soy_alternation": (
            _crop_planning_config,
            lambda i: [i % 2, 4],
        ),
        "corn/no_fertilizer": (
            _corn_config,
            lambda i: 0,
        ),
        # Action 4 of 11 with maxN=150 is 60 kg N/ha.
        "corn/fixed_60kgN_weeks_0_3": (
            _corn_config,
            lambda i: 4 if i < 4 else 0,
        ),
    }
    results: Dict[str, Dict[str, Any]] = {}
    for name, (make_config, policy) in policies.items():
        env = CyclesEnv(make_config(cycles_gym_path, runtime_path))
        try:
            result = _rollout(env, policy)
        finally:
            env.close()
        result["checks"] = _check(result)
        results[name] = result
    return results


def _summary_rows(results: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for name, result in results.items():
        traj = result["traj"]
        row = {
            "policy": name,
            "steps": len(result["steps"]),
            "native_episode_reward": traj["native_episode_reward"],
            "n_loss_kg_ha": traj["n_loss_kg_ha"],
            "n_loss_pen_usd": traj["n_loss_pen_usd"],
            "episode_reward": traj["episode_reward"],
            "total_n_kg_ha": traj.get("total_n_kg_ha"),
        }
        for suffix in N_LOSS_METRIC_SUFFIX.values():
            row[f"n_loss__{suffix}"] = traj[f"n_loss__{suffix}"]
        if "pen_net_revenue" in traj:
            row["cumulative_gross_revenue"] = traj["cumulative_gross_revenue"]
            row["pen_net_revenue"] = traj["pen_net_revenue"]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    from agrimanager.env.cycles_gym.env_config import DEFAULT_CYCLES_GYM_PATH

    parser.add_argument("--cycles-gym-path", default=DEFAULT_CYCLES_GYM_PATH,
                        help="CycleGym checkout (default: $CYCLES_GYM_PATH or ../AgriManagerExternal/CyclesGym)")
    parser.add_argument("--runtime-path", default=None)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    results = run_smoke(args.cycles_gym_path, args.runtime_path)
    rows = _summary_rows(results)
    for row in rows:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
    print("all component checks passed")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"generated": datetime.datetime.now().isoformat(), "results": results}, indent=1, default=str),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
