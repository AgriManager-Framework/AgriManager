"""Tests for the pollution-penalized CycleGym objective.

The unit tests drive ``CyclesEnv`` with a small fake crop-planning simulator,
so they run without CycleGym. ``test_real_cycles_smoke_rollout`` runs the
CPU smoke in ``smoke_tests/cycles_gym/pollution_reward_smoke.py`` on the real
simulator and is skipped when no CycleGym checkout with the Cycles binary is
found (set ``CYCLES_GYM_PATH``).
"""

import datetime
import importlib.util
import math
import os
from pathlib import Path

import pandas as pd
import pytest

from agrimanager.env.cycles_gym.env import CyclesEnv
from agrimanager.env.cycles_gym.env_config import (
    DEFAULT_CYCLES_GYM_PATH,
    CyclesEnvConfig,
    _looks_like_cycles_gym_root,
)
from agrimanager.env.cycles_gym.n_loss import (
    DEFAULT_N_LOSS_COSTS_USD_PER_KG_N,
    N_LOSS_METRIC_SUFFIX,
    POLLUTION_OBJECTIVE_ID,
    n_loss_penalty,
    n_loss_penalty_params,
    n_loss_totals,
)
from agrimanager.env.cycles_gym.prompt import CropPlanningPromptGenerator

REPO_ROOT = Path(__file__).resolve().parents[1]

N_COLUMNS = [
    "ORG SOIL N", "PROF SOIL NO3", "PROF SOIL NH4", "MINERALIZATION", "IMMOBILIZATION",
    "NET MINERALIZ", "NH4 NITRIFICAT", "N2O FROM NITRIF", "NH3 VOLATILIZ", "NO3 DENITRIF",
    "N2O FROM DENIT", "NO3 LEACHING", "NH4 LEACHING", "NO3 BYPASS", "NH4 BYPASS",
]

# Daily N-loss fluxes (kg N/ha/day) used by the synthetic N.dat below.
DAILY_FLUX = {
    "NO3 LEACHING": 0.02,
    "NH4 LEACHING": 0.001,
    "NO3 BYPASS": 0.005,
    "NH4 BYPASS": 0.0005,
    "NH3 VOLATILIZ": 0.01,
    "N2O FROM NITRIF": 0.0004,
    "N2O FROM DENIT": 0.0006,
    "NO3 DENITRIF": 0.03,
}


def _soil_n_frame(start_year, end_year, flux_scale_by_year=None):
    rows = []
    day = datetime.date(start_year, 1, 1)
    while day.year <= end_year:
        scale = (flux_scale_by_year or {}).get(day.year, 1.0)
        row = {column: 0.0 for column in N_COLUMNS}
        row["ORG SOIL N"] = 8000.0
        for column, value in DAILY_FLUX.items():
            row[column] = value * scale
        row["YEAR"] = day.year
        row["DOY"] = day.timetuple().tm_yday
        rows.append(row)
        day += datetime.timedelta(days=1)
    return pd.DataFrame(rows)[["YEAR", "DOY", *N_COLUMNS]]


def _year_days(year):
    return (datetime.date(year + 1, 1, 1) - datetime.date(year, 1, 1)).days


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------


def test_penalty_params_defaults_and_overrides():
    defaults = n_loss_penalty_params({})
    assert defaults["costs"] == DEFAULT_N_LOSS_COSTS_USD_PER_KG_N
    assert defaults["include_bypass"] is True
    assert set(defaults["costs"]) == {"no3_leaching", "nh4_leaching", "nh3_volatilization", "n2o_emission"}

    shared = n_loss_penalty_params({"n_loss_cost_usd_per_kg_n": 2.0, "n_loss_cost_n2o_emission_usd_per_kg_n": 30.0})
    assert shared["costs"] == {
        "no3_leaching": 2.0,
        "nh4_leaching": 2.0,
        "nh3_volatilization": 2.0,
        "n2o_emission": 30.0,
    }

    short_keys = n_loss_penalty_params({"n_loss_cost_no3_leaching": 0.0, "n_loss_include_bypass": "false"})
    assert short_keys["costs"]["no3_leaching"] == 0.0
    assert short_keys["include_bypass"] is False

    with pytest.raises(ValueError):
        n_loss_penalty_params({"n_loss_cost_nh3_volatilization_usd_per_kg_n": -1.0})


def test_n_loss_totals_half_open_window_and_bypass():
    df = _soil_n_frame(1990, 1991)
    year = n_loss_totals(df, datetime.date(1990, 1, 1), datetime.date(1991, 1, 1))
    days = _year_days(1990)
    assert year["no3_leaching"] == pytest.approx(days * (0.02 + 0.005))
    assert year["nh4_leaching"] == pytest.approx(days * (0.001 + 0.0005))
    assert year["nh3_volatilization"] == pytest.approx(days * 0.01)
    assert year["n2o_emission"] == pytest.approx(days * (0.0004 + 0.0006))
    # Total denitrification is not a penalty term.
    assert "denitrification" not in year

    no_bypass = n_loss_totals(df, datetime.date(1990, 1, 1), datetime.date(1991, 1, 1), include_bypass=False)
    assert no_bypass["no3_leaching"] == pytest.approx(days * 0.02)

    # Consecutive windows tile the horizon without double counting.
    first = n_loss_totals(df, datetime.date(1990, 1, 1), datetime.date(1990, 7, 1))
    second = n_loss_totals(df, datetime.date(1990, 7, 1), datetime.date(1991, 1, 1))
    for term in year:
        assert first[term] + second[term] == pytest.approx(year[term])

    # Days past the simulated table and missing inputs contribute zero.
    beyond = n_loss_totals(df, datetime.date(1992, 1, 1), datetime.date(1993, 1, 1))
    assert all(value == 0.0 for value in beyond.values())
    assert all(value == 0.0 for value in n_loss_totals(None, None, None).values())
    partial = n_loss_totals(df.drop(columns=["NH4 BYPASS", "NH4 LEACHING"]), datetime.date(1990, 1, 1), datetime.date(1991, 1, 1))
    assert partial["nh4_leaching"] == 0.0


def test_n_loss_penalty_is_weighted_sum():
    totals = {"no3_leaching": 3.0, "nh4_leaching": 0.5, "nh3_volatilization": 2.0, "n2o_emission": 0.25}
    costs = {"no3_leaching": 10.0, "nh4_leaching": 4.0, "nh3_volatilization": 1.0, "n2o_emission": 100.0}
    penalty, parts = n_loss_penalty(totals, costs)
    assert parts == {"no3_leaching": 30.0, "nh4_leaching": 2.0, "nh3_volatilization": 2.0, "n2o_emission": 25.0}
    assert penalty == pytest.approx(59.0)


# --------------------------------------------------------------------------
# CyclesEnv with a fake crop-planning simulator
# --------------------------------------------------------------------------


class _FakeSpace:
    nvec = [2, 14]


class _FakeObserver:
    obs_names = ["ORG SOIL N", "PROF SOIL NO3"]


class _FakeManager:
    pass


class _FakeCropPlanning:
    """Minimal stand-in for CycleGym's CropPlanningFixedPlanting.

    One step plants one season; the native reward is gross revenue
    ``yield x price`` and N.dat holds one row per simulated day. Soybean
    seasons have half the N losses of corn seasons.
    """

    YIELDS = {"CornRM.100": 9.0, "SoybeanMG.3": 3.0}

    def __init__(self, start_year=1990, end_year=1992):
        self.rotation_crops = ["CornRM.100", "SoybeanMG.3"]
        self.crop_prices = {"CornRM.100": 200.0, "SoybeanMG.3": 400.0}
        self.action_space = _FakeSpace()
        self.observer = _FakeObserver()
        self.start_year = start_year
        self.end_year = end_year
        self.season_manager = _FakeManager()
        self.soil_n_manager = _FakeManager()
        self.reset()

    def reset(self):
        self.date = datetime.date(self.start_year, 1, 1)
        self._harvests = []
        self.season_manager.season_df = pd.DataFrame(self._harvests, columns=["YEAR", "DOY", "CROP", "GRAIN YIELD"])
        self.soil_n_manager.soil_n_state = _soil_n_frame(self.start_year, self.end_year)
        self._scales = {}
        return [8000.0, 30.0]

    def step(self, action):
        crop = self.rotation_crops[int(action[0])]
        year = self.date.year
        self._scales[year] = 1.0 if crop.startswith("Corn") else 0.5
        self.soil_n_manager.soil_n_state = _soil_n_frame(self.start_year, self.end_year, self._scales)
        self._harvests.append({"YEAR": year, "DOY": 280, "CROP": crop, "GRAIN YIELD": self.YIELDS[crop]})
        self.season_manager.season_df = pd.DataFrame(self._harvests)
        self.date = datetime.date(year + 1, 1, 1)
        reward = self.YIELDS[crop] * self.crop_prices[crop]
        done = self.date.year > self.end_year
        return [8000.0, 30.0], reward, done, {}

    def close(self):
        pass


T13_SHAPING = {
    "crop_planning_same_crop_penalty": 0.05,
    "crop_planning_change_crop_bonus": 0.01,
}


@pytest.fixture
def make_env(monkeypatch, tmp_path):
    monkeypatch.setattr(CyclesEnv, "_prepare_runtime_tree", staticmethod(lambda config: None))
    monkeypatch.setattr(
        CyclesEnv,
        "_create_cyclesgym_env",
        classmethod(lambda cls, config: (None, _FakeCropPlanning())),
    )

    def _make(objective_id="profit_max", llm_mode=False, reward_params=None, **kwargs):
        config = CyclesEnvConfig(
            env_id="CropPlanningRockSpringsRW-v1",
            cycles_gym_path=str(tmp_path),
            cycles_runtime_path=str(tmp_path / "runtime"),
            llm_mode=llm_mode,
            objective_id=objective_id,
            reward_scale=1000.0,
            reward_params=dict(T13_SHAPING, **(reward_params or {})),
            turn_num=3,
            env_kwargs={},
            **kwargs,
        )
        return CyclesEnv(config)

    return _make


def _run(env, actions):
    env.reset()
    steps = []
    for action in actions:
        _, reward, done, info = env.step(action)
        steps.append((reward, info))
    return steps


def _expected_season_penalty(year, scale, costs=DEFAULT_N_LOSS_COSTS_USD_PER_KG_N):
    days = _year_days(year)
    totals = {
        "no3_leaching": days * (0.02 + 0.005) * scale,
        "nh4_leaching": days * (0.001 + 0.0005) * scale,
        "nh3_volatilization": days * 0.01 * scale,
        "n2o_emission": days * (0.0004 + 0.0006) * scale,
    }
    return sum(costs[term] * value for term, value in totals.items()), sum(totals.values())


def test_default_objective_reward_is_unchanged(make_env):
    """profit_max on crop planning stays native revenue / scale + shaping, with no N-loss keys."""
    env = make_env("profit_max")
    steps = _run(env, [[0, 4], [1, 4], [1, 4]])
    native = [9.0 * 200.0, 3.0 * 400.0, 3.0 * 400.0]
    shaping = [0.0, 0.01, -0.05]
    for (reward, info), revenue, shape in zip(steps, native, shaping):
        assert reward == pytest.approx(revenue / 1000.0 + shape)
        assert not any(key.startswith("n_loss") for key in info["turn_metrics"])
    traj = steps[-1][1]["trajectory_metrics"]
    assert not any(key.startswith(("n_loss", "pen_net")) for key in traj)
    assert traj["episode_reward"] == pytest.approx(sum(r for r, _ in steps))


def test_pollution_objective_subtracts_per_season_penalty(make_env):
    env = make_env(POLLUTION_OBJECTIVE_ID)
    steps = _run(env, [[0, 4], [1, 4], [1, 4]])
    native = [9.0 * 200.0, 3.0 * 400.0, 3.0 * 400.0]
    shaping = [0.0, 0.01, -0.05]
    scales = [1.0, 0.5, 0.5]
    total_penalty = 0.0
    total_kg = 0.0
    for idx, ((reward, info), revenue, shape, scale) in enumerate(zip(steps, native, shaping, scales)):
        penalty, kg = _expected_season_penalty(1990 + idx, scale)
        total_penalty += penalty
        total_kg += kg
        tm = info["turn_metrics"]
        assert tm["n_loss_pen_usd"] == pytest.approx(penalty)
        assert tm["n_loss_kg_ha"] == pytest.approx(kg)
        assert tm["n_loss_kg_ha"] == pytest.approx(sum(tm[f"n_loss__{s}"] for s in N_LOSS_METRIC_SUFFIX.values()))
        assert tm["n_loss_pen_usd"] == pytest.approx(sum(tm[f"n_loss_pen__{s}"] for s in N_LOSS_METRIC_SUFFIX.values()))
        assert tm["n_loss_penalty_reward"] == pytest.approx(penalty / 1000.0)
        assert reward == pytest.approx(revenue / 1000.0 + shape - penalty / 1000.0)
        assert reward == pytest.approx(
            tm["normalized_env_reward"] + tm["format_reward"] + tm["shaping_reward"] - tm["n_loss_penalty_reward"]
        )
        assert tm["crop_planning/n_loss_penalty"] == pytest.approx(penalty)
        assert tm["crop_planning/net_revenue"] == pytest.approx(revenue - penalty)

    traj = steps[-1][1]["trajectory_metrics"]
    assert traj["episode_reward"] == pytest.approx(sum(r for r, _ in steps))
    assert traj["n_loss_pen_usd"] == pytest.approx(total_penalty)
    assert traj["n_loss_kg_ha"] == pytest.approx(total_kg)
    assert traj["n_loss_pen_reward"] == pytest.approx(total_penalty / 1000.0)
    assert traj["cumulative_gross_revenue"] == pytest.approx(sum(native))
    assert traj["pen_net_revenue"] == pytest.approx(sum(native) - total_penalty)
    assert traj["pen_net_revenue_norm"] == pytest.approx((sum(native) - total_penalty) / 1000.0)
    assert traj["n_loss_cost__no3_leach"] == pytest.approx(10.0)


def test_pollution_objective_custom_weights_and_zero_weights(make_env):
    weights = {
        "n_loss_cost_no3_leaching_usd_per_kg_n": 2.0,
        "n_loss_cost_nh4_leaching_usd_per_kg_n": 0.0,
        "n_loss_cost_nh3_volatilization_usd_per_kg_n": 5.0,
        "n_loss_cost_n2o_emission_usd_per_kg_n": 50.0,
    }
    env = make_env(POLLUTION_OBJECTIVE_ID, reward_params=weights)
    (reward, info), = _run(env, [[0, 4]])
    costs = {"no3_leaching": 2.0, "nh4_leaching": 0.0, "nh3_volatilization": 5.0, "n2o_emission": 50.0}
    penalty, _ = _expected_season_penalty(1990, 1.0, costs)
    assert info["turn_metrics"]["n_loss_pen_usd"] == pytest.approx(penalty)
    assert reward == pytest.approx(1.8 - penalty / 1000.0)

    zero = make_env(POLLUTION_OBJECTIVE_ID, reward_params={"n_loss_cost_usd_per_kg_n": 0.0})
    (reward_zero, info_zero), = _run(zero, [[0, 4]])
    assert reward_zero == pytest.approx(1.8)
    assert info_zero["turn_metrics"]["n_loss_kg_ha"] > 0.0


def test_pollution_objective_invalid_step_tallies_losses_without_reward(make_env):
    env = make_env(POLLUTION_OBJECTIVE_ID, llm_mode=True, require_think=False)
    env.reset()
    _, reward, _, info = env.step("not an answer")
    tm = info["turn_metrics"]
    assert info["invalid_action"] is True
    assert reward == pytest.approx(0.0)
    assert tm["n_loss_penalty_reward"] == 0.0
    assert tm["n_loss_pen_usd"] > 0.0

    _, reward, _, info = env.step("<answer>1, 4</answer>")
    penalty, _ = _expected_season_penalty(1991, 0.5)
    # Soybean after the fallback corn season: format bonus 0.1, change-crop bonus 0.01.
    assert reward == pytest.approx(1.2 + 0.1 + 0.01 - penalty / 1000.0)
    _, _, done, info = env.step("<answer>0, 4</answer>")
    assert done
    traj = info["trajectory_metrics"]
    assert traj["invalid_steps"] == 1.0
    # The penalty in the score covers valid steps; the reported total covers all seasons.
    assert traj["n_loss_pen_reward"] < traj["n_loss_pen_usd"] / 1000.0
    assert traj["pen_net_revenue"] == pytest.approx(traj["cumulative_gross_revenue"] - traj["n_loss_pen_usd"])


def test_pollution_objective_rejects_yield_only_reward_mode(make_env):
    with pytest.raises(ValueError, match=POLLUTION_OBJECTIVE_ID):
        make_env(POLLUTION_OBJECTIVE_ID, reward_mode="final_yield")


def test_config_round_trip_keeps_objective_and_weights():
    config = CyclesEnvConfig(
        env_id="CropPlanningRockSpringsRW-v1",
        objective_id=POLLUTION_OBJECTIVE_ID,
        reward_params={"n_loss_cost_usd_per_kg_n": 3.0},
    )
    as_dict = config.to_dict()
    assert as_dict["objective_id"] == POLLUTION_OBJECTIVE_ID
    assert as_dict["reward_params"] == {"n_loss_cost_usd_per_kg_n": 3.0}


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------


def _prompt_generator(**kwargs):
    return CropPlanningPromptGenerator(
        rotation_crops=["CornRM.100", "SoybeanMG.3"],
        action_nvec=[2, 14],
        obs_names=["ORG SOIL N"],
        require_think=True,
        think_tag="tool_call",
        **kwargs,
    )


PROMPT_CONTEXT = {
    "current_year": 1991,
    "years_remaining": 2,
    "crop_prices": {"CornRM.100": 200.0, "SoybeanMG.3": 400.0},
    "past_trajectory": [
        {"year": 1990, "crop": "CornRM.100", "planting_week": 4, "yield_tonnes": 9.0, "revenue": 1800.0,
         "n_loss_kg_ha": 13.0, "n_loss_penalty": 130.0},
    ],
}


def test_default_crop_planning_prompt_is_unchanged():
    baseline = _prompt_generator()
    for objective_id in (None, "profit_max", "yield_max"):
        other = _prompt_generator(objective_id=objective_id, reward_params={"n_loss_cost_usd_per_kg_n": 99.0})
        assert other.get_system_prompt() == baseline.get_system_prompt()
        assert other.get_turn_prompt([8000.0], PROMPT_CONTEXT) == baseline.get_turn_prompt([8000.0], PROMPT_CONTEXT)
    turn = baseline.get_turn_prompt([8000.0], PROMPT_CONTEXT)
    assert "management objective" not in turn
    assert "n_loss" not in turn
    assert baseline.get_system_prompt().startswith(
        "You are an agricultural planning expert. Your goal is to maximize long-run gross revenue "
        "over a multi-year crop rotation"
    )


def test_pollution_prompt_states_objective_and_weights():
    generator = _prompt_generator(
        objective_id=POLLUTION_OBJECTIVE_ID,
        reward_params={"n_loss_cost_n2o_emission_usd_per_kg_n": 25.0, "n_loss_cost_nh4_leaching_usd_per_kg_n": 0.0},
    )
    system = generator.get_system_prompt()
    turn = generator.get_turn_prompt([8000.0], PROMPT_CONTEXT)
    assert "nitrogen-loss penalty" in system
    assert f'<management objective id="{POLLUTION_OBJECTIVE_ID}">' in turn
    assert "N2O emission from nitrification and denitrification: 25 dollars per kg N/ha." in turn
    assert "Nitrate (NO3) leaching (including bypass flow): 10 dollars per kg N/ha." in turn
    assert "Ammonium" not in turn  # zero-weight terms are not listed
    assert "cumulative nitrogen-loss penalty: 130.00; cumulative net score: 1670.00." in turn
    assert "n_loss_kg_ha: 13.00, n_loss_penalty: 130.00" in turn


# --------------------------------------------------------------------------
# Real simulator smoke (skipped without CycleGym)
# --------------------------------------------------------------------------


def _cycles_gym_root():
    for candidate in (os.environ.get("CYCLES_GYM_PATH"), DEFAULT_CYCLES_GYM_PATH):
        if candidate and _looks_like_cycles_gym_root(Path(candidate)):
            return str(Path(candidate).resolve())
    return None


@pytest.mark.skipif(_cycles_gym_root() is None, reason="CycleGym checkout with the Cycles binary not found")
def test_real_cycles_smoke_rollout(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "pollution_reward_smoke",
        REPO_ROOT / "smoke_tests" / "cycles_gym" / "pollution_reward_smoke.py",
    )
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)

    results = smoke.run_smoke(_cycles_gym_root(), str(tmp_path / "runtime"))

    for name, result in results.items():
        traj = result["traj"]
        assert traj["n_loss_kg_ha"] > 0.0, name
        assert traj["n_loss_pen_usd"] > 0.0, name
        unpenalized = sum(
            s["normalized_env_reward"] + s["format_reward"] + s["shaping_reward"] for s in result["steps"]
        )
        assert traj["episode_reward"] + traj["n_loss_pen_reward"] == pytest.approx(unpenalized), name
    no_n = results["corn/no_fertilizer"]["traj"]
    fertilized = results["corn/fixed_60kgN_weeks_0_3"]["traj"]
    assert fertilized["total_n_kg_ha"] == pytest.approx(240.0)
    assert fertilized["n_loss_kg_ha"] > no_n["n_loss_kg_ha"]
    assert fertilized["n_loss_pen_usd"] > no_n["n_loss_pen_usd"]
    assert math.isfinite(results["crop_planning/corn_soy_alternation"]["traj"]["pen_net_revenue"])


def test_defaulted_terms_and_notice(caplog):
    import logging

    from agrimanager.env.cycles_gym import n_loss as n_loss_module

    params = n_loss_module.n_loss_penalty_params({"n_loss_cost_no3_leaching_usd_per_kg_n": 5.0})
    assert params["defaulted_terms"] == ["nh4_leaching", "nh3_volatilization", "n2o_emission"]
    assert n_loss_module.n_loss_penalty_params({"n_loss_cost_usd_per_kg_n": 1.0})["defaulted_terms"] == []

    n_loss_module._DEFAULT_WEIGHT_NOTICE_SHOWN = False
    with caplog.at_level(logging.INFO, logger=n_loss_module.__name__):
        n_loss_module.log_default_weights_once(params)
        n_loss_module.log_default_weights_once(params)
    notices = [r for r in caplog.records if "illustrative default" in r.getMessage()]
    assert len(notices) == 1
