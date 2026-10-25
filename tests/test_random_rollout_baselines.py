import sys
from types import SimpleNamespace

import numpy as np
import pytest

from agrimanager.rollout.inference.inference_rollout import _extract_validation_metric_dict
from agrimanager.rollout.inference.random_rollout import (
    create_env,
    _flatten_numeric,
    _log_trial_to_wandb,
    _normalize_mode,
    _run_single_trial,
    _zero_action,
)


class _DiscreteSpace:
    def __init__(self, action: int = 3):
        self.action = action

    def sample(self):
        return self.action


class _MultiDiscreteSpace:
    nvec = np.array([2, 3, 4])


class _FakeEnv:
    def __init__(self):
        self.actions = []
        self.steps = 0

    def reset(self):
        self.steps = 0
        return "prompt", {"observation": {"step": 0}}

    def step(self, action):
        self.steps += 1
        action_id = int(action)
        self.actions.append(action_id)
        done = self.steps >= 2
        info = {
            "observation": {"step": self.steps},
            "executed_action_id": action_id,
            "invalid_action": False,
            "turn_metrics": {"wso": float(action_id), "reward": float(action_id)},
        }
        if done:
            info["trajectory_metrics"] = {
                "target_yield": float(sum(self.actions)),
                "invalid_action_rate": 0.0,
                "total_steps": float(self.steps),
            }
        return "next prompt", float(action_id), done, info


def test_no_action_uses_native_zero_action():
    assert _normalize_mode("no-action") == "no_action"
    np.testing.assert_array_equal(_zero_action(_MultiDiscreteSpace()), np.array([0, 0, 0]))

    env = _FakeEnv()
    results = _run_single_trial(
        envs=[env],
        env_configs=[{}],
        action_spaces=[_DiscreteSpace(action=3)],
        turn_num=2,
        mode="no_action",
    )

    assert env.actions == [0, 0]
    assert results[0]["turns"][-1]["trajectory_metrics"]["target_yield"] == 0.0


def test_random_action_samples_action_space():
    env = _FakeEnv()
    results = _run_single_trial(
        envs=[env],
        env_configs=[{}],
        action_spaces=[_DiscreteSpace(action=3)],
        turn_num=2,
        mode="random",
    )

    assert env.actions == [3, 3]
    assert results[0]["turns"][-1]["trajectory_metrics"]["target_yield"] == 6.0


def test_ppv0_mode_uses_native_zero_action():
    assert _normalize_mode("pp-v0") == "ppv0"

    env = _FakeEnv()
    _run_single_trial(
        envs=[env],
        env_configs=[{}],
        action_spaces=[_DiscreteSpace(action=3)],
        turn_num=1,
        mode="ppv0",
    )

    assert env.actions == [0]


def test_ag_heuristic_mode_dispatches_policy(monkeypatch):
    assert _normalize_mode("ag-heuristic") == "ag_heuristic"

    from agrimanager.rollout.inference import ag_heuristic_rollout

    def fake_select_ag_heuristic_action(env, action_space, observation):
        assert env is not None
        assert action_space is not None
        assert observation == {"step": 0}
        return 2

    monkeypatch.setattr(
        ag_heuristic_rollout,
        "select_ag_heuristic_action",
        fake_select_ag_heuristic_action,
    )

    env = _FakeEnv()
    results = _run_single_trial(
        envs=[env],
        env_configs=[{}],
        action_spaces=[_DiscreteSpace(action=3)],
        turn_num=1,
        mode="ag_heuristic",
    )

    assert env.actions == [2]
    assert results[0]["turns"][-1]["executed_action_id"] == 2


def test_ppv0_create_env_switches_to_potential_production_env(monkeypatch):
    captured = {}

    def fake_create_environment(env_name, env_config):
        captured["env_name"] = env_name
        captured["env_config"] = dict(env_config)
        return object(), env_config

    monkeypatch.setattr(
        "agrimanager.rollout.inference.random_rollout.create_environment",
        fake_create_environment,
    )

    create_env(
        "wofost_gym",
        {
            "env_id": "lnpk-v0",
            "prompt_action_schema_env_id": "lnpk-v0",
            "include_crop_traits": True,
            "include_variety_traits": True,
            "require_think": True,
        },
        mode="ppv0",
    )

    assert captured["env_name"] == "wofost_gym"
    assert captured["env_config"]["env_id"] == "pp-v0"
    assert captured["env_config"]["prompt_action_schema_env_id"] == "pp-v0"
    assert captured["env_config"]["include_crop_traits"] is False
    assert captured["env_config"]["include_variety_traits"] is False
    assert captured["env_config"]["require_think"] is False
    assert captured["env_config"]["llm_mode"] is True

    with pytest.raises(ValueError, match="only supported for WOFOST-Gym"):
        create_env("other_env", {}, mode="ppv0")


def test_wandb_payload_flattening_keeps_only_numeric_leaves():
    payload = _flatten_numeric(
        "baseline",
        {
            "primary_metric": {"key": "objective_reward", "mean": -1.5},
            "invalid_action_rate_mean": 0.0,
            "baseline_mode": "random",
            "ok": True,
            "bad_value": float("nan"),
        },
    )

    assert payload == {
        "baseline/primary_metric/mean": -1.5,
        "baseline/invalid_action_rate_mean": 0.0,
        "baseline/ok": 1,
    }


def test_wandb_logs_od_and_each_ood_validation_metric(monkeypatch):
    records = []

    fake_wandb = SimpleNamespace(
        log=lambda payload, step=None: records.append((dict(payload), step)),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    results = [
        {
            "env_config": {
                "trajectory_group_labels": {"weather_regime": "id"},
                "validation_set": "id",
            },
            "turns": [
                {
                    "reward": 1.0,
                    "trajectory_metrics": {
                        "target_yield": 100.0,
                        "invalid_action_rate": 0.0,
                        "total_steps": 2.0,
                    },
                }
            ],
        },
        {
            "env_config": {
                "trajectory_group_labels": {"weather_regime": "drought"},
                "validation_set": "drought",
            },
            "turns": [
                {
                    "reward": 2.0,
                    "trajectory_metrics": {
                        "target_yield": 200.0,
                        "invalid_action_rate": 0.0,
                        "total_steps": 2.0,
                    },
                }
            ],
        },
        {
            "env_config": {
                "trajectory_group_labels": {"weather_regime": "heat"},
                "validation_set": "heat",
            },
            "turns": [
                {
                    "reward": 3.0,
                    "trajectory_metrics": {
                        "target_yield": 300.0,
                        "invalid_action_rate": 0.0,
                        "total_steps": 2.0,
                    },
                }
            ],
        },
    ]
    validation_metrics = _extract_validation_metric_dict(
        results,
        validation_axis="weather_regime",
    )

    for mode in ("random", "no_action", "ag_heuristic"):
        _log_trial_to_wandb(
            object(),
            trial_metrics={
                "primary_metric": {"mean": 1.0, "std": 0.0},
                "invalid_action_rate_mean": 0.0,
            },
            validation_metrics=validation_metrics,
            mode=mode,
            trial=0,
            seed=7,
            step=11,
        )

    assert len(records) == 3
    for payload, step in records:
        assert step == 11
        assert payload["val-env-weather_regime/all/target_yield/mean"] == pytest.approx(200.0)
        assert payload["val-env-weather_regime/id/target_yield/mean"] == pytest.approx(100.0)
        assert payload["val-env-weather_regime/drought/target_yield/mean"] == pytest.approx(200.0)
        assert payload["val-env-weather_regime/heat/target_yield/mean"] == pytest.approx(300.0)
    assert records[0][0]["baseline/is_random"] == 1
    assert records[0][0]["baseline/is_no_action"] == 0
    assert records[1][0]["baseline/is_random"] == 0
    assert records[1][0]["baseline/is_no_action"] == 1
    assert records[2][0]["baseline/is_ag_heuristic"] == 1
