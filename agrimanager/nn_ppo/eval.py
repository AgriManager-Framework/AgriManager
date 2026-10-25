"""Framework-native parquet-driven evaluation for numeric-policy environments.

The algorithm is read from the checkpoint's own saved config, so a DQN bundle
is never loaded as PPO; see `agrimanager.nn_ppo.algorithms`.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import OmegaConf
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch

from agrimanager.adapter.trainer.validation_sets import (
    annotate_env_config_with_validation_set,
    normalize_val_sets,
)
from agrimanager.env.base import create_nn_env_adapter, load_env_configs_from_parquet
from agrimanager.nn_ppo.algorithms import algorithm_class_for_checkpoint
from agrimanager.nn_ppo.common import (
    evaluate_model_on_dataset,
    resolve_repo_path,
    save_rollout_results,
)


def _resolve_model_path(path_str: str) -> Path:
    path = resolve_repo_path(path_str)
    if path.exists():
        return path
    if path.suffix != ".zip":
        zipped = path.with_suffix(".zip")
        if zipped.exists():
            return zipped
    raise FileNotFoundError(f"Model file not found: {path}")


def _resolve_vecnormalize_path(model_path: Path, configured_path: str | None) -> Path | None:
    if configured_path:
        candidate = resolve_repo_path(configured_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"VecNormalize stats file not found: {candidate}")

    sibling = model_path.parent / "vecnormalize.pkl"
    if sibling.exists():
        return sibling
    return None


def _load_vecnormalize(
    adapter,
    env_configs: list[dict[str, Any]],
    stats_path: Path | None,
) -> VecNormalize | None:
    if stats_path is None:
        return None

    first_env_config = dict(env_configs[0])
    dummy_vec_env = DummyVecEnv([lambda: adapter.make_env(first_env_config)])
    vecnormalize = VecNormalize.load(str(stats_path), dummy_vec_env)
    vecnormalize.training = False
    vecnormalize.norm_reward = False
    return vecnormalize


def _load_named_inference_sets(raw_sets: Any):
    inference_sets = normalize_val_sets(raw_sets)
    if not inference_sets:
        return None

    all_configs: list[dict[str, Any]] = []
    env_name: str | None = None
    first_path: Path | None = None
    for validation_set, files in inference_sets.items():
        configs, set_env_name, dataset_path = load_env_configs_from_parquet(files)
        if env_name is None:
            env_name = set_env_name
            first_path = dataset_path
        elif set_env_name != env_name:
            raise ValueError(
                f"NN inference env_name mismatch across data.inference_sets: "
                f"{env_name!r} vs {set_env_name!r}"
            )
        all_configs.extend(
            annotate_env_config_with_validation_set(config, validation_set)
            for config in configs
        )
    assert env_name is not None and first_path is not None
    return all_configs, env_name, first_path


# Overriding any of these would change which scenario is being scored rather than
# how it is driven, which is the whole point of a frozen suite. They are refused
# instead of applied, so "only the wrapper changed" is enforced rather than
# promised in a comment.
SCENARIO_IDENTITY_KEYS = frozenset({
    "scenario_id",
    "paired_scenario_id",
    "env_id",
    "env_name",
    "crop_name",
    "objective_id",
    "agro_file",
    "agro_params",
    "reward_params",
    "dataset_id",
    "dataset_role",
    "dataset_split",
    "intvn_interval",
    "scale_action_amounts_by_interval",
    "fert_amount",
    "irrig_amount",
})


def _apply_env_overrides(
    env_configs: list[dict[str, Any]], env_overrides: dict[str, Any]
) -> None:
    """Apply wrapper-level overrides in place, refusing scenario-identifying keys."""
    if not env_overrides:
        return
    forbidden = sorted(SCENARIO_IDENTITY_KEYS.intersection(env_overrides))
    if forbidden:
        raise ValueError(
            "data.env_overrides may not set scenario-identifying keys: "
            f"{', '.join(forbidden)}. Those define which episode runs, not how it is driven."
        )
    before = [str(config.get("scenario_id")) for config in env_configs]
    for config in env_configs:
        config.update(env_overrides)
    after = [str(config.get("scenario_id")) for config in env_configs]
    if before != after:
        raise AssertionError("Scenario identity changed while applying data.env_overrides.")
    applied = ", ".join(f"{key}={value!r}" for key, value in sorted(env_overrides.items()))
    print(f"[nn_eval] env overrides applied to {len(env_configs)} scenarios: {applied}")


@hydra.main(config_path="config", config_name="default", version_base=None)
def main(config) -> None:
    OmegaConf.resolve(config)
    cfg = OmegaConf.to_container(config, resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected dict config, got {type(cfg)!r}")

    data_cfg = cfg.get("data") or {}
    agent_cfg = cfg.get("agent") or {}
    runtime_cfg = cfg.get("runtime") or {}
    output_cfg = cfg.get("output") or {}

    inference_file = data_cfg.get("inference_file")
    inference_sets = data_cfg.get("inference_sets") or data_cfg.get("val_sets")
    if not inference_file and not inference_sets:
        raise ValueError("Config must specify data.inference_file or data.inference_sets.")
    output_dir = output_cfg.get("dir")
    if not output_dir:
        raise ValueError("Config must specify output.dir.")
    agent_path = agent_cfg.get("path")
    if not agent_path:
        raise ValueError("Config must specify agent.path.")

    named_inference_load = _load_named_inference_sets(inference_sets)
    if named_inference_load is not None:
        env_configs, env_name, dataset_path = named_inference_load
    else:
        env_configs, env_name, dataset_path = load_env_configs_from_parquet(inference_file)
    _apply_env_overrides(env_configs, dict(data_cfg.get("env_overrides") or {}))
    try:
        from agrimanager.env.wofost_gym.weather_pool import ensure_weather_caches

        ensure_weather_caches(env_configs)
    except ImportError:  # pragma: no cover - WOFOST extras absent
        pass
    adapter = create_nn_env_adapter(env_name)

    model_path = _resolve_model_path(str(agent_path))
    vecnormalize_path = _resolve_vecnormalize_path(model_path, agent_cfg.get("vecnormalize_path"))
    output_path = resolve_repo_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    seed = int(runtime_cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    vecnormalize = _load_vecnormalize(adapter, env_configs, vecnormalize_path)
    algorithm_name, algorithm_cls = algorithm_class_for_checkpoint(
        model_path, agent_cfg.get("type")
    )
    model = algorithm_cls.load(str(model_path), device=str(runtime_cfg.get("device", "auto")))

    print("=" * 80)
    print("AgriManager NN Policy Evaluation")
    print("=" * 80)
    print(f"Env name: {env_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_path}")
    print(f"Algorithm: {algorithm_name}")
    print(f"VecNormalize: {vecnormalize_path if vecnormalize_path else 'none'}")
    print(f"Output dir: {output_path}")
    print(f"Num scenarios: {len(env_configs)}")
    print("=" * 80)

    try:
        results, _, metric_dict = evaluate_model_on_dataset(
            model,
            adapter,
            env_configs,
            vecnormalize=vecnormalize,
            deterministic=bool(runtime_cfg.get("deterministic", False)),
            seed=seed,
            num_repeats=int(runtime_cfg.get("num_repeats", 1)),
            validation_axis=data_cfg.get("validation_axis"),
        )
        split_by_group = runtime_cfg.get("split_by_group")
        if not split_by_group and named_inference_load is not None:
            split_by_group = "validation_set"
        if not split_by_group and bool(runtime_cfg.get("split_by_crop", False)):
            split_by_group = "crop"
        save_rollout_results(
            output_path,
            results,
            split_by_group=None if split_by_group is None else str(split_by_group),
        )
        metrics_path = output_path / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metric_dict, f, indent=2, sort_keys=True)
    finally:
        if vecnormalize is not None:
            vecnormalize.venv.close()


if __name__ == "__main__":
    main()
