"""Algorithm selection for the numerical policies.

`agent.type` existed in the configs but was never read: both training and
evaluation hardcoded PPO. This module makes it the single place that decides
which Stable-Baselines3 algorithm a numerical run uses.

Each algorithm reads only the hyperparameters it actually accepts. PPO's
rollout parameters (`n_steps`, `n_epochs`, `num_minibatches`, `clip_range`)
have no meaning for DQN, and silently passing them would raise deep inside
SB3 rather than here. The shared interaction budget is *not* set here: it comes
from `train_epochs` and the dataset size, so every algorithm consumes the same
number of environment steps and the comparison stays same-budget.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm


DEFAULT_ALGORITHM = "PPO"

# MaskablePPO lives in sb3-contrib, which is an optional dependency: importing it
# eagerly would make every numerical run fail on a machine that only needs PPO.
# The import is deferred to the point where a run actually asks for it, so the
# error names the missing package instead of surfacing as an unknown agent.type.
try:  # pragma: no cover - import guard, exercised by environment rather than tests
    from sb3_contrib import MaskablePPO
except ModuleNotFoundError:  # pragma: no cover
    MaskablePPO = None


def _cfg(agent_cfg: dict[str, Any], *keys_then_default: Any) -> Any:
    """First key present with a non-None value, else the trailing default.

    A shared config declares every algorithm's knobs and leaves the inapplicable
    ones `null`, so a present-but-null key has to mean "not set" rather than
    "set to None".
    """
    *keys, default = keys_then_default
    for key in keys:
        value = agent_cfg.get(key)
        if value is not None:
            return value
    return default


def _rollout_batch_size(agent_cfg: dict[str, Any], num_envs: int, n_steps: int) -> int:
    if agent_cfg.get("batch_size") is not None:
        return int(agent_cfg["batch_size"])
    num_minibatches = int(_cfg(agent_cfg, "num_minibatches", 4))
    rollout_batch = num_envs * n_steps
    if rollout_batch % num_minibatches != 0:
        raise ValueError(
            "num_envs * n_steps must be divisible by num_minibatches, got "
            f"{rollout_batch} and {num_minibatches}."
        )
    return rollout_batch // num_minibatches


def _ppo_kwargs(agent_cfg: dict[str, Any], *, num_envs: int, gamma: float) -> dict[str, Any]:
    n_steps = int(_cfg(agent_cfg, "n_steps", "num_steps", 128))
    return {
        "learning_rate": float(_cfg(agent_cfg, "learning_rate", 2.5e-4)),
        "n_steps": n_steps,
        "batch_size": _rollout_batch_size(agent_cfg, num_envs, n_steps),
        "n_epochs": int(_cfg(agent_cfg, "n_epochs", "update_epochs", 4)),
        "gamma": gamma,
        "gae_lambda": float(_cfg(agent_cfg, "gae_lambda", 0.95)),
        "clip_range": float(_cfg(agent_cfg, "clip_range", "clip_coef", 0.2)),
        "ent_coef": float(_cfg(agent_cfg, "ent_coef", 0.0)),
        "vf_coef": float(_cfg(agent_cfg, "vf_coef", 0.5)),
        "max_grad_norm": float(_cfg(agent_cfg, "max_grad_norm", 0.5)),
    }


def _dqn_kwargs(agent_cfg: dict[str, Any], *, num_envs: int, gamma: float) -> dict[str, Any]:
    # `train_freq` counts steps per environment, so the same value means the
    # same number of gradient updates per environment step regardless of
    # `num_envs`, which is what keeps a DQN run budget-comparable to PPO.
    n_steps = int(_cfg(agent_cfg, "n_steps", "num_steps", 128))
    return {
        "learning_rate": float(_cfg(agent_cfg, "learning_rate", 1e-4)),
        "buffer_size": int(_cfg(agent_cfg, "buffer_size", 100_000)),
        "learning_starts": int(_cfg(agent_cfg, "learning_starts", num_envs * n_steps)),
        "batch_size": int(_cfg(agent_cfg, "batch_size", 64)),
        "tau": float(_cfg(agent_cfg, "tau", 1.0)),
        "gamma": gamma,
        "train_freq": int(_cfg(agent_cfg, "train_freq", 4)),
        "gradient_steps": int(_cfg(agent_cfg, "gradient_steps", 1)),
        "target_update_interval": int(_cfg(agent_cfg, "target_update_interval", 1_000)),
        "exploration_fraction": float(_cfg(agent_cfg, "exploration_fraction", 0.1)),
        "exploration_initial_eps": float(_cfg(agent_cfg, "exploration_initial_eps", 1.0)),
        "exploration_final_eps": float(_cfg(agent_cfg, "exploration_final_eps", 0.05)),
        "max_grad_norm": float(_cfg(agent_cfg, "max_grad_norm", 10.0)),
    }


def _a2c_kwargs(agent_cfg: dict[str, Any], *, num_envs: int, gamma: float) -> dict[str, Any]:
    return {
        "learning_rate": float(_cfg(agent_cfg, "learning_rate", 7e-4)),
        "n_steps": int(_cfg(agent_cfg, "n_steps", "num_steps", 5)),
        "gamma": gamma,
        "gae_lambda": float(_cfg(agent_cfg, "gae_lambda", 1.0)),
        "ent_coef": float(_cfg(agent_cfg, "ent_coef", 0.0)),
        "vf_coef": float(_cfg(agent_cfg, "vf_coef", 0.5)),
        "max_grad_norm": float(_cfg(agent_cfg, "max_grad_norm", 0.5)),
    }


KwargsBuilder = Callable[..., dict[str, Any]]

ALGORITHMS: dict[str, tuple[type[BaseAlgorithm] | None, KwargsBuilder]] = {
    "PPO": (PPO, _ppo_kwargs),
    "DQN": (DQN, _dqn_kwargs),
    "A2C": (A2C, _a2c_kwargs),
    # MaskablePPO takes PPO's constructor signature verbatim, so it shares the
    # builder. What differs is the environment, which must expose action_masks();
    # that is the masked SB3 env selected by `env.nn_action_space`, not anything
    # this module decides.
    "MASKABLEPPO": (MaskablePPO, _ppo_kwargs),
}

# `agent.type` is written by hand in run scripts, so accept the spellings people
# actually type rather than making a run fail on a hyphen.
ALGORITHM_ALIASES = {
    "SB3_PPO": "PPO",
    "SB3_DQN": "DQN",
    "SB3_A2C": "A2C",
    "MASKABLE_PPO": "MASKABLEPPO",
    "MASKED_PPO": "MASKABLEPPO",
    "MASK_PPO": "MASKABLEPPO",
}


def _canonical_name(raw: Any) -> str:
    name = str(raw or DEFAULT_ALGORITHM).strip().upper().replace("-", "_")
    name = ALGORITHM_ALIASES.get(name, name)
    if name not in ALGORITHMS:
        raise ValueError(
            f"Unknown agent.type {raw!r}. Available: {', '.join(sorted(ALGORITHMS))}."
        )
    return name


def _algorithm_class(name: str) -> type[BaseAlgorithm]:
    cls, _ = ALGORITHMS[name]
    if cls is None:
        raise ModuleNotFoundError(
            f"agent.type resolves to {name}, which needs sb3-contrib. Install it into an "
            "environment of its own with `pip install --no-deps sb3-contrib`; installing it "
            "with dependencies would replace the editable stable-baselines3 fork that every "
            "other numerical run depends on."
        )
    return cls


def resolve_algorithm_name(agent_cfg: dict[str, Any]) -> str:
    return _canonical_name(agent_cfg.get("type"))


def build_algorithm(
    agent_cfg: dict[str, Any],
    *,
    env: Any,
    num_envs: int,
    gamma: float,
    seed: int,
    tensorboard_log: str,
    device: str,
) -> tuple[str, BaseAlgorithm]:
    name = resolve_algorithm_name(agent_cfg)
    _, build_kwargs = ALGORITHMS[name]
    model = _algorithm_class(name)(
        policy="MlpPolicy",
        env=env,
        seed=seed,
        verbose=int(_cfg(agent_cfg, "verbose", 1)),
        tensorboard_log=tensorboard_log,
        device=device,
        **build_kwargs(agent_cfg, num_envs=num_envs, gamma=gamma),
    )
    return name, model


def algorithm_class_for_checkpoint(
    model_path: Path,
    configured_type: str | None = None,
) -> tuple[str, type[BaseAlgorithm]]:
    """Pick the class that can load a saved bundle.

    An SB3 archive does not name the algorithm that wrote it, and loading a DQN
    bundle with `PPO.load` fails in a way that points at the weights rather than
    at the mismatch. The trainer saves its resolved config next to every
    checkpoint, so read `agent.type` from there and let an explicit override win.
    """
    if configured_type:
        name = _canonical_name(configured_type)
        return name, _algorithm_class(name)

    sidecar = model_path.parent / "config.yaml"
    if sidecar.is_file():
        try:
            saved = yaml.safe_load(sidecar.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            saved = {}
        recorded = ((saved.get("agent") or {}).get("type")) if isinstance(saved, dict) else None
        if recorded:
            name = _canonical_name(recorded)
            return name, _algorithm_class(name)

    return DEFAULT_ALGORITHM, _algorithm_class(DEFAULT_ALGORITHM)


__all__ = [
    "ALGORITHMS",
    "DEFAULT_ALGORITHM",
    "algorithm_class_for_checkpoint",
    "build_algorithm",
    "resolve_algorithm_name",
]
