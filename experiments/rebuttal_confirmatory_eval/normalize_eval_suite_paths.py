#!/usr/bin/env python3
"""Rewrite machine-specific resource paths in the E4 suites to repo-relative form.

The dataset builder resolves `crop_traits_dir` and `weather_cache_dir` to
absolute paths through `WOFOSTEnvConfig`, so every rendered parquet carries the
absolute path of the machine that built it. That is a problem twice over: the
suites cannot be used on any other machine, and an uploaded artifact would
carry a local path containing a username.

`WOFOSTEnvConfig._resolve_repo_relative_path` resolves a relative path against
the consumer's own repository root, so storing the relative form makes the
suites portable and identity-neutral at once.

This touches only those two resource paths. Scenario selection, prompts,
labels, seeds and rewards are untouched, and the script verifies that by
comparing the physical scenario keys and `paired_scenario_id` values before and
after. Re-run the audit scripts afterwards so the recorded checksums match.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from agrimanager.env.wofost_gym.weather_pool import find_pool_meteo_cache_dir


EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"
REPO_ROOT = EXPERIMENT_DIR.parents[1]

# Repo-relative targets. The crop-traits value is exactly
# agrimanager.env.wofost_gym.env_config.DEFAULT_CROP_TRAITS_DIR.
CROP_TRAITS_REL = "agrimanager/env/wofost_gym/crop_traits"

# The weather cache is resolved per suite from that suite's own source pool.
# It used to be one constant pointing at the maize pool, which was correct while
# maize was the only E4 suite and silently wrong the moment the T1.1 potato and
# chickpea suites were added: their scenarios sit at latitudes the maize cache
# does not contain, so evaluation died with a cache-only miss rather than with
# anything that named the real cause.

# `../AgriManagerExternal/WOFOSTGym` is the value the E4 configs themselves
# declare, so restoring it is faithful rather than a new choice.
WOFOST_GYM_REL = "../AgriManagerExternal/WOFOSTGym"

REWRITE_FIELDS = {
    "crop_traits_dir": CROP_TRAITS_REL,
    "wofost_gym_path": WOFOST_GYM_REL,
}

# `weather_cache_dir` is not in REWRITE_FIELDS because its target depends on the
# suite; suite_weather_cache_rel() supplies it per file.
WEATHER_CACHE_FIELD = "weather_cache_dir"

# gym-dssat is installed through spack, so its path carries a build hash that
# differs on every machine and no relative form can be portable. The DSSAT env
# config already discovers it at construction time via DSSAT_GYM_PATH, then
# REPO_ROOT/spack/gym-dssat-pdi, then the external checkout. Dropping the baked
# value lets that discovery run. `save_folder` is likewise recomputed from the
# consumer's repository root.
DROP_FIELDS = ("dssat_gym_path", "save_folder")

PATH_FIELDS = tuple(REWRITE_FIELDS) + (WEATHER_CACHE_FIELD,) + DROP_FIELDS


def suite_weather_cache_rel(suite: str) -> str | None:
    """Repo-relative meteo cache for one E4 suite, read from its own config.

    The suite directory name and its config file name are the same string, so
    the mapping needs no table that a new suite could be left out of. Suites
    whose source is not a WOFOST weather pool -- the Cycles and DSSAT T3.2
    suites -- return None and keep whatever the builder wrote.
    """
    config_path = EXPERIMENT_DIR / "config" / f"{suite}.yaml"
    if not config_path.is_file():
        raise SystemExit(f"No config for suite {suite!r}; cannot resolve its weather cache.")

    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    source = config.get("source") or {}
    if source.get("kind") != "weather_pool" or not source.get("path"):
        return None

    cache_dir = find_pool_meteo_cache_dir(REPO_ROOT / source["path"])
    if cache_dir is None:
        raise SystemExit(
            f"Suite {suite!r} declares pool {source['path']!r}, which has no meteo cache. "
            "Freeze the pool before normalizing the suite."
        )
    return os.path.relpath(cache_dir.resolve(), REPO_ROOT)


def to_plain(value):
    """Deep-copy a parquet record into plain Python containers.

    pandas hands back nested numpy objects, so the record has to be copied
    before it can be edited. Doing that with
    `json.loads(json.dumps(record, default=str))` looks equivalent and is not:
    `default=str` silently replaces every value json cannot encode with its
    `str()`. A `prompt_observation_fields` array became the literal text
    "['FIN' 'DVS' ...]", which the environment then iterated character by
    character. Convert by type instead, so an unexpected type raises rather
    than being quietly turned into prose.
    """
    if isinstance(value, np.ndarray):
        return [to_plain(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


def scenario_fingerprint(frame: pd.DataFrame) -> tuple[set, set]:
    """Identity of the test set: which scenarios, and how they pair up."""
    physical, paired = set(), set()
    for info in frame["extra_info"]:
        env = info["interaction_kwargs"]["env_config"]
        physical.add(str(env.get("scenario_id")))
        if env.get("paired_scenario_id") is not None:
            paired.add(str(env["paired_scenario_id"]))
    return physical, paired


def normalize_env_config(env: dict, weather_cache_rel: str | None) -> int:
    changed = 0
    targets = dict(REWRITE_FIELDS)
    if weather_cache_rel is not None:
        targets[WEATHER_CACHE_FIELD] = weather_cache_rel
    for field, target in targets.items():
        value = env.get(field)
        # A previous run of this script may have written the wrong constant, so
        # rewrite a relative value that disagrees with the target as well.
        if isinstance(value, str) and (Path(value).is_absolute() or value != target):
            env[field] = target
            changed += 1
    for field in DROP_FIELDS:
        value = env.get(field)
        if isinstance(value, str) and Path(value).is_absolute():
            del env[field]
            changed += 1
    return changed


def normalize_file(path: Path, apply: bool) -> dict:
    frame = pd.read_parquet(path)
    if "extra_info" not in frame.columns:
        return {"path": str(path.relative_to(DATA_DIR)), "skipped": "no extra_info"}

    before_physical, before_paired = scenario_fingerprint(frame)

    weather_cache_rel = suite_weather_cache_rel(path.parent.name)

    rewrites = 0
    records = []
    for info in frame["extra_info"]:
        info = to_plain(info)
        rewrites += normalize_env_config(
            info["interaction_kwargs"]["env_config"], weather_cache_rel
        )
        records.append(info)

    if apply and rewrites:
        frame["extra_info"] = records
        frame.to_parquet(path, index=False)
        after = pd.read_parquet(path)
        after_physical, after_paired = scenario_fingerprint(after)
        if (after_physical, after_paired) != (before_physical, before_paired):
            raise SystemExit(f"Scenario identity changed while rewriting {path}")
        leftovers = [
            f
            for info in after["extra_info"]
            for f in PATH_FIELDS
            if Path(str(info["interaction_kwargs"]["env_config"].get(f) or "")).is_absolute()
        ]
        if leftovers:
            raise SystemExit(f"Absolute paths remain in {path}: {sorted(set(leftovers))}")

    return {
        "path": str(path.relative_to(DATA_DIR)),
        "rows": len(frame),
        "rewrites": rewrites,
        "scenarios": len(before_physical),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="rewrite the files; without it the script only reports",
    )
    args = parser.parse_args()

    targets = sorted(DATA_DIR.glob("e4_*/*.parquet"))
    total = 0
    for path in targets:
        report = normalize_file(path, args.apply)
        if "skipped" in report:
            continue
        total += report["rewrites"]
        print(
            f"  {report['path']:52s} rows={report['rows']:4d} "
            f"scenarios={report['scenarios']:4d} rewrites={report['rewrites']}"
        )

    verb = "rewrote" if args.apply else "would rewrite"
    print(f"\n{verb} {total} absolute path fields across {len(targets)} files")
    if not args.apply:
        print("re-run with --apply, then re-run every audit script")


if __name__ == "__main__":
    main()
