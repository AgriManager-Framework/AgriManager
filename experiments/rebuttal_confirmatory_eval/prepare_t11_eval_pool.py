"""Freeze disjoint T1.1 scenarios under an explicit regime protocol.

Both T1.1 crops follow the same protocol: take the public weather pool, remove
every scenario the submitted experiment consumed, partition what is left into
weather regimes using statistics over that crop's own growing season, and
select a fixed count per regime. Only the growing-season window differs between
crops, and it is fixed by the benchmark's crop calendar rather than chosen.

Pass `--crop` to select which one to freeze. Selection is deterministic, so
re-running a crop that is already frozen reproduces it byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
GENERAL_REPO = "AgriManager-Framework/wofost-weather-pool"
GENERAL_REVISION = "268da7d08a0f6e544469e56bfefd1dbb2b08ea64"
REGIME_REPO = "AgriManager-Framework/wofost-weather-regime-pool"
REGIME_REVISION = "32b799473b82e708b308f6b0654c02f39f3434b1"

GENERAL_SPLITS = ("train", "val", "test")
REGIME_SPLITS = (
    "train",
    "val",
    "val_id",
    "val_drought",
    "val_wet",
    "val_hot",
    "val_cold",
)
REGIMES = ("id", "drought", "wet", "hot", "cold")

EVAL_SELECTION_SEED = 20260725
EVAL_COUNT_PER_REGIME = 128

# The weather window is each crop's growing season, taken from the benchmark's
# own crop calendars in WOFOSTGym `env_config/agro/`. Aggregating temperature
# and rainfall from sowing to maturity is the only window over which weather can
# affect that crop, so the choice is fixed by the environment definition and not
# by anything observed in the data. In particular it is NOT chosen to maximise
# agreement with the submitted regime labels: that agreement is computed
# afterwards as a diagnostic, and for potato the sowing-to-maturity window
# scores marginally *lower* on the rainfall-derived groups than the best-fitting
# window would.
CROP_PROTOCOLS = {
    "potato": {
        "output_dirname": "t11_potato_e4",
        "protocol_id": "e4-t11-potato-explicit-regimes-v1",
        "weather_window": "jan01_jul01_inclusive",
        "weather_window_start": (1, 1),
        "weather_window_end": (7, 1),
        "weather_window_source": {
            "reason": "potato_crop_calendar_sowing_to_maturity",
            "source_file": "WOFOSTGym/env_config/agro/potato_agro.yaml",
            "crop_start_date": "01-01",
            "crop_start_type": "sowing",
            "crop_end_date": "07-01",
            "crop_end_type": "maturity",
            "selected_by_label_agreement": False,
        },
    },
    "chickpea": {
        "output_dirname": "t11_chickpea_e4",
        "protocol_id": "e4-t11-chickpea-explicit-regimes-v1",
        "weather_window": "mar01_aug01_inclusive",
        "weather_window_start": (3, 1),
        "weather_window_end": (8, 1),
        "weather_window_source": {
            "reason": "chickpea_crop_calendar_sowing_to_maturity",
            "source_file": "WOFOSTGym/env_config/agro/chickpea_agro.yaml",
            "crop_start_date": "03-01",
            "crop_start_type": "sowing",
            "crop_end_date": "08-01",
            "crop_end_type": "maturity",
            "selected_by_label_agreement": False,
        },
    },
}

# Set by configure() before main() does any work.
CROP = ""
PROTOCOL_ID = ""
WEATHER_WINDOW = ""
WEATHER_WINDOW_START = (0, 0)
WEATHER_WINDOW_END = (0, 0)
WEATHER_WINDOW_SOURCE: dict[str, Any] = {}
OUTPUT_DIR = EXPERIMENT_DIR / "data"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
AUDIT_PATH = OUTPUT_DIR / "audit.json"
DIAGNOSTIC_PATH = OUTPUT_DIR / "reconstruction_diagnostic.json"
STATS_PATH = OUTPUT_DIR / "weather_statistics.parquet"
OUTPUT_METEO_CACHE = OUTPUT_DIR / "meteo_cache"


def configure(crop: str) -> None:
    """Bind the module to one crop's protocol."""
    global CROP, PROTOCOL_ID, WEATHER_WINDOW, WEATHER_WINDOW_START
    global WEATHER_WINDOW_END, WEATHER_WINDOW_SOURCE, OUTPUT_DIR
    global MANIFEST_PATH, AUDIT_PATH, DIAGNOSTIC_PATH, STATS_PATH, OUTPUT_METEO_CACHE

    spec = CROP_PROTOCOLS[crop]
    CROP = crop
    PROTOCOL_ID = spec["protocol_id"]
    WEATHER_WINDOW = spec["weather_window"]
    WEATHER_WINDOW_START = spec["weather_window_start"]
    WEATHER_WINDOW_END = spec["weather_window_end"]
    WEATHER_WINDOW_SOURCE = spec["weather_window_source"]
    OUTPUT_DIR = EXPERIMENT_DIR / "data" / "source_pools" / spec["output_dirname"]
    MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
    AUDIT_PATH = OUTPUT_DIR / "audit.json"
    DIAGNOSTIC_PATH = OUTPUT_DIR / "reconstruction_diagnostic.json"
    STATS_PATH = OUTPUT_DIR / "weather_statistics.parquet"
    OUTPUT_METEO_CACHE = OUTPUT_DIR / "meteo_cache"


ASSIGNMENT_ORDER = ("cold", "hot", "drought", "wet")

METEO_CACHE_CANDIDATES = (
    REPO_ROOT
    / ".cache"
    / "wofost_gym"
    / "weather_pool_20crop_3200_val128_test512"
    / "meteo_cache",
    REPO_ROOT
    / ".cache"
    / "wofost_gym"
    / "AgriManager-Framework__wofost-weather-pool"
    / "meteo_cache",
    REPO_ROOT
    / ".cache"
    / "wofost_gym"
    / "agrimanager__wofost-weather-pool"
    / "meteo_cache",
)

ScenarioKey = tuple[str, int, float, float]


class WeatherRecord:
    """Validation-free pickle target for PCSE weather cache records."""


class WeatherCacheUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module in {"pcse.nasapower", "pcse.base.weather"} and name == "WeatherDataContainer":
            return WeatherRecord
        return super().find_class(module, name)


def scenario_key(
    crop: str,
    year: int,
    latitude: float,
    longitude: float,
) -> ScenarioKey:
    return (
        str(crop),
        int(year),
        round(float(latitude), 2),
        round(float(longitude), 2),
    )


def dataframe_keys(df: pd.DataFrame, crop: str | None = None) -> set[ScenarioKey]:
    crop = CROP if crop is None else crop
    return {
        scenario_key(crop, row.year, row.latitude, row.longitude)
        for row in df.itertuples(index=False)
    }


def key_dataframe(keys: Iterable[ScenarioKey]) -> pd.DataFrame:
    rows = [
        {
            "year": year,
            "latitude": latitude,
            "longitude": longitude,
        }
        for _, year, latitude, longitude in sorted(keys)
    ]
    return pd.DataFrame(rows, columns=["year", "latitude", "longitude"])


def source_parquet(
    repo_id: str,
    revision: str,
    split: str,
) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"{split}/{CROP}.parquet",
            revision=revision,
            token=False,
        )
    )


def find_meteo_cache() -> Path:
    for candidate in METEO_CACHE_CANDIDATES:
        if candidate.is_dir():
            return candidate.resolve()
    candidates = "\n".join(f"  - {path}" for path in METEO_CACHE_CANDIDATES)
    raise FileNotFoundError(
        "The public 20-crop weather cache is required for regime reconstruction. "
        f"Checked:\n{candidates}"
    )


def cache_path(cache_dir: Path, latitude: float, longitude: float) -> Path:
    filename = (
        "NASAPowerWeatherDataProvider_"
        f"LAT{int(latitude * 10):05d}_LON{int(longitude * 10):05d}.cache"
    )
    return cache_dir / filename


def load_weather_cache(path: Path) -> dict[tuple[date, int], WeatherRecord]:
    with path.open("rb") as handle:
        payload = WeatherCacheUnpickler(handle).load()
    if not isinstance(payload, tuple) or not payload or not isinstance(payload[0], dict):
        raise ValueError(f"Unexpected PCSE weather-cache payload: {path}")
    return payload[0]


def compute_weather_statistics(
    task: tuple[ScenarioKey, str, tuple[int, int], tuple[int, int]],
) -> dict[str, Any]:
    # The window travels in the task rather than being read from module state.
    # These run under a "spawn" context, so a worker re-imports this module and
    # never sees what configure() bound in the parent.
    key, raw_cache_dir, window_start, window_end = task
    crop, year, latitude, longitude = key
    path = cache_path(Path(raw_cache_dir), latitude, longitude)
    if not path.is_file():
        raise FileNotFoundError(f"Missing weather cache for {key}: {path}")
    weather = load_weather_cache(path)

    result: dict[str, Any] = {
        "crop": crop,
        "year": year,
        "latitude": latitude,
        "longitude": longitude,
    }
    start = date(year, *window_start)
    end = date(year, *window_end)
    records = []
    for raw_day, record in weather.items():
        day = raw_day[0] if isinstance(raw_day, tuple) else raw_day
        if start <= day <= end:
            records.append(record)
    if not records:
        raise ValueError(f"No weather records for {key} in the protocol window.")
    result["rain"] = sum(float(record.RAIN) for record in records)
    result["temp"] = sum(float(record.TEMP) for record in records) / len(records)
    result["days"] = len(records)
    return result


def compute_statistics(
    keys: set[ScenarioKey],
    cache_dir: Path,
) -> dict[ScenarioKey, dict[str, Any]]:
    workers = max(1, min(int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), 32))
    tasks = [
        (key, str(cache_dir), WEATHER_WINDOW_START, WEATHER_WINDOW_END)
        for key in sorted(keys)
    ]
    if workers == 1:
        rows = [compute_weather_statistics(task) for task in tasks]
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            rows = list(executor.map(compute_weather_statistics, tasks, chunksize=8))

    stats: dict[ScenarioKey, dict[str, Any]] = {}
    for row in rows:
        key = scenario_key(
            row["crop"],
            row["year"],
            row["latitude"],
            row["longitude"],
        )
        stats[key] = row
    return stats


def rank_key(
    key: ScenarioKey,
    stats: dict[ScenarioKey, dict[str, Any]],
    *,
    regime: str,
) -> tuple[Any, ...]:
    if regime == "drought":
        return (stats[key]["rain"], key)
    if regime == "wet":
        return (-stats[key]["rain"], key)
    if regime == "cold":
        return (stats[key]["temp"], key)
    if regime == "hot":
        return (-stats[key]["temp"], key)
    raise ValueError(f"Unsupported extreme regime: {regime}")


def partition_regimes(
    keys: set[ScenarioKey],
    stats: dict[ScenarioKey, dict[str, Any]],
    *,
    assignment_order: tuple[str, ...],
) -> dict[str, set[ScenarioKey]]:
    if len(keys) % 5:
        raise ValueError(f"Regime candidate count must be divisible by five: {len(keys)}")
    group_size = len(keys) // 5
    remaining = set(keys)
    groups: dict[str, set[ScenarioKey]] = {}
    for regime in assignment_order:
        ranked = sorted(
            remaining,
            key=lambda key: rank_key(
                key,
                stats,
                regime=regime,
            ),
        )
        selected = set(ranked[:group_size])
        groups[regime] = selected
        remaining.difference_update(selected)
    groups["id"] = remaining
    return groups


def reconstruction_score(
    predicted: dict[str, set[ScenarioKey]],
    actual: dict[str, set[ScenarioKey]],
) -> tuple[int, dict[str, int]]:
    matches = {
        regime: len(predicted[regime].intersection(actual[regime]))
        for regime in REGIMES
    }
    return sum(matches.values()), matches


def compare_with_submitted_labels(
    old_validation_keys: set[ScenarioKey],
    actual_groups: dict[str, set[ScenarioKey]],
    stats: dict[ScenarioKey, dict[str, Any]],
) -> dict[str, Any]:
    predicted = partition_regimes(
        old_validation_keys,
        stats,
        assignment_order=ASSIGNMENT_ORDER,
    )
    score, matches = reconstruction_score(predicted, actual_groups)
    return {
        "purpose": "diagnostic_only_not_protocol_selection",
        "weather_window": WEATHER_WINDOW,
        "assignment_order": list(ASSIGNMENT_ORDER),
        "matches": matches,
        "total_matches": score,
        "total_labels": len(old_validation_keys),
        "accuracy": score / len(old_validation_keys),
        "exact_reconstruction": score == len(old_validation_keys),
    }


def select_eval_rows(
    group: set[ScenarioKey],
    *,
    regime: str,
) -> set[ScenarioKey]:
    ordered = sorted(group)
    regime_offset = REGIMES.index(regime) * 1009
    rng = np.random.RandomState(EVAL_SELECTION_SEED + regime_offset)
    indices = rng.permutation(len(ordered))[:EVAL_COUNT_PER_REGIME]
    return {ordered[int(index)] for index in indices}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_cache_link(cache_dir: Path) -> None:
    if OUTPUT_METEO_CACHE.is_dir():
        return
    if OUTPUT_METEO_CACHE.is_symlink():
        raise FileNotFoundError(f"Broken cache symlink: {OUTPUT_METEO_CACHE}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_METEO_CACHE.symlink_to(cache_dir, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", choices=sorted(CROP_PROTOCOLS), default="potato")
    configure(parser.parse_args().crop)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    general_frames = {
        split: pd.read_parquet(source_parquet(GENERAL_REPO, GENERAL_REVISION, split))
        for split in GENERAL_SPLITS
    }
    regime_frames = {
        split: pd.read_parquet(source_parquet(REGIME_REPO, REGIME_REVISION, split))
        for split in REGIME_SPLITS
    }
    general_keys = {
        split: dataframe_keys(frame)
        for split, frame in general_frames.items()
    }
    regime_keys = {
        split: dataframe_keys(frame)
        for split, frame in regime_frames.items()
    }

    general_all = set().union(*general_keys.values())
    old_used = regime_keys["train"].union(regime_keys["val"])
    old_named_union = set().union(
        regime_keys["val_id"],
        regime_keys["val_drought"],
        regime_keys["val_wet"],
        regime_keys["val_hot"],
        regime_keys["val_cold"],
    )
    if old_named_union != regime_keys["val"]:
        raise ValueError("The five submitted named regimes do not partition val.")
    if not old_used.issubset(general_all):
        raise ValueError("Submitted T1.1 scenarios are not a subset of the public pool.")

    unused = general_all.difference(old_used)
    if len(general_all) != 3840 or len(old_used) != 2240 or len(unused) != 1600:
        raise ValueError(
            "Unexpected T1.1 pool cardinalities: "
            f"general={len(general_all)}, old_used={len(old_used)}, unused={len(unused)}."
        )

    cache_dir = find_meteo_cache()
    stats = compute_statistics(regime_keys["val"].union(unused), cache_dir)
    stats_df = pd.DataFrame(stats.values()).sort_values(
        ["crop", "year", "latitude", "longitude"]
    )
    stats_df.to_parquet(STATS_PATH, index=False)

    actual_groups = {
        regime: regime_keys[f"val_{regime}"]
        for regime in REGIMES
    }
    diagnostic = compare_with_submitted_labels(
        regime_keys["val"],
        actual_groups,
        stats,
    )
    DIAGNOSTIC_PATH.write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    unused_groups = partition_regimes(
        unused,
        stats,
        assignment_order=ASSIGNMENT_ORDER,
    )
    selected_groups = {
        regime: select_eval_rows(unused_groups[regime], regime=regime)
        for regime in REGIMES
    }
    selected_union = set().union(*selected_groups.values())
    if len(selected_union) != EVAL_COUNT_PER_REGIME * len(REGIMES):
        raise ValueError("Selected T1.1 E4 groups are not mutually disjoint.")
    if selected_union.intersection(old_used):
        raise ValueError("Selected T1.1 E4 scenarios overlap submitted data.")

    split_checksums: dict[str, str] = {}
    for regime, keys in selected_groups.items():
        split = f"test_{regime}"
        path = OUTPUT_DIR / split / f"{CROP}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        key_dataframe(keys).to_parquet(path, index=False)
        split_checksums[f"{split}/{CROP}.parquet"] = f"sha256:{sha256_file(path)}"

    ensure_cache_link(cache_dir)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "general_pool": {
                "repo_id": GENERAL_REPO,
                "revision": GENERAL_REVISION,
                "splits": list(GENERAL_SPLITS),
            },
            "submitted_regime_pool": {
                "repo_id": REGIME_REPO,
                "revision": REGIME_REVISION,
                "splits": list(REGIME_SPLITS),
            },
        },
        "physical_key_fields": ["crop", "year", "latitude", "longitude"],
        "regime_protocol": {
            "definition": "new_explicit_confirmatory_protocol",
            "weather_window": {
                "name": WEATHER_WINDOW,
                "start_month_day": list(WEATHER_WINDOW_START),
                "end_month_day": list(WEATHER_WINDOW_END),
                "end_inclusive": True,
                "justification": WEATHER_WINDOW_SOURCE,
            },
            "temperature_statistic": "mean_daily_temperature",
            "rainfall_statistic": "cumulative_daily_rainfall",
            "assignment_order": list(ASSIGNMENT_ORDER),
            "fraction_per_group": 0.20,
            "id_definition": "remaining_scenarios_after_four_tail_assignments",
            "legacy_label_comparison_is_diagnostic_only": True,
        },
        "eval_selection": {
            "generation_seed": EVAL_SELECTION_SEED,
            "candidate_count_after_exclusion": len(unused),
            "candidate_count_per_regime": {
                regime: len(keys)
                for regime, keys in unused_groups.items()
            },
            "selected_count_per_regime": EVAL_COUNT_PER_REGIME,
        },
        "selected_scenarios": {
            regime: [
                {
                    "crop": crop,
                    "year": year,
                    "latitude": latitude,
                    "longitude": longitude,
                }
                for crop, year, latitude, longitude in sorted(keys)
            ]
            for regime, keys in selected_groups.items()
        },
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    audit = {
        "protocol_id": PROTOCOL_ID,
        "status": "pass",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "public_general_pool": len(general_all),
            "submitted_train_and_validation": len(old_used),
            "unused_candidates": len(unused),
            "submitted_validation": len(regime_keys["val"]),
            "selected_total": len(selected_union),
            "selected_per_regime": {
                regime: len(keys)
                for regime, keys in selected_groups.items()
            },
        },
        "legacy_label_comparison": diagnostic,
        "overlap": {
            "selected_vs_submitted_train_and_validation": len(
                selected_union.intersection(old_used)
            ),
            "between_selected_regimes": (
                sum(len(keys) for keys in selected_groups.values())
                - len(selected_union)
            ),
        },
        "checksums": {
            **split_checksums,
            "manifest.json": f"sha256:{sha256_file(MANIFEST_PATH)}",
            "weather_statistics.parquet": f"sha256:{sha256_file(STATS_PATH)}",
            "reconstruction_diagnostic.json": f"sha256:{sha256_file(DIAGNOSTIC_PATH)}",
        },
    }
    AUDIT_PATH.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"T1.1 {CROP} E4 source pool frozen")
    print(f"Protocol: {PROTOCOL_ID}")
    print(f"Weather window: {WEATHER_WINDOW}")
    print(f"Assignment order: {', '.join(ASSIGNMENT_ORDER)}")
    print(
        "Legacy-label comparison (diagnostic only): "
        f"{diagnostic['total_matches']}/{diagnostic['total_labels']}"
    )
    print("Unused public-pool candidates: 1600")
    print("Selected: 128 per regime, 640 total")
    print("Overlap with submitted train/validation: 0")
    print(f"Manifest: {MANIFEST_PATH}")
    print(f"Audit: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
