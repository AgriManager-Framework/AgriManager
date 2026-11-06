"""Freeze the shared disjoint maize source pool for the held-out test suite."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EXPERIMENT_DIR / "data" / "source_pools" / "maize_e4"
OUTPUT_PARQUET = OUTPUT_DIR / "test" / "maize.parquet"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
AUDIT_PATH = OUTPUT_DIR / "audit.json"
OUTPUT_METEO_CACHE = OUTPUT_DIR / "meteo_cache"

SOURCE_REPO = "AgriManager-Framework/weather_pool_maize"
SOURCE_REVISION = "3389bc2d75acddbd6740f30198167e8d3e4559c3"
SOURCE_TRAIN_FILE = "train/maize.parquet"
SOURCE_VAL_FILE = "val/maize.parquet"

CROP = "maize"
SUBMITTED_VALIDATION_SEED = 42
SUBMITTED_VALIDATION_COUNT = 128
EVAL_SELECTION_SEED = 20260725
EVAL_COUNT = 128

LOCAL_RENDERED_REFERENCE = (
    REPO_ROOT
    / "experiments"
    / "t3_2_unified_prompt_conditioned_policy"
    / "data"
    / "t32_unified_wofost"
    / "val_wofost_lnw_profit.parquet"
)
# A pool fetched under the release name, or under the first owner's name that
# frozen suites record (both hold the same files).
LOCAL_METEO_CACHE_CANDIDATES = (
    REPO_ROOT
    / ".cache"
    / "wofost_gym"
    / "AgriManager-Framework__weather_pool_maize"
    / "meteo_cache",
    REPO_ROOT
    / ".cache"
    / "wofost_gym"
    / "agrimanager__weather_pool_maize"
    / "meteo_cache",
)

ScenarioKey = tuple[str, int, float, float]


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


def dataframe_keys(df: pd.DataFrame, crop: str = CROP) -> set[ScenarioKey]:
    required = {"year", "latitude", "longitude"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Weather-pool dataframe is missing columns: {sorted(missing)}")
    return {
        scenario_key(crop, row.year, row.latitude, row.longitude)
        for row in df.itertuples(index=False)
    }


def select_rows(
    df: pd.DataFrame,
    *,
    count: int,
    seed: int,
    exclude: Iterable[ScenarioKey] = (),
    crop: str = CROP,
) -> pd.DataFrame:
    """Mirror the submitted one-crop weather-pool sampling semantics."""
    excluded = set(exclude)
    selected_indices: list[int] = []
    selected_keys: set[ScenarioKey] = set()
    rng = np.random.RandomState(seed)

    for raw_index in rng.permutation(len(df)):
        index = int(raw_index)
        row = df.iloc[index]
        key = scenario_key(
            crop,
            row["year"],
            row["latitude"],
            row["longitude"],
        )
        if key in excluded or key in selected_keys:
            continue
        selected_indices.append(index)
        selected_keys.add(key)
        if len(selected_indices) == count:
            break

    if len(selected_indices) != count:
        raise ValueError(
            f"Requested {count} disjoint scenarios, but selected "
            f"{len(selected_indices)} from {len(df)} candidates."
        )

    result = df.iloc[selected_indices][["year", "latitude", "longitude"]].copy()
    result["year"] = result["year"].astype(int)
    result["latitude"] = result["latitude"].astype(float).round(2)
    result["longitude"] = result["longitude"].astype(float).round(2)
    return result.reset_index(drop=True)



def relative_to_repo(path: Path) -> str:
    """Repo-relative string, so audits never name the building machine."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return resolved.name

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sorted_key_records(keys: Iterable[ScenarioKey]) -> list[dict[str, Any]]:
    return [
        {
            "crop": crop,
            "year": year,
            "latitude": latitude,
            "longitude": longitude,
        }
        for crop, year, latitude, longitude in sorted(keys)
    ]


def rendered_reference_keys(path: Path) -> set[ScenarioKey]:
    records = pd.read_parquet(path)
    keys: set[ScenarioKey] = set()
    for row in records.to_dict("records"):
        env_config = (
            row["extra_info"]["interaction_kwargs"]["env_config"]
        )
        agro_params = env_config["agro_params"]
        keys.add(
            scenario_key(
                env_config.get("crop_name", CROP),
                agro_params["year"],
                agro_params["latitude"],
                agro_params["longitude"],
            )
        )
    return keys


def download_source_parquet(filename: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=SOURCE_REPO,
            repo_type="dataset",
            filename=filename,
            revision=SOURCE_REVISION,
            token=False,
        )
    )


def load_existing_manifest() -> dict[str, Any] | None:
    if not MANIFEST_PATH.is_file():
        return None
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def ensure_local_meteo_cache_link() -> Path:
    """Attach a verified local cache without copying its large payload."""
    if OUTPUT_METEO_CACHE.is_dir():
        return OUTPUT_METEO_CACHE.resolve()
    if OUTPUT_METEO_CACHE.is_symlink():
        raise FileNotFoundError(
            f"Broken weather-cache symlink: {OUTPUT_METEO_CACHE}"
        )

    for candidate in LOCAL_METEO_CACHE_CANDIDATES:
        if candidate.is_dir():
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            OUTPUT_METEO_CACHE.symlink_to(candidate.resolve(), target_is_directory=True)
            return candidate.resolve()

    candidates = "\n".join(f"  - {path}" for path in LOCAL_METEO_CACHE_CANDIDATES)
    raise FileNotFoundError(
        "No local maize weather cache is available. Download the public "
        f"{SOURCE_REPO} weather bundle first. Checked:\n{candidates}"
    )


def main() -> None:
    train_df = pd.read_parquet(download_source_parquet(SOURCE_TRAIN_FILE))
    val_df = pd.read_parquet(download_source_parquet(SOURCE_VAL_FILE))

    train_keys = dataframe_keys(train_df)
    val_keys = dataframe_keys(val_df)
    source_overlap = train_keys.intersection(val_keys)
    if source_overlap:
        raise ValueError(
            f"Public maize train and validation pools overlap in "
            f"{len(source_overlap)} physical scenarios."
        )

    submitted_validation = select_rows(
        val_df,
        count=SUBMITTED_VALIDATION_COUNT,
        seed=SUBMITTED_VALIDATION_SEED,
    )
    submitted_validation_keys = dataframe_keys(submitted_validation)

    local_reference_status = "not_available"
    if LOCAL_RENDERED_REFERENCE.is_file():
        local_reference_keys = rendered_reference_keys(LOCAL_RENDERED_REFERENCE)
        if local_reference_keys != submitted_validation_keys:
            raise ValueError(
                "Reconstructed submitted validation keys do not match the "
                f"local rendered reference: {LOCAL_RENDERED_REFERENCE}"
            )
        local_reference_status = "matched"

    selected = select_rows(
        val_df,
        count=EVAL_COUNT,
        seed=EVAL_SELECTION_SEED,
        exclude=submitted_validation_keys,
    )
    selected_keys = dataframe_keys(selected)
    overlap_train = selected_keys.intersection(train_keys)
    overlap_submitted_validation = selected_keys.intersection(
        submitted_validation_keys
    )
    if overlap_train or overlap_submitted_validation:
        raise ValueError(
            "Disjoint selection failed: "
            f"train overlap={len(overlap_train)}, "
            f"submitted-validation overlap={len(overlap_submitted_validation)}."
        )

    expected_keys = sorted_key_records(selected_keys)
    existing_manifest = load_existing_manifest()
    if existing_manifest is not None:
        if existing_manifest.get("selected_scenarios") != expected_keys:
            raise ValueError(
                "The frozen maize E4 manifest does not match the deterministic "
                "selection. Refusing to overwrite it."
            )
        if not OUTPUT_PARQUET.is_file():
            raise FileNotFoundError(
                f"Frozen manifest exists but source Parquet is missing: {OUTPUT_PARQUET}"
            )
        if dataframe_keys(pd.read_parquet(OUTPUT_PARQUET)) != selected_keys:
            raise ValueError(
                "The frozen maize E4 Parquet does not match its manifest."
            )
    else:
        OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
        selected.to_parquet(OUTPUT_PARQUET, index=False)
        manifest = {
            "protocol_id": "e4-shared-maize-v1",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "repo_id": SOURCE_REPO,
                "revision": SOURCE_REVISION,
                "train_file": SOURCE_TRAIN_FILE,
                "validation_file": SOURCE_VAL_FILE,
            },
            "physical_key_fields": [
                "crop",
                "year",
                "latitude",
                "longitude",
            ],
            "submitted_validation": {
                "generation_seed": SUBMITTED_VALIDATION_SEED,
                "num_scenarios": SUBMITTED_VALIDATION_COUNT,
            },
            "eval_selection": {
                "generation_seed": EVAL_SELECTION_SEED,
                "num_scenarios": EVAL_COUNT,
                "source_split": "val",
                "eligible_after_exclusion": len(
                    val_keys.difference(submitted_validation_keys)
                ),
            },
            "selected_scenarios": expected_keys,
        }
        MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    parquet_checksum = sha256_file(OUTPUT_PARQUET)
    manifest_checksum = sha256_file(MANIFEST_PATH)
    meteo_cache_path = ensure_local_meteo_cache_link()
    audit = {
        "protocol_id": "e4-shared-maize-v1",
        "status": "pass",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "public_train": len(train_keys),
            "public_validation": len(val_keys),
            "submitted_validation": len(submitted_validation_keys),
            "eligible_after_exclusion": len(
                val_keys.difference(submitted_validation_keys)
            ),
            "selected_eval": len(selected_keys),
        },
        "overlap": {
            "public_train_vs_public_validation": len(source_overlap),
            "eval_vs_public_train": len(overlap_train),
            "eval_vs_submitted_validation": len(
                overlap_submitted_validation
            ),
        },
        "local_rendered_reference": {
            "path": str(LOCAL_RENDERED_REFERENCE.relative_to(REPO_ROOT)),
            "status": local_reference_status,
        },
        # Record this repo-relative. The resolved path names the building
        # machine and the account its pool cache was downloaded under, and this
        # audit is uploaded alongside the suites.
        "local_meteo_cache": {
            "path": relative_to_repo(meteo_cache_path),
            "status": "available",
        },
        "checksums": {
            "test/maize.parquet": f"sha256:{parquet_checksum}",
            "manifest.json": f"sha256:{manifest_checksum}",
        },
    }
    AUDIT_PATH.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Frozen E4 maize source: {OUTPUT_PARQUET}")
    print(f"Selected scenarios: {len(selected_keys)}")
    print(f"Eligible after exclusion: {audit['counts']['eligible_after_exclusion']}")
    print(f"Overlap with public train: {len(overlap_train)}")
    print(
        "Overlap with submitted validation: "
        f"{len(overlap_submitted_validation)}"
    )
    print(f"Local rendered-reference check: {local_reference_status}")
    print(f"Parquet SHA-256: {parquet_checksum}")
    print(f"Manifest SHA-256: {manifest_checksum}")


if __name__ == "__main__":
    main()
