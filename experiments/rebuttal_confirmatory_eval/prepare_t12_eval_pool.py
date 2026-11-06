"""Freeze the disjoint T1.2 cross-crop source pool for the held-out test suite.

T1.2 varies the crop, so unlike T1.1 there is nothing to partition: the axis is
already categorical and the public pool already separates the crops. What the
suite needs is a fixed, auditable draw that provably never reached training.

That is easier here than anywhere else in E4. The general 20-crop pool ships a
real `test/` split of 512 scenarios per crop, disjoint from both `train/` and
`val/`. The maize pool has no test split, which is why the shared maize base had
to be carved out of validation with the submitted draw excluded; no such
manoeuvre is needed for T1.2. This script selects from `test/` directly and
records the three-way disjointness it relies on rather than assuming it.

Selection is deterministic, so re-running reproduces the pool byte for byte.
"""

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
OUTPUT_DIR = EXPERIMENT_DIR / "data" / "source_pools" / "t12_cross_crop_e4"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
AUDIT_PATH = OUTPUT_DIR / "audit.json"
OUTPUT_METEO_CACHE = OUTPUT_DIR / "meteo_cache"

PROTOCOL_ID = "e4-t12-cross-crop-v1"
SOURCE_REPO = "AgriManager-Framework/wofost-weather-pool"
SOURCE_REVISION = "268da7d08a0f6e544469e56bfefd1dbb2b08ea64"

# The crop partition is the experiment. It is copied from
# experiments/t1_2_cross_crop_trait_shift/config/cross_crop_4id_llm_traits_v1_23d_think.yaml
# and verified against that file at audit time, so the two cannot drift apart.
ID_CROPS = ("cotton", "millet", "rapeseed", "rice")
HELDOUT_CROPS = ("barley", "groundnut", "pigeonpea", "seed_onion")
ALL_CROPS = ID_CROPS + HELDOUT_CROPS

TRAINING_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "t1_2_cross_crop_trait_shift"
    / "config"
    / "cross_crop_4id_llm_traits_v1_23d_think.yaml"
)

EVAL_SELECTION_SEED = 20260725
EVAL_COUNT_PER_CROP = 128

WOFOST_CACHE_ROOT = REPO_ROOT / ".cache" / "wofost_gym"

# The canonical location first, then any org-prefixed download of the same pool.
# The fallbacks are globbed rather than listed because the prefix is whichever
# account downloaded the pool, and hardcoding an account name in order to find
# it would embed that name in the repository permanently.
LOCAL_METEO_CACHE_CANDIDATES = (
    WOFOST_CACHE_ROOT / "weather_pool_20crop_3200_val128_test512" / "meteo_cache",
)
LOCAL_METEO_CACHE_GLOB = "*wofost-weather-pool*/meteo_cache"

ScenarioKey = tuple[str, int, float, float]


def scenario_key(crop: str, year: int, latitude: float, longitude: float) -> ScenarioKey:
    return (str(crop), int(year), round(float(latitude), 2), round(float(longitude), 2))


def dataframe_keys(df: pd.DataFrame, crop: str) -> set[ScenarioKey]:
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
    crop: str,
    count: int,
    seed: int,
    exclude: Iterable[ScenarioKey] = (),
) -> pd.DataFrame:
    """Mirror the submitted one-crop weather-pool sampling semantics."""
    excluded = set(exclude)
    selected_indices: list[int] = []
    selected_keys: set[ScenarioKey] = set()
    rng = np.random.RandomState(seed)

    for raw_index in rng.permutation(len(df)):
        index = int(raw_index)
        row = df.iloc[index]
        key = scenario_key(crop, row["year"], row["latitude"], row["longitude"])
        if key in excluded or key in selected_keys:
            continue
        selected_indices.append(index)
        selected_keys.add(key)
        if len(selected_indices) == count:
            break

    if len(selected_indices) != count:
        raise ValueError(
            f"Requested {count} disjoint {crop} scenarios, but selected "
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
        {"crop": crop, "year": year, "latitude": latitude, "longitude": longitude}
        for crop, year, latitude, longitude in sorted(keys)
    ]


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


def check_crop_partition() -> dict[str, Any]:
    """Refuse to freeze a partition that disagrees with the training config."""
    import yaml

    config = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    splits = config["sampling"]["splits"]
    trained = set(splits["train"]["crop_budgets"])
    heldout_declared = set(splits["val_heldout_crops"]["crop_budgets"])

    if trained != set(ID_CROPS):
        raise ValueError(
            f"{TRAINING_CONFIG.name} trains on {sorted(trained)} but this pool "
            f"labels {sorted(ID_CROPS)} as in-distribution."
        )
    if heldout_declared != set(HELDOUT_CROPS):
        raise ValueError(
            f"{TRAINING_CONFIG.name} holds out {sorted(heldout_declared)} but this "
            f"pool labels {sorted(HELDOUT_CROPS)}."
        )
    if trained & set(HELDOUT_CROPS):
        raise ValueError(
            f"Crops labelled held-out were trained on: {sorted(trained & set(HELDOUT_CROPS))}."
        )
    return {
        "training_config": str(TRAINING_CONFIG.relative_to(REPO_ROOT)),
        "id_crops": sorted(ID_CROPS),
        "heldout_crops": sorted(HELDOUT_CROPS),
    }


def load_existing_manifest() -> dict[str, Any] | None:
    if not MANIFEST_PATH.is_file():
        return None
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def ensure_local_meteo_cache_link() -> Path:
    """Attach a verified local cache without copying its large payload."""
    if OUTPUT_METEO_CACHE.is_dir():
        return OUTPUT_METEO_CACHE.resolve()
    if OUTPUT_METEO_CACHE.is_symlink():
        raise FileNotFoundError(f"Broken weather-cache symlink: {OUTPUT_METEO_CACHE}")

    candidates = list(LOCAL_METEO_CACHE_CANDIDATES) + sorted(
        WOFOST_CACHE_ROOT.glob(LOCAL_METEO_CACHE_GLOB)
    )
    for candidate in candidates:
        if candidate.is_dir():
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            OUTPUT_METEO_CACHE.symlink_to(candidate.resolve(), target_is_directory=True)
            return candidate.resolve()

    checked = "\n".join(f"  - {path}" for path in candidates) or "  (nothing)"
    raise FileNotFoundError(
        "No local 20-crop weather cache is available. Download the public "
        f"{SOURCE_REPO} weather bundle first. Checked:\n{checked}"
    )


def main() -> None:
    partition = check_crop_partition()

    per_crop: dict[str, dict[str, Any]] = {}
    selected_keys_all: set[ScenarioKey] = set()
    train_keys_all: set[ScenarioKey] = set()
    val_keys_all: set[ScenarioKey] = set()
    test_keys_all: set[ScenarioKey] = set()

    for crop in ALL_CROPS:
        train_df = pd.read_parquet(download_source_parquet(f"train/{crop}.parquet"))
        val_df = pd.read_parquet(download_source_parquet(f"val/{crop}.parquet"))
        test_df = pd.read_parquet(download_source_parquet(f"test/{crop}.parquet"))

        train_keys = dataframe_keys(train_df, crop)
        val_keys = dataframe_keys(val_df, crop)
        test_keys = dataframe_keys(test_df, crop)

        # The whole disjointness argument rests on this. Check it per crop
        # rather than trusting the pool's split names.
        overlaps = {
            "train_vs_val": len(train_keys & val_keys),
            "train_vs_test": len(train_keys & test_keys),
            "val_vs_test": len(val_keys & test_keys),
        }
        if any(overlaps.values()):
            raise ValueError(f"Public {crop} pool splits overlap: {overlaps}")

        selected = select_rows(
            test_df, crop=crop, count=EVAL_COUNT_PER_CROP, seed=EVAL_SELECTION_SEED
        )
        selected_keys = dataframe_keys(selected, crop)
        if selected_keys & (train_keys | val_keys):
            raise ValueError(f"Selected {crop} scenarios reach train or validation.")

        output_parquet = OUTPUT_DIR / "test" / f"{crop}.parquet"
        existing = load_existing_manifest()
        if existing is not None:
            frozen = [
                record
                for record in existing["selected_scenarios"]
                if record["crop"] == crop
            ]
            if frozen != sorted_key_records(selected_keys):
                raise ValueError(
                    f"The frozen T1.2 manifest does not match the deterministic "
                    f"selection for {crop}. Refusing to overwrite it."
                )
            if not output_parquet.is_file():
                raise FileNotFoundError(
                    f"Frozen manifest exists but source Parquet is missing: {output_parquet}"
                )
            if dataframe_keys(pd.read_parquet(output_parquet), crop) != selected_keys:
                raise ValueError(f"The frozen T1.2 {crop} Parquet does not match its manifest.")
        else:
            output_parquet.parent.mkdir(parents=True, exist_ok=True)
            selected.to_parquet(output_parquet, index=False)

        per_crop[crop] = {
            "crop_regime": "id" if crop in ID_CROPS else "heldout",
            "public_train": len(train_keys),
            "public_validation": len(val_keys),
            "public_test": len(test_keys),
            "selected_eval": len(selected_keys),
            "overlap": overlaps,
        }
        selected_keys_all |= selected_keys
        train_keys_all |= train_keys
        val_keys_all |= val_keys
        test_keys_all |= test_keys

    if load_existing_manifest() is None:
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": {
                "repo_id": SOURCE_REPO,
                "revision": SOURCE_REVISION,
                "source_split": "test",
            },
            "physical_key_fields": ["crop", "year", "latitude", "longitude"],
            "crop_partition": partition,
            "eval_selection": {
                "generation_seed": EVAL_SELECTION_SEED,
                "num_scenarios_per_crop": EVAL_COUNT_PER_CROP,
                "num_scenarios": len(selected_keys_all),
                "source_split": "test",
                # No exclusion list. The pool ships a real test split that is
                # disjoint from train and val, so there is nothing to exclude.
                "excluded": None,
            },
            "selected_scenarios": sorted_key_records(selected_keys_all),
        }
        MANIFEST_PATH.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    checksums = {
        f"test/{crop}.parquet": f"sha256:{sha256_file(OUTPUT_DIR / 'test' / f'{crop}.parquet')}"
        for crop in ALL_CROPS
    }
    checksums["manifest.json"] = f"sha256:{sha256_file(MANIFEST_PATH)}"

    meteo_cache_path = ensure_local_meteo_cache_link()
    audit = {
        "protocol_id": PROTOCOL_ID,
        "status": "pass",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "crop_partition": partition,
        "counts": {
            "crops": len(ALL_CROPS),
            "public_train": len(train_keys_all),
            "public_validation": len(val_keys_all),
            "public_test": len(test_keys_all),
            "selected_eval": len(selected_keys_all),
        },
        "overlap": {
            "eval_vs_public_train": len(selected_keys_all & train_keys_all),
            "eval_vs_public_validation": len(selected_keys_all & val_keys_all),
            "public_train_vs_public_validation": len(train_keys_all & val_keys_all),
            "public_train_vs_public_test": len(train_keys_all & test_keys_all),
            "public_validation_vs_public_test": len(val_keys_all & test_keys_all),
        },
        "per_crop": per_crop,
        # Repo-relative: the resolved path names the building machine and the
        # account its pool cache was downloaded under, and this audit ships
        # alongside the suites.
        "local_meteo_cache": {
            "path": relative_to_repo(meteo_cache_path),
            "status": "available",
        },
        "checksums": checksums,
    }
    AUDIT_PATH.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"Frozen E4 T1.2 source: {OUTPUT_DIR}")
    print(f"Crops: {len(ALL_CROPS)} ({len(ID_CROPS)} in-distribution, {len(HELDOUT_CROPS)} held-out)")
    print(f"Selected scenarios: {len(selected_keys_all)} ({EVAL_COUNT_PER_CROP} per crop)")
    print(f"Overlap with public train: {audit['overlap']['eval_vs_public_train']}")
    print(f"Overlap with public validation: {audit['overlap']['eval_vs_public_validation']}")
    print(f"Manifest SHA-256: {checksums['manifest.json'].removeprefix('sha256:')}")


if __name__ == "__main__":
    main()
