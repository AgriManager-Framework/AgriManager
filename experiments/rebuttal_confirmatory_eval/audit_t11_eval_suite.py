"""Audit a rendered T1.1 suite against the frozen pool and the training data.

Two properties have to hold before this suite can be called a test set.

Disjointness is the one the meta-review asked for: no scenario here may appear
in any split of the T1.1 training dataset, including its validation splits,
because the submitted paper reported validation numbers as final results.

Pairing is what makes the LLM-versus-numerical comparison meaningful: pass
`--paired-with` and the two suites must select the same physical scenarios.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent.parent
CROPS = ("potato", "chickpea")


def pool_dir(crop: str) -> Path:
    return EXPERIMENT_DIR / "data" / "source_pools" / f"t11_{crop}_e4"


def default_dataset_dir(crop: str) -> Path:
    return EXPERIMENT_DIR / "data" / f"e4_t11_{crop}_weather_regime"


def training_data_dir(crop: str) -> Path:
    return (
        REPO_ROOT
        / "experiments"
        / "t1_1_weather_regime_shift"
        / "data"
        / f"weather_regime_{crop}_nn_without_traits"
    )

EXPECTED_SPLITS = {
    "test_id": "id",
    "test_drought": "drought",
    "test_wet": "wet",
    "test_hot": "hot",
    "test_cold": "cold",
}

PhysicalKey = tuple[int, float, float]


def physical_key(year: Any, latitude: Any, longitude: Any) -> PhysicalKey:
    return (int(year), round(float(latitude), 4), round(float(longitude), 4))


def pool_keys(crop: str, regime: str) -> set[PhysicalKey]:
    frame = pd.read_parquet(pool_dir(crop) / f"test_{regime}" / f"{crop}.parquet")
    return {
        physical_key(row["year"], row["latitude"], row["longitude"])
        for _, row in frame.iterrows()
    }


def rendered_keys(path: Path) -> set[PhysicalKey]:
    keys: set[PhysicalKey] = set()
    for record in pd.read_parquet(path).to_dict("records"):
        env_config = record["extra_info"]["interaction_kwargs"]["env_config"]
        agro_params = env_config["agro_params"]
        keys.add(
            physical_key(
                agro_params["year"], agro_params["latitude"], agro_params["longitude"]
            )
        )
    return keys


def training_keys(crop: str) -> tuple[set[PhysicalKey], list[str]]:
    """Every scenario the T1.1 experiment trained or validated on."""
    directory = training_data_dir(crop)
    keys: set[PhysicalKey] = set()
    consumed: list[str] = []
    for path in sorted(directory.glob("*.parquet")):
        keys |= rendered_keys(path)
        consumed.append(path.name)
    if not consumed:
        raise FileNotFoundError(
            f"No T1.1 training parquets under {directory}. The disjointness "
            "check cannot be skipped: build that dataset first."
        )
    return keys, consumed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", choices=CROPS, default="potato")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--paired-with", type=Path, default=None)
    args = parser.parse_args()

    crop = args.crop
    dataset_dir = args.dataset_dir or default_dataset_dir(crop)
    manifest_path = dataset_dir / "manifest.json"
    audit_path = dataset_dir / "audit.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing rendered dataset manifest: {manifest_path}")

    train_keys, consumed = training_keys(crop)

    split_report: dict[str, Any] = {}
    all_keys: set[PhysicalKey] = set()
    for split, regime in EXPECTED_SPLITS.items():
        path = dataset_dir / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing T1.1 E4 split: {path}")

        keys = rendered_keys(path)
        num_rows = len(pd.read_parquet(path))
        if num_rows != 128 or len(keys) != 128:
            raise ValueError(
                f"{split} is not a one-to-one 128-scenario rendering: "
                f"rows={num_rows}, physical_keys={len(keys)}."
            )
        if keys != pool_keys(crop, regime):
            raise ValueError(f"{split} does not match the frozen {regime} pool group.")

        leaked = keys & train_keys
        if leaked:
            raise ValueError(
                f"{split} shares {len(leaked)} scenarios with the T1.1 training "
                f"dataset. This suite would not be a held-out test set."
            )

        overlap_with_earlier = keys & all_keys
        if overlap_with_earlier:
            raise ValueError(
                f"{split} shares {len(overlap_with_earlier)} scenarios with another "
                "regime group; the groups must be disjoint."
            )
        all_keys |= keys

        split_report[split] = {
            "weather_regime": regime,
            "num_rows": num_rows,
            "num_physical_keys": len(keys),
            "scenarios_shared_with_training": 0,
            "sha256": sha256_file(path),
        }

    cross_suite = None
    if args.paired_with is not None:
        other: set[PhysicalKey] = set()
        for split in EXPECTED_SPLITS:
            other |= rendered_keys(args.paired_with / f"{split}.parquet")
        if other != all_keys:
            raise ValueError(
                f"{dataset_dir.name} and {args.paired_with.name} do not select the "
                f"same scenarios: {len(all_keys ^ other)} differ. The "
                "LLM-versus-numerical comparison would not be paired."
            )
        cross_suite = args.paired_with.name

    audit = {
        "protocol_id": f"e4-t11-{crop}-weather-regime-v1",
        "crop": crop,
        "status": "pass",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "paired_with": cross_suite,
        "total_scenarios": len(all_keys),
        "regime_groups_are_disjoint": True,
        "disjoint_from_training": True,
        "training_parquets_checked": consumed,
        "training_scenarios_checked": len(train_keys),
        "manifest_sha256": sha256_file(manifest_path),
        "splits": split_report,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"T1.1 {crop} E4 audit passed: {len(split_report)} weather-regime groups")
    print(f"Rows per group: 128, total scenarios: {len(all_keys)}")
    print(
        f"Disjoint from {len(train_keys)} training/validation scenarios across "
        f"{len(consumed)} parquets"
    )
    if cross_suite:
        print(f"Paired with: {cross_suite}")
    print(f"Audit report: {audit_path}")


if __name__ == "__main__":
    main()
