"""Audit the three seen-control T3.2 WOFOST confirmatory cells."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
SOURCE_PARQUET = (
    EXPERIMENT_DIR
    / "data"
    / "source_pools"
    / "maize_e4"
    / "test"
    / "maize.parquet"
)
DATASET_DIR = EXPERIMENT_DIR / "data" / "e4_t32_wofost_seen"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
AUDIT_PATH = DATASET_DIR / "audit.json"

EXPECTED_SPLITS = {
    "test_wofost_lnpkw_profit": {
        "schema_tuple": "wofost_s1_lnpkw_profit",
        "action_menu": "lnpkw",
        "env_id": "lnpkw-v0",
        "objective_id": "profit_max",
    },
    "test_wofost_lnpk_profit": {
        "schema_tuple": "wofost_s1_lnpk_profit",
        "action_menu": "lnpk",
        "env_id": "lnpk-v0",
        "objective_id": "profit_max",
    },
    "test_wofost_lnw_yield": {
        "schema_tuple": "wofost_s1_lnw_yield",
        "action_menu": "lnw",
        "env_id": "lnw-v0",
        "objective_id": "yield_max",
    },
}

PhysicalKey = tuple[str, int, float, float]


def physical_key(
    crop: str,
    year: int,
    latitude: float,
    longitude: float,
) -> PhysicalKey:
    return (
        str(crop),
        int(year),
        round(float(latitude), 2),
        round(float(longitude), 2),
    )


def source_keys() -> set[PhysicalKey]:
    df = pd.read_parquet(SOURCE_PARQUET)
    return {
        physical_key("maize", row.year, row.latitude, row.longitude)
        for row in df.itertuples(index=False)
    }


def rendered_key(record: dict[str, Any]) -> PhysicalKey:
    env_config = record["extra_info"]["interaction_kwargs"]["env_config"]
    agro_params = env_config["agro_params"]
    return physical_key(
        env_config["crop_name"],
        agro_params["year"],
        agro_params["latitude"],
        agro_params["longitude"],
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    expected_keys = source_keys()
    if len(expected_keys) != 128:
        raise ValueError(
            f"Expected 128 frozen maize scenarios, found {len(expected_keys)}."
        )
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing rendered dataset manifest: {MANIFEST_PATH}")

    split_report: dict[str, Any] = {}
    reference_paired_ids: set[str] | None = None
    for split, expected in EXPECTED_SPLITS.items():
        path = DATASET_DIR / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing T3.2 WOFOST E4 split: {path}")
        records = pd.read_parquet(path).to_dict("records")
        keys = {rendered_key(record) for record in records}
        paired_ids: set[str] = set()
        observed = {
            "schema_tuple": set(),
            "action_menu": set(),
            "env_id": set(),
            "objective_id": set(),
            "dataset_role": set(),
            "y_ref": set(),
        }
        for record in records:
            env_config = record["extra_info"]["interaction_kwargs"]["env_config"]
            labels = env_config["trajectory_group_labels"]
            paired_ids.add(str(env_config["paired_scenario_id"]))
            observed["schema_tuple"].add(str(labels["schema_tuple"]))
            observed["action_menu"].add(str(labels["action_menu"]))
            observed["env_id"].add(str(env_config["env_id"]))
            observed["objective_id"].add(str(env_config["objective_id"]))
            observed["dataset_role"].add(str(env_config["dataset_role"]))
            observed["y_ref"].add(float(env_config["y_ref"]))

        if len(records) != 128 or len(keys) != 128 or len(paired_ids) != 128:
            raise ValueError(
                f"{split} is not a one-to-one 128-scenario rendering: "
                f"rows={len(records)}, physical_keys={len(keys)}, "
                f"paired_ids={len(paired_ids)}."
            )
        if keys != expected_keys:
            raise ValueError(f"{split} does not match the frozen maize source keys.")
        for field, value in expected.items():
            if observed[field] != {value}:
                raise ValueError(
                    f"{split} has {field}={sorted(observed[field])}, expected {value}."
                )
        if observed["dataset_role"] != {"test"}:
            raise ValueError(
                f"{split} has roles {sorted(observed['dataset_role'])}, expected test."
            )
        if observed["y_ref"] != {8500.0}:
            raise ValueError(
                f"{split} has y_ref={sorted(observed['y_ref'])}, expected 8500.0."
            )
        if reference_paired_ids is None:
            reference_paired_ids = paired_ids
        elif paired_ids != reference_paired_ids:
            raise ValueError(
                f"{split} does not share paired_scenario_id values with the "
                "other T3.2 WOFOST cell."
            )

        split_report[split] = {
            **expected,
            "num_rows": len(records),
            "num_physical_keys": len(keys),
            "num_paired_scenario_ids": len(paired_ids),
            "y_ref": 8500.0,
            "sha256": sha256_file(path),
        }

    audit = {
        "protocol_id": "e4-t32-wofost-seen-v1",
        "status": "pass",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_scenario_count": len(expected_keys),
        "all_splits_share_physical_scenarios": True,
        "all_splits_share_paired_scenario_ids": True,
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "splits": split_report,
    }
    AUDIT_PATH.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"T3.2 WOFOST E4 audit passed: {len(split_report)} seen-control cells")
    print("Rows per cell: 128")
    print("Shared physical scenarios: 128")
    print("Shared paired_scenario_id values: 128")
    print("Fixed y_ref: 8500.0")
    print(f"Audit report: {AUDIT_PATH}")


if __name__ == "__main__":
    main()

