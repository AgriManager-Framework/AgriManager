"""Audit the rendered T2.1 confirmatory suite against the frozen maize base."""

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
DATASET_DIR = EXPERIMENT_DIR / "data" / "e4_t21_observation_schema"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
AUDIT_PATH = DATASET_DIR / "audit.json"

EXPECTED_SPLITS = {
    "test_s1": "S1_full_current",
    "test_s2a": "S2a_no_stage_time",
    "test_s2b": "S2b_no_resource_state",
    "test_s2c": "S2c_no_management_history",
    "test_s2d": "S2d_no_weather_context",
    "test_s3": "S3_domain_synonym_rename",
    "test_s4": "S4_compact_growth_superset",
    "test_s5": "S5_anonymous_label_rename",
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


def rendered_records(path: Path) -> list[dict[str, Any]]:
    return pd.read_parquet(path).to_dict("records")


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
    for split, schema in EXPECTED_SPLITS.items():
        path = DATASET_DIR / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing T2.1 E4 split: {path}")
        records = rendered_records(path)
        keys = {rendered_key(record) for record in records}
        paired_ids: set[str] = set()
        observed_schemas: set[str] = set()
        observed_roles: set[str] = set()
        for record in records:
            env_config = record["extra_info"]["interaction_kwargs"]["env_config"]
            labels = env_config["trajectory_group_labels"]
            paired_ids.add(str(env_config["paired_scenario_id"]))
            observed_schemas.add(str(labels["observation_schema"]))
            observed_roles.add(str(env_config["dataset_role"]))

        if len(records) != 128 or len(keys) != 128 or len(paired_ids) != 128:
            raise ValueError(
                f"{split} is not a one-to-one 128-scenario rendering: "
                f"rows={len(records)}, physical_keys={len(keys)}, "
                f"paired_ids={len(paired_ids)}."
            )
        if keys != expected_keys:
            raise ValueError(f"{split} does not match the frozen maize source keys.")
        if observed_schemas != {schema}:
            raise ValueError(
                f"{split} has schemas {sorted(observed_schemas)}, expected {schema}."
            )
        if observed_roles != {"test"}:
            raise ValueError(
                f"{split} has dataset roles {sorted(observed_roles)}, expected test."
            )
        if reference_paired_ids is None:
            reference_paired_ids = paired_ids
        elif paired_ids != reference_paired_ids:
            raise ValueError(
                f"{split} does not share paired_scenario_id values with test_s1."
            )

        split_report[split] = {
            "observation_schema": schema,
            "num_rows": len(records),
            "num_physical_keys": len(keys),
            "num_paired_scenario_ids": len(paired_ids),
            "sha256": sha256_file(path),
        }

    audit = {
        "protocol_id": "e4-t21-observation-schema-v1",
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

    print(f"T2.1 E4 audit passed: {len(split_report)} schema splits")
    print("Rows per split: 128")
    print("Shared physical scenarios: 128")
    print("Shared paired_scenario_id values: 128")
    print(f"Audit report: {AUDIT_PATH}")


if __name__ == "__main__":
    main()

