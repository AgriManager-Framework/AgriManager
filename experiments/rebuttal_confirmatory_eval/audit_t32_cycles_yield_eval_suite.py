"""Audit the disjoint T3.2 Cycles confirmatory scenario block."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
DATASET_DIR = EXPERIMENT_DIR / "data" / "e4_t32_cycles_yield"
PARQUET_PATH = DATASET_DIR / "test.parquet"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
AUDIT_PATH = DATASET_DIR / "audit.json"

ENV_ID = "CornShortRockSpringsFW-v1"


def scenario_block(
    years: range,
    *,
    seeds_per_year: int,
    seed_start: int,
) -> set[tuple[str, int, int, int]]:
    scenarios: set[tuple[str, int, int, int]] = set()
    for year_index, year in enumerate(years):
        for seed_offset in range(seeds_per_year):
            seed = seed_start + year_index * seeds_per_year + seed_offset
            scenarios.add((ENV_ID, year, year, seed))
    return scenarios


SUBMITTED_TRAIN = scenario_block(
    range(1980, 1996),
    seeds_per_year=25,
    seed_start=42,
)
SUBMITTED_VALIDATION = scenario_block(
    range(1996, 2004),
    seeds_per_year=16,
    seed_start=100042,
)
EXPECTED_TEST = scenario_block(
    range(1996, 2004),
    seeds_per_year=16,
    seed_start=200042,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not PARQUET_PATH.is_file():
        raise FileNotFoundError(f"Missing T3.2 Cycles E4 split: {PARQUET_PATH}")
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing T3.2 Cycles manifest: {MANIFEST_PATH}")

    records = pd.read_parquet(PARQUET_PATH).to_dict("records")
    scenarios: set[tuple[str, int, int, int]] = set()
    scenario_ids: set[str] = set()
    observed = {
        "schema_tuple": set(),
        "objective_id": set(),
        "dataset_role": set(),
    }
    for record in records:
        env_config = record["extra_info"]["interaction_kwargs"]["env_config"]
        env_kwargs = env_config["env_kwargs"]
        labels = env_config["trajectory_group_labels"]
        scenarios.add(
            (
                str(env_config["env_id"]),
                int(env_kwargs["start_year"]),
                int(env_kwargs["end_year"]),
                int(env_config["seed"]),
            )
        )
        scenario_ids.add(str(env_config["scenario_id"]))
        observed["schema_tuple"].add(str(labels["schema_tuple"]))
        observed["objective_id"].add(str(env_config["objective_id"]))
        observed["dataset_role"].add(str(env_config["dataset_role"]))

    overlap_train = scenarios.intersection(SUBMITTED_TRAIN)
    overlap_validation = scenarios.intersection(SUBMITTED_VALIDATION)
    if len(records) != 128 or len(scenarios) != 128 or len(scenario_ids) != 128:
        raise ValueError(
            "Cycles E4 is not a one-to-one 128-scenario block: "
            f"rows={len(records)}, scenarios={len(scenarios)}, "
            f"scenario_ids={len(scenario_ids)}."
        )
    if scenarios != EXPECTED_TEST:
        raise ValueError(
            "Cycles E4 scenarios do not match the frozen 1996-2003, "
            "seed 200042-200169 block."
        )
    if overlap_train or overlap_validation:
        raise ValueError(
            "Cycles E4 overlap detected: "
            f"train={len(overlap_train)}, validation={len(overlap_validation)}."
        )

    expected_observed = {
        "schema_tuple": {"cycles_yield"},
        "objective_id": {"yield_max"},
        "dataset_role": {"test"},
    }
    if observed != expected_observed:
        raise ValueError(
            f"Unexpected Cycles E4 metadata: {observed}; expected {expected_observed}."
        )

    audit = {
        "protocol_id": "e4-t32-cycles-yield-v1",
        "status": "pass",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_key_fields": [
            "env_id",
            "start_year",
            "end_year",
            "seed",
        ],
        "submitted_train": {
            "year_range": [1980, 1995],
            "seeds_per_year": 25,
            "seed_range": [42, 441],
        },
        "submitted_validation": {
            "year_range": [1996, 2003],
            "seeds_per_year": 16,
            "seed_range": [100042, 100169],
        },
        "confirmatory_test": {
            "year_range": [1996, 2003],
            "seeds_per_year": 16,
            "seed_range": [200042, 200169],
        },
        "counts": {
            "rows": len(records),
            "unique_scenarios": len(scenarios),
            "unique_scenario_ids": len(scenario_ids),
        },
        "overlap": {
            "test_vs_submitted_train": len(overlap_train),
            "test_vs_submitted_validation": len(overlap_validation),
        },
        "checksums": {
            "test.parquet": f"sha256:{sha256_file(PARQUET_PATH)}",
            "manifest.json": f"sha256:{sha256_file(MANIFEST_PATH)}",
        },
    }
    AUDIT_PATH.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("T3.2 Cycles E4 audit passed")
    print("Rows: 128")
    print("Test years: 1996-2003")
    print("Test seed range: 200042-200169")
    print("Overlap with submitted train: 0")
    print("Overlap with submitted validation: 0")
    print(f"Audit report: {AUDIT_PATH}")


if __name__ == "__main__":
    main()

