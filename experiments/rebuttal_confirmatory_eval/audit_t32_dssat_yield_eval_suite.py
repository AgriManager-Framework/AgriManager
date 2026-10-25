"""Audit the disjoint T3.2 DSSAT confirmatory seed block."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


EXPERIMENT_DIR = Path(__file__).resolve().parent
DATASET_DIR = EXPERIMENT_DIR / "data" / "e4_t32_dssat_yield"
PARQUET_PATH = DATASET_DIR / "test.parquet"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
AUDIT_PATH = DATASET_DIR / "audit.json"

SUBMITTED_TRAIN_SEEDS = set(range(42, 42 + 400))
SUBMITTED_VALIDATION_SEEDS = set(range(100042, 100042 + 128))
EXPECTED_TEST_SEEDS = set(range(200042, 200042 + 128))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if not PARQUET_PATH.is_file():
        raise FileNotFoundError(f"Missing T3.2 DSSAT E4 split: {PARQUET_PATH}")
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing T3.2 DSSAT manifest: {MANIFEST_PATH}")

    records = pd.read_parquet(PARQUET_PATH).to_dict("records")
    seeds: set[int] = set()
    scenario_ids: set[str] = set()
    observed = {
        "env_id": set(),
        "schema_tuple": set(),
        "objective_id": set(),
        "dataset_role": set(),
    }
    for record in records:
        env_config = record["extra_info"]["interaction_kwargs"]["env_config"]
        labels = env_config["trajectory_group_labels"]
        seeds.add(int(env_config["seed"]))
        scenario_ids.add(str(env_config["scenario_id"]))
        observed["env_id"].add(str(env_config["env_id"]))
        observed["schema_tuple"].add(str(labels["schema_tuple"]))
        observed["objective_id"].add(str(env_config["objective_id"]))
        observed["dataset_role"].add(str(env_config["dataset_role"]))

    overlap_train = seeds.intersection(SUBMITTED_TRAIN_SEEDS)
    overlap_validation = seeds.intersection(SUBMITTED_VALIDATION_SEEDS)
    if len(records) != 128 or len(seeds) != 128 or len(scenario_ids) != 128:
        raise ValueError(
            "DSSAT E4 is not a one-to-one 128-scenario block: "
            f"rows={len(records)}, seeds={len(seeds)}, "
            f"scenario_ids={len(scenario_ids)}."
        )
    if seeds != EXPECTED_TEST_SEEDS:
        raise ValueError("DSSAT E4 seeds do not match the frozen 200042-200169 block.")
    if overlap_train or overlap_validation:
        raise ValueError(
            "DSSAT E4 seed overlap detected: "
            f"train={len(overlap_train)}, validation={len(overlap_validation)}."
        )

    expected_observed = {
        "env_id": {"maize-irrigation-v0"},
        "schema_tuple": {"dssat_yield"},
        "objective_id": {"yield_max"},
        "dataset_role": {"test"},
    }
    if observed != expected_observed:
        raise ValueError(
            f"Unexpected DSSAT E4 metadata: {observed}; expected {expected_observed}."
        )

    audit = {
        "protocol_id": "e4-t32-dssat-yield-v1",
        "status": "pass",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario_key_fields": ["env_id", "seed"],
        "submitted_train_seed_range": [42, 441],
        "submitted_validation_seed_range": [100042, 100169],
        "confirmatory_test_seed_range": [200042, 200169],
        "counts": {
            "rows": len(records),
            "unique_seeds": len(seeds),
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

    print("T3.2 DSSAT E4 audit passed")
    print("Rows: 128")
    print("Test seed range: 200042-200169")
    print("Overlap with submitted train: 0")
    print("Overlap with submitted validation: 0")
    print(f"Audit report: {AUDIT_PATH}")


if __name__ == "__main__":
    main()

