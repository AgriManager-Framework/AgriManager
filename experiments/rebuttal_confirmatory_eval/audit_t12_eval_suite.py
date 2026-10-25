"""Audit the rendered T1.2 cross-crop suite against the frozen source pool.

The claim a T1.2 number supports is that a policy trained on four crops
transfers to four it never saw. That needs three things checked:

1. Each split renders exactly the frozen scenarios for its own crops, one row
   per scenario, so neither split is silently short.
2. The two splits' crop sets are disjoint, and match what the submitted
   training config actually trained on. A crop that appears in both would make
   the held-out number meaningless.
3. Everything except the crop is held fixed -- same action menu, same
   objective, same trait schema -- so the axis under test is the crop alone.

Scenario-level disjointness from training is inherited from the pool audit and
re-checked here rather than assumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
SOURCE_POOL_DIR = EXPERIMENT_DIR / "data" / "source_pools" / "t12_cross_crop_e4"
POOL_AUDIT = SOURCE_POOL_DIR / "audit.json"
TRAINING_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "t1_2_cross_crop_trait_shift"
    / "config"
    / "cross_crop_4id_llm_traits_v1_23d_think.yaml"
)
DEFAULT_DATASET_DIR = EXPERIMENT_DIR / "data" / "e4_t12_cross_crop"

EXPECTED_ENV_ID = "lnpkw-v0"
EXPECTED_OBJECTIVE = "profit_max"
EXPECTED_TRAIT_SCHEMA = "traits_v1_23d"
EXPECTED_ROWS_PER_CROP = 128

EXPECTED_SPLITS = {
    "test_id_crops": ("id", ("cotton", "millet", "rapeseed", "rice")),
    "test_heldout_crops": ("heldout", ("barley", "groundnut", "pigeonpea", "seed_onion")),
}

PhysicalKey = tuple[str, int, float, float]


def physical_key(crop: str, year: int, latitude: float, longitude: float) -> PhysicalKey:
    return (str(crop), int(year), round(float(latitude), 2), round(float(longitude), 2))


def source_keys(crops: tuple[str, ...]) -> set[PhysicalKey]:
    keys: set[PhysicalKey] = set()
    for crop in crops:
        path = SOURCE_POOL_DIR / "test" / f"{crop}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Frozen pool is missing {crop}: {path}")
        df = pd.read_parquet(path)
        keys |= {
            physical_key(crop, row.year, row.latitude, row.longitude)
            for row in df.itertuples(index=False)
        }
    return keys


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


def check_training_disjointness() -> dict[str, Any]:
    if not POOL_AUDIT.is_file():
        raise FileNotFoundError(f"Missing frozen pool audit: {POOL_AUDIT}")
    pool_audit = json.loads(POOL_AUDIT.read_text(encoding="utf-8"))
    for field in ("eval_vs_public_train", "eval_vs_public_validation"):
        overlap = pool_audit["overlap"][field]
        if overlap != 0:
            raise ValueError(
                f"The frozen T1.2 pool overlaps {field} in {overlap} scenarios."
            )

    config = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    trained = set(config["sampling"]["splits"]["train"]["crop_budgets"])
    id_crops = set(EXPECTED_SPLITS["test_id_crops"][1])
    heldout_crops = set(EXPECTED_SPLITS["test_heldout_crops"][1])

    if trained != id_crops:
        raise ValueError(
            f"{TRAINING_CONFIG.name} trains on {sorted(trained)} but this suite "
            f"labels {sorted(id_crops)} as in-distribution."
        )
    leaked = trained & heldout_crops
    if leaked:
        raise ValueError(f"Crops labelled held-out were trained on: {sorted(leaked)}.")

    return {
        "pool_eval_vs_public_train_overlap": pool_audit["overlap"]["eval_vs_public_train"],
        "pool_eval_vs_public_validation_overlap": pool_audit["overlap"][
            "eval_vs_public_validation"
        ],
        "training_config": str(TRAINING_CONFIG.relative_to(REPO_ROOT)),
        "trained_crops": sorted(trained),
        "held_out_crops": sorted(heldout_crops),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    manifest_path = dataset_dir / "manifest.json"
    audit_path = dataset_dir / "audit.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing rendered dataset manifest: {manifest_path}")

    disjointness = check_training_disjointness()

    split_report: dict[str, Any] = {}
    crops_seen: dict[str, str] = {}

    for split, (regime, crops) in EXPECTED_SPLITS.items():
        path = dataset_dir / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing T1.2 E4 split: {path}")
        records = pd.read_parquet(path).to_dict("records")
        keys = {rendered_key(record) for record in records}
        expected_keys = source_keys(crops)

        per_crop: dict[str, int] = {crop: 0 for crop in crops}
        observed_regimes: set[str] = set()
        observed_env_ids: set[str] = set()
        observed_objectives: set[str] = set()
        observed_roles: set[str] = set()
        observed_trait_schemas: set[str] = set()
        observed_traits_flag: set[bool] = set()

        for record in records:
            env_config = record["extra_info"]["interaction_kwargs"]["env_config"]
            labels = env_config["trajectory_group_labels"]
            crop = str(env_config["crop_name"])
            if crop not in per_crop:
                raise ValueError(f"{split} contains unexpected crop {crop!r}.")
            per_crop[crop] += 1
            observed_regimes.add(str(labels["crop_regime"]))
            observed_env_ids.add(str(env_config["env_id"]))
            observed_objectives.add(str(env_config["objective_id"]))
            observed_roles.add(str(env_config["dataset_role"]))
            observed_trait_schemas.add(str(env_config.get("trait_schema")))
            observed_traits_flag.add(bool(env_config.get("include_crop_traits")))

        expected_rows = EXPECTED_ROWS_PER_CROP * len(crops)
        if len(records) != expected_rows or len(keys) != expected_rows:
            raise ValueError(
                f"{split} is not a one-to-one {expected_rows}-scenario rendering: "
                f"rows={len(records)}, physical_keys={len(keys)}."
            )
        if keys != expected_keys:
            raise ValueError(f"{split} does not match the frozen T1.2 source keys.")
        short = {c: n for c, n in per_crop.items() if n != EXPECTED_ROWS_PER_CROP}
        if short:
            raise ValueError(
                f"{split} is not balanced across crops; expected "
                f"{EXPECTED_ROWS_PER_CROP} each, got {short}."
            )
        if observed_regimes != {regime}:
            raise ValueError(
                f"{split} has crop regimes {sorted(observed_regimes)}, expected {regime}."
            )
        if observed_env_ids != {EXPECTED_ENV_ID}:
            raise ValueError(
                f"{split} has env IDs {sorted(observed_env_ids)}, expected "
                f"{EXPECTED_ENV_ID}. T1.2 must vary the crop alone."
            )
        if observed_objectives != {EXPECTED_OBJECTIVE}:
            raise ValueError(
                f"{split} has objectives {sorted(observed_objectives)}, expected "
                f"{EXPECTED_OBJECTIVE}."
            )
        if observed_roles != {"test"}:
            raise ValueError(
                f"{split} has dataset roles {sorted(observed_roles)}, expected test."
            )
        if observed_traits_flag != {True}:
            raise ValueError(
                f"{split} does not carry crop traits. T1.2 is the trait-shift axis; "
                "without traits the held-out crops are unidentifiable."
            )
        if observed_trait_schemas != {EXPECTED_TRAIT_SCHEMA}:
            raise ValueError(
                f"{split} has trait schemas {sorted(observed_trait_schemas)}, expected "
                f"{EXPECTED_TRAIT_SCHEMA}."
            )

        for crop in crops:
            if crop in crops_seen:
                raise ValueError(
                    f"Crop {crop!r} appears in both {crops_seen[crop]} and {split}; "
                    "the held-out split would not be held out."
                )
            crops_seen[crop] = split

        split_report[split] = {
            "crop_regime": regime,
            "crops": sorted(crops),
            "env_id": EXPECTED_ENV_ID,
            "objective_id": EXPECTED_OBJECTIVE,
            "trait_schema": EXPECTED_TRAIT_SCHEMA,
            "num_rows": len(records),
            "num_physical_keys": len(keys),
            "rows_per_crop": EXPECTED_ROWS_PER_CROP,
            "sha256": sha256_file(path),
        }

    audit = {
        "protocol_id": "e4-t12-cross-crop-v1",
        "status": "pass",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "splits_have_disjoint_crops": True,
        "all_splits_share_env_id": EXPECTED_ENV_ID,
        "all_splits_share_objective": EXPECTED_OBJECTIVE,
        "all_splits_share_trait_schema": EXPECTED_TRAIT_SCHEMA,
        "training_disjointness": disjointness,
        "manifest_sha256": sha256_file(manifest_path),
        "splits": split_report,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    total = sum(report["num_rows"] for report in split_report.values())
    print(f"T1.2 E4 audit passed: {len(split_report)} splits, {len(crops_seen)} crops")
    print(f"Rows: {total} ({EXPECTED_ROWS_PER_CROP} per crop)")
    print(f"Shared env_id: {EXPECTED_ENV_ID}   objective: {EXPECTED_OBJECTIVE}")
    print(f"Trait schema: {EXPECTED_TRAIT_SCHEMA}")
    print(f"Eval-vs-training overlap: {disjointness['pool_eval_vs_public_train_overlap']}")
    print(f"Audit report: {audit_path}")


if __name__ == "__main__":
    main()
