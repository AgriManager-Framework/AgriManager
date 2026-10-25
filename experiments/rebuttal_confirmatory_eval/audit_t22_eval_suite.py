"""Audit a rendered T2.2 action-menu suite against the frozen maize base.

Defaults to the LLM suite. Pass `--dataset-dir` to audit the numerical suite,
and `--paired-with` to additionally require that the two suites select the same
scenarios, which is what makes the LLM-versus-numerical comparison paired
rather than merely distributional.
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
SOURCE_PARQUET = (
    EXPERIMENT_DIR
    / "data"
    / "source_pools"
    / "maize_e4"
    / "test"
    / "maize.parquet"
)
DEFAULT_DATASET_DIR = EXPERIMENT_DIR / "data" / "e4_t22_action_menu"

EXPECTED_SPLITS = {
    "test_lnw": ("lnw", "lnw-v0"),
    "test_ln": ("ln", "ln-v0"),
    "test_lw": ("lw", "lw-v0"),
    "test_lnpkw": ("lnpkw", "lnpkw-v0"),
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


def paired_ids_of(dataset_dir: Path) -> set[str]:
    ids: set[str] = set()
    for split in EXPECTED_SPLITS:
        for record in pd.read_parquet(dataset_dir / f"{split}.parquet").to_dict("records"):
            ids.add(str(record["extra_info"]["interaction_kwargs"]["env_config"]["paired_scenario_id"]))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--paired-with", type=Path, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    manifest_path = dataset_dir / "manifest.json"
    audit_path = dataset_dir / "audit.json"

    expected_keys = source_keys()
    if len(expected_keys) != 128:
        raise ValueError(
            f"Expected 128 frozen maize scenarios, found {len(expected_keys)}."
        )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing rendered dataset manifest: {manifest_path}")

    split_report: dict[str, Any] = {}
    reference_paired_ids: set[str] | None = None
    for split, (menu, env_id) in EXPECTED_SPLITS.items():
        path = dataset_dir / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing T2.2 E4 split: {path}")
        records = pd.read_parquet(path).to_dict("records")
        keys = {rendered_key(record) for record in records}
        paired_ids: set[str] = set()
        observed_menus: set[str] = set()
        observed_env_ids: set[str] = set()
        observed_roles: set[str] = set()
        for record in records:
            env_config = record["extra_info"]["interaction_kwargs"]["env_config"]
            labels = env_config["trajectory_group_labels"]
            paired_ids.add(str(env_config["paired_scenario_id"]))
            observed_menus.add(str(labels["action_menu"]))
            observed_env_ids.add(str(env_config["env_id"]))
            observed_roles.add(str(env_config["dataset_role"]))

        if len(records) != 128 or len(keys) != 128 or len(paired_ids) != 128:
            raise ValueError(
                f"{split} is not a one-to-one 128-scenario rendering: "
                f"rows={len(records)}, physical_keys={len(keys)}, "
                f"paired_ids={len(paired_ids)}."
            )
        if keys != expected_keys:
            raise ValueError(f"{split} does not match the frozen maize source keys.")
        if observed_menus != {menu}:
            raise ValueError(
                f"{split} has action menus {sorted(observed_menus)}, expected {menu}."
            )
        if observed_env_ids != {env_id}:
            raise ValueError(
                f"{split} has env IDs {sorted(observed_env_ids)}, expected {env_id}."
            )
        if observed_roles != {"test"}:
            raise ValueError(
                f"{split} has dataset roles {sorted(observed_roles)}, expected test."
            )
        if reference_paired_ids is None:
            reference_paired_ids = paired_ids
        elif paired_ids != reference_paired_ids:
            raise ValueError(
                f"{split} does not share paired_scenario_id values with test_lnw."
            )

        split_report[split] = {
            "action_menu": menu,
            "env_id": env_id,
            "num_rows": len(records),
            "num_physical_keys": len(keys),
            "num_paired_scenario_ids": len(paired_ids),
            "sha256": sha256_file(path),
        }

    cross_suite = None
    if args.paired_with is not None:
        other = paired_ids_of(args.paired_with)
        if other != reference_paired_ids:
            raise ValueError(
                f"{dataset_dir.name} and {args.paired_with.name} do not select the "
                f"same scenarios: {len(reference_paired_ids ^ other)} paired_scenario_id "
                "values differ. The LLM-versus-numerical comparison would not be paired."
            )
        cross_suite = args.paired_with.name

    audit = {
        "protocol_id": "e4-t22-action-menu-v1",
        "paired_with": cross_suite,
        "status": "pass",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_scenario_count": len(expected_keys),
        "all_splits_share_physical_scenarios": True,
        "all_splits_share_paired_scenario_ids": True,
        "manifest_sha256": sha256_file(manifest_path),
        "splits": split_report,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"T2.2 E4 audit passed: {len(split_report)} action-menu splits")
    print("Rows per split: 128")
    print("Shared physical scenarios: 128")
    print("Shared paired_scenario_id values: 128")
    print(f"Audit report: {audit_path}")


if __name__ == "__main__":
    main()

