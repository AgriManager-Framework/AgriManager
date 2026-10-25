"""Audit the rendered T2.3 reward-formulation suite against the frozen maize base.

Three things have to hold for a T2.3 number to mean what the response will say
it means:

1. All four splits render the same 128 frozen scenarios, so seen-versus-held-out
   varies the objective and nothing else.
2. `env_id` is identical across splits. T2.3 is the reward axis; if the action
   menu moved too, the comparison would confound two axes.
3. The eval scenarios never appear in T2.3 training. That is checked
   structurally rather than by loading the training parquets, which are not
   kept: the frozen pool's own audit records zero overlap between the eval
   block and the public train split, and this script asserts that every T2.3
   training split does in fact draw from that train split.
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
SOURCE_POOL_DIR = EXPERIMENT_DIR / "data" / "source_pools" / "maize_e4"
SOURCE_PARQUET = SOURCE_POOL_DIR / "test" / "maize.parquet"
POOL_AUDIT = SOURCE_POOL_DIR / "audit.json"
TRAINING_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "t2_3_reward_formulation_shift"
    / "config"
    / "t23_reward_formulation_shift.yaml"
)
DEFAULT_DATASET_DIR = EXPERIMENT_DIR / "data" / "e4_t23_reward_formulation"

EXPECTED_ENV_ID = "lnpkw-v0"

# objective -> regime. The regime is the claim under test: three formulations
# were trained on, the fourth never was.
EXPECTED_SPLITS = {
    "test_yield_max": ("yield_max", "seen"),
    "test_profit_max": ("profit_max", "seen"),
    "test_water_stewardship": ("water_stewardship", "seen"),
    "test_nutrient_stewardship": ("nutrient_stewardship", "heldout"),
}

PhysicalKey = tuple[str, int, float, float]


def physical_key(crop: str, year: int, latitude: float, longitude: float) -> PhysicalKey:
    return (str(crop), int(year), round(float(latitude), 2), round(float(longitude), 2))


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


def check_training_disjointness() -> dict[str, Any]:
    """Prove the eval block cannot appear in T2.3 training, without the parquets.

    The pool audit establishes eval ∩ public_train = 0. This adds the missing
    half of the argument: that T2.3's training splits are drawn from
    public_train and from nowhere else.
    """
    if not POOL_AUDIT.is_file():
        raise FileNotFoundError(f"Missing frozen pool audit: {POOL_AUDIT}")
    pool_audit = json.loads(POOL_AUDIT.read_text(encoding="utf-8"))
    overlap = pool_audit["overlap"]["eval_vs_public_train"]
    if overlap != 0:
        raise ValueError(
            f"The frozen maize pool overlaps the public train split in {overlap} "
            "scenarios; the T2.3 eval block is not disjoint from training."
        )

    config = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    splits = config["sampling"]["splits"]
    train_sources = {
        name: spec.get("source_split")
        for name, spec in splits.items()
        if spec.get("role", "train") == "train"
    }
    if not train_sources:
        raise ValueError(f"{TRAINING_CONFIG} declares no training split.")
    off_pool = {name: src for name, src in train_sources.items() if src != "train"}
    if off_pool:
        raise ValueError(
            "T2.3 training draws from a split other than the public train split, "
            f"so the pool-level disjointness proof does not carry: {off_pool}."
        )

    trained_objectives = {
        splits[name]["env"]["objective_id"] for name in train_sources
    }
    declared_seen = {obj for obj, regime in EXPECTED_SPLITS.values() if regime == "seen"}
    if trained_objectives != declared_seen:
        raise ValueError(
            f"T2.3 trained on {sorted(trained_objectives)} but this suite labels "
            f"{sorted(declared_seen)} as seen."
        )
    held_out = {obj for obj, regime in EXPECTED_SPLITS.values() if regime == "heldout"}
    if held_out & trained_objectives:
        raise ValueError(
            f"Objectives labelled held-out were trained on: {sorted(held_out & trained_objectives)}."
        )

    return {
        "pool_eval_vs_public_train_overlap": overlap,
        "training_splits": sorted(train_sources),
        "training_source_splits": sorted(set(train_sources.values())),
        "trained_objectives": sorted(trained_objectives),
        "held_out_objectives": sorted(held_out),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
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

    disjointness = check_training_disjointness()

    split_report: dict[str, Any] = {}
    reference_paired_ids: set[str] | None = None
    reference_reward_params: str | None = None

    for split, (objective, regime) in EXPECTED_SPLITS.items():
        path = dataset_dir / f"{split}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing T2.3 E4 split: {path}")
        records = pd.read_parquet(path).to_dict("records")
        keys = {rendered_key(record) for record in records}

        paired_ids: set[str] = set()
        observed_objectives: set[str] = set()
        observed_prompt_objectives: set[str] = set()
        observed_regimes: set[str] = set()
        observed_env_ids: set[str] = set()
        observed_roles: set[str] = set()
        observed_reward_params: set[str] = set()

        for record in records:
            env_config = record["extra_info"]["interaction_kwargs"]["env_config"]
            labels = env_config["trajectory_group_labels"]
            paired_ids.add(str(env_config["paired_scenario_id"]))
            observed_objectives.add(str(env_config["objective_id"]))
            observed_prompt_objectives.add(str(env_config.get("prompt_objective_id")))
            observed_regimes.add(str(labels["reward_regime"]))
            observed_env_ids.add(str(env_config["env_id"]))
            observed_roles.add(str(env_config["dataset_role"]))
            observed_reward_params.add(
                json.dumps(env_config["reward_params"], sort_keys=True, default=str)
            )

        if len(records) != 128 or len(keys) != 128 or len(paired_ids) != 128:
            raise ValueError(
                f"{split} is not a one-to-one 128-scenario rendering: "
                f"rows={len(records)}, physical_keys={len(keys)}, "
                f"paired_ids={len(paired_ids)}."
            )
        if keys != expected_keys:
            raise ValueError(f"{split} does not match the frozen maize source keys.")
        if observed_objectives != {objective}:
            raise ValueError(
                f"{split} has objectives {sorted(observed_objectives)}, expected {objective}."
            )
        if observed_prompt_objectives != {objective}:
            raise ValueError(
                f"{split} states {sorted(observed_prompt_objectives)} in the prompt but "
                f"scores {objective}. The policy would be told to optimise the wrong thing."
            )
        if observed_regimes != {regime}:
            raise ValueError(
                f"{split} has reward regimes {sorted(observed_regimes)}, expected {regime}."
            )
        if observed_env_ids != {EXPECTED_ENV_ID}:
            raise ValueError(
                f"{split} has env IDs {sorted(observed_env_ids)}, expected "
                f"{EXPECTED_ENV_ID}. T2.3 must vary the reward alone."
            )
        if observed_roles != {"test"}:
            raise ValueError(
                f"{split} has dataset roles {sorted(observed_roles)}, expected test."
            )
        if len(observed_reward_params) != 1:
            raise ValueError(f"{split} renders more than one reward_params block.")

        if reference_paired_ids is None:
            reference_paired_ids = paired_ids
            reference_reward_params = next(iter(observed_reward_params))
        else:
            if paired_ids != reference_paired_ids:
                raise ValueError(
                    f"{split} does not share paired_scenario_id values with test_yield_max."
                )
            if next(iter(observed_reward_params)) != reference_reward_params:
                raise ValueError(
                    f"{split} uses different reward_params from test_yield_max, so the "
                    "objectives are not being compared on a common coefficient set."
                )

        split_report[split] = {
            "objective_id": objective,
            "reward_regime": regime,
            "env_id": EXPECTED_ENV_ID,
            "num_rows": len(records),
            "num_physical_keys": len(keys),
            "num_paired_scenario_ids": len(paired_ids),
            "sha256": sha256_file(path),
        }

    audit = {
        "protocol_id": "e4-t23-reward-formulation-v1",
        "status": "pass",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_scenario_count": len(expected_keys),
        "all_splits_share_physical_scenarios": True,
        "all_splits_share_paired_scenario_ids": True,
        "all_splits_share_env_id": EXPECTED_ENV_ID,
        "all_splits_share_reward_params": True,
        "training_disjointness": disjointness,
        "manifest_sha256": sha256_file(manifest_path),
        "splits": split_report,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    seen = [s for s, (_, r) in EXPECTED_SPLITS.items() if r == "seen"]
    held = [s for s, (_, r) in EXPECTED_SPLITS.items() if r == "heldout"]
    print(f"T2.3 E4 audit passed: {len(seen)} seen + {len(held)} held-out objective splits")
    print("Rows per split: 128")
    print(f"Shared env_id across splits: {EXPECTED_ENV_ID}")
    print("Shared physical scenarios: 128")
    print("Shared paired_scenario_id values: 128")
    print(f"Eval-vs-training overlap: {disjointness['pool_eval_vs_public_train_overlap']}")
    print(f"Audit report: {audit_path}")


if __name__ == "__main__":
    main()
