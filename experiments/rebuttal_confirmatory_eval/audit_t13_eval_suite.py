"""Audit the rendered T1.3 Cycles price-regime suite.

A price-shock number is only interpretable if the shock is the only thing that
moved. This checks that directly rather than trusting the generator:

1. Every `paired_scenario_id` appears once per regime, and the three rows agree
   on env, year window and seed. Only the price vector may differ.
2. The shocked crop's price is actually above its base range in the shocked
   regime, and the unshocked crop's price is unchanged from `id_test`. A regime
   whose override silently failed to apply would otherwise read as "the policy
   is robust to price shocks".
3. The seed block does not touch the submitted training or validation blocks.
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
DEFAULT_DATASET_DIR = EXPERIMENT_DIR / "data" / "e4_t13_price_regime"
TRAINING_CONFIG = (
    REPO_ROOT
    / "experiments"
    / "t1_3_cycles_gym_price_regime"
    / "config"
    / "price_regime_llm_think.yaml"
)

EXPECTED_REGIMES = {
    "id_test": {"family": "id", "shocked_crop": None},
    "high_corn": {"family": "ood", "shocked_crop": "CornRM.100"},
    "high_soybean": {"family": "ood", "shocked_crop": "SoybeanMG.3"},
}
EXPECTED_PAIRS = 128
EXPECTED_TURN_NUM = 19
EXPECTED_SEED_START = 300000
EXPECTED_SEED_COUNT = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def submitted_seed_blocks() -> dict[str, set[int]]:
    """Reconstruct which seeds the submitted T1.3 splits consumed.

    The generator assigns `seed_start + combo_index * seeds_per_combo +
    seed_offset` over env_ids x year_windows, so the block is fully determined
    by the config and needs no stored artifact to check against.
    """
    config = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    blocks: dict[str, set[int]] = {}
    for split in ("train", "val"):
        spec = config.get(f"{split}_generation")
        if not spec:
            continue
        combos = len(spec["env_ids"]) * len(spec["year_windows"])
        per = int(spec["seeds_per_combo"])
        start = int(spec.get("seed_start", 0))
        blocks[split] = {
            start + combo * per + offset
            for combo in range(combos)
            for offset in range(per)
        }
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    parquet_path = dataset_dir / "test.parquet"
    manifest_path = dataset_dir / "manifest.json"
    audit_path = dataset_dir / "audit.json"
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing T1.3 E4 split: {parquet_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing T1.3 manifest: {manifest_path}")

    records = pd.read_parquet(parquet_path).to_dict("records")

    # pair_id -> regime -> (identity, prices)
    pairs: dict[str, dict[str, tuple[tuple, dict[str, float]]]] = {}
    observed_roles: set[str] = set()
    observed_turn_nums: set[int] = set()
    observed_env_ids: set[str] = set()
    seeds: set[int] = set()

    for record in records:
        env_config = record["extra_info"]["interaction_kwargs"]["env_config"]
        env_kwargs = env_config["env_kwargs"]
        regime = str(env_config["price_regime"])
        if regime not in EXPECTED_REGIMES:
            raise ValueError(f"Unexpected price regime {regime!r}.")
        pair_id = str(env_config["paired_scenario_id"])
        identity = (
            str(env_config["env_id"]),
            int(env_kwargs["start_year"]),
            int(env_kwargs["end_year"]),
            int(env_config["seed"]),
        )
        prices = {
            str(crop): float(value)
            for crop, value in dict(env_kwargs["crop_prices"]).items()
        }
        slot = pairs.setdefault(pair_id, {})
        if regime in slot:
            raise ValueError(f"Regime {regime!r} appears twice for pair {pair_id}.")
        slot[regime] = (identity, prices)

        observed_roles.add(str(env_config["dataset_role"]))
        observed_turn_nums.add(int(env_config["turn_num"]))
        observed_env_ids.add(str(env_config["env_id"]))
        seeds.add(int(env_config["seed"]))

    if len(pairs) != EXPECTED_PAIRS:
        raise ValueError(f"Expected {EXPECTED_PAIRS} paired scenarios, found {len(pairs)}.")
    expected_rows = EXPECTED_PAIRS * len(EXPECTED_REGIMES)
    if len(records) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(records)}.")
    if observed_roles != {"test"}:
        raise ValueError(f"Dataset roles {sorted(observed_roles)}, expected test.")
    if observed_turn_nums != {EXPECTED_TURN_NUM}:
        raise ValueError(
            f"Turn counts {sorted(observed_turn_nums)}, expected {EXPECTED_TURN_NUM}."
        )

    # The seed block must be fresh.
    expected_seeds = set(range(EXPECTED_SEED_START, EXPECTED_SEED_START + EXPECTED_SEED_COUNT))
    if seeds != expected_seeds:
        raise ValueError(
            f"Seed block is {min(seeds)}-{max(seeds)} ({len(seeds)} seeds), expected "
            f"{EXPECTED_SEED_START}-{EXPECTED_SEED_START + EXPECTED_SEED_COUNT - 1}."
        )
    submitted = submitted_seed_blocks()
    seed_overlap = {name: len(seeds & block) for name, block in submitted.items()}
    if any(seed_overlap.values()):
        raise ValueError(
            f"The E4 seed block collides with submitted T1.3 splits: {seed_overlap}."
        )

    # Each pair must vary the prices and nothing else.
    shock_report: dict[str, dict[str, Any]] = {}
    for pair_id, slot in pairs.items():
        missing = set(EXPECTED_REGIMES) - set(slot)
        if missing:
            raise ValueError(f"Pair {pair_id} is missing regimes {sorted(missing)}.")
        identities = {identity for identity, _ in slot.values()}
        if len(identities) != 1:
            raise ValueError(
                f"Pair {pair_id} does not hold env, years and seed fixed across regimes: "
                f"{sorted(identities)}. The price shock would be confounded."
            )

        base_prices = slot["id_test"][1]
        for regime, spec in EXPECTED_REGIMES.items():
            shocked = spec["shocked_crop"]
            if shocked is None:
                continue
            regime_prices = slot[regime][1]
            if regime_prices[shocked] <= base_prices[shocked]:
                raise ValueError(
                    f"Pair {pair_id} regime {regime}: {shocked} price "
                    f"{regime_prices[shocked]} is not above the base "
                    f"{base_prices[shocked]}. The override did not apply."
                )
            unshocked = [crop for crop in base_prices if crop != shocked]
            for crop in unshocked:
                if regime_prices[crop] != base_prices[crop]:
                    raise ValueError(
                        f"Pair {pair_id} regime {regime}: unshocked crop {crop} moved from "
                        f"{base_prices[crop]} to {regime_prices[crop]}."
                    )
            entry = shock_report.setdefault(
                regime,
                {"shocked_crop": shocked, "min_ratio": None, "max_ratio": None},
            )
            ratio = regime_prices[shocked] / base_prices[shocked]
            entry["min_ratio"] = ratio if entry["min_ratio"] is None else min(entry["min_ratio"], ratio)
            entry["max_ratio"] = ratio if entry["max_ratio"] is None else max(entry["max_ratio"], ratio)

    audit = {
        "protocol_id": "e4-t13-price-regime-v1",
        "status": "pass",
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "num_paired_scenarios": len(pairs),
        "num_rows": len(records),
        "regimes": sorted(EXPECTED_REGIMES),
        "env_ids": sorted(observed_env_ids),
        "turn_num": EXPECTED_TURN_NUM,
        "pairs_hold_env_years_seed_fixed": True,
        "unshocked_prices_unchanged": True,
        "seed_block": {
            "start": EXPECTED_SEED_START,
            "count": EXPECTED_SEED_COUNT,
            "overlap_with_submitted": seed_overlap,
        },
        "shock_magnitude": {
            regime: {
                "shocked_crop": entry["shocked_crop"],
                "price_ratio_min": round(entry["min_ratio"], 4),
                "price_ratio_max": round(entry["max_ratio"], 4),
            }
            for regime, entry in shock_report.items()
        },
        "manifest_sha256": sha256_file(manifest_path),
        "checksums": {"test.parquet": f"sha256:{sha256_file(parquet_path)}"},
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print(f"T1.3 E4 audit passed: {len(pairs)} paired scenarios x {len(EXPECTED_REGIMES)} regimes")
    print(f"Rows: {len(records)}   env_ids: {len(observed_env_ids)}   turns: {EXPECTED_TURN_NUM}")
    print(f"Seed block: {EXPECTED_SEED_START}-{EXPECTED_SEED_START + EXPECTED_SEED_COUNT - 1}")
    print(f"Overlap with submitted seed blocks: {seed_overlap}")
    for regime, entry in audit["shock_magnitude"].items():
        print(
            f"  {regime:14s} {entry['shocked_crop']:14s} "
            f"price x{entry['price_ratio_min']}-{entry['price_ratio_max']}"
        )
    print(f"Audit report: {audit_path}")


if __name__ == "__main__":
    main()
