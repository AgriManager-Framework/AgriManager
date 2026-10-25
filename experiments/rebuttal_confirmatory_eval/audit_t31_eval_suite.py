"""Audit the rendered T3.1 cross-simulator suites.

T3.1's eval side is one target cell per simulator. Each is checked against the
frozen block it is supposed to occupy, and then against its T3.2 counterpart.

That second check is the point of this script. The three T3.1 target cells are
built to sit on the same scenario blocks as three already-frozen T3.2 cells, so
if the two agree, T3.1 and T3.2 numbers are directly comparable and the
duplication is verified rather than a place for the two to drift apart. Where
they disagree, the audit records exactly which env fields differ instead of
failing: the submitted T3.1 configs leave some prompt fields at their defaults
where T3.2 pins them, and that difference is a fact about the submitted
experiments, not an error in this suite.
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
DATA_DIR = EXPERIMENT_DIR / "data"
SOURCE_PARQUET = DATA_DIR / "source_pools" / "maize_e4" / "test" / "maize.parquet"
POOL_AUDIT = DATA_DIR / "source_pools" / "maize_e4" / "audit.json"
T31_CONFIG_DIR = (
    REPO_ROOT / "experiments" / "t3_1_cross_simulator_maize_transfer" / "config"
)

EXPECTED_ROWS = 128
EXPECTED_OBJECTIVE = "profit_max"

# suite -> (simulator label, T3.2 counterpart suite, counterpart split)
SUITES = {
    "e4_t31_cross_sim_wofost": (
        "wofost_gym",
        "e4_t32_wofost_heldout",
        "test_wofost_lnw_profit",
    ),
    "e4_t31_cross_sim_dssat": ("gym_dssat", "e4_t32_dssat_profit", "test"),
    "e4_t31_cross_sim_cycles": ("cycles_gym", "e4_t32_cycles_profit", "test"),
}
SPLIT_NAMES = {
    "e4_t31_cross_sim_wofost": "test_wofost",
    "e4_t31_cross_sim_dssat": "test",
    "e4_t31_cross_sim_cycles": "test",
}

EXPECTED_SEED_BLOCK = set(range(200042, 200042 + EXPECTED_ROWS))

# Fields that name the suite rather than the physical scenario. Two renderings
# of the same scenario are expected to disagree on these.
IDENTITY_FIELDS = {
    "dataset_id",
    "trajectory_group_labels",
    "validation_set",
    "scenario_id",
    "paired_scenario_id",
    "save_folder",
    "dataset_split",
}

# Fields where a T3.1-versus-T3.2 difference is real but provably has no effect
# on what the policy sees or is scored on. Anything differing that is NOT
# listed here fails the audit, so a substantive divergence cannot be waved
# through as bookkeeping.
#
# Each reason was checked rather than assumed. In particular the two prompt
# fields: WOFOSTEnv falls back to `output_vars` when
# `prompt_observation_fields` is unset (env.py:453), and for lnw-v0 that
# resolves to exactly the fifteen fields the T3.2 config pins, so the observed
# prompt is the same either way.
INERT_DIFFERENCES = {
    "observation_schema_name": "label only; the env reads prompt_observation_fields",
    "prompt_observation_fields": (
        "T3.1 leaves it unset and the env falls back to output_vars, which for "
        "lnw-v0 is the same fifteen fields T3.2 pins"
    ),
    "prompt_objective_id": "falls back to objective_id, which both suites set to profit_max",
    "y_ref": "only read by the yield-ratio stewardship rewards, not by profit_max",
    "reward_params": (
        "T3.2 additionally carries the stewardship coefficients; the profit "
        "coefficients are shared and equal, and the rest are unread under profit_max"
    ),
    "seed": (
        "per-row generation seed derived from the dataset id; WOFOST weather is "
        "determined by year, latitude and longitude, which the scenario key pins"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def env_configs(path: Path) -> list[dict[str, Any]]:
    return [
        info["interaction_kwargs"]["env_config"]
        for info in pd.read_parquet(path)["extra_info"]
    ]


def wofost_keys(configs: list[dict[str, Any]]) -> set[tuple]:
    return {
        (
            str(cfg["crop_name"]),
            int(cfg["agro_params"]["year"]),
            round(float(cfg["agro_params"]["latitude"]), 2),
            round(float(cfg["agro_params"]["longitude"]), 2),
        )
        for cfg in configs
    }


def source_keys() -> set[tuple]:
    df = pd.read_parquet(SOURCE_PARQUET)
    return {
        ("maize", int(row.year), round(float(row.latitude), 2), round(float(row.longitude), 2))
        for row in df.itertuples(index=False)
    }


def scenario_identity(simulator: str, cfg: dict[str, Any]) -> tuple:
    """What physically pins a scenario, per simulator."""
    if simulator == "wofost_gym":
        agro = cfg["agro_params"]
        return (
            str(cfg["crop_name"]),
            int(agro["year"]),
            round(float(agro["latitude"]), 2),
            round(float(agro["longitude"]), 2),
        )
    if simulator == "gym_dssat":
        return (str(cfg["env_id"]), int(cfg["seed"]))
    if simulator == "cycles_gym":
        env_kwargs = cfg["env_kwargs"]
        return (
            str(cfg["env_id"]),
            int(env_kwargs["start_year"]),
            int(env_kwargs["end_year"]),
            int(cfg["seed"]),
        )
    raise ValueError(f"Unknown simulator {simulator!r}")


def submitted_seed_blocks(simulator: str) -> dict[str, set[int]]:
    """Seeds the submitted T3.1 splits consumed, reconstructed from their configs."""
    if simulator == "gym_dssat":
        config = yaml.safe_load(
            (T31_CONFIG_DIR / "t31_cross_sim_gym_dssat.yaml").read_text(encoding="utf-8")
        )
        blocks = {}
        for split, key in (("train", "train_seeds"), ("val", "val_seeds")):
            spec = config.get(key)
            if spec:
                base = int(spec["base_seed"])
                blocks[split] = set(range(base, base + int(spec["count"])))
        return blocks
    if simulator == "cycles_gym":
        config = yaml.safe_load(
            (T31_CONFIG_DIR / "t31_cross_sim_cycles_gym.yaml").read_text(encoding="utf-8")
        )
        blocks = {}
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
    return {}


def compare_with_counterpart(
    simulator: str,
    configs: list[dict[str, Any]],
    counterpart_path: Path,
) -> dict[str, Any]:
    if not counterpart_path.is_file():
        return {"counterpart": str(counterpart_path.name), "status": "not_available"}

    other = env_configs(counterpart_path)
    mine = {scenario_identity(simulator, cfg): cfg for cfg in configs}
    theirs = {scenario_identity(simulator, cfg): cfg for cfg in other}

    shared = set(mine) & set(theirs)
    result: dict[str, Any] = {
        "counterpart": counterpart_path.parent.name + "/" + counterpart_path.stem,
        "scenarios_here": len(mine),
        "scenarios_there": len(theirs),
        "scenarios_shared": len(shared),
        "same_scenario_block": set(mine) == set(theirs),
    }
    if not shared:
        result["differing_env_fields"] = None
        return result

    differing: set[str] = set()
    for key in shared:
        a, b = mine[key], theirs[key]
        for field in set(a) | set(b):
            if field in IDENTITY_FIELDS:
                continue
            if json.dumps(a.get(field), sort_keys=True, default=str) != json.dumps(
                b.get(field), sort_keys=True, default=str
            ):
                differing.add(field)
    substantive = sorted(differing - set(INERT_DIFFERENCES))
    if substantive:
        raise ValueError(
            f"{counterpart_path.parent.name} and this suite render the same scenarios "
            f"but disagree on {substantive}, which is not a known-inert difference. "
            "The T3.1 and T3.2 numbers would not be comparable."
        )

    result["differing_env_fields"] = sorted(differing)
    result["env_identical"] = not differing
    result["behaviourally_identical"] = True
    result["inert_differences"] = {
        field: INERT_DIFFERENCES[field] for field in sorted(differing)
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    pool_audit = json.loads(POOL_AUDIT.read_text(encoding="utf-8"))
    report: dict[str, Any] = {}

    for suite, (simulator, counterpart_suite, counterpart_split) in SUITES.items():
        dataset_dir = args.data_dir / suite
        split = SPLIT_NAMES[suite]
        parquet_path = dataset_dir / f"{split}.parquet"
        manifest_path = dataset_dir / "manifest.json"
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Missing T3.1 E4 split: {parquet_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing T3.1 manifest: {manifest_path}")

        configs = env_configs(parquet_path)
        if len(configs) != EXPECTED_ROWS:
            raise ValueError(f"{suite} has {len(configs)} rows, expected {EXPECTED_ROWS}.")

        identities = {scenario_identity(simulator, cfg) for cfg in configs}
        if len(identities) != EXPECTED_ROWS:
            raise ValueError(
                f"{suite} renders {len(identities)} distinct scenarios across "
                f"{EXPECTED_ROWS} rows; scenarios are duplicated."
            )

        observed_objectives = {str(cfg["objective_id"]) for cfg in configs}
        observed_roles = {str(cfg["dataset_role"]) for cfg in configs}
        observed_simulators = {
            str(cfg["trajectory_group_labels"]["simulator"]) for cfg in configs
        }
        if observed_objectives != {EXPECTED_OBJECTIVE}:
            raise ValueError(
                f"{suite} has objectives {sorted(observed_objectives)}, expected "
                f"{EXPECTED_OBJECTIVE}. T3.1 varies the simulator alone."
            )
        if observed_roles != {"test"}:
            raise ValueError(f"{suite} has roles {sorted(observed_roles)}, expected test.")
        if observed_simulators != {simulator}:
            raise ValueError(
                f"{suite} labels simulators {sorted(observed_simulators)}, expected {simulator}."
            )

        suite_report: dict[str, Any] = {
            "simulator": simulator,
            "num_rows": len(configs),
            "num_scenarios": len(identities),
            "objective_id": EXPECTED_OBJECTIVE,
            "manifest_sha256": sha256_file(manifest_path),
            "checksums": {f"{split}.parquet": f"sha256:{sha256_file(parquet_path)}"},
        }

        if simulator == "wofost_gym":
            expected = source_keys()
            if wofost_keys(configs) != expected:
                raise ValueError(f"{suite} does not match the frozen maize source keys.")
            suite_report["source_pool"] = "maize_e4"
            suite_report["eval_vs_public_train_overlap"] = pool_audit["overlap"][
                "eval_vs_public_train"
            ]
        else:
            seeds = {int(cfg["seed"]) for cfg in configs}
            if seeds != EXPECTED_SEED_BLOCK:
                raise ValueError(
                    f"{suite} seed block is {min(seeds)}-{max(seeds)}, expected "
                    f"{min(EXPECTED_SEED_BLOCK)}-{max(EXPECTED_SEED_BLOCK)}."
                )
            overlap = {
                name: len(seeds & block)
                for name, block in submitted_seed_blocks(simulator).items()
            }
            if any(overlap.values()):
                raise ValueError(
                    f"{suite} seed block collides with submitted T3.1 splits: {overlap}."
                )
            suite_report["seed_block"] = {
                "start": min(seeds),
                "count": len(seeds),
                "overlap_with_submitted": overlap,
            }

        suite_report["t32_counterpart"] = compare_with_counterpart(
            simulator,
            configs,
            args.data_dir / counterpart_suite / f"{counterpart_split}.parquet",
        )

        audit = {
            "protocol_id": f"e4-t31-{simulator.replace('_', '-')}-v1",
            "status": "pass",
            "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            **suite_report,
        }
        (dataset_dir / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report[suite] = suite_report

    print(f"T3.1 E4 audit passed: {len(report)} cross-simulator target cells")
    for suite, suite_report in report.items():
        counterpart = suite_report["t32_counterpart"]
        same = counterpart.get("same_scenario_block")
        differing = counterpart.get("differing_env_fields")
        if same and not differing:
            verdict = f"identical to {counterpart['counterpart']}"
        elif same:
            verdict = (
                f"equivalent to {counterpart['counterpart']} "
                f"({len(differing)} inert field differences)"
            )
        else:
            verdict = f"different scenario block from {counterpart['counterpart']}"
        print(f"  {suite:30s} {suite_report['num_rows']:4d} rows   {verdict}")


if __name__ == "__main__":
    main()
