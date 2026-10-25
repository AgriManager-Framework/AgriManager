"""Audit the three v2 CycleGym E4 suites on the physical scenario key.

    python experiments/rebuttal_confirmatory_eval/audit_cycles_v2_eval_suites.py

The v1 audits keyed a CycleGym scenario on (env_id, start_year, end_year, seed).
The seed does not change a CornShortRockSpringsFW-v1 episode -- the site is
fixed and the weather is the historical year -- so two scenarios that differ
only by seed are the same scenario, and v1 passed while repeating the
validation set's years. This audit keys on (env_id, start_year, end_year) and
compares against the scenarios the policies actually saw, read from the
training and validation parquets rather than restated here.

Checks, for each of the three v2 suites:
  * 130 rows, 130 distinct scenario ids, 13 physical scenarios, years 2004-2016;
  * no physical overlap with any CycleGym training or validation split of T3.1
    or T3.2;
  * the expected labels and objective;
and across them, that all three hold the identical (year, seed) block, so the
T3.1, T3.2-profit and T3.2-yield cells stay paired scenario by scenario.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO = EXPERIMENT_DIR.parents[1]
DATA = EXPERIMENT_DIR / "data"
ENV_ID = "CornShortRockSpringsFW-v1"
YEARS = set(range(2004, 2017))
ROWS = 130

SUITES = {
    "e4_t31_cross_sim_cycles_v2": {"objective_id": "profit_max", "reward_formulation": "profit_max"},
    "e4_t32_cycles_profit_v2": {"objective_id": "profit_max", "reward_formulation": "profit_max",
                                "schema_tuple": "cycles_profit"},
    "e4_t32_cycles_yield_v2": {"objective_id": "yield_max", "reward_formulation": "yield_max",
                               "schema_tuple": "cycles_yield"},
}

# Every CycleGym corn split a policy was trained or validated on.
SEEN = sorted(
    list((REPO / "experiments/t3_1_cross_simulator_maize_transfer/data/t31_cross_sim_cycles_gym").glob("*.parquet"))
    + list((REPO / "experiments/t3_2_unified_prompt_conditioned_policy/data/t32_unified_cycles_yield").glob("*.parquet"))
    + list((REPO / "experiments/t3_2_unified_prompt_conditioned_policy/data/t32_unified_cycles_profit").glob("*.parquet"))
)


def env_configs(path: Path) -> list[dict]:
    out = []
    for extra in pd.read_parquet(path)["extra_info"]:
        extra = extra if isinstance(extra, dict) else json.loads(extra)
        ik = extra["interaction_kwargs"]
        ik = ik if isinstance(ik, dict) else json.loads(ik)
        ec = ik["env_config"]
        out.append(ec if isinstance(ec, dict) else json.loads(ec))
    return out


def physical(ec: dict) -> tuple[str, int, int]:
    kw = ec["env_kwargs"]
    return (str(ec["env_id"]), int(kw["start_year"]), int(kw["end_year"]))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if not SEEN:
        raise FileNotFoundError("no CycleGym training or validation parquets found to audit against")
    seen = {}
    for p in SEEN:
        for ec in env_configs(p):
            if str(ec.get("env_id", "")).startswith("Corn"):
                seen.setdefault(physical(ec), set()).add(str(p.relative_to(REPO)))
    seen_years = sorted({k[1] for k in seen})

    blocks = {}
    for name, want in SUITES.items():
        d = DATA / name
        parquet, manifest = d / "test.parquet", d / "manifest.json"
        ecs = env_configs(parquet)
        phys = {physical(ec) for ec in ecs}
        ids = {str(ec["scenario_id"]) for ec in ecs}
        years = {k[1] for k in phys}
        if len(ecs) != ROWS or len(ids) != ROWS:
            raise ValueError(f"{name}: rows={len(ecs)}, scenario ids={len(ids)}; expected {ROWS} each")
        if {k[0] for k in phys} != {ENV_ID} or any(k[1] != k[2] for k in phys) or years != YEARS:
            raise ValueError(f"{name}: physical scenarios {sorted(phys)} are not {ENV_ID}, one year each, 2004-2016")
        overlap = sorted(k for k in phys if k in seen)
        if overlap:
            raise ValueError(f"{name}: physically overlaps training/validation: "
                             f"{[(k, sorted(seen[k])) for k in overlap]}")
        for key, value in want.items():
            got = {str(ec.get(key) or (ec.get("trajectory_group_labels") or {}).get(key)) for ec in ecs}
            if got != {value}:
                raise ValueError(f"{name}: {key} = {got}, expected {value}")
        blocks[name] = sorted((k[1], int(ec["seed"])) for ec, k in ((ec, physical(ec)) for ec in ecs))
        audit = {
            "protocol_id": f"{name}-audit",
            "status": "pass",
            "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "scenario_key_fields": ["env_id", "start_year", "end_year"],
            "scenario_key_note": "the seed does not change a CornShortRockSpringsFW-v1 episode, so it is not part of the key",
            "test": {"years": [min(YEARS), max(YEARS)], "physical_scenarios": len(phys),
                     "rows": len(ecs), "seeds_per_year": ROWS // len(YEARS)},
            "audited_against": {"files": sorted(str(p.relative_to(REPO)) for p in SEEN),
                                "years_seen": [seen_years[0], seen_years[-1]],
                                "physical_scenarios_seen": len(seen)},
            "overlap": {"physical": 0},
            "checksums": {"test.parquet": f"sha256:{sha256(parquet)}",
                          "manifest.json": f"sha256:{sha256(manifest)}"},
        }
        (d / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        print(f"{name}: pass  rows={len(ecs)}  physical={len(phys)}  years 2004-2016  overlap=0")

    ref = next(iter(blocks.values()))
    for name, b in blocks.items():
        if b != ref:
            raise ValueError(f"{name}: (year, seed) block differs from the other v2 suites; pairing would break")
    print(f"all three share one (year, seed) block; audited against {len(SEEN)} files, "
          f"years {seen_years[0]}-{seen_years[-1]} seen")


if __name__ == "__main__":
    main()
