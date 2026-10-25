"""Collect every frozen test-suite ("E4") evaluation into one per-episode table.

    python analysis/test_suite/extract_e4.py --data-root <dir> --out <dir> [--nn-root <dir>] [--workers N]

`--data-root` (or the environment variable E4_DATA_ROOT) is the directory that
holds the result roots below, laid out as

  <data-root>/agri_e4_scoring/<run>/bundle/<suite>/<split>/results.json
  <data-root>/agri_t12_results/t12_e4/evaluations/...   (same layout)
  <data-root>/agri_t31_results/t31_e4/evaluations/...   (same layout)
  <data-root>/agri_camera_ready/hub_results/<dataset>/<run>/<revision>/<suite>/<split>/results.json
  <data-root>/agri_e4_anchors/<mode>/<suite>/<split>/results.json

Missing roots are skipped.

E4 results live in five places, for historical reasons rather than by design:

  worker      the persistent scoring worker's root (most of the r4 checkpoints)
  local_t12   an earlier on-disk evaluation root for the T1.2 think replicates
  local_t31   the same for T3.1 WOFOST think r2
  hub:<ds>    the Hugging Face results datasets, which hold the rented-host
              replicates, the untrained floors and the T3.2 revisions
  nn          the numerical-RL evaluations, written next to each experiment
  anchor      the scripted anchors (no-action, both heuristics), written by
              experiments/rebuttal_confirmatory_eval/run_e4_anchors.sh

The worker root also holds the controls listed in results/scoring_manifest.tsv,
whose directory names carry a suffix (`--t12`, `--on-t32`) because one model is
scored there under several tiers; e4_tables.py reads the suffix.

Every row is one scenario of one split of one checkpoint. `primary` is the
metric the split itself declares (objective_reward for most suites, target_yield
for CycleGym price-regime and the T3.2 yield tuples), read from the final turn,
and every split's episode mean is reconciled against the summary its evaluator
wrote. Gross revenue and target yield are kept alongside for every row.

Per-episode rows are what make paired comparisons possible: the same scenario
identifier appears under every checkpoint scored on that split, so think versus
no-think, or trained versus untrained, can be differenced scenario by scenario.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SOURCES: dict[str, Path] = {}


def set_data_root(root: Path) -> None:
    SOURCES.update({
        "worker": root / "agri_e4_scoring",
        "local_t12": root / "agri_t12_results/t12_e4/evaluations",
        "local_t31": root / "agri_t31_results/t31_e4/evaluations",
        "hub": root / "agri_camera_ready/hub_results",
        "anchor": root / "agri_e4_anchors",
    })


# Neural-baseline evaluations are written next to each experiment
# (experiments/<setting>/results/nn_eval_e4/<run>/); --nn-root points at another
# checkout or copy that holds them.
NN_ROOTS: list[Path] = sorted(REPO.glob("experiments/*/results/nn_eval_e4"))
FIELDS = ("objective_reward", "target_yield", "gross_revenue", "final_wso", "total_n_kg_ha",
          "total_irrig_mm", "invalid_action_rate", "profit_ge_kg_ha")


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def split_jobs():
    """Yield (source, run, revision, suite, split, results.json, metrics.json)."""
    for src in ("worker", "local_t12", "local_t31"):
        root = SOURCES[src]
        for res in root.glob("*/bundle/*/*/results.json"):
            run = res.parents[3].name
            man = res.parents[2] / "eval_manifest.json"
            rev = None
            if man.is_file():
                rev = json.load(open(man)).get("checkpoint", {}).get("revision")
            yield (src, run, rev, res.parents[1].name, res.parent.name, str(res),
                   str(res.parent / "metrics.json"))
    for res in SOURCES["hub"].glob("*/*/*/*/*/results.json"):
        ds, run, rev, suite, split = res.parts[-6:-1]
        yield (f"hub:{ds}", run, rev, suite, split, str(res), str(res.parent / "metrics.json"))
    for res in SOURCES["anchor"].glob("*/*/*/results.json"):
        mode, suite, split = res.parts[-4:-1]
        yield ("anchor", f"anchor-{mode}", None, suite, split, str(res), str(res.parent / "metrics.json"))
    for root in NN_ROOTS:
        exp = root.parents[1].name
        for res in root.glob("*/results.json"):
            yield ("nn", res.parent.name, exp, None, None, str(res), str(res.parent / "metrics.json"))


def _episode_metrics(turns):
    """Episode-level metrics from the final turn.

    The LLM evaluator writes them as `trajectory_metrics`; the numerical-RL
    evaluator writes the same quantities as the last turn's `turn_metrics`.
    """
    if not turns:
        return {}
    last = turns[-1]
    return last.get("trajectory_metrics") or last.get("turn_metrics") or {}


def parse(job):
    src, run, rev, suite, split, res, met = job
    rows, problems = [], []
    try:
        episodes = json.load(open(res))
    except Exception as e:  # a truncated file is a finding, not a crash
        return [], [f"{res}: unreadable ({type(e).__name__})"]
    m = json.load(open(met)) if os.path.isfile(met) else {}
    # The split declares its own primary metric. Most WOFOST suites score
    # objective_reward, but CycleGym price-regime and the T3.2 yield tuples
    # declare target_yield, so hardcoding one key silently empties those.
    primary_key = (m.get("primary_metric") or {}).get("key") if split else None
    for ep in episodes:
        cfg = ep.get("env_config") or {}
        turns = ep.get("turns") or []
        tm = _episode_metrics(turns)
        gl = ep.get("group_labels") or {}
        row = {
            "source": src, "run": run, "revision": rev,
            "suite": suite or cfg.get("dataset_id"),
            "split": split or gl.get("validation_set") or cfg.get("dataset_split"),
            "scenario_id": cfg.get("scenario_id"),
            "crop": cfg.get("crop_name") or cfg.get("crop"),
            "n_turns": len(turns),
            "primary_key": primary_key or "objective_reward",
        }
        for f in FIELDS:
            row[f] = _num(tm.get(f))
        # The numerical-RL CycleGym evaluator writes no episode total; revenue
        # arrives one planting year at a time as crop_planning/revenue, and its
        # sum over turns is what that evaluator's own summary reports.
        if row["gross_revenue"] is None and any("crop_planning/revenue" in (t.get("turn_metrics") or {}) for t in turns):
            row["gross_revenue"] = sum(_num((t.get("turn_metrics") or {}).get("crop_planning/revenue")) or 0.0
                                       for t in turns)
        row["primary"] = _num(tm.get(row["primary_key"]))
        rows.append(row)
    problems.extend(_reconcile(res, rows, m, split))
    return rows, problems


def _reconcile(res, rows, m, split):
    """Check every split's episode mean against the summary its evaluator wrote.

    An earlier version compared against `objective_reward_mean` and skipped the
    check when that key was absent -- which is exactly the case for the suites
    whose metric is not objective_reward, so the check passed on the splits it
    was most needed for. A split with nothing to compare against is now itself
    reported.
    """
    out = []
    if split:  # LLM bundle: one metrics.json per split, with a primary metric
        want = _num((m.get("primary_metric") or {}).get("mean"))
        got = [r["primary"] for r in rows if r["primary"] is not None]
        if want is None:
            return [f"{res}: metrics.json has no primary_metric.mean"]
        if len(got) != len(rows):
            out.append(f"{res}: {len(rows) - len(got)} of {len(rows)} episodes lack {rows[0]['primary_key']}")
        if got and abs(sum(got) / len(got) - want) > 1e-6 * max(1.0, abs(want)):
            out.append(f"{res}: episode mean {sum(got)/len(got):.6f} != metrics.json {want:.6f}")
        return out
    # Numerical-RL: one metrics.json keyed val-env-<axis>/<group>/<metric>/mean.
    # The summary groups by the validation-axis value (`ln`), while the episode
    # label may carry the split prefix (`test_ln`), and a suite with a single
    # split is summarized only under `all`. CycleGym price-regime reports gross
    # revenue, not objective_reward.
    metric = "objective_reward" if any(r["objective_reward"] is not None for r in rows) else "gross_revenue"
    # T1.2's summary groups by crop, not by split, so its episodes are
    # reconciled crop by crop.
    group_by = "crop" if any(k.startswith("val-env-crop/") for k in m) else "split"
    by_group = {}
    for r in rows:
        by_group.setdefault(r[group_by], []).append(r[metric])
    for g, vals in by_group.items():
        cands = [g, re.sub(r"^test_", "", str(g))] + (["all"] if len(by_group) == 1 else [])
        key = next((k for c in cands for k in m if k.endswith(f"/{c}/{metric}/mean")), None)
        want = _num(m.get(key)) if key else None
        got = [v for v in vals if v is not None]
        if want is None:
            out.append(f"{res}: no {metric} summary for group {g}")
        elif len(got) != len(vals) or abs(sum(got) / len(got) - want) > 1e-6 * max(1.0, abs(want)):
            out.append(f"{res}: group {g} {metric} episode mean {sum(got)/max(1,len(got)):.6f} "
                       f"!= summary {want:.6f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("E4_DATA_ROOT"),
                    help="directory holding the result roots (default: $E4_DATA_ROOT)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--limit", type=int, default=0, help="parse only the first N splits (smoke test)")
    ap.add_argument("--nn-root", type=Path, default=REPO,
                    help="directory whose experiments/*/results/nn_eval_e4 hold the neural-baseline "
                         "evaluations (default: this repository)")
    a = ap.parse_args()
    NN_ROOTS[:] = sorted(a.nn_root.glob("experiments/*/results/nn_eval_e4"))
    if not a.data_root:
        ap.error("--data-root (or E4_DATA_ROOT) is required")
    set_data_root(Path(a.data_root))
    jobs = list(split_jobs())
    if a.limit:
        jobs = jobs[: a.limit]
    print(f"{len(jobs)} result files")
    rows, problems = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for r, p in ex.map(parse, jobs, chunksize=2):
            rows.extend(r)
            problems.extend(p)
    df = pd.DataFrame(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "e4_episodes.parquet", index=False)
    print(f"{len(df)} episodes from {df['run'].nunique()} checkpoints -> {out/'e4_episodes.parquet'}")
    print(f"{len(problems)} reconciliation problems")
    for p in problems[:50]:
        print("  " + p)
    (out / "extract_problems.txt").write_text("\n".join(problems))


if __name__ == "__main__":
    main()
