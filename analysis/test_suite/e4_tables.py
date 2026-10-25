"""Turn the per-episode test-suite table into the paper's summary tables.

"E4" in file and suite names is the internal name of the frozen held-out test
suite.

    python analysis/test_suite/e4_tables.py --episodes <e4_episodes.parquet> [--out results/tables]

Reads  e4_episodes.parquet (written by extract_e4.py; or $E4_EPISODES)
Writes results/tables/*.csv

Conventions:

* Raw objective first. Normalized scores (e4_normalized.csv) are computed only
  where the gap between the scripted reference and no-action on the same
  scenarios is usable (e4_anchors.csv, `gap_usable`).
* One checkpoint per training seed: the terminal (or last-valid) one.
* The unit of replication is the training seed. Each seed contributes one
  number per split, the mean over that split's scenarios, and cells report the
  mean and SD over seeds. With three seeds that is a direction and a spread,
  not a significance level.
* Comparisons between two policies are paired twice over: by seed where the
  two arms share one (think and no-think of a replicate index train on the same
  seed), and by scenario inside each seed.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

EPISODES = Path(os.environ.get("E4_EPISODES", "e4_episodes.parquet"))
OUT = Path(__file__).resolve().parents[2] / "results" / "tables"

SETTING = {"t11": "T1.1", "t12": "T1.2", "t13": "T1.3", "t21": "T2.1", "t22": "T2.2",
           "t23": "T2.3", "t31": "T3.1", "t32": "T3.2"}

# The three test seeds of each arm, fixed by a seed audit before scoring. R1 is
# absent for most arms because its checkpoint was trained on a rented host and
# is gone; T1.3 was trained fresh; CycleGym think skips r4, which never produced
# a checkpoint.
DEFAULT_SEEDS = (2, 3, 4)
SEEDS = {("t13-price", "think"): (1, 2, 3), ("t13-price", "no-think"): (1, 2, 3),
         ("t31-cycles", "think"): (2, 3, 5)}

LLM_RE = re.compile(r"^schema-(?P<cell>.+?)-(?P<arm>no-think|think)-r(?P<seed>\d+)$")
LLAMA_RE = re.compile(r"^schema-t32-llama32-(?P<arm>no-think|think)-s1$")
BASE_RE = re.compile(r"^baseline-(?P<model>qwen3-4b|llama32-3b)-untrained-(?P<arm>no-think|think)(?:--\w+)?$")
# A T3.1 specialist scored on a T3.2 callback suite (results/scoring_manifest.tsv, run names ending in --on-t32).
SPEC_RE = re.compile(r"^schema-t31-(?P<sim>wofost|dssat|cycles)-(?P<arm>no-think|think)-r(?P<seed>\d+)--on-t32$")
ANCHOR_METHOD = {"anchor-no_action": "No-action", "anchor-ag_heuristic": "Ag heuristic (current)",
                 "anchor-extension_guideline": "Ag heuristic (paper)",
                 # DSSAT and CycleGym: the paper's first implementation (paper_references/);
                 # the DSSAT branch is a reconstruction, see paper_sim_anchor_rollout.py.
                 "anchor-paper_sim_heuristic": "Ag heuristic (paper)"}


def _step(rev):
    m = re.search(r"(\d+)$", str(rev or ""))
    return int(m.group(1)) if m else -1


# The three CycleGym E4 suites were rebuilt on unused weather years (v2): v1
# repeated the validation set's eight physical years. v2 replaces v1 in every
# table; v1 survives only in e4_cycles_v1_v2.csv, to show what changed. v2
# results carry a --cyclesv2 suffix so the two never share a result path.
CYCLES_V1 = ("e4_t31_cross_sim_cycles", "e4_t32_cycles_profit", "e4_t32_cycles_yield")
V2_SUFFIX = "--cyclesv2"


def classify(run: str, source: str, suite: str) -> dict | None:
    """Map a checkpoint name to (setting, cell, method, arm, seed)."""
    if run.endswith(V2_SUFFIX):
        run = run[: -len(V2_SUFFIX)]
    if source == "nn":
        return _classify_nn(run, suite)
    if source == "anchor":
        return dict(setting=SETTING[suite[3:6]], cell=None, method=ANCHOR_METHOD[run], arm=None,
                    seed=0, canonical=True)
    m = SPEC_RE.match(run)
    if m:
        seed = int(m["seed"])
        return dict(setting="T3.2", cell=f"t31-{m['sim']}-specialist", method="T3.1 specialist",
                    arm=m["arm"], seed=seed,
                    canonical=seed in SEEDS.get((f"t31-{m['sim']}", m["arm"]), DEFAULT_SEEDS))
    m = LLM_RE.match(run)
    if m and not run.startswith("schema-t32-llama32"):
        cell, arm, seed = m["cell"], m["arm"], int(m["seed"])
        return dict(setting=SETTING[cell[:3]], cell=cell, method="LLM-RL", arm=arm, seed=seed,
                    canonical=seed in SEEDS.get((cell, arm), DEFAULT_SEEDS))
    m = LLAMA_RE.match(run)
    if m:
        return dict(setting="T3.2", cell="t32-llama32", method="LLM-RL (Llama)", arm=m["arm"],
                    seed=1, canonical=False)
    m = BASE_RE.match(run)
    if m:
        model = "Qwen" if m["model"].startswith("qwen") else "Llama"
        return dict(setting=SETTING[suite[3:6]], cell=None, method=f"zero-shot ({model})",
                    arm=m["arm"], seed=0, canonical=model == "Qwen")
    return None


def _classify_nn(run: str, suite: str) -> dict | None:
    seed = int(m.group(1)) if (m := re.search(r"[_-]s(\d)(?:__|$)", run)) else 1
    if run.startswith("wofost_weather_regime_ood_"):
        crop, algo = re.match(r"wofost_weather_regime_ood_(\w+?)_(nn|a2c|dqn)_", run).groups()
        n = "n8" if run.endswith("n8") else "n1"
        method = {"nn": "PPO", "a2c": "A2C", "dqn": "DQN"}[algo] + f" ({n})"
        return dict(setting="T1.1", cell=f"t11-{crop}", method=method, arm=None, seed=seed,
                    canonical=n == "n1")
    if "crosscrop" in run or "cross_crop" in run:
        cov = re.search(r"(\d+)id", run).group(1)
        n = re.search(r"(n[18])", run).group(1)
        return dict(setting="T1.2", cell=f"t12-crosscrop-{cov}id", method=f"PPO ({n})", arm=None,
                    seed=seed, canonical=True)
    if run.startswith("t1_3_"):
        return dict(setting="T1.3", cell="t13-price", method="PPO (n1)", arm=None, seed=seed,
                    canonical=True)
    if run.startswith("t22-nn-mask-family-"):
        fam = re.search(r"family-([ab])", run).group(1)
        return dict(setting="T2.2", cell=f"t22-family-{fam}", method="PPO masked", arm=None,
                    seed=seed, canonical=True)
    if run.startswith("t22-nn-target-"):
        return dict(setting="T2.2", cell="t22-target", method="PPO target-trained", arm=None,
                    seed=seed, canonical=True)
    return None


def target_of(setting: str, suite: str, split: str) -> str:
    """A short label for the evaluation target, comparable across sources."""
    s = re.sub(r"^test_?", "", split or "")
    if setting == "T3.1":
        return re.search(r"cross_sim_(\w+)$", suite).group(1)
    if setting == "T3.2":
        base = re.sub(r"^e4_t32_", "", suite)
        return re.sub(r"^wofost_(seen|heldout)$", "wofost", base) + (f"_{s}" if s else "")
    return s or "all"


def crop_of(suite: str) -> str | None:
    m = re.match(r"e4_t11_(\w+?)_weather_regime", suite or "")
    return m.group(1) if m else None


def load() -> pd.DataFrame:
    d = pd.read_parquet(EPISODES)
    meta = {}
    for (run, src, suite) in d[["run", "source", "suite"]].drop_duplicates().itertuples(index=False):
        meta[(run, src, suite)] = classify(run, src, suite)
    keys = list(zip(d.run, d.source, d.suite))
    for col in ("setting", "cell", "method", "arm", "seed", "canonical"):
        d[col] = [(meta[k] or {}).get(col) for k in keys]
    d = d[d.setting.notna()].copy()
    # Tag the CycleGym suite version and give v2 its v1 name, so targets,
    # anchors and figures address CycleGym the same way whichever version fed them.
    d["cycles_version"] = None
    d.loc[d.suite.isin(CYCLES_V1), "cycles_version"] = "v1"
    v2 = d.suite.isin([f"{x}_v2" for x in CYCLES_V1])
    d.loc[v2, "cycles_version"] = "v2"
    d.loc[v2, "suite"] = d.loc[v2, "suite"].str.replace(r"_v2$", "", regex=True)
    # T1.1 zero-shot is scored once per crop suite, so its cell comes from the suite.
    d["crop"] = [crop_of(s) for s in d.suite]
    t11 = d.setting.eq("T1.1") & d.cell.isna()
    d.loc[t11, "cell"] = "t11-" + d.loc[t11, "crop"]
    d["target"] = [target_of(st, su, sp) for st, su, sp in zip(d.setting, d.suite, d.split)]
    # The metric each cell reports: whatever the split declares as primary,
    # except T1.3, whose evaluator declares target_yield while the paper reports
    # the price-regime task on gross revenue, its training objective.
    d["value"] = np.where(d.setting.eq("T1.3"), d.gross_revenue, d.primary)
    d["metric"] = np.where(d.setting.eq("T1.3"), "gross_revenue", d.primary_key)
    # Terminal revision per checkpoint and suite. Only T3.2 carries several
    # training revisions of one suite. The grouping must include the suite: the
    # untrained floors are stored as revisions base-t11 ... base-t32, one per
    # tier, and taking the highest per run alone kept base-t32 and silently
    # dropped every other tier's floor.
    d["step"] = d.revision.map(_step)
    last = d.groupby(["source", "run", "suite"]).step.transform("max")
    d["terminal"] = d.step.eq(last) | d.source.eq("nn")
    return d


def seed_table(d: pd.DataFrame) -> pd.DataFrame:
    """One row per (checkpoint, target): the mean over that split's scenarios."""
    g = d[d.terminal].groupby(["setting", "cell", "method", "arm", "seed", "canonical", "target"],
                               dropna=False)
    return g.agg(objective=("value", "mean"),
                 metric=("metric", "first"),
                 n_scenarios=("value", "size"),
                 final_wso=("final_wso", "mean"),
                 total_n_kg_ha=("total_n_kg_ha", "mean"),
                 total_irrig_mm=("total_irrig_mm", "mean"),
                 invalid_rate=("invalid_action_rate", "mean")).reset_index()


def cell_table(s: pd.DataFrame) -> pd.DataFrame:
    """Mean and SD over training seeds, canonical seeds only."""
    c = s[s.canonical.fillna(False)]
    g = c.groupby(["setting", "cell", "method", "arm", "target"], dropna=False).objective
    t = g.agg(mean="mean", sd="std", n_seeds="size", lo="min", hi="max").reset_index()
    t["sd"] = t.sd.fillna(0.0)
    return t


# An anchor gap is usable as a normalization denominator only if the scripted
# reference clears no-action by a margin that matters next to the policies being
# normalized. The margin is a tenth of the think arm's own gain over no-action:
# below it, a normalized score mostly measures the noise in the denominator.
# T1.1 is the case this exists for -- the paper's heuristic sits at or below
# no-action on most T1.1 splits, as the submitted figure package also recorded.
MIN_GAP_FRACTION = 0.10
PAPER_ANCHOR = "Ag heuristic (paper)"


def anchor_table(c: pd.DataFrame) -> pd.DataFrame:
    """Anchor returns per (setting, cell, target), and whether the gap is usable.

    The paper's reference is the paper-era per-crop policy on WOFOST and the
    first implementation on DSSAT and CycleGym (both kept in
    experiments/rebuttal_confirmatory_eval/paper_references/; the DSSAT branch is a
    reconstruction); the current module is kept alongside for comparison.
    """
    anc = c[c.method.isin(["No-action", PAPER_ANCHOR, "Ag heuristic (current)"])]
    # groupby keeps the NaN cell of the shared anchors without forming the
    # cross-product of index levels that pivot_table(dropna=False) builds.
    w = anc.groupby(["setting", "cell", "target", "method"], dropna=False)["mean"].first() \
           .unstack("method").reset_index()
    w.columns.name = None
    w = w.rename(columns={"No-action": "no_action", PAPER_ANCHOR: "heuristic_paper",
                          "Ag heuristic (current)": "heuristic_current"})
    # T1.3 is crop planning, where the first implementation and the current module take the same
    # fallback, so run_e4_anchors.sh runs it once, as the current module.
    t13 = w.setting.eq("T1.3") & w.heuristic_paper.isna()
    w.loc[t13, "heuristic_paper"] = w.loc[t13, "heuristic_current"]
    w["gap_paper"] = w.heuristic_paper - w.no_action
    # The reference cell for the usability test: the canonical think arm, or
    # every think cell sharing the target when the anchor row carries no cell.
    think = c[c.method.eq("LLM-RL") & c.arm.eq("think")]
    ref = []
    for r in w.itertuples():
        t = think[think.setting.eq(r.setting) & think.target.eq(r.target)]
        if isinstance(r.cell, str):
            t = t[t.cell.eq(r.cell)]
        ref.append(t["mean"].max() if len(t) else np.nan)
    w["think_best"] = ref
    w["think_gain"] = w.think_best - w.no_action
    w["gap_usable"] = (w.gap_paper > 0) & (w.gap_paper >= MIN_GAP_FRACTION * w.think_gain.abs())
    return w


def normalized_table(s: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    """(policy - no-action) / (paper heuristic - no-action), per seed, then mean/SD.

    Only targets whose anchor gap is usable are normalized; the rest keep their
    raw score in e4_cells.csv and are listed here with a reason.
    """
    pol = s[s.canonical.fillna(False) & ~s.method.isin(["No-action", PAPER_ANCHOR, "Ag heuristic (current)"])]
    a_cell = anchors[anchors.cell.notna()].set_index(["setting", "cell", "target"])
    a_any = anchors[anchors.cell.isna()].set_index(["setting", "target"])
    rows = []
    for r in pol.itertuples():
        key = (r.setting, r.cell, r.target)
        if key in a_cell.index:
            a = a_cell.loc[key]
        elif (r.setting, r.target) in a_any.index:
            a = a_any.loc[(r.setting, r.target)]
        else:
            continue
        ok = bool(a.gap_usable)
        rows.append(dict(setting=r.setting, cell=r.cell, method=r.method, arm=r.arm, seed=r.seed,
                         target=r.target, raw=r.objective, no_action=a.no_action,
                         heuristic_paper=a.heuristic_paper,
                         normalized=(r.objective - a.no_action) / a.gap_paper if ok else np.nan,
                         status="ok" if ok else "anchor gap not usable"))
    n = pd.DataFrame(rows)
    g = n.groupby(["setting", "cell", "method", "arm", "target", "status"], dropna=False)
    out = g.agg(norm_mean=("normalized", "mean"), norm_sd=("normalized", "std"),
                n_seeds=("normalized", "size"), raw_mean=("raw", "mean"),
                no_action=("no_action", "first"), heuristic_paper=("heuristic_paper", "first")).reset_index()
    out["norm_sd"] = out.norm_sd.fillna(0.0)
    return out


def paired(d: pd.DataFrame, a: dict, b: dict, label: str) -> pd.DataFrame:
    """A minus B, per target, differenced by scenario inside each seed.

    A and B are row filters. When both are trained arms they are matched on
    seed; when B is a single untrained model it is matched to every seed of A.
    """
    def pick(f):
        m = d.terminal.copy()
        for k, v in f.items():
            m &= d[k].eq(v) if not isinstance(v, (list, tuple)) else d[k].isin(v)
        return d[m]
    A, B = pick(a), pick(b)
    rows = []
    for tgt in sorted(set(A.target) & set(B.target)):
        At, Bt = A[A.target == tgt], B[B.target == tgt]
        b_seeds = sorted(Bt.seed.unique())
        per_seed = []
        for sd in sorted(At.seed.unique()):
            bs = sd if sd in b_seeds else (b_seeds[0] if len(b_seeds) == 1 else None)
            if bs is None:
                continue
            x = At[At.seed == sd].set_index("scenario_id").value
            y = Bt[Bt.seed == bs].set_index("scenario_id").value
            common = x.index.intersection(y.index)
            if len(common) == 0:
                continue
            diff = x.loc[common] - y.loc[common]
            per_seed.append((sd, diff.mean(), diff.std(ddof=1) / np.sqrt(len(diff)), len(diff)))
        if not per_seed:
            continue
        means = np.array([p[1] for p in per_seed])
        rows.append(dict(comparison=label, target=tgt, n_seeds=len(per_seed),
                         mean_diff=means.mean(), sd_over_seeds=means.std(ddof=1) if len(means) > 1 else 0.0,
                         seeds_positive=int((means > 0).sum()),
                         per_seed=";".join(f"s{p[0]}:{p[1]:+.3f}±{p[2]:.3f}(n={p[3]})" for p in per_seed)))
    return pd.DataFrame(rows)


def independent(d: pd.DataFrame, a: dict, b: dict, label: str) -> pd.DataFrame:
    """A minus B when the two share no seed, e.g. an LLM arm against a neural one.

    Pairing LLM seed 2 with PPO seed 2 would pair two unrelated random draws, so
    the seeds are treated as independent. The difference of seed means is
    reported against the propagated SD, sqrt(sd_A^2 + sd_B^2), and, separately, the paired SEM over scenarios after
    each policy's scenarios are averaged over its seeds.
    """
    def pick(f):
        m = d.terminal.copy()
        for k, v in f.items():
            m &= d[k].eq(v)
        return d[m]
    A, B = pick(a), pick(b)
    rows = []
    for tgt in sorted(set(A.target) & set(B.target)):
        At, Bt = A[A.target == tgt], B[B.target == tgt]
        ma, mb = At.groupby("seed").value.mean(), Bt.groupby("seed").value.mean()
        sda = ma.std(ddof=1) if len(ma) > 1 else 0.0
        sdb = mb.std(ddof=1) if len(mb) > 1 else 0.0
        diff = ma.mean() - mb.mean()
        prop = float(np.hypot(sda, sdb))
        xa = At.groupby("scenario_id").value.mean()
        xb = Bt.groupby("scenario_id").value.mean()
        common = xa.index.intersection(xb.index)
        sc = xa.loc[common] - xb.loc[common]
        rows.append(dict(comparison=label, target=tgt, n_seeds_a=len(ma), n_seeds_b=len(mb),
                         mean_diff=diff, propagated_sd=prop,
                         diff_in_prop_sd=diff / prop if prop > 0 else np.inf * np.sign(diff),
                         scenario_sem=sc.std(ddof=1) / np.sqrt(len(sc)) if len(sc) > 1 else np.nan,
                         n_scenarios=len(sc), all_a_seeds_above_all_b=bool(ma.min() > mb.max())))
    return pd.DataFrame(rows)


def _cat(xs):
    """Concatenate the non-empty frames; an empty frame when there are none (partial inputs)."""
    xs = [x for x in xs if len(x)]
    return pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()


def main():
    OUT.mkdir(exist_ok=True)
    d_all = load()
    # What the rebuild changed, per policy and target, before v1 is dropped.
    cv = d_all[d_all.cycles_version.notna() & d_all.terminal]
    cv = cv[cv.canonical.fillna(False) | cv.method.isin(["No-action", PAPER_ANCHOR, "Ag heuristic (current)"])]
    cvt = cv.groupby(["setting", "cell", "method", "arm", "target", "cycles_version"], dropna=False) \
            .agg(mean=("value", "mean"), n_seeds=("seed", "nunique")).reset_index()
    cvt = cvt.set_index(["setting", "cell", "method", "arm", "target", "cycles_version"])["mean"] \
             .unstack("cycles_version").reset_index()
    cvt.columns.name = None
    OUT.mkdir(exist_ok=True)
    cvt.to_csv(OUT / "e4_cycles_v1_v2.csv", index=False)
    d = d_all[d_all.cycles_version.ne("v1")].copy()
    s = seed_table(d)
    s.to_csv(OUT / "e4_by_seed.csv", index=False)
    c = cell_table(s)
    c.to_csv(OUT / "e4_cells.csv", index=False)

    llm = dict(method="LLM-RL", canonical=True)
    comps = []
    for cell in sorted(d.loc[d.method.eq("LLM-RL"), "cell"].dropna().unique()):
        comps.append(paired(d, {**llm, "cell": cell, "arm": "think"},
                            {**llm, "cell": cell, "arm": "no-think"}, f"{cell}: think - no-think"))
        st = SETTING[cell[:3]]
        for arm in ("think", "no-think"):
            zs = {"method": "zero-shot (Qwen)", "setting": st, "arm": arm}
            if st == "T1.1":
                zs["cell"] = cell
            comps.append(paired(d, {**llm, "cell": cell, "arm": arm}, zs, f"{cell}: {arm} - zero-shot"))
    p = _cat(comps)
    p.to_csv(OUT / "e4_paired.csv", index=False)

    ind = []
    nnm = d.method.fillna("").str.match(r"^(PPO|A2C|DQN)")
    for cell in sorted(d.loc[d.method.eq("LLM-RL"), "cell"].dropna().unique()):
        for nm in sorted(d.loc[d.cell.eq(cell) & nnm & d.canonical.eq(True), "method"].unique()):
            ind.append(independent(d, {**llm, "cell": cell, "arm": "think"},
                                   {"cell": cell, "method": nm, "canonical": True}, f"{cell}: think - {nm}"))
    for fam in ("a", "b"):
        ind.append(independent(d, {**llm, "cell": f"t22-family-{fam}", "arm": "think"},
                               {"cell": "t22-target", "method": "PPO target-trained"},
                               f"t22-family-{fam}: think - PPO target-trained"))
    q = _cat(ind)
    q.to_csv(OUT / "e4_vs_neural.csv", index=False)
    print(f"neural rows: {len(q)}  -> {OUT/'e4_vs_neural.csv'}")

    a = anchor_table(c)
    a.to_csv(OUT / "e4_anchors.csv", index=False)
    nt = normalized_table(s, a)
    nt.to_csv(OUT / "e4_normalized.csv", index=False)
    print(f"anchor rows: {len(a)} ({int(a.gap_usable.sum())} usable)  -> {OUT/'e4_anchors.csv'}")

    # Unified T3.2 policy against the matched T3.1 specialist, on the T3.2
    # callback suites both were scored on. Seeds are unrelated draws, so the
    # comparison is independent; scenario pairing uses seed-averaged scores.
    sp = []
    for arm in ("think", "no-think"):
        for sim in ("wofost", "dssat", "cycles"):
            sp.append(independent(d, {**llm, "cell": "t32-qwen", "arm": arm},
                                  {"method": "T3.1 specialist", "cell": f"t31-{sim}-specialist",
                                   "canonical": True},
                                  f"T3.2 unified {arm} - T3.1 {sim} specialist (think)"))
    sp = _cat(sp)
    sp.to_csv(OUT / "e4_specialist.csv", index=False)
    print(f"specialist rows: {len(sp)}  -> {OUT/'e4_specialist.csv'}")

    print(f"checkpoints: {s.groupby(['cell','method','arm','seed'], dropna=False).ngroups}")
    print(f"cell rows:   {len(c)}  -> {OUT/'e4_cells.csv'}")
    print(f"paired rows: {len(p)}  -> {OUT/'e4_paired.csv'}")
    short = c[c.method.eq("LLM-RL")].groupby(["cell", "arm"]).n_seeds.agg(["min", "max"])
    print("LLM arms with fewer than 3 canonical seeds on some target:")
    print(short[short["min"] < 3].to_string() or "  none")


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _ap.add_argument("--episodes", type=Path, default=EPISODES, help="per-episode parquet from extract_e4.py")
    _ap.add_argument("--out", type=Path, default=OUT, help="output directory for the CSV tables")
    _a = _ap.parse_args()
    EPISODES, OUT = _a.episodes, _a.out
    main()
