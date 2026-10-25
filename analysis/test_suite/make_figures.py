"""Camera-ready figures on the frozen E4 test suite.

    python analysis/test_suite/make_figures.py

Reads results/tables/*.csv (e4_tables.py; override with $E4_TABLES) and
writes results/figures/*.{pdf,png} (override with $E4_FIGURES).

The figures follow the submitted Tier 1-3 figures in layout and style so they
can replace them panel for panel, with three deliberate differences:

* Raw objective on the y-axis, with the two anchors drawn as ticks across each
  cluster (no-action solid, the paper's scripted reference dotted) rather than
  folded into a normalized score. The anchor gap is too small to divide by on
  T1.1 (see tables/e4_anchors.csv), and drawing it shows that directly; the
  normalized scores, where the gap is usable, are in tables/e4_normalized.csv.
  Panels do not share a y-scale across settings.
* Every bar is a mean over three training seeds, drawn with its SD and with the
  three seeds as dots, so the spread is visible rather than summarized.
* The untrained model is the open marker, as in the submitted figures. Every
  setting now has an untrained E4 evaluation under both protocols.
* On the three T3.2 callback targets the matched T3.1 specialist, scored on the
  same T3.2 scenarios, is the dashed line with its seed range shaded.

Palette: the submitted figures' Okabe-Ito triple, validated with the dataviz
skill's checker. Target-trained PPO adds #CC79A7, whose deuteranope distance
from the NN green is 7.6 (floor band), so it always carries a hatch as well.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
TAB = Path(os.environ.get("E4_TABLES", HERE.parents[1] / "results" / "tables"))
FIG = Path(os.environ.get("E4_FIGURES", HERE.parents[1] / "results" / "figures"))

THINK, NOTHINK, NN, TARGET = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
ID_SHADE, OOD_SHADE = "#e8f1ef", "#f5efe6"
INK, MUTED = "#222222", "#6b6b6b"

plt.rcParams.update({
    "font.family": "serif", "font.weight": "bold", "axes.labelweight": "bold",
    "axes.titleweight": "bold", "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK, "axes.linewidth": 0.8,
    "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

by_seed = pd.read_csv(TAB / "e4_by_seed.csv")
anchors = pd.read_csv(TAB / "e4_anchors.csv")


def seeds(cell=None, method="LLM-RL", arm=None, setting=None, canonical=True):
    q = by_seed[by_seed.method.eq(method)]
    if cell is not None:
        q = q[q.cell.eq(cell)]
    if setting is not None:
        q = q[q.setting.eq(setting)]
    if arm is not None:
        q = q[q.arm.eq(arm)]
    if canonical:
        q = q[q.canonical.eq(True)]
    return q


def bar_group(ax, x, series, width=0.8):
    """Draw one cluster of bars at x.

    series: list of (values_by_target_or_None, color, hatch, zero_shot_value)
    where values is an array of per-seed numbers for this x.
    """
    k = len(series)
    w = width / k
    for j, (vals, color, hatch, zs) in enumerate(series):
        if vals is None or len(vals) == 0:
            continue
        xc = x - width / 2 + w * (j + 0.5)
        m, sd = np.mean(vals), (np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        ax.bar(xc, m, w * 0.92, color=color, edgecolor=INK, linewidth=0.5, hatch=hatch, zorder=2)
        ax.errorbar(xc, m, yerr=sd, color=INK, lw=0.8, capsize=2, capthick=0.8, zorder=3)
        jit = np.linspace(-w * 0.22, w * 0.22, len(vals)) if len(vals) > 1 else [0]
        ax.scatter(xc + np.asarray(jit), vals, s=7, color=INK, zorder=4, linewidths=0)
        if zs is not None and np.isfinite(zs):
            ax.scatter([xc], [zs], s=26, facecolor="white", edgecolor=INK, linewidths=0.8, zorder=5)


def vals_at(q, target):
    return q[q.target.eq(target)].sort_values("seed").objective.to_numpy()


def zs_at(setting, arm, target, cell=None):
    q = seeds(cell=cell, method="zero-shot (Qwen)", arm=arm, setting=setting, canonical=False)
    q = q[q.target.eq(target)]
    return float(q.objective.iloc[0]) if len(q) else None


def anchor_ticks(ax, x, setting, target, cell=None, width=0.8):
    """No-action (solid) and the paper's scripted reference (dotted) across one cluster."""
    a = anchors[anchors.setting.eq(setting) & anchors.target.eq(target)]
    if cell is not None and a.cell.notna().any():
        a = a[a.cell.eq(cell)]
    if a.empty:
        return
    r = a.iloc[0]
    x0, x1 = x - width / 2, x + width / 2
    if np.isfinite(r.no_action):
        ax.hlines(r.no_action, x0, x1, colors=MUTED, lw=1.1, zorder=6)
    if np.isfinite(r.heuristic_paper):
        ax.hlines(r.heuristic_paper, x0, x1, colors=INK, lw=1.3, linestyles=(0, (1, 1.2)), zorder=6)


# The three callback targets, each matched to the T3.1 specialist of its
# simulator. The WOFOST specialist is also scored on the nutrient tuple, but it
# was trained for profit, so it is no matched reference there.
CALLBACK_TARGETS = {"wofost_wofost_lnw_profit", "dssat_profit", "cycles_profit"}


def specialist_band(ax, x, target, width=0.7):
    if target not in CALLBACK_TARGETS:
        return
    q = by_seed[by_seed.method.eq("T3.1 specialist") & by_seed.canonical.eq(True) & by_seed.target.eq(target)]
    if q.empty:
        return
    v = q.objective.to_numpy()
    x0, x1 = x - width / 2, x + width / 2
    ax.fill_between([x0, x1], v.min(), v.max(), color=INK, alpha=0.10, lw=0, zorder=1)
    ax.hlines(v.mean(), x0, x1, colors=INK, lw=1.4, linestyles="--", zorder=6)


ANCHOR_HANDLES = [Line2D([], [], color=MUTED, lw=1.1, label="no-action anchor"),
                  Line2D([], [], color=INK, lw=1.3, ls=(0, (1, 1.2)), label="scripted reference (paper's anchor)")]


def shade(ax, spans):
    """spans: list of (x0, x1, kind) with kind 'id' or 'ood'."""
    for x0, x1, kind in spans:
        ax.axvspan(x0, x1, color=ID_SHADE if kind == "id" else OOD_SHADE, zorder=0, lw=0)


def finish(ax, labels, title, ylabel=None):
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title(title, pad=13)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlim(-0.6, len(labels) - 0.4)


def strongest_nn(cell):
    """The neural baseline with the highest mean over this cell's targets.

    Choosing the strongest on the test set favours the baseline, which is the
    conservative direction for an LLM-versus-neural claim.
    """
    q = by_seed[by_seed.cell.eq(cell) & by_seed.method.str.match(r"^(PPO|A2C|DQN)", na=False)
                & by_seed.canonical.eq(True)]
    if q.empty:
        return None, None
    best = q.groupby("method").objective.mean().idxmax()
    return best, q[q.method.eq(best)]


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.pdf / .png")


def method_legend(fig, extra=(), ncol=4, y=1.02):
    h = [Patch(facecolor=NN, edgecolor=INK, lw=0.5, label="NN-RL (strongest)"),
         Patch(facecolor=NOTHINK, edgecolor=INK, lw=0.5, label="LLM-RL (no-think)"),
         Patch(facecolor=THINK, edgecolor=INK, lw=0.5, label="LLM-RL (think)"),
         *extra,
         Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK,
                label="untrained model (step 0)"),
         *ANCHOR_HANDLES,
         Line2D([], [], marker="o", ls="", markersize=3, color=INK, label="one training seed"),
         Patch(facecolor=ID_SHADE, label="ID / seen"), Patch(facecolor=OOD_SHADE, label="OOD / held-out")]
    fig.legend(handles=h, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y))


# --------------------------------------------------------------------------- Tier 1
def tier1():
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), gridspec_kw={"width_ratios": [5, 5, 6, 1.3], "wspace": 0.28})
    regimes = ["id", "drought", "wet", "hot", "cold"]
    for ax, crop, tag in ((axes[0], "potato", "a"), (axes[1], "chickpea", "b")):
        cell = f"t11-{crop}"
        nn_name, nnq = strongest_nn(cell)
        th, nt = seeds(cell, arm="think"), seeds(cell, arm="no-think")
        for i, r in enumerate(regimes):
            bar_group(ax, i, [(vals_at(nnq, r) if nnq is not None else None, NN, None, None),
                              (vals_at(nt, r), NOTHINK, None, zs_at("T1.1", "no-think", r, cell)),
                              (vals_at(th, r), THINK, None, zs_at("T1.1", "think", r, cell))])
            anchor_ticks(ax, i, "T1.1", r, cell)
        shade(ax, [(-0.6, 0.5, "id"), (0.5, 4.6, "ood")])
        finish(ax, ["ID", "drought", "wet", "hot", "cold"],
               f"({tag}) Weather regime, {crop}", "E4 objective (raw)" if tag == "a" else None)
        ax.text(0.5, 1.004, f"strongest NN: {nn_name}", transform=ax.transAxes, ha="center",
                va="bottom", fontsize=7, color=MUTED, fontweight="normal")

    ax = axes[2]
    labels, spans, x = [], [], 0
    for cov in ("4", "8", "16"):
        cell = f"t12-crosscrop-{cov}id"
        nn_name, nnq = strongest_nn(cell)
        th, nt = seeds(cell, arm="think"), seeds(cell, arm="no-think")
        for tgt, lab, kind in (("id_crops", f"{cov}ID\nID", "id"), ("heldout_crops", f"{cov}ID\nheld-out", "ood")):
            bar_group(ax, x, [(vals_at(nnq, tgt) if nnq is not None else None, NN, None, None),
                              (vals_at(nt, tgt), NOTHINK, None, zs_at("T1.2", "no-think", tgt)),
                              (vals_at(th, tgt), THINK, None, zs_at("T1.2", "think", tgt))])
            anchor_ticks(ax, x, "T1.2", tgt)
            spans.append((x - 0.5, x + 0.5, kind)); labels.append(lab); x += 1
    shade(ax, spans)
    finish(ax, labels, "(c) Crop variety (4 / 8 / 16 training crops)")
    ax.text(0.5, 1.004, "strongest NN: PPO (n8) at every coverage", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7, color=MUTED, fontweight="normal")

    ax = axes[3]
    cell = "t13-price"
    nn_name, nnq = strongest_nn(cell)
    # Gross revenue sits near 30,000 for every policy, so bars from zero would
    # hide the differences and bars from 28,000 would exaggerate them. Points
    # with their seed spread show both the level and the gap honestly.
    for j, (vals, color) in enumerate(((vals_at(nnq, "all"), NN),
                                       (vals_at(seeds(cell, arm="no-think"), "all"), NOTHINK),
                                       (vals_at(seeds(cell, arm="think"), "all"), THINK))):
        xc = -0.25 + 0.25 * j
        m, sd = vals.mean(), vals.std(ddof=1)
        ax.errorbar(xc, m, yerr=sd, fmt="o", color=color, mec=INK, mew=0.6, ms=7, ecolor=INK,
                    elinewidth=0.8, capsize=2, zorder=3)
        ax.scatter(np.full(len(vals), xc + 0.07), vals, s=7, color=INK, zorder=4, linewidths=0)
        arm = {1: "no-think", 2: "think"}.get(j)
        zs = zs_at("T1.3", arm, "all") if arm else None
        if zs is not None:
            ax.scatter([xc], [zs], s=26, facecolor="white", edgecolor=INK, linewidths=0.8, zorder=5)
    anchor_ticks(ax, 0, "T1.3", "all", width=1.0)
    shade(ax, [(-0.6, 0.6, "ood")])
    finish(ax, ["all\nregimes"], "(d) Price regime")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:.1f}k"))
    ax.text(0.5, 1.004, "gross revenue", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=7, color=MUTED, fontweight="normal")
    method_legend(fig, ncol=7, y=1.08)
    save(fig, "e4_tier1")


# --------------------------------------------------------------------------- Tier 2
def _clustered(ax, targets, labels, seen, series_fn, title, ylabel=None, highlight=None):
    """series_fn(target) -> list of bar_group series for that target."""
    for i, t in enumerate(targets):
        bar_group(ax, i, series_fn(t))
    shade(ax, [(i - 0.5, i + 0.5, "id" if t in seen else "ood") for i, t in enumerate(targets)])
    finish(ax, labels, title, ylabel)
    if highlight in targets:
        i = targets.index(highlight)
        ax.get_xticklabels()[i].set_color("#b3261e")


def tier2():
    fig = plt.figure(figsize=(16, 7.6))
    gs = fig.add_gridspec(2, 14, height_ratios=[1, 1], hspace=0.6, wspace=1.4)

    # (a) observation schema
    ax = fig.add_subplot(gs[0, 0:9])
    cell = "t21-observation"
    th, nt = seeds(cell, arm="think"), seeds(cell, arm="no-think")
    tg = ["s1", "s2a", "s2b", "s2c", "s2d", "s3", "s4", "s5"]
    lab = ["S1 full\n(train)", "S2a\n-stage", "S2b\n-resource", "S2c\n-history", "S2d\n-weather",
           "S3\nsynonyms", "S4\nsuperset", "S5\nanonymous"]
    _clustered(ax, tg, lab, {"s1"},
               lambda t: [(vals_at(nt, t), NOTHINK, None, zs_at("T2.1", "no-think", t)),
                          (vals_at(th, t), THINK, None, zs_at("T2.1", "think", t))],
               "(a) Observation schema $\\Sigma_O$", "E4 objective (raw)", highlight="s2b")
    for i, t in enumerate(tg):
        anchor_ticks(ax, i, "T2.1", t)

    # (f) multi-objective minus yield-only, per objective, seed-matched
    ax = fig.add_subplot(gs[0, 9:14])
    objs = ["yield_max", "profit_max", "water_stewardship", "nutrient_stewardship"]
    olab = ["yield", "profit", "water", "nutrient\n(held out)"]
    for i, o in enumerate(objs):
        series = []
        for arm, color in (("no-think", NOTHINK), ("think", THINK)):
            a = seeds("t23-objectives", arm=arm); b = seeds("t23-yield-only", arm=arm)
            a = a[a.target.eq(o)].set_index("seed").objective
            b = b[b.target.eq(o)].set_index("seed").objective
            common = a.index.intersection(b.index)
            series.append(((a.loc[common] - b.loc[common]).to_numpy(), color, None, None))
        bar_group(ax, i, series)
    ax.axhline(0, color=INK, lw=0.7)
    shade(ax, [(-0.5, 2.5, "id"), (2.5, 3.5, "ood")])
    finish(ax, olab, "(f) Multi-objective $-$ yield-only")
    ax.text(0.5, 1.004, "same seed, same scenarios", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=7, color=MUTED, fontweight="normal")

    # (b, c) action menu
    menus = ["ln", "lnw", "lw", "lnpkw"]
    for k, (fam, seen) in enumerate((("a", {"lnpkw"}), ("b", {"lw"}))):
        ax = fig.add_subplot(gs[1, 4 * k:4 * k + 4])
        cell = f"t22-family-{fam}"
        th, nt = seeds(cell, arm="think"), seeds(cell, arm="no-think")
        mask = seeds(cell, method="PPO masked"); tgt = seeds("t22-target", method="PPO target-trained")
        _clustered(ax, menus, [m + ("\n(train)" if m in seen else "") for m in menus], seen,
                   lambda t: [(vals_at(mask, t), NN, None, None),
                              (vals_at(tgt, t), TARGET, "////", None),
                              (vals_at(nt, t), NOTHINK, None, zs_at("T2.2", "no-think", t)),
                              (vals_at(th, t), THINK, None, zs_at("T2.2", "think", t))],
                   f"({'bc'[k]}) Action menu $\\Sigma_A$, family {fam.upper()}",
                   "E4 objective (raw)" if k == 0 else None, highlight="lnw")
        for i, t in enumerate(menus):
            anchor_ticks(ax, i, "T2.2", t)

    # (d, e) reward formulation
    for k, (cell, seen, name) in enumerate((("t23-objectives", {"yield_max", "profit_max", "water_stewardship"}, "multi-objective"),
                                            ("t23-yield-only", {"yield_max"}, "yield-only"))):
        ax = fig.add_subplot(gs[1, 8 + 3 * k:11 + 3 * k])
        th, nt = seeds(cell, arm="think"), seeds(cell, arm="no-think")
        _clustered(ax, objs, ["yield", "profit", "water", "nutrient"], seen,
                   lambda t: [(vals_at(nt, t), NOTHINK, None, zs_at("T2.3", "no-think", t)),
                              (vals_at(th, t), THINK, None, zs_at("T2.3", "think", t))],
                   f"({'de'[k]}) $\\Sigma_R$, {name}", highlight="nutrient_stewardship")
        for i, t in enumerate(objs):
            anchor_ticks(ax, i, "T2.3", t)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    extra = [Patch(facecolor=NN, edgecolor=INK, lw=0.5, label="PPO masked (same menus as the LLM family)"),
             Patch(facecolor=TARGET, edgecolor=INK, lw=0.5, hatch="////", label="PPO trained on the test menu")]
    h = [*extra,
         Patch(facecolor=NOTHINK, edgecolor=INK, lw=0.5, label="LLM-RL (no-think)"),
         Patch(facecolor=THINK, edgecolor=INK, lw=0.5, label="LLM-RL (think)"),
         Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK, label="untrained model"),
         Line2D([], [], marker="o", ls="", markersize=3, color=INK, label="one training seed"),
         *ANCHOR_HANDLES,
         Patch(facecolor=ID_SHADE, label="seen in training"), Patch(facecolor=OOD_SHADE, label="held out")]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    save(fig, "e4_tier2")


# --------------------------------------------------------------------------- Tier 3
LLAMA_T, LLAMA_N = THINK, NOTHINK


def llama_at(arm, target):
    q = by_seed[by_seed.method.eq("LLM-RL (Llama)") & by_seed.arm.eq(arm) & by_seed.target.eq(target)]
    return float(q.objective.iloc[0]) if len(q) else None


def tier3():
    fig = plt.figure(figsize=(16, 7.4))
    gs = fig.add_gridspec(2, 12, hspace=0.62, wspace=1.3)
    sims = ["wofost", "dssat", "cycles"]
    names = {"wofost": "WOFOST", "dssat": "gym-DSSAT", "cycles": "CycleGym"}

    # (a-c) T3.1: one axis per target, since returns are not on a common scale
    for k, tgt in enumerate(sims):
        ax = fig.add_subplot(gs[0, 4 * k:4 * k + 4])
        for i, src in enumerate(sims):
            cell = f"t31-{src}"
            bar_group(ax, i, [(vals_at(seeds(cell, arm="no-think"), tgt), NOTHINK, None, zs_at("T3.1", "no-think", tgt)),
                              (vals_at(seeds(cell, arm="think"), tgt), THINK, None, zs_at("T3.1", "think", tgt))], width=0.7)
            anchor_ticks(ax, i, "T3.1", tgt, width=0.7)
        shade(ax, [(i - 0.5, i + 0.5, "id" if src == tgt else "ood") for i, src in enumerate(sims)])
        finish(ax, [f"from\n{names[s]}" for s in sims], f"({'abc'[k]}) T3.1, target {names[tgt]}",
               "E4 objective (raw)" if k == 0 else None)
        ax.text(0.5, 1.004, "policies compared within this target only", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=7, color=MUTED, fontweight="normal")

    # (d-f) T3.2 unified policy, grouped by what is on a common scale
    groups = [
        ("(d) T3.2 WOFOST tuples", ["wofost_wofost_lnpk_profit", "wofost_wofost_lnpkw_profit",
                                    "wofost_wofost_lnw_yield", "wofost_wofost_lnw_profit",
                                    "wofost_wofost_lnpkw_nutrient"],
         ["lnpk\nprofit", "lnpkw\nprofit", "lnw\nyield", "lnw\nprofit", "lnpkw\nnutrient"],
         {"wofost_wofost_lnpk_profit", "wofost_wofost_lnpkw_profit", "wofost_wofost_lnw_yield"}, "E4 objective (raw)"),
        ("(e) T3.2 profit, other simulators", ["dssat_profit", "cycles_profit"],
         ["gym-DSSAT\nprofit", "CycleGym\nprofit"], set(), "E4 objective (raw)"),
        ("(f) T3.2 yield, other simulators", ["dssat_yield", "cycles_yield"],
         ["gym-DSSAT\nyield", "CycleGym\nyield"], {"dssat_yield", "cycles_yield"}, "target yield (kg/ha)"),
    ]
    spans_cols = [(0, 6), (6, 9), (9, 12)]
    for (title, tg, lab, seen, yl), (c0, c1) in zip(groups, spans_cols):
        ax = fig.add_subplot(gs[1, c0:c1])
        th, nt = seeds("t32-qwen", arm="think"), seeds("t32-qwen", arm="no-think")
        for i, t in enumerate(tg):
            bar_group(ax, i, [(vals_at(nt, t), NOTHINK, None, zs_at("T3.2", "no-think", t)),
                              (vals_at(th, t), THINK, None, zs_at("T3.2", "think", t))], width=0.7)
            for dx, arm, col in ((-0.175, "no-think", LLAMA_N), (0.175, "think", LLAMA_T)):
                v = llama_at(arm, t)
                if v is not None:
                    ax.scatter([i + dx + 0.13], [v], marker="D", s=22, color=col, edgecolor=INK,
                               linewidths=0.6, zorder=6)
            anchor_ticks(ax, i, "T3.2", t, width=0.7)
            specialist_band(ax, i, t)
        shade(ax, [(i - 0.5, i + 0.5, "id" if t in seen else "ood") for i, t in enumerate(tg)])
        finish(ax, lab, title, yl)
        if "cycles_yield" in tg:
            ax.text(0.5, 1.004, "CycleGym yield is saturated: identical for every policy but one",
                    transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color=MUTED,
                    fontweight="normal")

    h = [Patch(facecolor=NOTHINK, edgecolor=INK, lw=0.5, label="LLM-RL (no-think), Qwen3-4B"),
         Patch(facecolor=THINK, edgecolor=INK, lw=0.5, label="LLM-RL (think), Qwen3-4B"),
         Line2D([], [], marker="D", ls="", color="white", markeredgecolor=INK, label="Llama-3.2-3B, one seed (colour = arm)"),
         Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK, label="untrained Qwen3-4B"),
         Line2D([], [], marker="o", ls="", markersize=3, color=INK, label="one training seed"),
         *ANCHOR_HANDLES,
         Line2D([], [], color=INK, lw=1.4, ls="--", label="matched T3.1 specialist, same scenarios (range shaded)"),
         Patch(facecolor=ID_SHADE, label="source = target / seen tuple"),
         Patch(facecolor=OOD_SHADE, label="transfer / held-out tuple")]
    fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03))
    save(fig, "e4_tier3")


if __name__ == "__main__":
    tier1()
    tier2()
    tier3()
