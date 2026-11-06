"""Print the small demo's score table and save its training curve.

Usage: python examples/small_demo/summarize.py --out <run directory>

Reads the evaluation outputs written by run_small_demo.sh under <out>/eval/, the
GRPO metrics logs (<out>/grpo/metrics_<launch>.jsonl) or the PPO validation log
(<out>/nn_train/**/validation_history.jsonl), and <out>/timings.tsv. Writes
<out>/summary.json, <out>/summary.md and <out>/training_curve.png.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

POLICY_LABELS = {
    "no_action": "No action",
    "random": "Random valid action",
    "ag_heuristic": "Scripted agronomic reference",
    "untrained": "LLM untrained, greedy",
    "trained": "LLM after GRPO, greedy",
    "untrained_sampled": "LLM untrained, sampled (T=1)",
    "trained_sampled": "LLM after GRPO, sampled (T=1)",
    "nn_ppo": "NN PPO, sampled actions",
    "nn_ppo_argmax": "NN PPO, most likely action",
}
VAL_OBJECTIVE_SUFFIX = "/objective_reward/mean"


def _find(path: Path, name: str) -> Path | None:
    direct = path / name
    if direct.is_file():
        return direct
    found = sorted(path.rglob(name))
    return found[0] if found else None


def _episode_objectives(results_path: Path) -> dict[str, float]:
    """Per-scenario objective reward keyed by scenario_id (LLM, scripted and NN outputs)."""
    data = json.loads(results_path.read_text())
    if isinstance(data, dict):
        data = data.get("results") or data.get("episodes") or []
    out: dict[str, float] = {}
    for index, result in enumerate(data):
        if not isinstance(result, dict):
            continue
        env_config = result.get("env_config") or {}
        key = str(env_config.get("scenario_id") or result.get("scenario_id") or index)
        value = None
        turns = [t for t in (result.get("turns") or []) if isinstance(t, dict)]
        for turn in turns:
            traj = turn.get("trajectory_metrics")
            if traj and traj.get("objective_reward") is not None:
                value = traj["objective_reward"]
        if value is None and turns:
            # The numeric-policy evaluator keeps per-turn metrics only; the
            # objective is terminal, so the last turn carries it.
            value = (turns[-1].get("turn_metrics") or {}).get("objective_reward")
        if value is None:
            traj = result.get("trajectory_metrics") or {}
            value = traj.get("objective_reward", result.get("objective_reward"))
        if value is not None and math.isfinite(float(value)):
            out[key] = float(value)
    return out


def _load_policy(policy_dir: Path) -> dict[str, Any] | None:
    metrics_path = _find(policy_dir, "metrics.json")
    if metrics_path is None:
        return None
    metrics = json.loads(metrics_path.read_text())
    row: dict[str, Any] = {
        "objective_mean": metrics.get("objective_reward_mean", metrics.get("val-env/objective_reward/mean")),
        "objective_std": metrics.get("objective_reward_std"),
        "invalid_action_rate": metrics.get(
            "invalid_action_rate_mean", metrics.get("val-env/invalid_action_rate/mean")
        ),
        "num_scenarios": metrics.get("num_envs") or metrics.get("num_episodes"),
        "metrics_file": str(metrics_path),
    }
    results_path = _find(policy_dir, "results.json")
    if results_path is not None:
        try:
            row["per_scenario"] = _episode_objectives(results_path)
        except Exception as exc:  # pragma: no cover - diagnostic only
            print(f"note: could not read per-scenario results from {results_path}: {exc}")
    values = list((row.get("per_scenario") or {}).values())
    if row["objective_mean"] is None and values:
        row["objective_mean"] = sum(values) / len(values)
    if row["objective_std"] is None and values:
        mean = sum(values) / len(values)
        row["objective_std"] = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    if not row["num_scenarios"] and row.get("per_scenario"):
        row["num_scenarios"] = len(row["per_scenario"])
    return row


def _paired(a: dict[str, float], b: dict[str, float]) -> dict[str, Any] | None:
    keys = sorted(set(a) & set(b))
    if len(keys) < 2:
        return None
    diffs = [a[k] - b[k] for k in keys]
    n = len(diffs)
    mean = sum(diffs) / n
    sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1))
    half = 1.96 * sd / math.sqrt(n)
    return {
        "n": n,
        "mean_diff": mean,
        "ci95": [mean - half, mean + half],
        "wins": sum(d > 1e-9 for d in diffs),
        "ties": sum(abs(d) <= 1e-9 for d in diffs),
        "losses": sum(d < -1e-9 for d in diffs),
    }


def _grpo_records(out: Path) -> list[tuple[int, dict]]:
    """Merge the per-launch GRPO logs (metrics_<n>.jsonl); a resumed launch
    repeats the steps after the last checkpoint and its records win."""
    grpo = out / "grpo"

    def launch_number(path: Path) -> int:
        tail = path.stem.rsplit("_", 1)[-1]
        return int(tail) if tail.isdigit() else 0

    paths = sorted(grpo.glob("metrics*.jsonl"), key=launch_number)
    by_step: dict[int, dict] = {}
    for path in paths:
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                by_step.setdefault(int(record["step"]), {}).update(record["data"])
    return sorted(by_step.items())


def _grpo_curve(out: Path) -> dict[str, list] | None:
    records = _grpo_records(out)
    if not records:
        return None
    val_steps, val_scores, val_invalid = [], [], []
    train_steps, train_scores = [], []
    train_label = "training reward"
    for step, data in records:
        val_keys = [k for k in data if k.startswith("val-env") and k.endswith(VAL_OBJECTIVE_SUFFIX)]
        val_keys.sort(key=lambda k: ("/all/" not in k, len(k)))
        if val_keys:
            val_steps.append(step)
            val_scores.append(float(data[val_keys[0]]))
            inv = [k for k in data if k.startswith("val-env") and k.endswith("/invalid_action_rate/mean")]
            inv.sort(key=lambda k: ("/all/" not in k, len(k)))
            val_invalid.append(float(data[inv[0]]) if inv else None)
        train_key = next((k for k in ("train-env/objective_reward/mean", "critic/score/mean") if k in data), None)
        if train_key:
            train_steps.append(step)
            train_scores.append(float(data[train_key]))
            train_label = (
                "training episodes, sampled (T=1)"
                if train_key.startswith("train-env")
                else "training reward per episode (objective + format bonus, sampled)"
            )
    return {
        "x_label": "GRPO step",
        "val_x": val_steps,
        "val_y": val_scores,
        "val_invalid": val_invalid,
        "train_x": train_steps,
        "train_y": train_scores,
        "train_label": train_label,
    }


def _nn_curve(out: Path) -> dict[str, list] | None:
    paths = sorted((out / "nn_train").rglob("validation_history.jsonl"))
    if not paths:
        return None
    xs, ys = [], []
    for line in paths[0].read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        metrics = record.get("metrics") or {}
        keys = [k for k in metrics if k.endswith(VAL_OBJECTIVE_SUFFIX) and "/all/" in k]
        keys = keys or [k for k in metrics if k.endswith(VAL_OBJECTIVE_SUFFIX)]
        if not keys:
            continue
        keys.sort(key=len)
        xs.append(int(record.get("global_step", 0)))
        ys.append(float(metrics[keys[0]]))
    return {
        "x_label": "PPO environment steps",
        "val_x": xs,
        "val_y": ys,
        "val_label": "validation objective (16 held-out scenarios, most likely action)",
    }


def _plot(curve: dict[str, list], val_refs: dict[str, float], path: Path, title: str) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping the plot")
        return False
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    if curve.get("train_x"):
        ax.plot(curve["train_x"], curve["train_y"], color="#9aa5b1", linewidth=1,
                label=curve.get("train_label", "training reward"))
    ax.plot(curve["val_x"], curve["val_y"], marker="o", color="#1f6feb", linewidth=2,
            label=curve.get("val_label", "validation objective (16 held-out scenarios, greedy)"))
    styles = {"ag_heuristic": ("#2da44e", "--"), "no_action": ("#cf222e", ":"), "random": ("#8250df", "-.")}
    for name, value in val_refs.items():
        color, style = styles.get(name, ("black", "-."))
        ax.axhline(value, color=color, linestyle=style, linewidth=1.2,
                   label=f"{POLICY_LABELS.get(name, name)} on the same 16")
    ax.set_xlabel(curve["x_label"])
    ax.set_ylabel("profit objective (higher is better)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out: Path = args.out.resolve()
    settings = json.loads((out / "settings.json").read_text()) if (out / "settings.json").is_file() else {}

    rows: dict[str, dict[str, Any]] = {}
    for name in ("no_action", "random", "ag_heuristic", "nn_ppo", "nn_ppo_argmax",
                 "untrained", "trained", "untrained_sampled", "trained_sampled"):
        policy_dir = out / "eval" / "test" / name
        if policy_dir.is_dir():
            row = _load_policy(policy_dir)
            if row is not None:
                rows[name] = row
    val_refs = {}
    for name in ("no_action", "random", "ag_heuristic"):
        policy_dir = out / "eval" / "val" / name
        if policy_dir.is_dir():
            row = _load_policy(policy_dir)
            if row and row["objective_mean"] is not None:
                val_refs[name] = row["objective_mean"]

    timings = {}
    if (out / "timings.tsv").is_file():
        for line in (out / "timings.tsv").read_text().splitlines():
            if "\t" in line:
                phase, seconds = line.split("\t", 1)
                timings[phase] = int(seconds)
    gpu_peaks = {}
    for path in sorted((out / "logs").glob("gpu_memory_*.txt")):
        values = [int(v) for v in path.read_text().split() if v.strip().isdigit()]
        if values:
            gpu_peaks[path.stem.replace("gpu_memory_", "")] = {"peak_mib": max(values), "first_mib": values[0]}

    comparisons = {}
    for focus in ("trained", "trained_sampled", "nn_ppo", "nn_ppo_argmax"):
        if focus not in rows:
            continue
        own_baseline = {"trained": "untrained", "trained_sampled": "untrained_sampled"}.get(focus)
        for other in (own_baseline, "no_action", "ag_heuristic"):
            if other is None:
                continue
            if other in rows:
                result = _paired(rows[focus].get("per_scenario", {}), rows[other].get("per_scenario", {}))
                if result:
                    comparisons[f"{focus}_minus_{other}"] = result

    lines = []
    n_test = next((r["num_scenarios"] for r in rows.values() if r.get("num_scenarios")), "?")
    lines.append(f"AgriManager small demo ({settings.get('mode', '?')}), {n_test} held-out maize scenarios")
    lines.append("DEMO ONLY: not a paper result. Objective = profit_max (higher is better).")
    lines.append("")
    lines.append(f"| {'Policy':<30} | {'Objective mean':>14} | {'SD':>7} | {'Invalid actions':>15} |")
    lines.append(f"|{'-' * 32}|{'-' * 16}|{'-' * 9}|{'-' * 17}|")
    for name, row in rows.items():
        mean = row["objective_mean"]
        sd = row["objective_std"]
        inv = row["invalid_action_rate"]
        lines.append(
            f"| {POLICY_LABELS.get(name, name):<30} | "
            f"{mean:>14.3f} | {(sd if sd is not None else float('nan')):>7.3f} | "
            f"{(f'{inv:.1%}' if inv is not None else 'n/a'):>15} |"
        )
    if comparisons:
        lines.append("")
        lines.append("Paired by scenario (same 64 scenarios), mean difference with a 95% normal interval:")
        for key, c in comparisons.items():
            a, b = key.split("_minus_")
            lines.append(
                f"  {POLICY_LABELS.get(a, a)} - {POLICY_LABELS.get(b, b)}: {c['mean_diff']:+.3f} "
                f"[{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}], better on {c['wins']}/{c['n']}, "
                f"tied {c['ties']}, worse {c['losses']}"
            )
    if timings:
        lines.append("")
        phases = {k: v for k, v in timings.items() if k != "total"}
        total = timings.get("total", sum(phases.values()))
        lines.append("Wall time per phase (latest run of each): "
                     + ", ".join(f"{k} {v / 60:.1f} min" for k, v in phases.items())
                     + f"; whole script {total / 60:.1f} min"
                     + (" (scripted baselines run beside the GPU phases)" if settings.get("mode") == "gpu" else ""))
    if gpu_peaks:
        lines.append("Peak GPU memory in use (whole GPU, nvidia-smi): "
                     + ", ".join(f"{k} {v['peak_mib'] / 1024:.1f} GiB" for k, v in gpu_peaks.items()))

    curve = _grpo_curve(out) or _nn_curve(out)
    plot_path = out / "training_curve.png"
    if curve and curve.get("val_x"):
        model = settings.get("model", "")
        think = " (think)" if settings.get("think") else ""
        title = (f"Small demo: GRPO, {model}{think}" if settings.get("mode") == "gpu"
                 else "Small demo: neural-network PPO (CPU)")
        if _plot(curve, val_refs, plot_path, title + " (demo, not a paper result)"):
            lines.append(f"Training curve: {plot_path}")

    text = "\n".join(lines)
    print()
    print(text)
    (out / "summary.md").write_text(text + "\n")
    summary = {
        "settings": settings,
        "test": {k: {kk: vv for kk, vv in v.items() if kk != "per_scenario"} for k, v in rows.items()},
        "val_references": val_refs,
        "paired": comparisons,
        "timings_s": timings,
        "gpu_memory": gpu_peaks,
        "curve": curve,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
