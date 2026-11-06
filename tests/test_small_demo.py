"""Checks for examples/small_demo: dataset config shape and the summary script."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "examples" / "small_demo"


def _load_summarize():
    spec = importlib.util.spec_from_file_location("small_demo_summarize", DEMO_DIR / "summarize.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dataset_config_keeps_val_and_test_disjoint():
    cfg = yaml.safe_load((DEMO_DIR / "config" / "small_demo_maize.yaml").read_text())
    splits = cfg["sampling"]["splits"]
    assert splits["train"]["role"] == "train"
    held = cfg["sampling"]["scenario_sets"]["heldout"]
    assert held["source_split"] == "val"
    val, test = splits["val"]["scenario_slice"], splits["test"]["scenario_slice"]
    assert splits["val"]["scenario_set"] == splits["test"]["scenario_set"] == "heldout"
    assert val["stop"] <= test["start"], "val and test slices must not overlap"
    assert test["stop"] <= held["num_samples"]
    # Same environment as the CPU quickstart, so its numbers are comparable.
    quick = yaml.safe_load(
        (REPO_ROOT / "smoke_tests" / "wofost_gym" / "config" / "quickstart_maize_cpu.yaml").read_text()
    )
    assert cfg["env"] == quick["env"]
    assert cfg["source"] == quick["source"]


def _write_policy(root: Path, name: str, values: list[float], invalid: float = 0.0, numeric: bool = False):
    policy_dir = root / "eval" / "test" / name
    policy_dir.mkdir(parents=True)
    results = []
    for i, value in enumerate(values):
        env_config = {"scenario_id": f"s{i}"}
        if numeric:
            turns = [{"turn": 0, "turn_metrics": {"objective_reward": value}}]
        else:
            turns = [{"turn": 0, "trajectory_metrics": {"objective_reward": value}}]
        results.append({"env_id": i, "env_config": env_config, "turns": turns})
    (policy_dir / "results.json").write_text(json.dumps(results))
    mean = sum(values) / len(values)
    if numeric:
        metrics = {"val-env/objective_reward/mean": mean, "val-env/invalid_action_rate/mean": invalid}
    else:
        metrics = {"objective_reward_mean": mean, "objective_reward_std": 0.0,
                   "invalid_action_rate_mean": invalid, "num_envs": len(values)}
    (policy_dir / "metrics.json").write_text(json.dumps(metrics))


def test_summary_table_and_paired_comparison(tmp_path, monkeypatch, capsys):
    summarize = _load_summarize()
    (tmp_path / "settings.json").write_text(json.dumps({"mode": "gpu", "model": "tiny"}))
    (tmp_path / "timings.tsv").write_text("grpo_train\t120\n")
    _write_policy(tmp_path, "no_action", [0.1, 0.2, 0.3])
    _write_policy(tmp_path, "untrained", [0.5, 0.5, 0.5], invalid=0.25)
    _write_policy(tmp_path, "trained", [1.0, 0.4, 1.5])
    _write_policy(tmp_path, "nn_ppo", [0.9, 0.8, 0.7], numeric=True)
    grpo = tmp_path / "grpo"
    grpo.mkdir()
    first_launch = [
        {"step": 0, "data": {"val-env/all/objective_reward/mean": 0.5}},
        {"step": 1, "data": {"critic/score/mean": 2.0}},
        {"step": 2, "data": {"critic/score/mean": 9.9}},
    ]
    # A resumed launch repeats step 2; its record wins.
    second_launch = [
        {"step": 2, "data": {"critic/score/mean": 2.5, "val-env/all/objective_reward/mean": 0.9}},
    ]
    for name, records in (("metrics_1.jsonl", first_launch), ("metrics_2.jsonl", second_launch)):
        (grpo / name).write_text("\n".join(json.dumps(r) for r in records) + "\n")

    monkeypatch.setattr(sys, "argv", ["summarize.py", "--out", str(tmp_path)])
    summarize.main()
    printed = capsys.readouterr().out
    assert "LLM after GRPO, greedy" in printed and "DEMO ONLY" in printed

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert abs(summary["test"]["nn_ppo"]["objective_mean"] - 0.8) < 1e-9
    paired = summary["paired"]["trained_minus_untrained"]
    assert paired["n"] == 3 and (paired["wins"], paired["losses"]) == (2, 1)
    assert summary["curve"]["val_x"] == [0, 2]
    assert summary["curve"]["train_y"] == [2.0, 2.5]


def test_demo_script_parses_and_lists_its_options():
    import subprocess

    script = DEMO_DIR / "run_small_demo.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    help_text = subprocess.run(
        ["bash", str(script), "--help"], check=True, capture_output=True, text=True
    ).stdout
    for option in ("--gpu", "--cpu-nn", "--think", "--steps", "--lora-rank", "--fresh"):
        assert option in help_text
