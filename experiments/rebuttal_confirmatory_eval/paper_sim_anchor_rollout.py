"""The submitted paper's DSSAT and CycleGym ag-heuristic anchors, run on E4.

    python paper_sim_anchor_rollout.py paper_references/random_rollout_paper_sim.py <parquet> <output_dir> <validation_axis>

The DSSAT and CycleGym anchors in the submitted figures were not produced by
the ag_heuristic now in the tree. The runs date from 2026-05-05 and used the
first implementation inside random_rollout.py, kept in this repository as
paper_references/random_rollout_paper_sim.py. The current ag_heuristic module
behaves differently on both simulators:

  CycleGym  the first implementation applies split nitrogen by season progress. The current
            module only knows the crop-planning fallback, so on the
            fertilization task it returns the zero action and scores exactly
            as no-action.
  DSSAT     the first implementation carries the gym-dssat expert schedule (3 fertilizations,
            16 irrigations) but matches days exactly, so at the weekly
            decision interval it applied nothing and its logged run scored
            exactly as no-action. The submitted figures then used a value
            set by hand, with no run or code recorded for it. The current
            module keeps the fertilizations but only 3 irrigations.

What this runs, and how close each is to the paper:

  CycleGym  the first implementation verbatim, loaded by the caller. Reproduces the
            paper's T3.1 anchor exactly on the submitted validation file
            (10.8282 vs 10828.199 ge kg/ha).
  DSSAT     the first implementation's schedule, applied at the decision step whose interval
            contains each scheduled day. Reproduces the hand-patched anchor's
            input cost exactly (116 kg N, 264 mm; 419.2 ge kg/ha) but not its
            yield: 7166.1 vs 7466.6 ge kg/ha profit on the T3.1 validation
            file. Applying each event after its day instead gives 7006.7. The
            patched value is most plausibly a daily-step run that no longer
            exists, so this is a reconstruction, and is labelled as one.

The caller passes the path of the paper-era module; only its DSSAT branch is
replaced here.
"""

from __future__ import annotations

import importlib.util
import sys
from typing import Any, Dict

DSSAT_FERTILIZATION = {40: 27.0, 45: 35.0, 80: 54.0}
DSSAT_IRRIGATION = {6: 13.0, 20: 10.0, 37: 10.0, 50: 13.0, 54: 18.0, 65: 25.0, 69: 25.0,
                    72: 13.0, 75: 15.0, 77: 19.0, 80: 20.0, 84: 20.0, 91: 15.0, 101: 19.0,
                    104: 4.0, 105: 25.0}


def _load(path: str):
    spec = importlib.util.spec_from_file_location("random_rollout_paper_sim", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    may5_path, parquet, output_dir, axis = sys.argv[1:5]
    rr = _load(may5_path)

    def dssat_action(env: Any, observation: Dict[str, Any]) -> Dict[str, float]:
        dap = int(round(rr._numeric_obs_value(observation, "dap", "DAP", default=0.0)))
        interval = max(1, int(getattr(env, "decision_interval", 1) or 1))
        done = getattr(env, "_paper_dssat_applied", None)
        if done is None:
            done = set()
            env._paper_dssat_applied = done
        anfer = amir = 0.0
        for day, amount in DSSAT_FERTILIZATION.items():
            if ("fert", day) not in done and day < dap + interval:
                anfer += amount
                done.add(("fert", day))
        for day, amount in DSSAT_IRRIGATION.items():
            if ("irrig", day) not in done and day < dap + interval:
                amir += amount
                done.add(("irrig", day))
        return {"anfer": anfer, "amir": amir, "pesticide": 0.0}

    original = rr._select_ag_heuristic_action

    def select(env, action_space, observation, turn, turn_num):
        env_name = str(getattr(getattr(env, "config", None), "env_name", "") or "").lower()
        if env_name == "gym_dssat" or "dssat" in type(env).__name__.lower():
            return dssat_action(env, observation or {})
        if env_name == "cycles_gym" or "cycles" in type(env).__name__.lower():
            return original(env, action_space, observation, turn, turn_num)
        raise RuntimeError(f"paper DSSAT/CycleGym anchor asked to act on {type(env).__name__}")

    rr._select_ag_heuristic_action = select
    rr.run_random_rollout(
        output_dir=output_dir,
        seed=42,
        mode="ag_heuristic",
        num_trials=1,
        validation_axis=None if axis == "none" else axis,
        wandb_config={"enabled": False},
        test_files=parquet,
    )


if __name__ == "__main__":
    main()
