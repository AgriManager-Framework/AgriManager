# Baselines and reference policies

Every test-suite score is reported next to two scripted reference policies and,
on the settings where they apply, neural baselines.

| Role | Policy | Code mode or script | Scored on the test suite |
|---|---|---|---|
| Lower reference (normalized 0) | **no-action** | `no_action` | every target |
| Upper reference (normalized ±1) | **scripted agronomic reference** | per simulator, see below | every target |
| Learned baseline | **fixed-interface PPO** (`n1`, and `n8` in the appendix) | `entrypoints/train/nn_train.sh` | Tier 1 settings |
| Learned baselines | DQN and A2C (potato weather-regime), MaskablePPO and target-trained PPO (action-menu) | `agrimanager/nn_ppo/` | as listed |
| Reference point | the **untrained model** (same backbone before RL, both response formats) | `--checkpoint-dir` control in the evaluator | every target |
| Diagnostics only | `random`, `ppv0` (potential production) | `entrypoints/eval/config/*.yaml` | no |

## No-action

The no-action policy emits action 0 for discrete action spaces and the zero
vector for continuous ones. On WOFOST, DSSAT and Cycles crop growth this
applies no fertilizer and no irrigation. On Cycles crop planning (T1.3) the
zero action is crop index 0 at planting-week index 0, so no-action plants corn
in the earliest week every year and still earns gross revenue. A policy below
no-action does worse than abstaining; under a costed profit objective this
happens easily.

```bash
bash entrypoints/eval/baseline_eval.sh --config-name no_action \
  data.inference_file=<split.parquet> output.dir=<out_dir> wandb.enabled=false
```

## Scripted agronomic reference

A fixed rule set per simulator, written from agronomic guidance. It marks
codified practice. It is uncalibrated against experts or optimal control, so it
is a reference point and not an upper bound.

### Which version the paper uses

The code has evolved since the submitted figures were made. The paper's
numbers use the following versions; `experiments/rebuttal_confirmatory_eval/run_e4_anchors.sh`
reproduces them on every test split and records the exact code next to each
result.

| Simulator | Reference used in the paper | How it runs on the test suite |
|---|---|---|
| WOFOST (T1.1, T1.2, T2.x, WOFOST parts of T3.1 and T3.2) | the paper-era per-crop policy (`extension_guideline` mode) | kept as `experiments/rebuttal_confirmatory_eval/paper_references/extension_guideline_rollout_paper.py`; copied next to the results with its hash in `PROVENANCE` |
| Cycles crop growth (T3.1, T3.2) | the first implementation of the generic scripted policy | kept as `experiments/rebuttal_confirmatory_eval/paper_references/random_rollout_paper_sim.py`, run by `paper_sim_anchor_rollout.py` |
| DSSAT (T3.1, T3.2) | a code reconstruction of the fixed schedule (see the disclosure below) | `paper_sim_anchor_rollout.py` |
| Cycles crop planning (T1.3) | the current module | `ag_heuristic` mode |

The current `ag_heuristic` mode (used by the [quickstart](../getting-started/quickstart.md))
is also run on every split and stored as a separate column of
`tables/e4_anchors.csv`. It differs from the paper's reference on WOFOST (by
sign on several weather-regime splits) and on Cycles crop growth (see below).

> **Release note.** The WOFOST and Cycles crop-growth references are recovered
> from git history at run time. A release with a shortened git history needs
> these two policies checked in under their own names.

### Three disclosures

1. **Weather-regime (T1.1) cannot be normalized.** On the T1.1 test splits the
   scripted reference sits at or below no-action: the gap
   `R_ref − R_no-action` ranges from −0.148 to +0.017 profit-equivalent t/ha
   and is negative on 8 of 10 crop-split targets. T1.1 is reported as the gain
   over no-action, `R(π) − R_no-action`.
2. **The submitted DSSAT reference values were set by hand.** The first DSSAT
   implementation matched schedule days exactly and, at the agent's 10-day
   decision interval, applied nothing, so it scored as no-action. The submitted
   figures replaced its values with a manual correction for which no run or
   code survives. Every DSSAT number in the final version uses a code
   reconstruction of the schedule: it reproduces the submitted input totals
   (116 kg/ha N, 264 mm irrigation) and returns 7.17 profit-equivalent t/ha on
   the submitted validation scenarios, against the hand-set 7.47; on the test
   suite it returns 7.36.
3. **The current Cycles heuristic degenerates to no-action on crop growth.** On
   the Cycles maize task the current `ag_heuristic` module has no nitrogen
   rule, returns the zero action and scores exactly as no-action. The first
   implementation (`paper_references/random_rollout_paper_sim.py`) is used instead; it reproduces the
   submitted reference exactly on the submitted validation scenarios.

A fourth caution concerns reading normalized WOFOST scores. The WOFOST
reference is a conservative schedule; the untrained think model already scores
above it on every Tier 2 target. A normalized score of 5 means five times the
gap between no-action and that schedule. It does not mean closeness to an
optimum.

### WOFOST rule set

Each crop resolves to a record of seasonal caps (N, P, K in kg/ha, irrigation
in mm, and a soil-moisture trigger); crops absent from the table use the
default row. The seven pulse legumes (chickpea, cowpea, faba bean, groundnut,
mung bean, pigeon pea, soybean) get no nitrogen.

| Crop | N | P | K | Irrigation | Trigger |
|---|---:|---:|---:|---:|---:|
| default | 60 | 20 | 20 | 80 | 0.31 |
| barley | 90 | 20 | 20 | 80 | 0.31 |
| cotton | 80 | 20 | 40 | 120 | 0.31 |
| maize | 120 | 20 | 20 | 100 | 0.31 |
| millet | 60 | 20 | 20 | 60 | 0.31 |
| potato | 120 | 40 | 80 | 180 | 0.37 |
| rapeseed | 120 | 20 | 40 | 80 | 0.31 |
| rice | 100 | 20 | 20 | 160 | 0.31 |
| sorghum | 70 | 20 | 20 | 70 | 0.31 |
| sugarbeet | 100 | 20 | 60 | 120 | 0.31 |
| sunflower | 60 | 20 | 20 | 80 | 0.31 |
| sweetpotato | 40 | 20 | 40 | 80 | 0.31 |
| wheat | 100 | 20 | 20 | 80 | 0.31 |
| pulse legume | 0 | 20 | 20 | 60 | 0.29 |
| chickpea | 0 | 20 | 0 | 50 | 0.28 |

The objective scales the record: `yield_max` multiplies all caps by 1.15 and
raises the trigger by 0.02; `nutrient_stewardship` multiplies the N/P/K caps by
0.70; `water_stewardship` multiplies the irrigation cap by 0.60 and lowers the
trigger by 0.03; `profit_max` uses the record unchanged.

At each turn the policy takes the first rule that applies: irrigate under acute
water stress; otherwise apply P, then K, early in the season when the soil pool
is below its threshold; otherwise apply nitrogen up to a stage target (40 % of
the cap for development stage 0.15 to 0.45, 80 % for 0.45 to 0.90, the full cap
late in the season under `yield_max` only; pulse legumes get only a starter
allowance, which is zero in this version); otherwise irrigate if rain is low
and soil moisture is below the trigger; otherwise do nothing. The chosen action
is projected onto the current menu: the largest available level that does not
exceed the remaining cap, or no-op if the action family is absent. The full
rule, thresholds and agronomic sources are in the paper appendix.

### Cycles

- **Crop growth (T3.1, T3.2).** Nitrogen split by season progress (fraction of
  decision turns elapsed): 40 kg/ha between 15 % and 35 % of the season while
  cumulative N is below 40 kg/ha, 60 kg/ha between 35 % and 65 % while below
  110 kg/ha, and, under the yield objective only, 30 kg/ha between 65 % and 80 %
  while below 160 kg/ha, each rounded to the nearest available level.
- **Crop planning (T1.3).** The current module (`_fallback_crop_planning_action`
  in `agrimanager/env/cycles_gym/env.py`) picks the higher-priced crop and
  switches crops when that choice would repeat the previous year's crop; with
  two crops this alternates corn and soybean. `TODO(check)`: the paper appendix
  describes this reference as a fixed corn-even, soybean-odd rotation at
  planting-week index 4; the two descriptions need to be reconciled.

### DSSAT

A fixed schedule applied on exact days after planting (DAP) inside the daily
DSSAT loop, so scheduled events are not skipped by the decision interval:
nitrogen at DAP 40 (27 kg/ha), 45 (35 kg/ha) and 80 (54 kg/ha); irrigation on 16
days (DAP 6, 20, 37, 50, 54, 65, 69, 72, 75, 77, 80, 84, 91, 101, 104, 105),
264 mm in total.

## Normalized return

```text
R̃(π) = (R(π) − R_no-action) / |R_ref − R_no-action|
```

Both references are scored on the same scenarios as the policy. No-action sits
at 0 and the scripted reference at ±1. The gap is used as a denominator only
where the reference clears no-action by at least a tenth of the largest gain
over no-action that a think arm reaches on that target; on every such target
the reference lies above no-action. Because the gap differs across targets,
normalized scores compare policies within a target only. Raw returns, both
reference returns and the gap are published for every headline target
(`tables/e4_anchors.csv`, `tables/e4_normalized.csv`).

## Neural baselines

All neural baselines use Stable-Baselines3 on the simulator's numeric
observation (`nn_adapter.py`), with the stock `MlpPolicy` (two hidden layers of
64 units, tanh, separate actor and critic), 32 parallel environments and
72-step rollouts. Each trains three seeds (`agent.seed` 1, 2, 3) unless noted.

| Baseline | Settings | What it is | Scripts |
|---|---|---|---|
| PPO `n1` | T1.1 (both crops), T1.2 (4/8/16ID), T1.3 | one gradient pass per batch, the same interaction budget and update schedule as the language-model runs (12,800 episodes); the main-text comparison | `experiments/t1_*/run_*nn_train_n1*.sh`, `run_nn_train_price_regime_n1_s*.sh` |
| PPO `n8` | T1.2 (three seeds), T1.1 (one seed per crop, exploratory) | eight gradient passes per batch; appendix only | `run_*nn_train_n8*.sh`, `run_t12_*_nn_n8_s*_train.sh` |
| DQN, A2C | T1.1 potato | same observation, action menu and episode budget; A2C keeps its own default learning rate (7e-4, RMSprop) | `run_wofost_weather_regime_ood_potato_{dqn,a2c}_train_n1*.sh` |
| MaskablePPO | T2.2 | acts over a shared 17-action vocabulary (no-op plus four levels each of N, P, K, water) with a legal-action mask from each native menu; trained per family on that family's training menus | `run_t22_nn_mask_family_{a,b}_s*_train.sh` |
| target-trained PPO | T2.2 | trains on each test menu alone (800 scenarios, sixteen passes), a reachable learned reference for that menu | `run_t22_nn_target_*_s*_train.sh` |

The fixed-interface PPO applies to the Tier 1 settings, whose observation
layout and action set match training. There are no neural results on T2.1,
T2.3 or the Tier 3 settings. Neural test-suite evaluation uses the `_nn`
variants of the suites (same scenarios, `llm_mode: false`) and the scripts
`experiments/*/run_eval_*nn*_e4.sh`.

## Diagnostic scripts

These exist in the code and are not part of the paper's test-suite results.

- `random`: a uniformly random legal action each turn, seeded by `runtime.seed`.
  The submitted figures showed it; it is not scored on the test suite.
- `ppv0`: switches WOFOST to `pp-v0` (potential production, no N/P/K/water
  limits). A yield-potential diagnostic; it pays no input costs, so it is not
  a profit baseline.

The runner for all scripted policies is `entrypoints/eval/baseline_eval.sh`
with a config from `entrypoints/eval/config/`; see `entrypoints/eval/README.md`.
