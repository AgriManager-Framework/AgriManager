# Held-out test suite: build, audit, anchors

This directory builds the frozen held-out test suite of the AgriManager
benchmark, audits each suite against the training and validation scenarios of
its experiment, uploads the suites, and scores the scripted references on them.
File names use the internal prefix `e4_`. The full description (contents,
timeline, seeds, keys, limitations) is in
[docs/benchmark/test_suite.md](../../docs/benchmark/test_suite.md).

Test scenarios are never used for training, validation, checkpoint selection,
prompt tuning or decoding changes. Building a suite needs no checkpoint: a
suite is a scenario set plus an environment config, and a policy enters only
at scoring time.

## Scripts

| Suite | Build | Audit | Config |
|---|---|---|---|
| Shared maize base (T2.x, WOFOST parts of T3.x) | `run_freeze_maize_eval_pool.sh` (`prepare_maize_eval_pool.py`) | in the build | — |
| T1.1 potato, chickpea | `run_freeze_t11_{potato,chickpea}_eval_pool.sh`, then `run_build_t11_{potato,chickpea}_eval_suite.sh` | `audit_t11_eval_suite.py` | `config/e4_t11_*_weather_regime{,_nn}.yaml` |
| T1.2 | `run_build_t12_eval_suite.sh`, `run_build_t12_nn_eval_suite.sh` (`prepare_t12_eval_pool.py`) | `audit_t12_eval_suite.py` | `config/e4_t12_cross_crop{,_nn}.yaml` |
| T1.3 | `run_build_t13_eval_suite.sh`, `run_build_t13_nn_eval_suite.sh` | `audit_t13_eval_suite.py` | `config/e4_t13_price_regime{,_nn}.yaml` |
| T2.1 | `run_build_t21_eval_suite.sh` | `audit_t21_eval_suite.py` | `config/e4_t21_observation_schema.yaml` |
| T2.2 | `run_build_t22_eval_suite.sh`, `run_build_t22_nn_eval_suite.sh` | `audit_t22_eval_suite.py` | `config/e4_t22_action_menu{,_nn}.yaml` |
| T2.3 | `run_build_t23_eval_suite.sh` | `audit_t23_eval_suite.py` | `config/e4_t23_reward_formulation.yaml` |
| T3.1 WOFOST, DSSAT targets | `run_build_t31_eval_suite.sh` | `audit_t31_eval_suite.py` | `config/e4_t31_cross_sim_{wofost,dssat}.yaml` |
| T3.2 WOFOST, DSSAT tuples | `run_build_t32_{wofost,wofost_seen,dssat,dssat_yield}_eval_suite.sh` | `audit_t32_*_eval_suite.py` | `config/e4_t32_*.yaml` |
| Cycles crop growth, rebuilt (T3.1 target, T3.2 profit and yield) | `run_build_cycles_v2_eval_suites.sh` | `audit_cycles_v2_eval_suites.py` | `config/e4_*cycles*_v2.yaml` |

Helpers: `normalize_eval_suite_paths.py` (rewrite absolute cache paths),
`repair_stringified_env_fields.py`, `upload_eval_suites.py` (publish to the
Hugging Face dataset repository), `run_e4_anchors.sh` with
`paper_sim_anchor_rollout.py` (no-action and the scripted references on every
split).

> **Release note.** The rebuilt Cycles scripts and configs exist in the
> cluster working copy and are not yet committed here. The first-build Cycles
> configs (`e4_t31_cross_sim_cycles.yaml`, `e4_t32_cycles_{profit,yield}.yaml`)
> kept the validation years and are superseded.

Built suites are written under `data/` (ignored by git) with a
`manifest.json`, an `audit.json` and checksums per suite.

## Shared maize base

```bash
bash experiments/rebuttal_confirmatory_eval/run_freeze_maize_eval_pool.sh
```

Downloads the public maize pool's train and validation files at a pinned
revision, reconstructs the 128 validation scenarios of the submitted protocol
(`generation_seed: 42`), removes them from the 640-row validation pool, samples
128 new scenarios with seed `20260725`, and verifies zero overlap with the full
training split and the submitted validation scenarios. Physical scenarios are
compared on `(crop, year, latitude, longitude)`; `scenario_id` is not used
because it also encodes the split name.

## Policy-neutral suites

Suite configs set `require_think: false`. The evaluator applies each
checkpoint's own response format, so think and no-think policies are scored on
the same rows.

## T3.2 reference yield

The two held-out WOFOST tuples keep the submitted T3.2 fallback `y_ref: 8500.0`;
the audit records the value explicitly.

## T1.1 weather regimes

The T1.1 source reconstruction reads many weather caches and must run on a CPU
compute node:

```bash
bash experiments/rebuttal_confirmatory_eval/run_freeze_t11_potato_eval_pool.sh
```

The submitted regime labels cannot be reconstructed exactly from the rule
documented with the public dataset, so the test groups use an explicit rule:

1. Remove all 2,240 potato scenarios used by the submitted T1.1 train and
   validation splits from the 3,840-scenario public pool.
2. For each of the 1,600 unused scenarios, compute mean daily temperature and
   cumulative rainfall from January 1 through July 1, inclusive.
3. Sequentially assign the lowest-temperature 20 % to `cold`, the
   highest-temperature 20 % of the remainder to `hot`, the lowest-rainfall 20 %
   of the remainder to `drought`, and the highest-rainfall 20 % of the
   remainder to `wet`. The remaining 20 % form `id`.
4. Deterministically sample 128 scenarios from each 320-scenario group with
   seed `20260725`.

The audit requires five mutually disjoint 128-scenario splits and zero overlap
with all submitted T1.1 training and validation scenarios.

### Why January 1 to July 1

The window is the potato growing season itself, read from the benchmark's own
crop calendar in `WOFOSTGym/env_config/agro/potato_agro.yaml`:

```yaml
crop_start_date: 2010-01-01   # crop_start_type: sowing
crop_end_date:   2010-07-01   # crop_end_type:   maturity
```

Sowing to maturity is the only interval over which weather can affect this
crop, so the window follows from the environment definition and is fixed before
any label is examined.

It is specifically **not** chosen to maximise agreement with the submitted
labels. Agreement is computed afterwards, as a descriptive diagnostic only. An
exploratory search over rainfall-accumulation windows was run while this
protocol was being drafted; it is preserved, clearly marked as rejected, under
`data/rejected_explorations/t11_rain_window_search.json`, and it did not
determine the protocol. On the rainfall-derived groups the adopted
sowing-to-maturity window agrees with the submitted labels on 311/384 scenarios
(81.0%), while the best window that search found reaches 313/384 (81.5%) — the
adopted window scores **lower**, which it could not do had the search driven
the choice.

Overall agreement with the submitted labels is 567/640 (88.6%). This number is
reported as an observation about how close the two rules happen to be. It is
not evidence that the new groups measure the same quantity as the submitted
ones, and test-suite T1.1 results are not a re-measurement of the submitted
T1.1 numbers.

### The `id` group is a residual, not a random sample

Because the four tail groups are assigned first, `id` consists of the scenarios
that are neither the coldest, hottest, driest, nor wettest. It is therefore
systematically more moderate than a random in-distribution draw, and the
in-distribution end of any ID-to-OOD comparison is correspondingly favourable.
The manifest records this as
`id_definition: remaining_scenarios_after_four_tail_assignments`, and any
reported retention or ID-to-OOD gap must state it.
