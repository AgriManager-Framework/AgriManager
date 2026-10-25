# The held-out test suite

Every main-text number of the paper comes from a frozen, prespecified held-out
test suite. This page documents how it was built, what each part contains,
which seeds generated it, how its disjointness from training and validation
was audited, and where its manifests live.

Code: `experiments/rebuttal_confirmatory_eval/` (configs under `config/`,
build scripts `run_build_*.sh` and `run_freeze_*.sh`, audits `audit_*.py`).
File names in that directory carry the internal prefix `e4_`.

## Principles

- **Frozen before scoring.** A suite is a scenario set plus an environment
  config. Building it needs no checkpoint, so every suite was frozen before any
  policy was scored on it. Test scenarios are never used for training,
  validation, checkpoint selection, prompt tuning or decoding changes.
- **Policy-neutral.** Suite configs are written with `require_think: false`.
  The evaluator applies each checkpoint's own response format at scoring time,
  so think and no-think policies are scored on the same rows.
- **Paired where the setting allows.** Variants of one setting render the same
  physical scenarios (T1.3 price regimes, T2.1 schemas, T2.2 menus, T2.3
  objectives, the WOFOST tuples of T3.2, the Cycles cells of T3.1 and T3.2).
- **Audited on physical keys.** Disjointness is checked on what determines the
  simulated episode, never on the dataset's `scenario_id` (which also encodes
  the split name).
- **Terminal checkpoints only.** Each run is scored at its terminal checkpoint;
  nothing is selected on validation.

## Timeline

| When | What |
|---|---|
| May 2026 | Paper submitted; results then came from validation splits |
| July 2026, during the author-response period | Test suites for all eight settings built, audited and frozen; seed `20260725` |
| July to September 2026 | Three training seeds per arm trained and scored on the frozen suites; scripted references and the untrained model scored on every split |
| September 2026 | The three Tier 3 Cycles crop-growth suites found to repeat the validation years; rebuilt from unused weather years (seed `20260926`) |
| October 2026 (in progress) | Remaining Cycles cells scored on the rebuilt suites; per-run records regenerated for release |

`TODO(F8)`: exact construction date of each suite (from git and the suite
manifests).

## Contents

6,278 scenarios over the eight settings, in groups of 128 (130 for the three
rebuilt Cycles suites): 2,688 in Tier 1, 2,048 in Tier 2 and 1,542 in Tier 3.

| Setting | Test groups | Scenarios | Config(s) | Source and seed |
|---|---|---:|---|---|
| T1.1 weather-regime | per crop (chickpea, potato): id, drought, wet, hot, cold, 128 each | 1,280 | `e4_t11_{chickpea,potato}_weather_regime.yaml` | unused scenarios of `agrimanager/wofost-weather-regime-pool`, regrouped by a predeclared rule (below); seed `20260725` |
| T1.2 crop-variety | 4 seen crops + 4 held-out crops, 128 each; shared by 4ID, 8ID, 16ID | 1,024 | `e4_t12_cross_crop.yaml` | public `test` split of `agrimanager/wofost-weather-pool`; seed `20260725` |
| T1.3 price-regime | id_test, high_corn, high_soybean, 128 each, paired | 384 | `e4_t13_price_regime.yaml` | Cycles crop planning: two sites × the four validation year windows (1986–2004 to 1989–2007) × 16 seeds from `300000` (validation used `200000`); seed `20260725` |
| T2.1 observation-schema | S1, S2a to S2d, S3, S4, S5, 128 each, paired | 1,024 | `e4_t21_observation_schema.yaml` | shared maize base |
| T2.2 action-menu | `lnpkw`, `lnw`, `ln`, `lw`, 128 each, paired; shared by Families A and B | 512 | `e4_t22_action_menu.yaml` | shared maize base |
| T2.3 reward-formulation | yield, profit, water, nutrient, 128 each, paired | 512 | `e4_t23_reward_formulation.yaml` | shared maize base |
| T3.1 cross-simulator | WOFOST 128, DSSAT 128, Cycles 130 | 386 | `e4_t31_cross_sim_{wofost,dssat}.yaml`, Cycles: `e4_t31_cross_sim_cycles_v2.yaml` | shared maize base; DSSAT seeds `200042` to `200169`; Cycles years 2004 to 2016 |
| T3.2 unified policy | 5 seen-control + 4 held-out tuples: seven of 128, Cycles profit and yield of 130 | 1,156 | `e4_t32_*.yaml`, Cycles: `e4_t32_cycles_{profit,yield}_v2.yaml` | as T3.1 for each simulator |

The numeric baselines use `_nn` variants of the T1.1, T1.2, T1.3 and T2.2
suites: the same scenarios with `llm_mode: false`.

`TODO(numbers)`: the totals assume every Cycles cell uses the rebuilt suites,
which is the state of the final paper. The first build of those three suites
had 128 scenarios each (6,272 in total).

## How each part was built

Build order, from the most shared source to the most specialized:

1. **Shared maize base** (`run_freeze_maize_eval_pool.sh`,
   `prepare_maize_eval_pool.py`). Downloads the public maize pool at a pinned
   revision, reconstructs the 128 validation scenarios of the submitted
   protocol (`generation_seed: 42`), removes them from the 640-row validation
   pool, samples 128 new scenarios with seed `20260725`, and verifies zero
   overlap with the full training split and the submitted validation
   scenarios. Writes `data/source_pools/maize_e4/{test/maize.parquet,manifest.json,audit.json}`.
2. **T2.1, T2.2, T2.3 and the WOFOST tuples of T3.1 and T3.2** render that base
   under each schema, menu, objective or tuple (`run_build_t21_eval_suite.sh`,
   `run_build_t22_eval_suite.sh`, `run_build_t23_eval_suite.sh`,
   `run_build_t31_eval_suite.sh`, `run_build_t32_wofost_eval_suite.sh`,
   `run_build_t32_wofost_seen_eval_suite.sh`). Each audit checks that every
   split holds exactly the same physical keys and `paired_scenario_id` values
   as the base. The T3.2 WOFOST tuples keep the submitted fixed reference yield
   `y_ref = 8500`, and the audit records it.
3. **DSSAT** (`run_build_t32_dssat_eval_suite.sh`,
   `run_build_t32_dssat_yield_eval_suite.sh`). The submitted training and
   validation seed blocks are `42–441` and `100042–100169`; the test block is
   `200042–200169`. The audit keys on `(env_id, seed)`.
4. **Cycles crop growth**: see [the rebuild](#cycles-rebuild).
5. **T1.1 weather regimes** (`run_freeze_t11_{potato,chickpea}_eval_pool.sh`,
   `prepare_t11_eval_pool.py`; reads many weather caches, run on a CPU node).
   The regime labels of the public pool cannot be reconstructed exactly from
   the rule documented with it, so the test suite uses an explicit rule, fixed
   before any label was examined:
   1. remove every scenario used by the submitted T1.1 training and validation
      splits (for potato, 2,240 of the 3,840-scenario pool);
   2. for each remaining scenario, compute mean daily temperature and
      cumulative rainfall over the crop's growing window (for potato, January 1
      to July 1, sowing to maturity in the benchmark's crop calendar);
   3. assign, in sequence, the coldest 20 % to `cold`, the hottest 20 % of the
      rest to `hot`, the driest 20 % of the rest to `drought`, the wettest
      20 % of the rest to `wet`; the remaining 20 % form `id`;
   4. sample 128 scenarios per group with seed `20260725`.

   The audit requires five mutually disjoint groups and zero overlap with the
   submitted T1.1 scenarios. Agreement with the pool's original labels
   (567/640 for potato) is a descriptive diagnostic only; the test groups do not
   re-measure the submitted T1.1 numbers.
6. **T1.2 crop pool** (`prepare_t12_eval_pool.py`,
   `run_build_t12_eval_suite.sh`) from the public `test` split of the 20-crop
   pool, then the seen and held-out crop groups.
7. **T1.3 price regimes** (`run_build_t13_eval_suite.sh`). The regime
   definitions and year windows are copied from the validation config; only
   the seed block moves (`seed_start: 300000`; validation used
   `200000–200127`). The audit checks that the seed block touches neither
   the training nor the validation block, that the three regimes of a scenario
   differ only in the price vector, and that each shock was applied. Training
   windows are 1980–1998 to 1983–2001; the year ranges overlap with the test
   windows by design, since the shift in this setting is the price regime.

Path normalization: built suites store absolute cache paths;
`normalize_eval_suite_paths.py` rewrites them so a downloaded suite runs on a
new machine.

### The T1.1 `id` group is a residual

The four tail groups are assigned first, so `id` holds scenarios that are
neither the coldest, hottest, driest nor wettest. It is more moderate than a
random in-distribution draw, which favours the in-distribution end of any
ID-to-OOD comparison. The manifest records
`id_definition: remaining_scenarios_after_four_tail_assignments`.

## Cycles rebuild

A Cycles crop-growth scenario (`CornShortRockSpringsFW-v1`) runs historical
weather at a single site, so it is fixed by its weather year; the seed changes
nothing in the dynamics. The first build of the three Tier 3 Cycles suites kept
the validation years 1996 to 2003 and changed only the seed block, so it held
the validation set's eight physical years sixteen times each. Its audit keyed
on the seed and could not see this.

The rebuilt suites use every Rock Springs weather year that no Cycles
experiment touched: **2004 to 2016**, 13 years (training uses 1980 to 1995,
validation 1996 to 2003). Ten seeds per year give 130 rows; the copies carry no
physical difference and only average the policy's own sampling. The three
suites (T3.1 target, T3.2 profit, T3.2 yield) share the block, so they stay
paired scenario by scenario.

| Item | Value |
|---|---|
| Configs | `config/e4_t31_cross_sim_cycles_v2.yaml`, `config/e4_t32_cycles_profit_v2.yaml`, `config/e4_t32_cycles_yield_v2.yaml` |
| Build | `run_build_cycles_v2_eval_suites.sh` |
| Audit | `audit_cycles_v2_eval_suites.py`: 130 rows, 13 physical scenarios, years 2004 to 2016, no physical overlap with any Cycles training or validation split of T3.1 or T3.2, identical block across the three suites; key `(env_id, start_year, end_year)` |
| Seed | `generation_seed: 20260926` |

> **Release note.** These four files exist in the cluster working copy and are
> not yet committed to this branch. They must be added before release.

Consequence for uncertainty: the effective sample on these three targets is 13
physical scenarios, so scenario-level standard errors there overstate
precision. Seed-level SDs are unaffected. T1.3 (384 distinct crop-planning
scenarios) and DSSAT are not affected.

## Manifests, checksums and audits

Each built suite directory holds the rendered split files, a `manifest.json`
(config, generation seed, source revision, row counts, labels, group
definitions) and an `audit.json` (the overlap checks and their results), plus
file checksums. `upload_eval_suites.py` publishes the suites as one Hugging
Face dataset repository.

| Item | Location |
|---|---|
| Suites, manifests, audits | Hugging Face dataset `agrimanager/<test-suite-dataset>` (`TODO(dataset-name)`; built from `experiments/rebuttal_confirmatory_eval/data/` by `upload_eval_suites.py`, which includes every suite below, the rebuilt Cycles suites and the two numeric-baseline suites `e4_t12_cross_crop_nn`, `e4_t13_price_regime_nn`) |
| Dataset card draft | [docs/release/test_suite_dataset_card.md](../release/test_suite_dataset_card.md) |

## Rebuilding a suite

```bash
# shared maize base, then the WOFOST Tier 2 and Tier 3 suites
bash experiments/rebuttal_confirmatory_eval/run_freeze_maize_eval_pool.sh
bash experiments/rebuttal_confirmatory_eval/run_build_t21_eval_suite.sh
python experiments/rebuttal_confirmatory_eval/audit_t21_eval_suite.py
```

Each `run_build_*.sh` builds one suite and is followed by its `audit_*.py`.
Rebuilding with the same seeds and source revisions reproduces the same rows;
compare against the published manifest checksums.

*Not executed for this documentation revision: the build and audit commands
were run on the cluster when the suites were frozen and were not re-run here.*
