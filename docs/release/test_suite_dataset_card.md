# Test-suite dataset card (draft)

Draft README for the Hugging Face dataset repository that holds the frozen
held-out test suite. Fields in `{braces}` and `TODO` markers must be filled
before release. Numbers come from the suite manifests and the paper; do not
add numbers that are not in those sources.

````markdown
---
license: cc-by-4.0
pretty_name: AgriManager held-out test suite
task_categories:
  - reinforcement-learning
tags:
  - agriculture
  - crop-management
  - benchmark
  - simulation
  - schema-shift
size_categories:
  - 1K<n<10K
configs:
  {one entry per suite directory, TODO}
---

# AgriManager held-out test suite

Frozen test scenarios for the eight settings of the AgriManager benchmark
(NeurIPS 2026, Evaluations and Datasets track). Every main-text score of the
paper is computed on these scenarios.

- Paper: TODO(release-url)
- Code, build scripts and audits: https://github.com/AgriManager-Framework/AgriManager (`experiments/rebuttal_confirmatory_eval/`)
- Documentation: https://github.com/AgriManager-Framework/AgriManager/blob/main/docs/benchmark/test_suite.md

## What a row is

One row is one episode scenario: a complete environment configuration for one
simulator (crop, site, weather year or generator seed, action menu, objective,
reward parameters, prompt options), plus labels (setting, test group,
`paired_scenario_id`). A row holds no policy output. Run it with the
AgriManager environment loop to obtain an episode return.

The rows are simulator inputs. WOFOST rows reference weather from the public
NASA POWER-derived pools (`AgriManager-Framework/weather_pool_maize`,
`AgriManager-Framework/wofost-weather-pool`, `AgriManager-Framework/wofost-weather-regime-pool`);
Cycles and DSSAT rows are generator configurations for those simulators.

## Contents

6,278 scenarios in groups of 128 (130 for the three Cycles crop-growth suites).

| Suite directory | Setting | Groups | Rows |
|---|---|---|---:|
| `e4_t11_chickpea_weather_regime` | T1.1 weather-regime, chickpea | id, drought, wet, hot, cold | 640 |
| `e4_t11_potato_weather_regime` | T1.1 weather-regime, potato | id, drought, wet, hot, cold | 640 |
| `e4_t12_cross_crop` | T1.2 crop-variety | 4 seen + 4 held-out crops | 1,024 |
| `e4_t13_price_regime` | T1.3 price-regime | id_test, high_corn, high_soybean | 384 |
| `e4_t21_observation_schema` | T2.1 observation-schema | S1, S2a–S2d, S3, S4, S5 | 1,024 |
| `e4_t22_action_menu` | T2.2 action-menu | lnpkw, lnw, ln, lw | 512 |
| `e4_t23_reward_formulation` | T2.3 reward-formulation | yield, profit, water, nutrient | 512 |
| `e4_t31_cross_sim_wofost`, `e4_t31_cross_sim_dssat`, `e4_t31_cross_sim_cycles_v2` | T3.1 cross-simulator targets | WOFOST, DSSAT, Cycles | 128 + 128 + 130 |
| `e4_t32_wofost_seen`, `e4_t32_wofost_heldout`, `e4_t32_dssat_{yield,profit}`, `e4_t32_cycles_{yield,profit}_v2` | T3.2 unified policy | 5 seen-control + 4 held-out tuples | 1,156 |
| `*_nn` variants of T1.1, T1.2, T1.3, T2.2 | numeric baselines | same scenarios, `llm_mode: false` | (duplicates) |

Each directory contains the split files, `manifest.json` (generation config,
seed, source revision, group definitions, row counts), `audit.json` (overlap
checks and results) and checksums.

## How it was built

- Built in July 2026, after the paper's submission, during the author-response
  period, and frozen before any policy was scored on it. The three Cycles
  crop-growth suites were rebuilt in September 2026 (see below).
  `TODO(F8)`: per-suite construction dates.
- Generation seed `20260725` for all suites except the rebuilt Cycles suites
  (`20260926`). DSSAT test seeds `200042–200169`; Cycles crop-planning seeds
  from `300000`; Cycles crop-growth weather years 2004–2016.
- Disjointness from each experiment's training and validation scenarios was
  audited on physical scenario keys: (crop, year, latitude, longitude) for
  WOFOST; the generator seed for DSSAT; the weather year for Cycles crop
  growth; site, year window and seed for Cycles crop planning.
- T1.1 groups follow an explicit rule fixed before labels were examined
  (sequential 20 % tails by growing-season temperature and rainfall, over
  scenarios unused by the submitted experiment). The `id` group is the residual
  after the four tails and is more moderate than a random draw.

## Intended use

Scoring crop-management policies for the AgriManager benchmark: one pass, at
the policy's final checkpoint, with no tuning on these scenarios. Report raw
returns with the no-action and scripted-reference returns on the same rows.

Out of scope: training, model selection, prompt tuning, or any use as
agronomic advice.

## Known limitations

- **Cycles crop growth.** A scenario is fixed by its weather year at one site.
  The three Cycles crop-growth suites hold 13 physical scenarios each (10 seeds
  per year), so scenario-level standard errors on those targets overstate
  precision. Their first build reused the eight validation years; the rebuilt
  suites in this repository use 2004–2016, years no Cycles experiment used.
- **T1.1 labels.** The test groups use a new explicit rule; they do not
  re-measure the submitted T1.1 groups.
- **T1.3.** The test windows equal the validation year windows; only the seed
  block (and with it the sampled prices) differs.
- **Simulation only.** All returns are simulator outputs.

## License and attribution

CC BY 4.0 for the suite files. The WOFOST weather comes from NASA POWER
through WOFOST-Gym/PCSE; Cycles and DSSAT are third-party simulators with their
own licenses (Cycles native binary: CC BY-NC-ND 4.0). See the code
repository's `docs/reference/upstream_citations.md`.

## Citation

```bibtex
{the paper's BibTeX, TODO(release-url)}
```
````

## Before publishing

- [ ] Upload the five suites that are not yet in the repository
      (`e4_t12_cross_crop_nn`, `e4_t13_price_regime_nn`, the three `_v2`
      Cycles suites).
- [ ] Decide whether the first-build Cycles suites stay in the repository
      (marked superseded) or are removed.
- [ ] Generate a Croissant file with Responsible AI fields, as for the three
      public pools.
- [ ] Rename the repository to its public name and make it public (needs
      approval).
- [ ] Run `tools/release_identity_scan.py` over the card and manifests.
