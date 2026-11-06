# AgriManager Benchmark

The AgriManager benchmark measures how well crop-management policies generalize
under **schema shift**: changes to the structure of the task — which variables
are observed, which actions are available, and which objective is rewarded — and,
when crossing simulators, changes to the underlying dynamics. This is distinct
from ordinary distributional shift (new weather, new years) under a fixed task
interface.

Experiments are organized in three tiers:

- **Tier 1 — within-schema distributional shift.** The task schema is fixed;
  only the data distribution (weather, crop, price) moves out of distribution.
- **Tier 2 — single-axis schema shift.** One schema axis (observation, action,
  or reward) shifts at a time while the rest stays fixed.
- **Tier 3 — compositional schema shift.** Multiple axes and/or the simulator
  change together, up to a single prompt-conditioned policy evaluated on unseen
  schema combinations.

This page is the benchmark reference. Reward objective definitions live in
[reward_objectives.md](reward_objectives.md); baseline policy definitions live in
[baselines.md](baselines.md); the frozen held-out test suite on which every
paper number is computed is documented in [test_suite.md](test_suite.md). For how plans become configs, prompts, and run
scripts, see the convention docs:
[dataset_config_conventions.md](../conventions/dataset_config_conventions.md),
[prompt_design_conventions.md](../conventions/prompt_design_conventions.md), and
[training_script_conventions.md](../conventions/training_script_conventions.md).

## Experiment index

| Card | Tier | Shift axis | Simulator | Runnable directory |
| --- | --- | --- | --- | --- |
| [T1.1](#t11-weather-regime-shift) | 1 | Weather regime | WOFOST-Gym | `experiments/t1_1_weather_regime_shift/` |
| [T1.2](#t12-cross-crop-trait-shift) | 1 | Crop / trait | WOFOST-Gym | `experiments/t1_2_cross_crop_trait_shift/` |
| [T1.3](#t13-price-regime-shift) | 1 | Price regime | CycleGym | `experiments/t1_3_cycles_gym_price_regime/` |
| [T2.1](#t21-observation-schema-shift) | 2 | Observation schema | WOFOST-Gym | `experiments/t2_1_observation_schema_shift/` |
| [T2.2](#t22-action-menu-shift) | 2 | Action menu | WOFOST-Gym | `experiments/t2_2_action_menu_shift/` |
| [T2.3](#t23-reward-formulation-shift) | 2 | Reward objective | WOFOST-Gym | `experiments/t2_3_reward_formulation_shift/` |
| [T2.4](#t24-schema-corruption-suite) | analysis, not in the paper | Prompt-vs-environment corruption | WOFOST-Gym | derived from T2.1/T2.2/T2.3 |
| [T3.1](#t31-cross-simulator-maize-transfer) | 3 | Simulator (compositional) | WOFOST / DSSAT / CycleGym | `experiments/t3_1_cross_simulator_maize_transfer/` |
| [T3.2](#t32-unified-prompt-conditioned-policy) | 3 | Unified schema closure | WOFOST / DSSAT / CycleGym | `experiments/t3_2_unified_prompt_conditioned_policy/` |
| [APP1](#app1-wofost-same-crop-variety-ood) | exploratory, not in the paper | Variety / ecotype | WOFOST-Gym | `experiments/app1_wofost_variety_ood/` |

The paper reports the eight settings T1.1 to T3.2. T2.4 and APP1 are kept in
the code as analyses; they have no test suite and no paper results. Each
experiment's `experiments/<exp>/` directory holds the runnable configs and fixed
`run_*.sh` scripts; the scripts and seeds behind the paper's runs are listed in
[Training](../guides/training.md#paper-runs-and-seeds).

## Shared defaults

A card repeats a setting only when it overrides one of these defaults.

| Policy family | Model / policy | Algorithm | Train size | Validation size | Runtime epochs | Optimizer epochs | Samples / rollouts | Eval freq | Save freq |
| --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| LLM | `Qwen/Qwen3-4B-Instruct-2507` | GRPO | `1600` | `128` per group | `2` (200 steps) | `1` actor PPO epoch | `4` rollouts per prompt | every `10` steps | every `100` steps |
| NN | Stable-Baselines3 `MlpPolicy` (`ActorCriticPolicy`) | PPO | `1600` | `128` per group | `8` | `1` PPO epoch per buffer | n/a | `4` per epoch | every epoch |

Source configs: `entrypoints/train/config/agri_grpo.yaml` (LLM),
`entrypoints/train/config/nn.yaml` (NN); the paper's replicate scripts set the
values above explicitly (T2.3 trains 400 steps and T3.2 800). Evaluation
defaults: `entrypoints/eval/config/default.yaml` (LLM, greedy decoding,
temperature `0`, `1` pass) and `entrypoints/eval/config/nn.yaml` (NN).

NN training: `agent.n_epochs=1`; set `runtime.train_epochs` from train-set size
(`16` for `3200`-scenario splits, `8` for `1600`-scenario splits);
`agent.checkpoint_mode=epoch`.

**WOFOST-Gym defaults** (unless a card states otherwise): environment
`WOFOSTGym`; action mode `lnpkw-v0`; reward `objective_id: profit_max`; decision
interval `10` days; horizon `24` turns; `scale_action_amounts_by_interval=true`;
`include_crop_traits=false`; policy modes LLM think, LLM no-think, NN.

## Dataset conventions

- Use `train` for optimization and named `val_*` sets for training-time
  monitoring. Validation sets are never used to report results: every reported
  score comes from the frozen held-out test suite ([test_suite.md](test_suite.md)),
  which is disjoint from each experiment's training and validation scenarios
  (with the Cycles caveat documented there). Each card below lists its test
  groups.
- One row is one episode scenario before rollout. If one base scenario is
  rendered under multiple schemas, count the rendered rows.
- `sampling.generation_seed: 42` for all paper-facing train and validation
  sampling; paired-scenario variants share the same seeded base set.
- Every validation row carries the experiment-axis label (`weather_regime`,
  `crop_regime`, `observation_schema`, `action_menu`, `reward_formulation`,
  `simulator`, …).
- Validation size is `128` rows per named group (ID and OOD), except T1.2 which
  uses the `384`/`384` plan.

| Source | Use |
| --- | --- |
| `AgriManager-Framework/wofost-weather-pool` | General WOFOST 20-crop weather source (cross-crop experiments). |
| `AgriManager-Framework/wofost-weather-regime-pool` | WOFOST weather-regime source for T1.1. |
| `AgriManager-Framework/weather_pool_maize` | Maize-only WOFOST source for schema-shift experiments. |
| CycleGym deterministic generators | CycleGym crop-planning / crop-growth datasets (fixed locations, year windows, price regimes, seed `42`). |
| DSSAT-Gym deterministic generators | DSSAT maize datasets (fixed environment parameters, seed `42`). |

Variants reuse the same base validation scenarios where the shift is not the
data itself: T1.3, T2.1, T2.2, and T2.3 reuse fixed validation weather across
variants; T1.1 (weather) and T1.2 (crop) do not, because the distribution is the
treatment; T3.1 reuses scenarios only within a source simulator.

---

## T1.1 Weather-regime shift

Tier 1 — within-schema distributional shift · WOFOST-Gym

Fixed schema (WOFOST observation, N/P/K/water actions, `profit_max`); only the
weather regime shifts out of distribution. Train on normal viable weather and
validate on drought, wet, hot, and cold regimes.

| Item | Setting |
| --- | --- |
| Environment | WOFOST-Gym, single-crop in-season resource management |
| Train | `1600` normal viable weather scenarios per crop |
| Validation | `640` per crop: `128` each for `id`, `drought`, `wet`, `hot`, `cold` |
| Test | `640` per crop (chickpea, potato): `128` each for `id`, `drought`, `wet`, `hot`, `cold`, regrouped from unused pool scenarios by a predeclared rule ([test_suite.md](test_suite.md#how-each-part-was-built)) |
| Dataset source | `AgriManager-Framework/wofost-weather-regime-pool`, split `train` plus named `val_*`, seed `42` |
| Fixed factors | Crop, variety, simulator, observation schema, action menu, reward |
| Methods | LLM think, LLM no-think, PPO `n1` (and `n8`, one seed, appendix); DQN and A2C on potato |
| Main metrics | Gain over no-action in profit (this setting is not normalized: the scripted reference is at or below no-action), final WSO |

| Axis | Train | Validation |
| --- | --- | --- |
| Observation / action / reward | unchanged | unchanged |
| Weather context | Normal viable weather | Drought, wet, hot, cold |

Run: [experiments/t1_1_weather_regime_shift/](../../experiments/t1_1_weather_regime_shift/)

## T1.2 Cross-crop trait shift

Tier 1 — within-schema distributional shift · WOFOST-Gym

The task schema (observation format, action menu, reward, simulator, weather
sampling) is fixed; the crop and its trait card shift out of distribution. Train
under `4ID`, `8ID`, or `16ID` crop coverage at a fixed total budget and validate
on both ID crops and held-out crops.

| Item | Setting |
| --- | --- |
| Environment | WOFOST-Gym, multi-crop in-season resource management |
| Train | `1600` total: `400` per crop (`4ID`), `200` per crop (`8ID`), or `100` per crop (`16ID`) |
| Validation | `768` total: `384` ID-crop rows + `384` held-out-crop rows |
| Test | `1024`: `128` each for 4 seen and 4 held-out crops, from the pool's public `test` split; shared by 4ID, 8ID and 16ID |
| Dataset source | `AgriManager-Framework/wofost-weather-pool`, train and pool validation splits, seed `42` |
| Fixed factors | Simulator, observation schema, action menu, reward |
| Methods | LLM think, LLM no-think, trait-conditioned PPO `n1` and `n8` where the trait input is a fixed vector |
| Main metrics | Per-crop target profit (ID and held-out), crop-level OOD gaps, final WSO |

| Axis | Train | Validation |
| --- | --- | --- |
| Observation schema | Fixed WOFOST observation + trait context | Same structure |
| Action / reward | N/P/K/water, `profit_max` | Same |
| Crop / trait context | Seen crops and traits | Held-out crops and traits |

Run: [experiments/t1_2_cross_crop_trait_shift/](../../experiments/t1_2_cross_crop_trait_shift/) ·
crop-selection support: [experiments/support_t1_2_cross_crop_trait_selection/](../../experiments/support_t1_2_cross_crop_trait_selection/)

## T1.3 Price-regime shift

Tier 1 — within-schema distributional shift · CycleGym

The reward form stays fixed at `yield x crop price` and the schema is constant;
only the price vector shifts into held-out regimes. The task is 19-year crop
rotation planning over `CornRM.100` and `SoybeanMG.3`.
Validation is paired: the same weather, initial state, and year window rendered
under each price regime.

| Item | Setting |
| --- | --- |
| Environment | CycleGym crop planning, 19-year crop rotation |
| Train | `1600` episodes: `2` locations x `4` year windows x `200` seeds |
| Validation | `384` paired rows: `128` base scenarios under `id_test`, `high_corn`, `high_soybean` |
| Test | `384` paired rows, same regimes and year windows, fresh seed block from `300000` |
| Dataset source | Deterministic CycleGym generators (`CropPlanningRockSpringsRW-v1`, `CropPlanningNewHollandRW-v1`); training windows 1980–1998 to 1983–2001, validation and test windows 1986–2004 to 1989–2007 |
| Fixed factors | Observation schema, action schema, reward form; paired evaluation holds weather/initial-state/year-window constant, varying only price |
| Methods | LLM think, LLM no-think, PPO `n1`; price-greedy, rotation-aware and soil-aware scripts exist as diagnostics and are not scored on the test suite |
| Main metrics | 19-year cumulative gross revenue (normalized against no-action and the scripted reference) |

| Axis | Train | Validation |
| --- | --- | --- |
| Observation / action / reward | unchanged | unchanged |
| Price parameters | Non-dominant ID price band | Held-out high-price regimes |

Run: [experiments/t1_3_cycles_gym_price_regime/](../../experiments/t1_3_cycles_gym_price_regime/)

## T2.1 Observation-schema shift

Tier 2 — single-axis schema shift · WOFOST-Gym

The action menu and reward stay fixed; the observation schema shifts out of
distribution — which fields are exposed, what they are called, and whether extra
fields appear. Train on the full current schema and validate across the
full-schema reference plus leave-one-out masks, a semantic rename, a compact
superset, and an anonymous-label rename.

| Item | Setting |
| --- | --- |
| Environment | WOFOST-Gym, maize |
| Train | `1600` maize rows rendered only with `S1_full_current` |
| Validation | `1024` rows: `128` each for `S1_full_current`, four `S2_component_leave_one_out` masks, `S3_domain_synonym_rename`, `S4_compact_growth_superset`, `S5_anonymous_label_rename` |
| Dataset source | `AgriManager-Framework/weather_pool_maize`; reuse the same `128` base maize scenarios across schema conditions, seed `42` |
| Fixed factors | Action menu, `profit_max` reward, simulator backend |
| Test | `1024`: the same `128` frozen maize scenarios rendered under all eight variants |
| Methods | LLM think, LLM no-think. The fixed-interface PPO cannot run when the observation layout changes, so it is not run here |
| Main metrics | Target profit by observation schema, final WSO, gap to target-schema specialist if available |

| Axis | Train | Validation |
| --- | --- | --- |
| Observation schema | Full WOFOST schema `S1` | Reference, component-missing subset, rename, compact superset, or anonymous-label rename |
| Action / reward / simulator | N/P/K/water, `profit_max`, WOFOST-Gym | Same |

Run: [experiments/t2_1_observation_schema_shift/](../../experiments/t2_1_observation_schema_shift/)

## T2.2 Action-menu shift

Tier 2 — single-axis schema shift · WOFOST-Gym

The observation schema and reward objective stay fixed; the action menu shifts
out of distribution using WOFOST-Gym's native action modes. Two transfer
families are reported: Family A trains on full-coverage menus and validates on
restricted menus; Family B trains only on restricted menus and validates on
larger composed menus. Native modes may change the WOFOST production wrapper
along with the action menu.

| Item | Setting |
| --- | --- |
| Environment | WOFOST-Gym annual native action modes |
| Train | Family A: `1600` rows, `800` each `lnpkw-v0` / `lnpk-v0`. Family B: `1600` rows, `800` each `lnpk-v0` / `lw-v0` |
| Validation | Family A and Family B: `640` rows, `128` per mode across `lnpkw-v0`, `lnpk-v0`, `lnw-v0`, `ln-v0`, `lw-v0` |
| Dataset source | `AgriManager-Framework/weather_pool_maize`; validation covers training and held-out menus on the same base maize scenarios, seed `42` |
| Fixed factors | Observation schema, unified profit objective, crop/weather protocol |
| Test | `512`: the frozen maize scenarios under `lnpkw`, `lnw`, `ln`, `lw` (`128` each), shared by both families |
| Methods | LLM think, LLM no-think; schema-capable neural baselines: MaskablePPO over a shared 17-action vocabulary with legal-action masks (per family), and target-trained PPO per test menu. The fixed-interface PPO cannot run on a changed menu |
| Main metrics | Target profit and final WSO by target menu, gap to target-menu specialist if available |

Native WOFOST action modes:

| Env id | Action menu |
| --- | --- |
| `lnpkw-v0` | no-op + N/P/K/water, four levels each |
| `lnpk-v0` | no-op + N/P/K, four levels each |
| `lnw-v0` | no-op + N/water, four levels each |
| `ln-v0` | no-op + N, four levels |
| `lw-v0` | no-op + water, four levels |
| `pp-v0` | no-op only (potential-production ceiling, not a train/validation mode) |

| Axis | Train | Validation |
| --- | --- | --- |
| Action schema | Seen native menus | Held-out restricted or expanded native menus |
| Observation / reward | Fixed WOFOST observation, unified profit | Same |
| Simulator | WOFOST-Gym family | WOFOST-Gym family; production wrapper may co-vary |

Run: [experiments/t2_2_action_menu_shift/](../../experiments/t2_2_action_menu_shift/)

## T2.3 Reward-formulation shift

Tier 2 — single-axis schema shift · WOFOST-Gym

The observation schema and action menu stay fixed; the management objective
shifts out of distribution. Train on `yield_max`, `profit_max`, and
`water_stewardship` and validate on those three plus held-out
`nutrient_stewardship`, which reuses yield maximization and input tradeoffs
while changing the protected resource from water to soil N/P/K. Reward objective
definitions are in [reward_objectives.md](reward_objectives.md).

| Item | Setting |
| --- | --- |
| Environment | WOFOST-Gym, maize, `lnpkw-v0` |
| Train | `1600` maize rows balanced `534` `yield_max`, `533` `profit_max`, `533` `water_stewardship` |
| Validation | `512` rows: `128` each for the three seen objectives plus held-out `nutrient_stewardship` |
| Dataset source | `AgriManager-Framework/weather_pool_maize`; validation rows carry `reward_formulation` and `validation_set` labels, seed `42` |
| Fixed factors | Maize, `lnpkw-v0`, observation schema, action menu, simulator backend |
| Test | `512`: the frozen maize scenarios under the four objectives (`128` each) |
| Methods | LLM think, LLM no-think, each trained multi-objective and yield-only. No neural results |
| Main metrics | Target objective score, gap to target-reward specialist if available |
| Runnable dataset config | `experiments/t2_3_reward_formulation_shift/config/t23_reward_formulation_shift.yaml` |

| Axis | Train | Validation |
| --- | --- | --- |
| Reward form | Yield, profit, water stewardship | Same three + held-out nutrient stewardship |
| Observation / action / simulator | Fixed WOFOST observation, N/P/K/water, WOFOST-Gym | Same |

Run: [experiments/t2_3_reward_formulation_shift/](../../experiments/t2_3_reward_formulation_shift/)

## T2.4 Schema-corruption suite

Analysis kept in the code, not part of the paper's eight settings; no test
suite · WOFOST-Gym

An analysis suite that tests whether the policy uses the schema text in the
prompt. In the correct condition the prompt schema matches the environment; in
the corrupted condition the prompt schema is wrong while the environment
implementation is unchanged. Corruptions cover observation, action, and reward
axes, paired on the same seeds, initial states, weather, crops, and target
schemas where possible. Corruption rows are separately labeled validation
variants, not mixed into the clean T2.1/T2.2/T2.3 counts.

| Item | Setting |
| --- | --- |
| Correct condition | Prompt schema and environment schema match |
| Corrupted condition | Prompt schema is wrong; environment implementation unchanged |
| Covered axes | Observation schema, action menu, reward formulation |
| Pairing | Same seeds, initial states, weather, crops, and target schemas where possible |
| Minimum suite | One observation or action corruption, plus one reward-form corruption |
| Methods | LLM think, LLM no-think |
| Main metrics | Correct-minus-corrupted score drop, paired confidence interval |

| Axis | Correct | Corrupted |
| --- | --- | --- |
| Observation schema | Prompt matches environment | Prompt mislabels a field, or claims a missing field exists |
| Action menu | Prompt matches environment | Prompt exposes an unavailable action, or hides an available one |
| Reward formulation | Prompt matches environment | Prompt states the wrong objective |

Run: this is an analysis suite over the T2.1/T2.2/T2.3 runs; it has no dedicated
`experiments/` directory.

## T3.1 Cross-simulator maize transfer

Tier 3 — compositional schema shift · WOFOST-Gym, DSSAT-Gym, CycleGym

Maize is the shared crop. Across simulators the observation schema, action
interface, objective implementation, and transition dynamics can all change; the
prompt describes the target schema but not the target transition kernel. The
main regime is `aligned_profit`: all targets optimize the same grain-equivalent
profit form over their available inputs. Three single-source specialists are each
validated on all three target simulators, producing a `3 x 3` matrix. No joint
mixed-simulator training.

| Item | Setting |
| --- | --- |
| Environment | WOFOST-Gym, DSSAT-Gym, CycleGym crop-growth; maize |
| Primary action menus | WOFOST `lnw-v0` `{N, Water}`; DSSAT `{N, Water}`; CycleGym `{N}` |
| Train | `1600` maize rows per source simulator (single-source specialists) |
| Validation | `128` rows per target simulator (training-time monitoring) |
| Test | WOFOST `128` (frozen maize base), DSSAT `128` (seeds `200042`–`200169`), Cycles `130` (weather years 2004–2016, rebuilt) |
| Dataset sources | WOFOST `AgriManager-Framework/weather_pool_maize`; DSSAT deterministic `maize-irrigation-v0`; CycleGym deterministic CornShort year-windows; seed `42` |
| Methods | LLM think, LLM no-think. No neural baseline: the fixed-interface PPO cannot run across simulators |
| Main metrics | Target profit score, target-normalized score, transfer gap |

Evaluation matrix:

| Train \ Validation | WOFOST-Gym | DSSAT-Gym | CycleGym |
| --- | --- | --- | --- |
| WOFOST-Gym | In-simulator reference | WOFOST → DSSAT | WOFOST → CycleGym |
| DSSAT-Gym | DSSAT → WOFOST | In-simulator reference | DSSAT → CycleGym |
| CycleGym | CycleGym → WOFOST | CycleGym → DSSAT | In-simulator reference |

| Axis | Train | Validation |
| --- | --- | --- |
| Simulator | Single source | All three targets |
| Observation / action | Source-native | Target-native |
| Transition dynamics | Source kernel | Target kernel (implicit, not prompt-described) |
| Objective | `aligned_profit` grain-equivalent | Same form over target's available inputs |

Run: [experiments/t3_1_cross_simulator_maize_transfer/](../../experiments/t3_1_cross_simulator_maize_transfer/)

## T3.2 Unified prompt-conditioned policy

Tier 3 — compositional schema shift · WOFOST-Gym, Gym-DSSAT, CycleGym

A single prompt-conditioned policy is trained over a minimal closure set of
schema tuples `(simulator, observation schema, action menu, objective)`.
Observation schema is fixed to full/native in training. The design is
leave-one-cell-out with full marginal coverage: callback target tuples from
T2.2, T2.3, and T3.1 are held out of training, while every primitive factor in
those tuples is seen in another context.

| Item | Setting |
| --- | --- |
| Training unit | Episode-level schema tuple `(simulator, observation schema, action menu, objective)` |
| Observation schema | Full/native only in training: WOFOST `S1_full_current`, DSSAT native, CycleGym native |
| Train | `3200` rendered rows: `8` tuples x `400` rows |
| Validation | `128` rows per named validation tuple (training-time monitoring) |
| Test | `1156`: seven tuples of `128` and the two Cycles tuples of `130` (rebuilt suites); held-out tuples also scored for the matching T3.1 specialists |
| Dataset sources | WOFOST `AgriManager-Framework/weather_pool_maize`; DSSAT and CycleGym deterministic generators; seed `42` |
| Methods | LLM think, LLM no-think (Qwen, three seeds); Llama-3.2 and other backbones as single-seed exploratory runs. No neural baseline |
| Main metrics | Target-normalized score on held-out schema tuples, gap to relevant specialists |

Training tuples:

| Tuple | Simulator | Observation | Action | Objective | Rows |
| --- | --- | --- | --- | --- | ---: |
| W1 | WOFOST-Gym | `S1_full_current` | `lnpkw` | `yield_max` | 400 |
| W2 | WOFOST-Gym | `S1_full_current` | `lnpkw` | `profit_max` | 400 |
| W3 | WOFOST-Gym | `S1_full_current` | `lnpkw` | `water_stewardship` | 400 |
| W4 | WOFOST-Gym | `S1_full_current` | `lnpk` | `profit_max` | 400 |
| W5 | WOFOST-Gym | `S1_full_current` | `lnpk` | `nutrient_stewardship` | 400 |
| W6 | WOFOST-Gym | `S1_full_current` | `lnw` | `yield_max` | 400 |
| D1 | Gym-DSSAT | native | native | `yield_max` | 400 |
| C1 | CycleGym | native | native | `yield_max` | 400 |

Validation tuples (what shifts — the held-out cells are the shift):

| Validation set | Status | Callback role |
| --- | --- | --- |
| `wofost_s1_lnpkw_profit` | Seen control | Canonical WOFOST profit anchor |
| `wofost_s1_lnpk_profit` | Seen control | Restricted-action profit anchor |
| `wofost_s1_lnw_yield` | Seen control | `lnw` action-menu marginal support |
| `dssat_yield` | Seen control | DSSAT simulator marginal support |
| `cycles_yield` | Seen control | CycleGym simulator marginal support |
| `wofost_s1_lnw_profit` | Held out | T2.2 action-menu callback and T3.1 WOFOST target |
| `wofost_s1_lnpkw_nutrient` | Held out | T2.3 reward-formulation callback |
| `dssat_profit` | Held out | T3.1 DSSAT target |
| `cycles_profit` | Held out | T3.1 CycleGym target and far-transfer stress test |

Run: [experiments/t3_2_unified_prompt_conditioned_policy/](../../experiments/t3_2_unified_prompt_conditioned_policy/)

## APP1 WOFOST same-crop variety OOD

Exploratory, not part of the paper's eight settings; no test suite · WOFOST-Gym

The schema is held fixed (crop species, simulator, action space, reward, weather
distribution); the variety / ecotype shifts out of distribution. The main crop
is rice. Train on seen rice varieties and validate on held-out rice varieties,
with variety traits such as `rice_variety_traits_v1` supplied as context where
available.

| Item | Setting |
| --- | --- |
| Environment | WOFOST-Gym |
| Main crop | Rice |
| OOD factor | Variety / ecotype |
| Trait condition | Variety traits such as `rice_variety_traits_v1` where available |
| Fixed factors | Crop species, simulator, action space, reward, weather distribution |
| Methods | LLM think, LLM no-think, NN-PPO |
| Main metrics | ID / OOD / per-variety target profit, OOD gap; final WSO |

| Axis | Train | Validation |
| --- | --- | --- |
| Observation schema | Fixed WOFOST observation + variety traits | Same structure |
| Action / reward | N/P/K/water, `profit_max` | Same |
| Variety context | Seen rice varieties | Held-out rice varieties |

Run: [experiments/app1_wofost_variety_ood/](../../experiments/app1_wofost_variety_ood/)
