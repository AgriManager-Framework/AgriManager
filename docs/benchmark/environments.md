# Environments and simulators

AgriManager wraps three process-based crop simulators. Each one is an
environment module under `agrimanager/env/<env_name>/`, selected by the
`env_name` field of a dataset config. The framework imports the module by name
(`agrimanager.env.<env_name>`); there is no per-simulator branch in the core
code. How to add a fourth simulator is described in
[environment_adapter_contract.md](../reference/environment_adapter_contract.md).

| Simulator | `env_name` | Upstream wrapper | Install | Settings |
|---|---|---|---|---|
| WOFOST (through PCSE) | `wofost_gym` | WOFOST-Gym | `install.sh` | T1.1, T1.2, T2.1, T2.2, T2.3, WOFOST parts of T3.1 and T3.2 |
| Cycles | `cycles_gym` | CycleGym | `install.sh` (also downloads the native Cycles binary) | T1.3, Cycles parts of T3.1 and T3.2 |
| DSSAT (DSSAT-CSM) | `gym_dssat` | Gym-DSSAT (`gym_dssat_pdi`) | [separate Spack build](../getting-started/install_dssat.md) | DSSAT parts of T3.1 and T3.2 |

The paper and these docs call the simulators DSSAT, WOFOST and Cycles.
CycleGym, WOFOST-Gym and Gym-DSSAT are the Gym wrappers AgriManager calls.

Every environment exposes two interfaces:

- a **text interface** (`env.py`, `prompt.py`) for language-model policies:
  the observation, action menu and objective are rendered into a prompt, and
  the model's answer is parsed back into a simulator action;
- a **numeric interface** (`nn_adapter.py`) for the neural baselines: a fixed
  observation vector and a discrete action space.

One dataset row is one episode scenario. It stores the full environment config
(simulator, crop, site, year or seed, action menu, objective, reward
parameters, prompt options), so a scenario can be replayed exactly by any
policy. Datasets are built with `entrypoints/dataset/build.sh` from a YAML
config ([dataset_config_conventions.md](../conventions/dataset_config_conventions.md)).

## WOFOST

- **Task.** In-season management of one crop from sowing to maturity. One
  decision every 10 days, 24 decisions per season.
- **Scenarios.** A scenario is a (crop, year, latitude, longitude) tuple. Its
  weather comes from a pre-built NASA POWER cache shipped with three public
  Hugging Face pools (see [Models and data](../release/models_and_data.md)).
  A candidate is kept only if a full no-action rollout has complete weather,
  reaches development stage 1.5 and produces a positive storage-organ weight.
- **Observation.** Crop status (development stage, storage-organ dry matter),
  available soil N, P and K, root-zone soil moisture, cumulative inputs, and
  daily weather. The full schema `S1` has 15 fields; Tier 2 renames, removes or
  adds fields ([Schema and prompts](schema_and_prompts.md)).
- **Actions.** WOFOST-Gym's native action modes:

  | Env id | Menu | Actions |
  |---|---|---:|
  | `lnpkw-v0` | no-op + N, P, K, water (four levels each) | 17 |
  | `lnpk-v0` | no-op + N, P, K | 13 |
  | `lnw-v0` | no-op + N, water | 9 |
  | `ln-v0` | no-op + N | 5 |
  | `lw-v0` | no-op + water | 5 |
  | `pp-v0` | potential production, no actions (diagnostic only) | 1 |

  With `scale_action_amounts_by_interval: true` (the benchmark default) the
  fertilizer and irrigation amounts are multiplied by the decision interval and
  the number of discrete actions stays the same; the prompt shows the scaled
  amounts.
- **Objectives.** `profit_max` (default), `yield_max`, `water_stewardship`,
  `nutrient_stewardship`; definitions in [reward_objectives.md](reward_objectives.md).
- **Code.** `agrimanager/env/wofost_gym/`, weather-pool tools in
  `integrations/wofost_gym/dataset_tools/`, dataset contract in
  [wofost_dataset_contract.md](../reference/wofost_dataset_contract.md).

## Cycles

Two Cycles tasks are used.

- **Crop planning (T1.3).** Environments `CropPlanningRockSpringsRW-v1` and
  `CropPlanningNewHollandRW-v1`, 19 yearly decisions. The action is the annual
  crop choice between `CornRM.100` and `SoybeanMG.3`; the objective is gross
  revenue (yield times crop price, no input costs). Scenarios vary the site,
  the year window, the price vector and a seed.
- **Maize crop growth (T3.1, T3.2).** Environment `CornShortRockSpringsFW-v1`,
  nitrogen only, eleven levels from 0 to 150 kg/ha. A scenario is fixed by its
  **weather year** at a single site (Rock Springs): the seed does not change the
  dynamics. Training uses 1980 to 1995, validation 1996 to 2003, and the
  rebuilt test suites 2004 to 2016 ([Test suite](test_suite.md#cycles-rebuild)).
- **Code.** `agrimanager/env/cycles_gym/`; smoke tests in `smoke_tests/cycles_gym/`.
- **License note.** CycleGym itself is BSD-3-Clause. The native Cycles binary
  it downloads is CC BY-NC-ND 4.0.

## DSSAT

- **Task.** Maize in-season management (`maize-irrigation-v0`), nitrogen and
  irrigation, one decision every 10 days, 24 decisions. The smoke tests also
  cover rice and cotton.
- **Scenarios.** A scenario is fixed by the generator seed. Training uses seeds
  from 42 upward (42 to 1641 in T3.1, 42 to 441 in T3.2), validation 100042 to
  100169, and the test suite 200042 to 200169.
- **Code.** `agrimanager/env/gym_dssat/`; smoke tests in `smoke_tests/gym_dssat/`.

## Cross-simulator objective

The cross-simulator settings use one grain-equivalent profit form on all three
simulators (`aligned_profit`): final yield minus the cost of the inputs each
simulator exposes, with the same unit costs (N 3.5, P 5.5, K 2.25 kg/ha of
grain per kg/ha of input; water at the simulator's unit). The prompt describes
the target schema; it does not describe the target simulator's transition
dynamics.

## Smoke tests

| Simulator | README | CPU-only scripts |
|---|---|---|
| WOFOST | `smoke_tests/wofost_gym/README.md` | `run_cpu_quickstart.sh`, `run_build_datasets.sh`, `run_nn_train.sh`, `run_nn_eval.sh` |
| Cycles | `smoke_tests/cycles_gym/README.md` | `run_build_datasets*.sh`, `run_nn_train*.sh`, `run_nn_eval*.sh` |
| DSSAT | `smoke_tests/gym_dssat/README.md` | `run_runtime_smoke.sh`, `run_build_datasets.sh` |

The language-model scripts in each smoke-test directory need a GPU.
