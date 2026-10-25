# Reward Objectives

Paper-facing reward plan for the experiment cards: what each experiment
optimizes and how reward definitions appear in runnable configs.

Reward objectives are part of the task definition, not the dataset source.
Dataset files may carry `objective_id` as metadata; the semantics live here.

## Config Contract

| Rule | Standard |
| --- | --- |
| Primary field | `objective_id` defines the true environment reward objective. |
| Legacy wrappers | Do not use `env_reward` as an experiment-definition field. Native simulator wrappers are implementation details only. |
| Prompt corruption | `prompt_objective_id` only for schema-corruption probes where the prompt-facing objective differs from the true reward. |
| Parameters | Put objective-specific constants in `reward_params`, not in run scripts. |
| Format reward | Action-format bonuses are training scaffolding, not the management objective reported in paper tables. |
| Labels | Reward-formulation validation rows carry `reward_formulation` and `validation_set` labels. |

In normal runs `objective_id` and the prompt-facing objective match. A mismatch
is valid only for explicit schema-corruption tests.

## Objective Definitions

| Objective id | Meaning | Main use |
| --- | --- | --- |
| `profit_max` | Maximize terminal grain/WSO-equivalent profit: yield or WSO minus input costs. | Default WOFOST objective, T2.2, T3.1 aligned-profit transfer. |
| `yield_max` | Maximize terminal yield or WSO, usually normalized by a calibrated `Y_ref`. | T2.3 train objective, yield-only crop-growth baselines. |
| `water_stewardship` | Maintain acceptable yield while penalizing irrigation use and water-budget violations. | T2.3 train objective. |
| `nutrient_stewardship` | Maintain acceptable yield while avoiding nutrient over-application and poor terminal N/P/K status. | T2.3 held-out OOD objective, reward-corruption probes. |
| CycleGym crop-planning revenue | Maximize multi-year gross revenue, `yield x crop price`, under price-regime variation. | T1.3 price-regime shift. |
| `pollution_penalized_revenue` | CycleGym only: the native reward (multi-year gross revenue on crop planning) minus a per-season penalty on nitrogen lost to water and air. | Supported option, not used in the paper. See [Pollution-Penalized Revenue](#pollution-penalized-revenue-cyclegym). |

`profit_max`, `yield_max`, `water_stewardship`, and `nutrient_stewardship` are
AgriManager objective IDs. CycleGym crop-planning revenue is a simulator-native
economic objective for T1.3, not a T2.3 reward-form variant unless a future
config exposes it through `objective_id`.

## Objectives By Tier

| Tier | Experiment | Reward objective |
| --- | --- | --- |
| 1 | T1.1 Weather-regime shift | `profit_max` for all WOFOST configs. |
| 1 | T1.2 Cross-crop trait shift | `profit_max` for all WOFOST configs, including support specialists. |
| 1 | T1.3 Price-regime shift | CycleGym gross revenue `yield x crop price`; reward form fixed while price parameters shift. |
| 2 | T2.1 Observation-schema shift | `profit_max`; schema changes, objective fixed. Final WSO reported as secondary yield readout. |
| 2 | T2.2 Action-menu shift | `profit_max`; action menu changes, objective fixed. |
| 2 | T2.3 Reward-formulation shift | Train on `yield_max`, `profit_max`, `water_stewardship`; validate on those plus held-out `nutrient_stewardship`. |
| 2 | T2.4 Schema-corruption suite | Parent T2.1/T2.2/T2.3 objectives, with prompt-side corruption where applicable. |
| 3 | T3.1 Cross-simulator maize transfer | `profit_max` aligned across WOFOST-Gym, DSSAT-Gym, CycleGym via shared economic constants in each action space. |
| 3 | T3.2 Unified prompt-conditioned policy | Unified training over `yield_max`, `profit_max`, `water_stewardship`, `nutrient_stewardship`; DSSAT/CycleGym profit targets are held-out validation callbacks. |

## WOFOST Profit Default

Default objective for WOFOST-Gym StoreV2 cards:

```yaml
env:
  objective_id: profit_max
```

Definition (grain/WSO-equivalent input costs):

```text
profit_wso_kg_ha =
  final_WSO_kg_ha
  - c_N * total_N_kg_ha
  - c_P * total_P_kg_ha
  - c_K * total_K_kg_ha
  - c_W * total_irrig_mm
```

Paper-wide constants:

| Parameter | Value | Meaning |
| --- | ---: | --- |
| `c_N` | `3.5` | kg WSO-equivalent cost per kg nitrogen |
| `c_P` | `5.5` | kg WSO-equivalent cost per kg phosphorus |
| `c_K` | `2.25` | kg WSO-equivalent cost per kg potassium |
| `c_W` | `0.05` | kg WSO-equivalent cost per mm irrigation |

Some prompts render cumulative irrigation in cm because the WOFOST prompt surface
reports depth in `cm`; `0.5` kg WSO-equivalent per cm equals `0.05` per mm, and a
`50 cm` budget equals `500 mm`.

T3.1 uses the same economic definition but prunes unavailable inputs (e.g.
CycleGym has no irrigation action in the main maize crop-growth setup, so the
water term is absent).

## Reward-Formulation Shift

T2.3 is the only main experiment whose axis is the reward formulation. Clean runs
have matching true and prompt objectives:

```yaml
env:
  objective_id: water_stewardship
```

Corruption probes deliberately separate them:

```yaml
env:
  objective_id: nutrient_stewardship
  prompt_objective_id: yield_max
```

Corruption-probe rows are labeled as such and not mixed into the clean T2.3
validation counts.

## Pollution-Penalized Revenue (CycleGym)

`objective_id: pollution_penalized_revenue` is a reward-schema option for
CycleGym environments. Every Cycles run already writes daily nitrogen fluxes
(`N.dat`), and the crop-planning prompt shows some of them in its
"Nitrogen losses" observation group; the default T1.3 objective does not score
them. This option keeps the native CycleGym reward and subtracts a weighted
nitrogen-loss penalty. It was added after review and is not used in any paper
table; no model has been trained on it.

```yaml
env:
  objective_id: pollution_penalized_revenue
  reward_params:                      # all optional; defaults shown
    n_loss_cost_no3_leaching_usd_per_kg_n: 10.0
    n_loss_cost_nh4_leaching_usd_per_kg_n: 10.0
    n_loss_cost_nh3_volatilization_usd_per_kg_n: 10.0
    n_loss_cost_n2o_emission_usd_per_kg_n: 10.0
    n_loss_include_bypass: true       # count bypass-flow leaching as leaching
    # n_loss_cost_usd_per_kg_n: 10.0  # one weight for every term without its own key
```

Definition, per environment step `t` (one season on crop planning):

```text
penalty_t = sum_k cost_k * loss_k,t            # $/ha
loss_k,t  = sum of daily N.dat column(s) for term k over the days the step simulated
reward_t  = (native_t - penalty_t) / reward_scale + format_t + shaping_t
```

| Term `k` | `N.dat` columns (kg N/ha/day) |
| --- | --- |
| `no3_leaching` | `NO3 LEACHING` (+ `NO3 BYPASS`) |
| `nh4_leaching` | `NH4 LEACHING` (+ `NH4 BYPASS`) |
| `nh3_volatilization` | `NH3 VOLATILIZ` |
| `n2o_emission` | `N2O FROM NITRIF` + `N2O FROM DENIT` |

- The step window is `[date before the step, date after the step)`, so the
  windows tile the simulated horizon and the episode penalty is the sum of the
  per-season penalties, the same way the episode's gross revenue is the sum of
  per-season revenue. `NO3 DENITRIF` is not penalized.
- On crop planning, `native_t` is gross revenue `yield x crop price` ($/ha).
  The rotation env has no fertilizer action, so its N losses come from soil
  mineralization, residues and soybean fixation. On the corn crop-growth env
  `native_t` is CycleGym's own dense reward (harvest revenue minus N fertilizer
  cost), and the penalty grows with applied N.
- Like the native reward, the penalty enters the scalar reward only on valid
  steps. Losses are tallied on every step.
- **Default weights are illustrative.** The 10 $/kg N defaults were chosen so the
  penalty is visible next to a gross revenue of about 1,000 to 2,000 $/ha per
  season. They are not damage-cost estimates and have no cited source. Set the
  weights explicitly for any reported experiment. When a term falls back to
  the default, the environment logs one info line per process saying so.
- The option cannot be combined with a yield-only `reward_mode`; the env raises
  an error.
- The crop-planning prompt states the objective and the nonzero weights in a
  `<management objective id="pollution_penalized_revenue">` block and adds the
  per-season N loss and penalty to the trajectory summary. Other objective ids
  keep the gross-revenue prompt unchanged. The corn crop-growth prompt has no
  dedicated text for this objective; use `prompt_objective_text` there.

Logged metrics: per step `n_loss_kg_ha`, `n_loss_pen_usd`,
`n_loss_penalty_reward`, `n_loss__<term>` (kg N/ha) and `n_loss_pen__<term>`
($/ha), plus `crop_planning/n_loss_kg_ha`, `crop_planning/n_loss_penalty` and
`crop_planning/net_revenue` on crop planning. Trajectory metrics add the episode
totals, the weights (`n_loss_cost__<term>`), and on crop planning
`pen_net_revenue = cumulative_gross_revenue - n_loss_pen_usd` and its
`reward_scale`-normalized form `pen_net_revenue_norm`. Term suffixes are
`no3_leach`, `nh4_leach`, `nh3_volat`, `n2o`.

Example dataset config: `smoke_tests/cycles_gym/config/crop_planning_pollution_llm.yaml`.
Tests: `tests/test_cycles_pollution_reward.py` (unit tests run without
CycleGym; the real-simulator smoke runs when `CYCLES_GYM_PATH` points to a
CycleGym checkout with the Cycles binary). The CPU smoke rollout prints the
reward components and checks that they add up:

```bash
python smoke_tests/cycles_gym/pollution_reward_smoke.py \
  --cycles-gym-path ../AgriManagerExternal/CyclesGym
```

Implementation: `agrimanager/env/cycles_gym/n_loss.py` (terms, weights,
window sums), `agrimanager/env/cycles_gym/env.py` (reward and metrics),
`agrimanager/env/cycles_gym/prompt.py` (crop-planning prompt).

## Open Decisions

- Whether T1.3's gross-revenue CycleGym objective should get a formal
  `objective_id` or remain a simulator-native special case.
- Keep T3.2 objective sampling small enough that held-out tuple results stay
  interpretable.
