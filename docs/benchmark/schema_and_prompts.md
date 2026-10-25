# Schema and prompt specification

A policy in AgriManager sees the task through its **schema**
`Σ = (Σ_O, Σ_A, Σ_R)`: the observed variables, the available actions and the
rewarded objective. A language-model policy reads `Σ` from the prompt of each
episode. A fixed-interface neural policy has `Σ_O` and `Σ_A` built into its
input and output layers. **Schema shift** means that `Σ` differs between
training and deployment.

## Prompt structure

Every decision turn is one user message rendered by the environment. On WOFOST
it has these blocks, in this order (the [quickstart](../getting-started/quickstart.md#2-look-at-what-the-policy-sees)
prints a real example):

| Block | Content | Schema component |
|---|---|---|
| System message | Role and the instruction to follow the objective and format | none |
| Intro | Crop, season length, decision interval, current day | context |
| Management objective | Objective id, the end-of-season score formula, unit input costs | `Σ_R` |
| Crop traits (optional) | Trait card of the crop; used by the crop-variety setting | context |
| Current observation | Grouped fields with names, units and values | `Σ_O` |
| Available actions | The legal actions of the current menu with their amounts | `Σ_A` |
| Decision guidance | State grounding, then the decision | none |
| Response format | Required output tags and one example | none |

Cycles and DSSAT prompts use the same block order with simulator-native
observations and actions. Full prompt listings for every simulator are in the
paper appendix; the code is in `agrimanager/env/<env_name>/prompt.py` and
`agrimanager/env/base/objective_prompt.py`. Editing conventions for the WOFOST
prompt are in [prompt_design_conventions.md](../conventions/prompt_design_conventions.md).

## Response formats: think and no-think

Each language-model arm is trained and evaluated with one response format.

| Arm | Env config | Required answer |
|---|---|---|
| no-think | `require_think: false` | `<answer> one action </answer>` |
| think | `require_think: true`, `think_tag: tool_call` | `<tool_call> short reasoning </tool_call> <answer> one action </answer>` |

The reasoning tag is literally `tool_call` in the released code and
checkpoints. A checkpoint is always scored with the format it was trained
with; the evaluator reads the format from the checkpoint's `run_manifest.json`
and refuses a mismatch ([Evaluation](../guides/evaluation.md)). An answer that
cannot be parsed into a legal action counts as invalid; the invalid rate is
reported next to every score.

## How each setting shifts the schema

| Setting | Component that shifts | How it is realized |
|---|---|---|
| T1.1 weather-regime, T1.2 crop-variety, T1.3 price-regime | none (distributional shift of the scenario) | same prompt template; new weather, crops or prices |
| T2.1 observation-schema | `Σ_O` | `prompt_observation_fields` and `prompt_field_aliases` in the env config select, rename or add observation fields |
| T2.2 action-menu | `Σ_A` | WOFOST-Gym native action mode (`env_id`), which sets the available-actions block |
| T2.3 reward-formulation | `Σ_R` | `objective_id` and `prompt_objective_id`, which set the objective block and the reward |
| T3.1 cross-simulator | `Σ` and the dynamics | a different simulator with its native schema |
| T3.2 unified policy | combinations of `Σ_O × Σ_A × Σ_R × simulator` | held-out combinations of components each seen during training |

### T2.1 observation variants

All variants render the same 128 physical scenarios. `S1` is the training
schema.

| Variant | Role | Prompt change |
|---|---|---|
| `S1_full_current` | in-distribution reference | all 15 fields |
| `S2a_no_stage_time` | missing information | remove `DVS`, `DAYS` |
| `S2b_no_resource_state` | missing information | remove `NAVAIL`, `PAVAIL`, `KAVAIL`, `SM` (15 to 11 fields) |
| `S2c_no_management_history` | missing information | remove `TOTN`, `TOTP`, `TOTK`, `TOTIRRIG` |
| `S2d_no_weather_context` | missing information | remove `IRRAD`, `TEMP`, `RAIN` |
| `S3_domain_synonym_rename` | semantic rename | same values, agronomic synonyms as labels, with a glossary |
| `S4_compact_growth_superset` | strict superset | add `LAI`, `TAGP`, `RD`, `RFTRA`, `NUPTAKETOTAL`, `PUPTAKETOTAL`, `KUPTAKETOTAL` (22 fields) |
| `S5_anonymous_label_rename` | label-grounding probe | same values, labels `A` to `O`, with a glossary |

### T2.2 action menus

Family A trains on `lnpkw` and `lnpk`; Family B trains on `lnpk` and `lw`. Both
are tested on `lnpkw`, `lnw`, `ln` and `lw` over the same scenarios. Menu
definitions are in [Environments](environments.md#wofost).

### T2.3 objectives

The multi-objective arm trains on `yield_max`, `profit_max` and
`water_stewardship`; the yield-only arm trains on `yield_max` alone. Both are
tested on those three and on the held-out `nutrient_stewardship`. Formulas are
in [reward_objectives.md](reward_objectives.md).

### T3.2 schema tuples

The unified policy trains on eight tuples `(simulator, observation, action,
objective)` and is tested on five seen-control tuples and four held-out
tuples; the list is in the [benchmark overview](README.md#t32-unified-prompt-conditioned-policy).

## What the neural baselines can and cannot do

The fixed-interface PPO baseline reads a fixed-length numeric vector and emits
one index of a fixed action set. It runs on the Tier 1 settings, where the
observation layout and action set match training. It cannot run when `Σ_O` or
`Σ_A` changes size or meaning, and it needs retraining when `Σ_R` changes. This
statement is about the fixed-interface baseline only: schema-capable neural
designs exist, and the action-menu setting includes one (MaskablePPO over a
shared 17-action vocabulary with a legal-action mask); see
[Baselines](baselines.md#neural-baselines).
