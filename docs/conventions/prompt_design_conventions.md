# Prompt Design Conventions

This document is a short reference for the current WOFOST-Gym prompt structure.
The implementation lives in code; this page records what the prompt looks like,
what each block means, and where to edit when an experiment changes one schema
axis.

Main code references:

- `agrimanager/env/wofost_gym/prompt.py`
- `agrimanager/env/base/objective_prompt.py`
- WOFOST dataset configs under `experiments/*/config/*.yaml`

## Prompt Shape

WOFOST-Gym prompts have two levels.

System prompt, fixed across normal WOFOST-Gym experiments:

```text
You are an agricultural management expert. Optimize the management objective specified in the user message. Use only the available actions and respond in the required format.
```

User prompt, rebuilt every decision step:

```text
Intro

<management objective id="...">
...
</management objective>

Bridge sentence

<crop traits>
...
</crop traits>

<current observation>
...
</current observation>

Available actions ...

Decision guidance ...

Response format ...
```

`<crop traits>` is optional and appears only when crop traits are enabled.

## Blocks

### Intro Block

Stable episode context: crop, growing window, intervention interval, current
day, and optional calendar date. Not the place to encode an experiment
condition.

```text
We are growing maize from sow to maturity. The planned growing window spans 241 days, actions are taken every 10 days, and today is day 31 of the season.
```

### Management Objective Block

Reward/objective schema shown to the model, built by
`agrimanager/env/base/objective_prompt.py`.

```text
<management objective id="profit_max">
Goal: maximize terminal WSO-equivalent profit, not raw yield.
Score: profit_wso_kg_ha = final_WSO_kg_ha - costs on N, P, K, and irrigation.
Decision rule: apply an input only if it raises final WSO by more than its cost.
</management objective>
```

Relevant config fields:

```yaml
env:
  objective_id: profit_max          # true environment reward
  prompt_objective_id: null         # optional prompt-only override
  prompt_objective_text: null       # optional free-form prompt text
  reward_params: {}
```

In normal runs `objective_id` and `prompt_objective_id` should match; they
differ only for explicit prompt-corruption probes.

### Crop Traits Block

Optional crop trait context. Used by cross-crop trait experiments, not by
default Tier 2 single-axis schema-shift runs.

```text
<crop traits>
- Crop: maize.
- Maturity class: medium.
- Water stress sensitivity: moderate.
</crop traits>
```

Relevant config fields:

```yaml
env:
  include_crop_traits: true
  trait_schema: traits_v1_23d
```

### Current Observation Block

Observation schema shown to the model. WOFOST-Gym renders field names, units,
physical descriptions, and current values here.

```text
<current observation>
[Crop status]
- Development stage index: 0.732 (DVS=1 flowering; DVS=2 maturity)
- Storage organ dry matter (kg/ha; yield biomass proxy): 1420
[Soil nutrients]
- Available soil nitrogen (kg/ha): 21.5
[Soil & water]
- Root-zone soil moisture (fraction): 0.31
</current observation>
```

Current no-traits baseline fields:

```text
FIN, DVS, WSO, NAVAIL, PAVAIL, KAVAIL, SM, TOTN, TOTP, TOTK,
TOTIRRIG, IRRAD, TEMP, RAIN, DAYS
```

`DAYS` is normally used by the intro rather than repeated as a regular
observation line.

### Available Actions Block

Action schema shown to the model: valid action types and amounts, and may also
list unavailable action types.

```text
Available actions (pick exactly one):
- Do nothing.
- Apply nitrogen fertilizer (2.0, 4.0, 6.0, or 8.0 kg/ha).
- Irrigate with 0.5, 1.0, 1.5, or 2.0 cm of water.
Unavailable actions: phosphorus fertilizer or potassium fertilizer.
```

Relevant config fields:

```yaml
env:
  env_id: lnw-v0                    # true simulator action mode
  prompt_action_schema_env_id: null # optional prompt-only override
```

In normal runs the prompt action schema should match `env_id`. Use
`prompt_action_schema_env_id` only for explicit prompt-corruption probes.

Native WOFOST-Gym action menus:

| Env id | Prompt action menu |
| --- | --- |
| `lnpkw-v0` | N, P, K, water |
| `lnpk-v0` | N, P, K |
| `lnw-v0` | N, water |
| `ln-v0` | N |
| `lw-v0` | water |
| `pp-v0` | no management action |

### Decision Guidance Block

Short instruction telling the model how to use the observation, objective, and
actions. The decision line is derived from the objective id. Do not hide new
experiment-specific information here.

```text
Please reason briefly before action about:
1. State grounding: describe the current agronomic state and the main limiting factor
2. Decision: choose the action that best balances final-yield gains against input costs
```

### Response Format Block

Parser contract.

No-think:

```text
<answer> ... </answer>
```

Think:

```text
<tool_call> ... </tool_call> <answer> ... </answer>
```

Relevant config fields:

```yaml
env:
  require_think: true
  thinking_mode: think
  think_tag: tool_call
```

The response format should stay fixed within a think/no-think condition.

## Where To Edit For Experiments

| Experiment axis | Prompt block | Usual edit point |
| --- | --- | --- |
| Observation schema | `<current observation>` | observation field list / descriptions in WOFOST prompt or schema adapter |
| Action menu | `Available actions` | `env.env_id`, split-level `env.env_id`, or `prompt_action_schema_env_id` for corruption |
| Reward formulation | `<management objective>` | `objective_id`, `prompt_objective_id`, `prompt_objective_text`, `reward_params` |
| Crop traits | `<crop traits>` | `include_crop_traits`, `trait_schema` |
| Think/no-think format | `Response format` | `require_think`, `thinking_mode`, `think_tag` |

For Tier 2 schema-shift experiments, change one prompt block at a time:

- T2.1 changes observation text.
- T2.2 changes action-menu text.
- T2.3 changes objective text.

The system prompt should stay unchanged unless we are deliberately changing the
global agent role, which is not part of the current Tier 2 design.
