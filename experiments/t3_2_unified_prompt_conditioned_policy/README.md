# T3.2 Unified Prompt-Conditioned Policy

This experiment implements the full-observation T3.2 capstone design from
[docs/benchmark/README.md](../../docs/benchmark/README.md#t32-unified-prompt-conditioned-policy) (T3.2).

The main training run uses `3200` rendered training rows: eight schema tuples
with `400` rows each. Observation schema is fixed to the full/native observation
interface. The experimental shift is compositional transfer across simulator,
action-menu, and objective tuples.

## Dataset Design

Training tuples:

- WOFOST `S1/lnpkw/yield_max`
- WOFOST `S1/lnpkw/profit_max`
- WOFOST `S1/lnpkw/water_stewardship`
- WOFOST `S1/lnpk/profit_max`
- WOFOST `S1/lnpk/nutrient_stewardship`
- WOFOST `S1/lnw/yield_max`
- Gym-DSSAT `yield_max`
- CycleGym `yield_max`

Held-out callback validation tuples:

- WOFOST `S1/lnw/profit_max`
- WOFOST `S1/lnpkw/nutrient_stewardship`
- Gym-DSSAT `profit_max`
- CycleGym `profit_max`

The training scripts also validate seen controls for WOFOST, DSSAT, and
CycleGym. All validation sets use `128` rows and are logged with
`data.validation_axis=schema_tuple`.

## Train

Think:

```bash
bash experiments/t3_2_unified_prompt_conditioned_policy/run_t32_unified_llm_think_train.sh
```

No-think:

```bash
bash experiments/t3_2_unified_prompt_conditioned_policy/run_t32_unified_llm_no_think_train.sh
```

Both scripts build missing datasets inline before launching training.
They are self-contained fixed run definitions, following the T3.1
Cross-Simulator script style: each script spells out the dataset configs,
parquet paths, named validation sets, model, and trainer overrides directly.
Validation is passed through `data.val_sets` with
`data.validation_axis=schema_tuple`. The fixed training scripts run validation
before training and every `10` trainer steps, with `4` validation generations
logged per validation pass.

## Paper replicates and exploratory backbones

The scored Qwen3-4B seeds are `run_t32_unified_qwen3_{think,no_think}_r{2,3,4}_train.sh`.
Exploratory runs with one seed per arm:

- `run_t32_unified_llama32_{think,no_think}_train.sh`: Llama-3.2-3B-Instruct.
- `run_t32_backbone_sweep_train.sh --algo {grpo,drgrpo,ppo} --think {think,tool_call,no_think} --model <id>`:
  Qwen3-4B-Instruct-2507 under DrGRPO and PPO, Qwen2.5-3B-Instruct and Qwen3-8B
  under GRPO. The script header lists the arguments, GPU count and save
  interval of each published `t32-diffmodels-*` checkpoint.

## OpenRouter held-out model evaluation

The fixed `run_openrouter_*_heldout134_eval.sh` scripts compare external models
on held-out validation tuples 1, 3, and 4:

- WOFOST `S1/lnw/profit_max`
- Gym-DSSAT `profit_max`
- CycleGym `profit_max`

Each script evaluates all `128` rows from each tuple, enables the model's native
reasoning without selecting a provider-specific effort level, and writes
provider-reported token/cost accounting plus local attempt/failure counts to
`usage.json`. A malformed response may have been billed without exposing usable
cost metadata, so `unreported_attempt_count` must be checked before treating the
reported cost as complete. The five model definitions are Hy3 free, GLM 5.2,
Qwen3.7 Plus, MiniMax M3, and DeepSeek V4 Pro.

For the rate-limited free Hunyuan endpoint,
`run_openrouter_hy3_free_heldout134_quota_eval.sh` evaluates 13 complete episodes
from each held-out tuple (39 episodes total, at most 962 normal requests). This
leaves a small part of the 1,000-request daily quota for preflights and retries.

`run_openrouter_hy3_free_heldout34_quota_resume.sh` is the fixed recovery run
for held-out tuples 3 and 4 after a truncated-response failure. It evaluates 11
episodes from each remaining tuple and uses the provider's JSON-response retry.

Run one model at a time in the `agrimanager` environment. For example:

```bash
conda run -n agrimanager \
  bash experiments/t3_2_unified_prompt_conditioned_policy/run_openrouter_glm52_heldout134_eval.sh
```
