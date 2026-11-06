# Quickstart

This page gets you from a fresh install to scored episodes in a few minutes,
on a CPU, without downloading a model.

## 1. Score two scripted policies on eight maize scenarios (CPU)

Prerequisite: AgriManager and WOFOST-Gym installed, either the
[full install](install.md#full-install) or the
[CPU-only minimal install](install.md#cpu-only-minimal-install).

```bash
bash smoke_tests/wofost_gym/run_cpu_quickstart.sh
```

The script does three things:

1. builds a tiny dataset from `smoke_tests/wofost_gym/config/quickstart_maize_cpu.yaml`:
   eight training and eight validation maize scenarios sampled with
   `generation_seed: 42` from the public pool `AgriManager-Framework/weather_pool_maize`
   (the first run downloads about 0.25 GB into `.cache/wofost_gym/`);
2. rolls out the **no-action** policy and the **scripted agronomic reference**
   (`ag_heuristic` mode) on the eight validation scenarios, with the WOFOST
   `lnpkw-v0` action menu (no-op plus four levels each of N, P, K and water),
   a 10-day decision interval, 24 decisions per season and the `profit_max`
   objective;
3. prints the mean objective of each policy.

Output files go to `smoke_tests/wofost_gym/results/quickstart/<policy>/`:
`results.json` (every turn of every episode), `metrics.json` (aggregate) and
`validation_metrics.json` (split by validation group).

Expected tail of the output (run on 2026-10-24, CPU only, about 20 seconds
after the download):

```text
no_action: smoke_tests/wofost_gym/results/quickstart/no_action/metrics.json
  objective_reward_mean = 0.3341123096222971
  objective_reward_std = 0.4346545424250296
ag_heuristic: smoke_tests/wofost_gym/results/quickstart/ag_heuristic/metrics.json
  objective_reward_mean = 1.5955195245941627
  objective_reward_std = 1.5794688312679235
```

These eight scenarios are a smoke check. They are not benchmark numbers.

> **Which scripted reference.** The `ag_heuristic` mode is the *current*
> scripted reference in the code. The paper's WOFOST reference is an earlier
> version of the same per-crop policy, recovered from git when the test-suite
> anchors are computed. See
> [Baselines](../benchmark/baselines.md#which-version-the-paper-uses).

## 2. Look at what the policy sees

Each row of a built dataset is one episode scenario with its environment
config. The text a language model reads is rendered by the environment at
every decision turn. To print the system prompt and the first user turn of one
scenario (run from the repository root):

```bash
python - <<'PY'
import pandas as pd
from agrimanager.env.base import create_environment

df = pd.read_parquet("smoke_tests/wofost_gym/data/quickstart_maize_cpu/val.parquet")
env_config = df.iloc[0]["extra_info"]["interaction_kwargs"]["env_config"]
env, _ = create_environment("wofost_gym", dict(env_config))
observation, info = env.reset()
print(env.system_prompt())
print(observation)
PY
```

Shortened output:

```text
You are an agricultural management expert. Optimize the management objective
specified in the user message. Use only the available actions and respond in
the required format.

We are growing maize from sow to maturity. [...] actions are taken every 10 days.

<management objective id="profit_max">
Goal: maximize terminal WSO-equivalent profit, not raw yield.
profit_wso_kg_ha = final_WSO_kg_ha - 3.5 * total_N_kg_ha - 5.5 * total_P_kg_ha
                   - 2.25 * total_K_kg_ha - 0.5 * total_irrig_cm
[...]
</management objective>

<current observation>
[Crop status] [Soil nutrients] [Soil & water] [Cumulative actions] [Weather]
[...]
</current observation>

Available actions (pick exactly one):
- Do nothing.
- Apply nitrogen fertilizer (20.0, 40.0, 60.0, or 80.0 kg/ha).
[...]
Respond using the exact format: <answer> ... </answer> with no extra text.
```

This scenario was built with `require_think: false` (the no-think format). With
`require_think: true` the response format asks for a short reasoning block
first. The prompt blocks and how each schema shift edits them are described in
[Schema and prompts](../benchmark/schema_and_prompts.md).

## 3. Next steps

| Next | Needs | Where |
|---|---|---|
| Build the full smoke datasets and run a 5-step GRPO training | GPU, full install, the 20-crop pool (17.6 GB download) | `smoke_tests/wofost_gym/README.md` |
| Train a numeric PPO baseline on the smoke data | CPU, full install | `smoke_tests/wofost_gym/run_nn_train.sh`, then `run_nn_eval.sh` |
| Evaluate a hosted model on one scenario without a GPU | an OpenRouter key | `smoke_tests/wofost_gym/run_llm_eval_openrouter_t32_free.sh` |
| Score a released checkpoint on the held-out test suite | GPU, full install | [Evaluation](../guides/evaluation.md) |
| Run a Cycles or DSSAT setting | CycleGym (from `install.sh`) or the [DSSAT runtime](install_dssat.md) | [Environments](../benchmark/environments.md) |

*Verified for this revision: steps 1 and 2, on a CPU-only Linux machine with
the CPU-only minimal install. The commands in step 3 were not executed.*
