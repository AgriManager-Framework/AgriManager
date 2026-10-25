# W&B Logging Overview

What we log to Weights & Biases during training, how it is organized, and what to
watch while a run is live. Written as a meeting-ready reference.

Source of truth: metrics are defined in
[`agrimanager/adapter/trainer/trainer.py`](../../agrimanager/adapter/trainer/trainer.py)
and [`agrimanager/adapter/trainer/validation_metrics.py`](../../agrimanager/adapter/trainer/validation_metrics.py).
The example numbers below come from a real export of the run
`wofost_weather_regime_ood_potato_llm_think_train` (**170 metric keys**).

---

## Background: what a run looks like

- **Organization**: `entity = ` → `project = agrimanager_<experiment>`
  (e.g. `agrimanager_wofost_weather_regime_ood`) → `run name = experiment_name`
  (e.g. `wofost_weather_regime_ood_potato_llm_think_train`).
- **Foundation**: built on **VERL's PPO/GRPO logging**, on top of which we add a layer
  of **agricultural environment trajectory metrics** (`train-env/` and `val-env/`
  prefixes).
- **Cadence**: training metrics logged every step; validation every `test_freq`
  steps (10 in the paper's runs); `save_freq=100`; `log_val_generations=4` (4 sample trajectories uploaded as
  text per validation).
- **Algorithm**: GRPO (`adv_estimator=grpo`), with `per_turn_training` and
  `rollout_filter` enabled.

---

## 1. Unified structure: metrics grouped by namespace

| Namespace | What it records | Origin |
|---|---|---|
| **`training/`** | global_step, epoch, early_stop, skipped_update | Training progress |
| **`actor/`** | pg_loss, kl_loss, entropy, grad_norm, lr, pg_clipfrac, ppo_kl | Policy optimization (VERL native) |
| **`critic/`** | score / rewards / advantages / returns, each as min·mean·max | GRPO in-group reward/advantage stats (no value network) |
| **`rollout/` + `train-diag/`** | in_group_reward, filter_kept_ratio, filter_zero_count, consecutive_all_filtered_steps, advantage_abs_mean | GRPO sampling & sample-filtering diagnostics |
| **`response_length/` `prompt_length/` `num_turns/`** | min·mean·max·clip_ratio | Generation length / interaction rounds |
| **`train-env/` `val-env/`** (our core additions) | **final_wso** (final yield), **reward**, **invalid_action_rate**, invalid_steps, total_steps | Real agricultural task outcomes |
| **`perf/` `timing_s/` `timing_per_token_ms/`** | throughput, time_per_step, mfu, memory, per-stage timings | Performance / cost |

Plus two additional categories:

- **Sliced metrics (the key to the generalization story)**: `val-env-by-split/*`,
  `val-env-by-weather_regime/{id,cold,hot,drought,wet}/*`, `val-env-by-crop/*` — the
  same `val-env` metrics broken out along the **OOD axis**, so you can read the
  ID-vs-OOD gap directly.
- **`val-core/` `val-aux/`**: VERL's native validation aggregation (partly overlaps
  `val-env/`, kept mainly for compatibility).
- **Generations text table**: raw prompt/response text for sampled trajectories, used
  for qualitative spot-checks of reasoning quality.

> One-line summary of the structure:
> **"training mechanics" (actor/critic/rollout) + "task outcomes" (\*-env, incl. OOD
> slices) + "generation behavior" (length/turns) + "cost" (perf/timing) + "qualitative
> samples" (generations table).**

---

## 2. What to watch during training (by priority)

1. **Is it learning / converging** → `critic/score/mean` (= reward mean),
   `rollout/in_group_reward_mean`
2. **Is the task outcome good** → `val-env/final_wso/mean` (yield),
   `val-env/reward/mean` — the "agronomic" bottom-line score
3. **Generalization (research core)** → `val-env-by-weather_regime/*`,
   `val-env-by-split/*`, comparing the gap between ID and each OOD distribution
4. **Is behavior valid** → `val-env/invalid_action_rate`, `invalid_steps` (is the LLM
   emitting illegal actions)
5. **Is training stable** → `actor/entropy` (exploration collapse?),
   `actor/grad_norm`, `actor/ppo_kl` / `kl_loss` (drifting off?),
   `training/skipped_update` / `early_stop`
6. **GRPO sampling health** → `rollout/filter_kept_ratio`, `filter_zero_count`,
   `consecutive_all_filtered_steps` (in-group reward with no variance gets fully
   filtered = no gradient)
7. **Generation length** → `response_length/mean` (is "think" growing / collapsing),
   `clip_ratio` (is it being truncated at the 512 cap)
8. **Cost / will it finish** → `perf/throughput`, `perf/time_per_step`, `perf/mfu`
   (estimate how many steps fit in 24h — directly tied to the earlier timeout issue)

**Example** (the same potato run, which finished at step 200):
`critic/score/mean=12.06`, `train-env/final_wso=9824`, `val-env/final_wso=11004`,
`val-env/reward=10.97`, `val-env/invalid_action_rate=0`, `response_length/mean=185`,
`num_turns/mean=1`.
