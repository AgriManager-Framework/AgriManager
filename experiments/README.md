# Experiments

One directory per benchmark setting. Each holds the dataset configs
(`config/`) and fixed run scripts (`run_*.sh`); running a script is the
experiment. Generated data, results and logs stay local (`data/`, `results/`,
`logs/` are ignored by git).

Setting definitions: [docs/benchmark/README.md](../docs/benchmark/README.md).
Training runs and seeds behind the paper: [docs/guides/training.md](../docs/guides/training.md).
Held-out test suite: [docs/benchmark/test_suite.md](../docs/benchmark/test_suite.md).

## The eight settings of the paper

| Tier | Setting | Directory | Paper runs (scripts) | Test-suite scoring |
|---|---|---|---|---|
| 1 | T1.1 weather-regime (chickpea, potato) | `t1_1_weather_regime_shift/` | `run_t11_{chickpea,potato}_{think,no_think}_r{2,3,4}_train.sh`; NN `run_wofost_weather_regime_ood_*_{nn,a2c,dqn}_train_*.sh` | `run_eval_t11_{chickpea,potato}_nn_e4.sh` (NN) |
| 1 | T1.2 crop-variety (4ID, 8ID, 16ID) | `t1_2_cross_crop_trait_shift/` | `run_t12_{4,8,16}id_{think,no_think}_r{2,3,4}_train.sh`; NN `run_t12_*_nn_*_train.sh` | `run_eval_t12_nn_e4.sh` (NN) |
| 1 | T1.3 price-regime (Cycles) | `t1_3_cycles_gym_price_regime/` | `run_t13_price_{think,no_think}_r{1,2,3}_train.sh`; NN `run_nn_train_price_regime_n1_s*.sh` | `run_eval_t13_nn_e4.sh` (NN) |
| 2 | T2.1 observation-schema | `t2_1_observation_schema_shift/` | `run_t21_observation_{think,no_think}_r{2,3,4}_train.sh` | — |
| 2 | T2.2 action-menu (Families A, B) | `t2_2_action_menu_shift/` | `run_t22_family_{a,b}_{think,no_think}_r{2,3,4}_train.sh`; NN `run_t22_nn_{mask,target}_*_train.sh` | `run_eval_t22_nn_{mask,target}_e4.sh` (NN) |
| 2 | T2.3 reward-formulation | `t2_3_reward_formulation_shift/` | `run_t23_{objectives,yield_only}_{think,no_think}_r{2,3,4}_train.sh` | — |
| 3 | T3.1 cross-simulator (WOFOST, DSSAT, Cycles sources) | `t3_1_cross_simulator_maize_transfer/` | `run_t31_{wofost,dssat,cycles}_{think,no_think}_r*_train.sh` | — |
| 3 | T3.2 unified policy | `t3_2_unified_prompt_conditioned_policy/` | `run_t32_unified_qwen3_{think,no_think}_r{2,3,4}_train.sh`; exploratory `run_t32_unified_llama32_*_train.sh` | — |

Language-model checkpoints of every setting are scored on the test suite with
`entrypoints/eval/e5_evaluate_checkpoint.sh` ([docs/guides/evaluation.md](../docs/guides/evaluation.md)).
The test suite itself is built by the scripts in `rebuttal_confirmatory_eval/`
(file prefix `e4_`).

Scripts named `run_*_llm_{think,no_think}_train.sh` without a replicate suffix
are the configurations of the submitted runs; the scored runs are the seeded
replicates listed above.

## Other directories

| Directory | What |
|---|---|
| `rebuttal_confirmatory_eval/` | builds, audits and uploads the held-out test suite; scores the scripted references on it (`run_e4_anchors.sh`) |
| `app1_wofost_variety_ood/` | exploratory rice variety experiment; not part of the paper |
| `support_t1_2_cross_crop_trait_selection/` | crop-selection support for T1.2 |
| `wofost_gym_datasets.md` | definitions of the three public WOFOST weather pools |
