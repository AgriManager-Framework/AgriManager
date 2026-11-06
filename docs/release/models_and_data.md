# Models and data

All released artifacts live on Hugging Face under the `agrimanager`
organization. `results/scoring_manifest.tsv` lists every checkpoint revision
scored for the paper's tables.

## By setting

Each benchmark setting with its Hugging Face collection, W&B training runs and test-suite folder.

| Tier | Setting | Models | Training curves | Test set |
|---|---|---|---|---|
| Tier 1 · Within Schema | T1.1 Weather regime | [Models](https://huggingface.co/collections/AgriManager-Framework/t11-weather-regime-6ade3b9d93dd1f22d1a1a49b) | [Training curves](https://wandb.ai/agrimanager/t1-1-weather-regime) | [Test set](https://huggingface.co/datasets/AgriManager-Framework/agrimanager-test-suite/tree/main) |
|  | T1.2 Crop variety | [Models](https://huggingface.co/collections/AgriManager-Framework/t12-crop-variety-6ade3b9fffe165f18441da59) | [Training curves](https://wandb.ai/agrimanager/t1-2-crop-variety) | [Test set](https://huggingface.co/datasets/AgriManager-Framework/agrimanager-test-suite/tree/main/e4_t12_cross_crop) |
|  | T1.3 Price regime | [Models](https://huggingface.co/collections/AgriManager-Framework/t13-price-regime-6ade3ba1f90654ac2bd63ca7) | [Training curves](https://wandb.ai/agrimanager/t1-3-price-regime) | [Test set](https://huggingface.co/datasets/AgriManager-Framework/agrimanager-test-suite/tree/main/e4_t13_price_regime) |
| Tier 2 · Single-Axis Schema | T2.1 Observation schema | [Models](https://huggingface.co/collections/AgriManager-Framework/t21-observation-schema-6ade3ba28e3e2ca978d0ba6d) | [Training curves](https://wandb.ai/agrimanager/t2-1-observation-schema) | [Test set](https://huggingface.co/datasets/AgriManager-Framework/agrimanager-test-suite/tree/main/e4_t21_observation_schema) |
|  | T2.2 Action menu | [Models](https://huggingface.co/collections/AgriManager-Framework/t22-action-menu-6ade3ba2d49bbdd8dd7ef3ba) | [Training curves](https://wandb.ai/agrimanager/t2-2-action-menu) | [Test set](https://huggingface.co/datasets/AgriManager-Framework/agrimanager-test-suite/tree/main/e4_t22_action_menu) |
|  | T2.3 Objective | [Models](https://huggingface.co/collections/AgriManager-Framework/t23-objective-6ade3ba4f90654ac2bd63df8) | [Training curves](https://wandb.ai/agrimanager/t2-3-objective) | [Test set](https://huggingface.co/datasets/AgriManager-Framework/agrimanager-test-suite/tree/main/e4_t23_reward_formulation) |
| Tier 3 · Compositional Schema | T3.1 Cross-simulator transfer | [Models](https://huggingface.co/collections/AgriManager-Framework/t31-cross-simulator-transfer-6ade3ba5b04b81f86ad2e4c7) | [Training curves](https://wandb.ai/agrimanager/t3-1-cross-simulator) | [Test set](https://huggingface.co/datasets/AgriManager-Framework/agrimanager-test-suite/tree/main) |
|  | T3.2 Unified policy | [Models](https://huggingface.co/collections/AgriManager-Framework/t32-unified-policy-6ade3ba7acc7c9c710e0bb0b) | [Training curves](https://wandb.ai/agrimanager/t3-2-unified-policy) | [Test set](https://huggingface.co/datasets/AgriManager-Framework/agrimanager-test-suite/tree/main) |
| Exploratory | Other backbones and RL algorithms | [Models](https://huggingface.co/collections/AgriManager-Framework/exploratory-backbones-and-rl-algorithms-6ade3ba8d49bbdd8dd7ef45f) | [Training curves](https://wandb.ai/agrimanager/exploratory-backbones-algorithms) | — |

## Data

| Artifact | Repository | Content | License | Status |
|---|---|---|---|---|
| 20-crop WOFOST weather pool | `AgriManager-Framework/wofost-weather-pool` | (crop, year, latitude, longitude) scenarios for 20 crops, `train` 3,200 / `val` 128 / `test` 512 per crop, plus the NASA POWER weather cache (17.6 GB archive) | CC BY 4.0 | public, Croissant metadata |
| WOFOST weather-regime pool | `AgriManager-Framework/wofost-weather-regime-pool` | chickpea and potato scenarios with weather-regime groups, weather cache | CC BY 4.0 | public |
| WOFOST maize pool | `AgriManager-Framework/weather_pool_maize` | maize scenarios, `train` 3,200 / `val` 640, weather cache | CC BY 4.0 | public |
| Held-out test suite | `AgriManager-Framework/agrimanager-test-suite` | all frozen test splits with manifests, checksums and audits | CC BY 4.0 | to be published; uploaded with `experiments/rebuttal_confirmatory_eval/upload_eval_suites.py`, which includes all suites ([Test suite](../benchmark/test_suite.md#manifests-checksums-and-audits)) |
| Per-run result bundles | 11 repositories `AgriManager-Framework/schema-results-*` (t11-chickpea, t11-potato, t12-crosscrop, t13-price, t21-observation, t22-llm, t23-reward, t31, t32-qwen, t32-llama32, t32-diffmodels) | per-scenario outputs and `eval_manifest.json` for every scored checkpoint | CC BY 4.0 | to be published with the paper ([Reproducing](../guides/reproduce.md)) |
| Aggregate tables | this repository, `results/tables/` | the CSV tables behind every paper number | CC BY 4.0 (data) | in the repository |

Cycles and DSSAT scenarios are generated on the fly from configs and seeds;
no static dataset is claimed for them beyond the test-suite splits.

Card updates still needed on the public pools: the `weather_pool_maize` card
says there is no separate test split (add a pointer to the held-out test
suite), and the `wofost-weather-regime-pool` card's regime rule does not
reproduce its labels exactly (add the disclosure from
[Test suite](../benchmark/test_suite.md#how-each-part-was-built)).

## Language-model checkpoints

Every checkpoint is a full Transformers model fine-tuned from
`Qwen/Qwen3-4B-Instruct-2507` with GRPO, one repository per training run, one
branch per saved step (`step-N`). Each repository carries a
`run_manifest.json` with the base model, seed, response format, setting, step
and file checksums. Repository names follow
`AgriManager-Framework/schema-<setting>-<variant>-<arm>-r<replicate>`.

| Setting | Repositories | Scored replicates | Terminal revisions (think) | Terminal revisions (no-think) |
|---|---|---|---|---|
| T1.1 chickpea | `schema-t11-chickpea-{think,no-think}-r{2,3,4}` | r2–r4 | step-200 ×3 | step-18, step-60, step-124 |
| T1.1 potato | `schema-t11-potato-{think,no-think}-r{2,3,4}` | r2–r4 | step-200 ×3 | step-14, step-17, step-27 |
| T1.2 4ID | `schema-t12-crosscrop-4id-…` | r2–r4 | step-200 ×3 | step-18, step-33, step-13 |
| T1.2 8ID | `schema-t12-crosscrop-8id-…` | r2–r4 | step-200 ×3 | step-13, step-13, step-134 |
| T1.2 16ID | `schema-t12-crosscrop-16id-…` | r2–r4 | step-200 ×3 | step-21, step-160, step-19 |
| T1.3 | `schema-t13-price-…` | r1–r3 | step-141, step-154, step-191 | step-54, step-43, step-110 |
| T2.1 | `schema-t21-observation-…` | r2–r4 | step-200 ×3 | step-116, step-88, step-92 |
| T2.2 Family A | `schema-t22-family-a-…` | r2–r4 | step-200 ×3 | step-99, step-161, step-141 |
| T2.2 Family B | `schema-t22-family-b-…` | r2–r4 | step-200 ×3 | step-66, step-149, step-187 |
| T2.3 multi-objective | `schema-t23-objectives-…` | r2–r4 | step-400 ×3 | step-400 ×3 |
| T2.3 yield-only | `schema-t23-yield-only-…` | r2–r4 | step-400 ×3 | step-400 ×3 |
| T3.1 WOFOST source | `schema-t31-wofost-…` | r2–r4 | step-200 ×3 | step-200, step-180, step-200 |
| T3.1 DSSAT source | `schema-t31-dssat-…` | r2–r4 | step-200 ×3 | step-114, step-76, step-78 |
| T3.1 Cycles source | `schema-t31-cycles-…` | think r2, r3, r5; no-think r2–r4 | step-200 ×3 | step-75, step-92, step-65 |
| T3.2 unified (Qwen) | `schema-t32-qwen-{think,no-think}-r{2,3,4}` | r2–r4 | step-800 ×3 | step-800 ×3 |

Terminal revisions were read from the repositories' `step-N` branches on
2026-10-24, in replicate order. A terminal step below the planned length means
the run was stopped early by the rollout filter; that step is the run's final
policy and the one scored.

Exploratory checkpoints (one seed each, reported as exploratory):

| Repositories | What |
|---|---|
| `Llama-3.2-3B-AgriManager-t32-unified-{think,no-think}` (training run names `schema-t32-llama32-*-s1`) | T3.2 unified policy on `Llama-3.2-3B-Instruct` |
| `t32-diffmodels-*` (7 repositories) | T3.2 with other backbones and algorithms: Qwen2.5-3B, Qwen3-8B; GRPO, Dr. GRPO, PPO (the Qwen3-0.6B run is not reported in the paper and is not released) |

Licenses: checkpoints fine-tuned from Qwen3 models are released under
Apache-2.0. `Llama-3.2-3B-AgriManager-t32-unified-*` follow the Llama 3.2 Community License, and
`t32-diffmodels-qwen25-3b-*` follow the Qwen Research License of
`Qwen/Qwen2.5-3B-Instruct`.

The model card template for these repositories is
[model_card_template.md](model_card_template.md).

## Neural-baseline checkpoints

The terminal `agent.zip` of every neural baseline run (PPO `n1` and `n8`, DQN,
A2C, MaskablePPO, target-trained PPO) is kept with its config and is planned
for upload with the release under `AgriManager-Framework/` (`TODO(nn-checkpoints)`).

## Licenses of what the checkpoints are built on

| Component | License |
|---|---|
| Qwen3-4B-Instruct-2507 (base model) | Apache 2.0 (upstream model card) |
| Qwen3-8B, Qwen3-0.6B (exploratory bases) | Apache 2.0 |
| Qwen2.5-3B-Instruct (exploratory base) | Qwen Research License |
| Llama-3.2-3B-Instruct (exploratory base) | Llama 3.2 Community License |
| Training and evaluation code | Apache-2.0 |

Simulator and wrapper licenses are listed in
[THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md).
