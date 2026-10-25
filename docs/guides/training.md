# Training

AgriManager trains two kinds of policy: prompt-conditioned language-model
policies with GRPO on VeRL, and numeric neural baselines with
Stable-Baselines3. Every paper run is a fixed script under
`experiments/<setting>/`; running the script is the experiment.

## Language-model policies (GRPO)

Entry point: `entrypoints/train/train.sh` (Hydra `key=value` overrides on
`entrypoints/train/config/agri_grpo.yaml`). Needs one or more CUDA GPUs and the
full install.

Paper defaults (the replicate scripts set them explicitly):

| Item | Value |
|---|---|
| Backbone | `Qwen/Qwen3-4B-Instruct-2507` (T3.2 also has exploratory `Llama-3.2-3B-Instruct` runs) |
| Algorithm | GRPO (`algorithm.adv_estimator=grpo`), no KL loss, actor learning rate 1e-6 |
| Rollouts | 4 per prompt, sampling temperature 1 |
| Training data | 1,600 scenarios per run (3,200 for T3.2) |
| Length | 200 training steps (`trainer.total_training_steps`); 400 for T2.3 and 800 for T3.2 |
| Validation | before training and every 10 steps on the named validation groups |
| Checkpoints | saved every 100 steps; each published repository keeps the last one or two as `step-N` branches (T3.2: steps 200, 400, 600, 800); the **terminal** checkpoint is the one scored |
| Response format | think or no-think, fixed per arm ([Schema and prompts](../benchmark/schema_and_prompts.md#response-formats-think-and-no-think)) |
| Early stop | a rollout filter stops a run after 5 consecutive steps in which every group was filtered out; this ended several no-think runs between steps 12 and 17 |

Per-setting overrides (reward, horizon, batch sizes) are in each setting's run
scripts and in the paper appendix.

### Paper runs and seeds

Each arm has three training seeds. The think and no-think arms of a replicate
share the seed, so their paired difference is not confounded with seed
variation. A seed sets both `data.seed` (training-data order) and
`actor_rollout_ref.actor.data_loader_seed` (minibatch order).

| Setting | Scripts (`experiments/...`) | Replicates scored | Seeds |
|---|---|---|---|
| T1.1 weather-regime (chickpea, potato) | `t1_1_weather_regime_shift/run_t11_{crop}_{think,no_think}_r{2,3,4}_train.sh` | r2, r3, r4 | 51, 52, 53 |
| T1.2 crop-variety (4ID, 8ID, 16ID) | `t1_2_cross_crop_trait_shift/run_t12_{4,8,16}id_{think,no_think}_r{2,3,4}_train.sh` | r2, r3, r4 | 81, 82, 83 |
| T1.3 price-regime | `t1_3_cycles_gym_price_regime/run_t13_price_{think,no_think}_r{1,2,3}_train.sh` | r1, r2, r3 | 110, 111, 112 |
| T2.1 observation-schema | `t2_1_observation_schema_shift/run_t21_observation_{think,no_think}_r{2,3,4}_train.sh` | r2, r3, r4 | 61, 62, 63 |
| T2.2 action-menu (Families A, B) | `t2_2_action_menu_shift/run_t22_family_{a,b}_{think,no_think}_r{2,3,4}_train.sh` | r2, r3, r4 | 71, 72, 73 |
| T2.3 reward-formulation (multi-objective, yield-only) | `t2_3_reward_formulation_shift/run_t23_{objectives,yield_only}_{think,no_think}_r{2,3,4}_train.sh` | r2, r3, r4 | 91, 92, 93 |
| T3.1 cross-simulator (WOFOST, DSSAT, Cycles source) | `t3_1_cross_simulator_maize_transfer/run_t31_{wofost,dssat,cycles}_{think,no_think}_r*_train.sh` | r2, r3, r4; Cycles think r2, r3, r5 | 71, 72, 73; Cycles think 71, 72, 74 |
| T3.2 unified policy (Qwen) | `t3_2_unified_prompt_conditioned_policy/run_t32_unified_qwen3_{think,no_think}_r{2,3,4}_train.sh` | r2, r3, r4 | 43, 44, 45 |

Notes:

- The submitted runs (r1) were trained without a fixed `data.seed`, and most of
  their checkpoints no longer exist, so the scored seeds are r2 to r4. T1.3 was
  retrained from scratch with three seeded runs.
- T3.1 Cycles think r4 (seed 73) never produced a checkpoint: it aborted on a
  fault in vLLM's FlashAttention-3 Hopper kernel. Its replacement r5 (seed 74)
  ran with FlashAttention 2 (`VLLM_FLASH_ATTN_VERSION=2`); the deviation is
  recorded in its script.
- The `run_*_llm_{think,no_think}_train.sh` scripts without a replicate suffix
  are the submitted reference configurations.
- The exploratory T3.2 backbone runs (one seed per arm) have their own
  launchers: `run_t32_unified_llama32_{think,no_think}_train.sh` for
  Llama-3.2-3B, and `run_t32_backbone_sweep_train.sh --algo --think --model`
  for the Qwen3-4B DrGRPO and PPO pairs, the Qwen2.5-3B pair and Qwen3-8B; its
  header lists the arguments, GPU count and save interval of each published
  `t32-diffmodels-*` checkpoint.

### Your own run

```bash
bash entrypoints/train/train.sh \
  data.train_files=experiments/my_exp/data/my_dataset/train.parquet \
  data.val_files=experiments/my_exp/data/my_dataset/val.parquet \
  actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
  trainer.experiment_name=my_run \
  trainer.logger='["console"]'
```

Build the dataset first with `entrypoints/dataset/build.sh --config <yaml>`
([dataset_config_conventions.md](../conventions/dataset_config_conventions.md)).
Override conventions for run scripts are in
[training_script_conventions.md](../conventions/training_script_conventions.md).

## Neural baselines

Entry point: `entrypoints/train/nn_train.sh --config-name nn` (Hydra overrides
on `entrypoints/train/config/nn.yaml`). Runs on CPU.

| Item | Value |
|---|---|
| Algorithm | `agent.type`: `PPO` (default), `MaskablePPO`, `A2C`, `DQN` |
| Policy | Stable-Baselines3 `MlpPolicy`, two hidden layers of 64, tanh |
| Gradient passes per batch | `agent.n_epochs=1` (`n1`, main comparison) or `8` (`n8`, appendix) |
| Epochs over the training set | `runtime.train_epochs=8` for 1,600-scenario splits (16 for 3,200) |
| Seeds | `agent.seed` 1, 2, 3 |
| Checkpoint | `agent.checkpoint_mode=epoch`; the terminal `agent.zip` is scored |

Environment behaviour comes from the dataset parquet, so change environment
parameters in the dataset config and rebuild; keep NN overrides for training
settings. Paper scripts are listed in [Baselines](../benchmark/baselines.md#neural-baselines).

## Publishing a checkpoint

`entrypoints/tools/publish_checkpoint.sh` merges a sharded checkpoint into
Transformers layout, strips host paths from `config.json`, writes
`run_manifest.json` (provenance, seed, response format, checksums), refuses to
upload if any text would leak an identity, and creates the repository private.
Run it with `--no-upload` first. Usage is in `entrypoints/README.md`.

*Checked on 2026-10-25 from a fresh `install.sh --gpu` on one A40: the "Your own
run" command with the CPU-quickstart maize dataset
(`smoke_tests/wofost_gym/config/quickstart_maize_cpu.yaml`), `Qwen/Qwen2.5-0.5B-Instruct`
and the batch settings of `smoke_tests/wofost_gym/run_llm_train.sh` ran validation,
five GRPO steps (about 18 s each, non-zero policy gradients in steps 1 to 3) and the
final validation. The paper-scale run scripts were not re-run; their defaults were
read from the configs and scripts.*
