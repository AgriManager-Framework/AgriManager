# Training Script Conventions

This document records the standard training-script override pattern used by
StoreV2 WOFOST experiments. It keeps `experiments/<exp>/run_*.sh` files
consistent across experiments.

Canonical examples:

- LLM GRPO: `experiments/t1_1_weather_regime_shift/run_wofost_weather_regime_ood_potato_llm_think_train.sh`
- NN PPO N1: `experiments/t1_1_weather_regime_shift/run_wofost_weather_regime_ood_potato_nn_train_n1.sh`
- NN PPO N8: `experiments/t1_1_weather_regime_shift/run_wofost_weather_regime_ood_potato_nn_train_n8.sh`

## Script Boundary

Each `run_*.sh` is one fixed, reproducible run definition.

It should hardcode: `RUN_NAME`, dataset config path, generated dataset paths,
named validation sets, model / agent settings, W&B project and experiment name,
output location, and all Hydra overrides.

It should not expose free-form experiment-defining flags such as
`--dataset-config`, `--crop`, `--run-name`, or arbitrary override passthrough.

## Shared Dataset Pattern

Every training script should derive dataset files from one dataset config:

```bash
RUN_NAME="..."
DATASET_CONFIG="$SCRIPT_DIR/config/<dataset_config>.yaml"
DATASET_ID="$(basename "$DATASET_CONFIG" .yaml)"
DATASET_DIR="$SCRIPT_DIR/data/${DATASET_ID}"
TRAIN_FILE="$DATASET_DIR/train.parquet"
```

For OOD validation, use named validation sets instead of anonymous
`data.val_files`:

```bash
VAL_SET_OVERRIDES=(
    "+data.val_sets.id=${VAL_ID_FILE}"
    "+data.val_sets.drought=${VAL_DROUGHT_FILE}"
    "+data.val_sets.wet=${VAL_WET_FILE}"
    "+data.val_sets.hot=${VAL_HOT_FILE}"
    "+data.val_sets.cold=${VAL_COLD_FILE}"
)
```

Training overrides should then include:

```bash
"data.train_files=${TRAIN_FILE}" \
"data.val_files=null" \
"data.validation_axis=weather_regime" \
"${VAL_SET_OVERRIDES[@]}"
```

Change `data.validation_axis` to match the experiment axis, such as
`crop_regime`, `action_menu`, `reward_formulation`, or `simulator`.

## Inline Dataset Build

Training scripts should build the dataset inline if required files are missing
or older than the dataset config:

```bash
DATASET_FILES=(
    "$TRAIN_FILE"
    "$VAL_ID_FILE"
    "$VAL_DROUGHT_FILE"
    "$VAL_WET_FILE"
    "$VAL_HOT_FILE"
    "$VAL_COLD_FILE"
)

NEEDS_BUILD=0
for dataset_file in "${DATASET_FILES[@]}"; do
    if [[ ! -f "$dataset_file" || "$DATASET_CONFIG" -nt "$dataset_file" ]]; then
        NEEDS_BUILD=1
        break
    fi
done

if [[ "$NEEDS_BUILD" -eq 1 ]]; then
    bash entrypoints/dataset/build.sh \
        --config "$DATASET_CONFIG" \
        --num-workers 1 \
        2>&1 | tee "$DATASET_BUILD_LOG"
fi
```

This keeps the run script self-contained while avoiding unnecessary rebuilds.

## Standard LLM GRPO Overrides

Use `entrypoints/train/train.sh` with `--config-name agri_grpo`. The default LLM
rollout backend in the entrypoint config is `vllm`; only write
`actor_rollout_ref.rollout.name=...` explicitly when a run intentionally tests a
different backend.

```bash
exec bash entrypoints/train/train.sh \
    --log-file "$LOG_FILE" \
    --config-name agri_grpo \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=null" \
    "data.validation_axis=weather_regime" \
    "${VAL_SET_OVERRIDES[@]}" \
    "data.gen_batch_size=16" \
    "data.max_response_length=512" \
    "actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507" \
    "actor_rollout_ref.rollout.n=4" \
    "actor_rollout_ref.rollout.gpu_memory_utilization=0.7" \
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32" \
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=64" \
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=64" \
    "trainer.logger=[\"console\", \"wandb\"]" \
    "trainer.project_name=<wandb_project>" \
    "trainer.default_local_dir=${RESULT_DIR}" \
    "trainer.experiment_name=${RUN_NAME}" \
    "trainer.total_epochs=2" \
    "trainer.n_gpus_per_node=2" \
    "trainer.save_freq=100" \
    "ray_kwargs.ray_init.num_cpus=32" \
    "+ray_kwargs.ray_init.include_dashboard=False" \
    "+ray_kwargs.ray_init._temp_dir=/tmp/ray_${SLURM_JOB_ID:-${SLURM_JOBID:-$$}}"
```

The standard LLM/VERL validation cadence is `trainer.test_freq=10`. Use the
default from `entrypoints/train/config/agri_grpo.yaml` or write the same value
explicitly in smoke scripts. Do not use a more frequent cadence unless the
experiment card explicitly requires it. Keep the core GRPO scale fixed unless
the experiment card explicitly defines a different budget.

## Standard NN PPO Overrides

Use `entrypoints/train/nn_train.sh` with `--config-name nn`.

```bash
bash entrypoints/train/nn_train.sh \
    --config-name nn \
    "data.train_files=${TRAIN_FILE}" \
    "data.val_files=null" \
    "data.validation_axis=weather_regime" \
    "${VAL_SET_OVERRIDES[@]}" \
    "agent.track=true" \
    "agent.wandb_project_name=<wandb_project>" \
    "agent.num_envs=32" \
    "agent.num_steps=72" \
    "agent.num_minibatches=4" \
    "agent.n_epochs=<1_or_8>" \
    "agent.checkpoint_mode=epoch" \
    "agent.learning_rate=0.00025" \
    "agent.gamma=1" \
    "agent.gae_lambda=0.99" \
    "agent.seed=1" \
    "agent.exp_name=${RUN_NAME}" \
    "runtime.vec_env=subproc" \
    "runtime.subproc_start_method=fork" \
    "runtime.sample_with_replacement=false" \
    "runtime.shard_train_scenarios_across_envs=true" \
    "runtime.train_epochs=8" \
    "runtime.resume.mode=auto_latest" \
    "runtime.validation.enabled=true" \
    "runtime.validation.val_before_train=true" \
    "runtime.validation.evals_per_epoch=4" \
    "output.save_folder=${SAVE_ROOT}" \
    2>&1 | tee "$LOG_FILE"
```

The two standard NN settings:

| Setting | Override | Meaning |
| --- | --- | --- |
| N1 | `agent.n_epochs=1` | Optimize each rollout buffer once. Default NN comparison setting. |
| N8 | `agent.n_epochs=8` | Optimize each rollout buffer eight times. Higher-optimization NN setting. |

N1 and N8 should otherwise keep the same data, validation, seed, environment
count, rollout length, learning rate, and validation cadence.

## Keep Consistent Across Runs

Change only the experimental axis under study. For example, a T2.2 action-menu
experiment should change action-menu data/configs, not model scale, validation
cadence, or objective text. Across comparable runs keep aligned:

- same dataset config family and named validation sets;
- same `data.validation_axis` (must match the paper-facing validation label);
- same train/validation split sizes;
- same W&B project;
- same seed when comparing NN N1 and N8;
- same LLM model and rollout count when comparing think/no-think;
- LLM scripts on the standard GRPO overrides, NN scripts differing only in
  `agent.n_epochs=1` versus `agent.n_epochs=8`, unless the card says otherwise.

## Pre-Launch Checklist

- dataset build happens inline when needed;
- `data.val_files=null` and `data.val_sets.*` are used for OOD validation;
- `trainer.experiment_name` or `agent.exp_name` equals `RUN_NAME`;
- logs and outputs go to run-specific locations;
- no free-form CLI passthrough changes the experiment definition.
