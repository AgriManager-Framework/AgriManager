# WOFOST-Gym Smoke Test

This is the canonical environment-level smoke test for WOFOST-Gym.

It exercises the WOFOST-Gym entrypoints through fixed `run_*.sh` scripts and
keeps everything local to this smoke-test directory:

- dataset configs in `config/`
- generated parquet artifacts in `data/`
- run logs in `logs/`
- model outputs and evaluation results in `results/`

## Entry Points Covered

- `entrypoints/dataset/build.sh`
- `entrypoints/train/train.sh`
- `entrypoints/eval/eval.sh`
- `entrypoints/train/nn_train.sh`
- `entrypoints/eval/nn_eval.sh`

## Fixed Run Scripts

- `run_cpu_quickstart.sh`
- `run_build_datasets.sh`
- `run_llm_train.sh`
- `run_llm_eval_qwen25_3b_instruct.sh`
- `run_llm_eval_openrouter_t32_free.sh`
- `run_nn_train.sh`
- `run_nn_eval.sh`

Each script is a complete smoke-test definition. It does not accept external
overrides. Running the script is the experiment.

## CPU Quickstart

`run_cpu_quickstart.sh` needs no GPU and no model. It builds eight maize
scenarios from the small public maize pool
(`config/quickstart_maize_cpu.yaml`) and scores the no-action policy and the
scripted agronomic reference on them:

```bash
bash smoke_tests/wofost_gym/run_cpu_quickstart.sh
```

See [docs/getting-started/quickstart.md](../../docs/getting-started/quickstart.md)
for the expected output.

## Typical Usage

The scripts below use the 20-crop weather pool (a 17.6 GB download on first
use).

Build both dataset variants:

```bash
bash smoke_tests/wofost_gym/run_build_datasets.sh
```

Run the LLM training smoke on a GPU node:

```bash
bash smoke_tests/wofost_gym/run_llm_train.sh
```

This smoke run performs one validation pass before training, runs 5 training
steps, and triggers a final validation pass on the last step.

Run the LLM evaluation smoke on a GPU node:

```bash
bash smoke_tests/wofost_gym/run_llm_eval_qwen25_3b_instruct.sh
```

Run one checked-in T3.2 held-out validation trajectory through OpenRouter
without a local GPU:

```bash
conda run -n agrimanager \
  bash smoke_tests/wofost_gym/run_llm_eval_openrouter_t32_free.sh
```

This uses the pinned free model `nvidia/nemotron-3-nano-30b-a3b:free` and
`data.max_samples=1`. It verifies the full
dataset-to-environment-to-model-to-validation-metrics path. Treat it as a
connectivity smoke result rather than a benchmark estimate because it contains
only one validation trajectory.

Run framework-native NN training and evaluation:

```bash
bash smoke_tests/wofost_gym/run_nn_train.sh
bash smoke_tests/wofost_gym/run_nn_eval.sh
```
