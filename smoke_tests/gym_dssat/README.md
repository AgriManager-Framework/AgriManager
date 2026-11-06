# Gym-DSSAT Quickstart

Commands assume you are running from the repo root. Supported crops are
`maize`, `rice`, and `cotton`; smoke scripts default to `maize` if no crop is
passed.

## Setup

Build the DSSAT runtime first: follow
[docs/getting-started/install_dssat.md](../../docs/getting-started/install_dssat.md).
That page covers the Spack build, the `gym_dssat_pdi` bridge, and the
`DSSAT_GYM_PATH` / `DSSAT_PDI_BRIDGE_DIR` variables. In short, after
`bash install.sh` and the Spack build:

```bash
source .venv/bin/activate    # or: conda activate agrimanager
export DSSAT_GYM_PATH=/path/to/spack/gym-dssat-pdi          # view containing bin/run_dssat
export DSSAT_PDI_BRIDGE_DIR=/path/to/gym_dssat_pdi/gym-dssat-pdi   # only if not found automatically
```

Check the environment:

```bash
source smoke_tests/gym_dssat/_activate_spack.sh
python - <<'PY'
import os, torch
print("cuda:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
print("DSSAT_GYM_PATH:", os.environ.get("DSSAT_GYM_PATH"))
PY
```

LLM train/eval needs a GPU. Example interactive SLURM session:

```bash
srun -A <account> --partition=<gpu_partition> --gpus=1 --cpus-per-task=16 --mem=120g --time=02:00:00 --pty bash
cd /path/to/AgriManager
source .venv/bin/activate    # or: conda activate agrimanager
```

## Smoke Tests

First validate the live DSSAT runtime. This is separate from bridge import:
it imports `gym_dssat_pdi`, constructs `DSSATEnv(config)`, and runs
`env.reset()` with a timeout.

```bash
bash smoke_tests/gym_dssat/run_runtime_smoke.sh --crop maize --timeout 240
```

If this times out at `construct_start` or `reset_start`, the DSSAT runtime has
not been validated yet even if bridge import succeeded. Check
`smoke_tests/gym_dssat/logs/gym_dssat_runtime_smoke_<crop>.log` and the DSSAT
log path printed by `DSSATEnv`.

```bash
# Replace maize with rice or cotton as needed.
CROP=maize

bash smoke_tests/gym_dssat/run_runtime_smoke.sh "$CROP"
bash smoke_tests/gym_dssat/run_build_datasets.sh "$CROP"
bash smoke_tests/gym_dssat/run_llm_train.sh "$CROP"
bash smoke_tests/gym_dssat/run_llm_eval_qwen3_4b_instruct.sh "$CROP"
```

Run all three crops manually:

```bash
bash smoke_tests/gym_dssat/run_build_datasets.sh maize
bash smoke_tests/gym_dssat/run_llm_train.sh maize
bash smoke_tests/gym_dssat/run_llm_eval_qwen3_4b_instruct.sh maize

bash smoke_tests/gym_dssat/run_build_datasets.sh rice
bash smoke_tests/gym_dssat/run_llm_train.sh rice
bash smoke_tests/gym_dssat/run_llm_eval_qwen3_4b_instruct.sh rice

bash smoke_tests/gym_dssat/run_build_datasets.sh cotton
bash smoke_tests/gym_dssat/run_llm_train.sh cotton
bash smoke_tests/gym_dssat/run_llm_eval_qwen3_4b_instruct.sh cotton
```

## Outputs

Smoke outputs: `smoke_tests/gym_dssat/{data,logs,results}/`

## Common Fixes

Ray socket path too long:

```bash
export RAY_LOG_DIR=$HOME/ray
```

LLM eval says required arguments are missing: run the wrapper, for example:

```bash
bash smoke_tests/gym_dssat/run_llm_eval_qwen3_4b_instruct.sh maize
```
