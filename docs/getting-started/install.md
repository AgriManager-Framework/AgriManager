# Installation

AgriManager runs on Linux with Python 3.12. The full install covers model
training (VeRL with vLLM), WOFOST-Gym and CycleGym. DSSAT is a separate native
runtime with its own guide.

## What you need

| Task | Hardware and runtime | Extra install |
|---|---|---|
| Read the docs and configs | browser | none |
| CPU quickstart: build WOFOST scenarios, score scripted policies | CPU, about 1 GB disk for the maize pool | `install.sh --cpu` |
| Score a scripted or numeric (NN) baseline on the test suite | CPU | `install.sh --cpu` (DSSAT settings: also the DSSAT runtime) |
| Score a language-model checkpoint on the test suite | one CUDA GPU (the paper used A100, A40 and H200 class GPUs) | `install.sh --gpu` |
| Train a language-model policy (GRPO) | one or more CUDA GPUs; paper runs used multi-GPU nodes | `install.sh --gpu` |
| Train a numeric baseline (PPO, MaskablePPO, A2C, DQN) | CPU node | `install.sh --cpu` |
| Any Cycles setting (T1.3, Cycles parts of T3.1 and T3.2) | CPU for the simulator | CycleGym, installed by `install.sh` |
| Any DSSAT setting (DSSAT parts of T3.1 and T3.2) | Linux, GCC 11 with Fortran (or let Spack build it), a Spack path of at most 37 characters; 20 to 40 minutes to build on 16 cores, plus about 15 if Spack builds GCC | [DSSAT runtime](install_dssat.md) |

Disk for the WOFOST weather pools (downloaded on first use into
`.cache/wofost_gym/` at the repository root, measured 2026-10-24):

| Pool | Used by | Download | On disk after extraction |
|---|---|---:|---:|
| `AgriManager-Framework/weather_pool_maize` | T2.1, T2.2, T2.3, WOFOST parts of T3.1 and T3.2, quickstart | 0.25 GB | about 0.95 GB |
| `AgriManager-Framework/wofost-weather-regime-pool` | T1.1 | 0.29 GB | not measured |
| `AgriManager-Framework/wofost-weather-pool` (20 crops) | T1.2, the WOFOST smoke tests | 17.6 GB | about 65 GB (archive plus extracted cache) |

If the repository sits on a small disk, make `.cache` a symbolic link to a
larger one before the first run.

## Install

1. Clone the repository with its VeRL submodule.

   ```bash
   git clone --recursive https://github.com/AgriManager-Framework/AgriManager.git
   cd AgriManager
   ```

2. Create a clean Python 3.12 environment, with Conda or with uv.

   ```bash
   conda create -n agrimanager python=3.12 -y && conda activate agrimanager
   # or
   uv venv --seed -p 3.12 .venv && source .venv/bin/activate
   ```

3. Run the installer in one of two modes.

   ```bash
   bash install.sh --cpu   # no GPU: WOFOST + Cycles, scripted and numeric baselines, tests
   bash install.sh --gpu   # adds CUDA 12.8 PyTorch, vLLM and FlashAttention for GRPO and model scoring
   ```

Both modes:

- initialize the `verl` git submodule (upstream VeRL `v0.7.0-1`, commit
  `7522bef0`) and install it in editable mode;
- clone WOFOST-Gym into `../AgriManagerExternal/WOFOSTGym` from
  `github.com/AgriManager-Framework/WOFOSTGym` at commit `253cf546` and install
  its `pcse`, `pcse_gym`, `imitation` and `stable-baselines3` packages;
- clone CycleGym into `../AgriManagerExternal/CyclesGym` from
  `github.com/AgriManager-Framework/cyclesgym` at commit `2eb87583`, install it
  next to the legacy `gym==0.26.2` it imports, run its native bootstrap
  (`install_cycles.py`, which downloads the Cycles `v0.12.9-alpha` binary) and
  check that the binary and input files exist;
- install AgriManager in editable mode and check the imports.

The two simulator forks are upstream WOFOSTGym and CyclesGym plus one patch
commit each; [THIRD_PARTY_NOTICES.md](../../THIRD_PARTY_NOTICES.md) lists what
the patches change and the license of every component.

The package sets differ by mode:

| Mode | Python packages |
|---|---|
| `--cpu` | `requirements/cpu.txt` (PyTorch 2.8.0 CPU build, Ray 2.53.0, Transformers 4.56.1, Hydra 1.3.2, ...), constrained by the frozen `requirements/locks/agrimanager-py312-cpu.lock.txt` |
| `--gpu` | the locked set `requirements/locks/agrimanager-py312-cu128.lock.txt` (CUDA 12.8, PyTorch 2.8.0, vLLM 0.11.0, Ray 2.53.0, Transformers 4.56.1) and a pinned FlashAttention 2.8.1 wheel |

Options: `--external-dir DIR` clones the simulator wrappers elsewhere (then
export `WOFOST_GYM_PATH` and `CYCLES_GYM_PATH`), `--skip-cycles` leaves out
CycleGym. Set `AGRI_USE_LOCKFILE=0` to let pip resolve versions (results may
then differ). An existing checkout in the target directory is reused and moved
to the pinned commit.

On a shared cluster, create the environment on a fast file system. On a
hard-disk-backed parallel file system we measured `import vllm` at about ten
minutes and `install.sh --gpu` at 45 minutes; on an NVMe-backed one the same
install took ten minutes and `import vllm` four seconds.

## DSSAT

DSSAT-backed settings need the DSSAT-Gym runtime built with Spack. It is not
part of `install.sh`. Follow [install_dssat.md](install_dssat.md).

## Environment variables

Most users need none of these. They override defaults or enable a provider.

| Variable | When | Purpose |
|---|---|---|
| `WOFOST_GYM_REMOTE`, `WOFOST_GYM_REF` | install | Git remote and commit for WOFOST-Gym |
| `CYCLES_GYM_REMOTE`, `CYCLES_GYM_REF` | install | The same for CycleGym |
| `AGRI_EXTERNAL_DIR` | install | Same as `--external-dir` |
| `AGRI_USE_LOCKFILE`, `AGRI_PIP_CONSTRAINT` | install | Skip the lock file, or use another constraint file |
| `WOFOST_GYM_PATH` | runtime | WOFOST-Gym checkout to use; default `../AgriManagerExternal/WOFOSTGym`. A `wofost_gym_path` in a dataset config takes precedence |
| `CYCLES_GYM_PATH` | runtime | CycleGym checkout to use; default `../AgriManagerExternal/CyclesGym` |
| `DSSAT_GYM_PATH` | runtime, DSSAT only | Root of the DSSAT-Gym runtime (the Spack view that contains `bin/run_dssat`); default `spack/gym-dssat-pdi` in the checkout |
| `DSSAT_PDI_BRIDGE_DIR` | runtime, DSSAT only | Checkout of the `gym_dssat_pdi` Python bridge when it is not found automatically |
| `HF_TOKEN` | publishing only | Needed to upload checkpoints or result bundles; reading public repositories needs no token |
| `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY` | hosted models only | API keys for hosted-model evaluation |
| `WANDB_MODE=offline` or `wandb.enabled=false` | any run | Keep experiment tracking local or off |
| `VLLM_USE_FLASHINFER_SAMPLER` | GPU runs | The train and eval entry points set it to `0`, so vLLM does not JIT-compile FlashInfer sampling kernels (that needs a CUDA toolkit with `nvcc` on the GPU node). Set `1` to use FlashInfer |

## Verified install

On 2026-10-25 the CPU path was run from a fresh one-commit clone in a fresh
Python 3.12 venv (uv) on Linux x86_64, with an empty pip cache and no GPU:

```bash
uv venv --seed -p 3.12 venv && source venv/bin/activate
bash install.sh --cpu                              # about 2 minutes
bash smoke_tests/wofost_gym/run_cpu_quickstart.sh  # about 45 s incl. the 0.25 GB maize pool download
python -m pytest -q tests                          # 194 passed in about 17 s
python smoke_tests/cycles_gym/pollution_reward_smoke.py   # Cycles simulator check, about 8 s
```

The quickstart printed `objective_reward_mean` 0.334 for no-action and 1.596
for the scripted reference (8 smoke scenarios; not benchmark numbers). The
resulting environment is frozen in `requirements/locks/agrimanager-py312-cpu.lock.txt`
(PyTorch 2.8.0+cpu, Transformers 4.56.1, Ray 2.53.0, Gymnasium 1.0.0,
NumPy 2.2.0). The DSSAT runtime was not re-run for this revision.

The GPU path was run the same way on 2026-10-25 inside a one-GPU job on a
Linux cluster (NVIDIA A40, driver 595, no CUDA toolkit or modules loaded, uv
venv, empty pip cache): `bash install.sh --gpu` finished in ten minutes on an
NVMe-backed file system; `pytest` passed (194), the CPU quickstart printed the
numbers above, `pip freeze` matched `requirements/locks/agrimanager-py312-cu128.lock.txt`
apart from `gym==0.26.2` (installed by `install.sh` for CycleGym), and checkpoint
scoring and a short GRPO run worked ([Evaluation](../guides/evaluation.md),
[Training](../guides/training.md)).

## Check the install

```bash
python -c "import agrimanager, pcse_gym, cyclesgym; print('ok')"
bash smoke_tests/wofost_gym/run_cpu_quickstart.sh
python -m pytest -q tests
```

For the per-simulator smoke tests (dataset build, short GRPO training, model
evaluation, NN training) see `smoke_tests/wofost_gym/README.md`,
`smoke_tests/cycles_gym/README.md` and `smoke_tests/gym_dssat/README.md`.
