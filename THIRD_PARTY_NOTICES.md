# Third-party code, simulators, models and data

AgriManager's own code is licensed under Apache-2.0 (`LICENSE`). The datasets
we publish (weather pools, held-out test suites, result bundles) are licensed
under CC BY 4.0. Released model weights are Apache-2.0, except checkpoints
fine-tuned from a base model whose license requires otherwise (see the table at
the end).

AgriManager does **not** vendor the simulators or their Gym wrappers.
`install.sh` fetches each one from its own repository at a pinned commit, and
each keeps its own license. Licenses below were checked on 2026-10-25 in the
place given in the last column.

## Code fetched by `install.sh`

| Component | Used as | Version / commit | License | Where the license was read |
|---|---|---|---|---|
| VeRL | git submodule `verl/` (RL training framework) | upstream `verl-project/verl` @ `7522bef0eb5c5761500fa8652e7ed45936f5323d` (`v0.7.0-1-g7522bef0`) | Apache-2.0 | `LICENSE` and `Notice.txt` in the repository; GitHub license API |
| WOFOSTGym | Gym wrapper around WOFOST | AgriManager fork `AgriManager-Framework/WOFOSTGym` @ `1096a6d7018fd94857f30456a8403c644894f2eb` = upstream `Intelligent-Reliable-Autonomous-Systems/WOFOSTGym` @ `281f0af` plus one patch commit | MIT | `LICENSE.md` (W. H. Solow, 2024); GitHub license API |
| PCSE / WOFOST (inside WOFOSTGym, `pcse/`) | crop model | modified PCSE 5.5.6 shipped in WOFOSTGym | European Union Public Licence (EUPL) | `pcse/pcse/__init__.py` (Copyright Alterra, Wageningen-UR; `__license__ = "European Union Public License"`; "all files in pcse directory fall under this license"). Upstream `ajwdewit/pcse` `LICENSE`: EUPL v1.1 or later. WOFOSTGym's `pcse/LICENSE.md` is an MIT text; we treat the EUPL header as governing |
| `pcse_gym`, `imitation`, `rl_algs` (inside WOFOSTGym) | Gym environments, helper code | same commit | MIT | `pcse_gym/LICENSE.md`, `imitation/LICENSE.md` (W. H. Solow) |
| Stable-Baselines3 (inside WOFOSTGym, `stable-baselines3/`) | numerical-RL baselines | 2.6.0a1 copy | MIT | `stable-baselines3/LICENSE` (A. Raffin); upstream `DLR-RM/stable-baselines3` |
| CyclesGym | Gym wrapper around Cycles | AgriManager fork `AgriManager-Framework/cyclesgym` @ `c5f4ba1249c2c2bf2e3442dd717b3d9772a27fbe` = upstream `kora-labs/cyclesgym` @ `44e7dbc` plus one patch commit | BSD-3-Clause | GitHub license API for `kora-labs/cyclesgym` |
| Cycles crop simulator | native binary, downloaded by CyclesGym's `install_cycles.py` | `v0.12.9-alpha` release archives of `PSUmodeling/Cycles` | CC BY-NC-ND 4.0 | `LICENSE.md` of `PSUmodeling/Cycles` (Copyright 2023 Armen Kemanian). Not redistributed by AgriManager; non-commercial terms apply to the binary |
| OpenAI Gym | legacy API imported by CyclesGym | `gym==0.26.2` (PyPI) | MIT | PyPI metadata |

The two fork commits keep the upstream history and add one commit each. Their
trees are byte-identical to the working forks the paper used (WOFOSTGym
`071f8a61cf81c8621304139cf36ae9b2665da923`, CyclesGym
`0c0a73986ad291f27564ec3a49b865af8268b4dd`). What the patches change:

- **WOFOSTGym**: agro/crop configuration files for the additional crops of the
  cross-crop settings (16 new `env_config/agro/*_agro.yaml`, edits to 10
  existing ones, 4 crop files); PCSE weather-cache handling and NASA POWER cache
  logging; resource-blackout hooks; trajectory-level reward range and reward
  wrappers in `pcse_gym`; PPO/DQN/SAC changes in `rl_algs` (parallel PPO, VeRL
  data format, extra reward functions). About 2,400 changed lines in 59 files.
- **CyclesGym**: runtime paths configurable through `CYCLES_GYM_CYCLESGYM_ROOT`,
  `CYCLES_GYM_CYCLES_ROOT` and `CYCLES_GYM_WORKING_DIR`; per-episode crop-price
  overrides in the rewarders and pricing utilities (used by the price-regime
  setting); crop-planning and corn environment compatibility fixes;
  `CYCLES_GYM_SKIP_INSTALL` in `setup.py`. 276 changed lines in 13 files.

## Optional: DSSAT (installed separately)

| Component | Version | License | Where read |
|---|---|---|---|
| gym-DSSAT-PDI (`gitlab.inria.fr/rgautron/gym_dssat_pdi`) | commit `63f2c529e0bd339b4553beb9aa56d56af83b5e2b`, Spack package `gym-dssat` 0.0.8 | BSD-3-Clause | GitLab project license |
| DSSAT-CSM (`DSSAT/dssat-csm-os`) and its PDI build | `dssat-pdi` 4.8.0.24_2, `dssat-csm-data` v4.8.0.28 | BSD-3-Clause | GitHub license API for `DSSAT/dssat-csm-os` |

## Python packages

The pinned PyPI dependencies (`requirements/cpu.txt`, `requirements/locks/`)
are installed from PyPI and the PyTorch index under their own licenses; none is
redistributed here.

## Base models and the license of fine-tuned checkpoints

| Base model | License | Applies to |
|---|---|---|
| `Qwen/Qwen3-4B-Instruct-2507` | Apache-2.0 | all confirmatory checkpoints |
| `Qwen/Qwen3-8B`, `Qwen/Qwen3-0.6B` | Apache-2.0 | exploratory backbone comparison |
| `Qwen/Qwen2.5-3B-Instruct` | Qwen Research License (`qwen-research`) | exploratory backbone comparison; checkpoints fine-tuned from it keep that license |
| `meta-llama/Llama-3.2-3B-Instruct` | Llama 3.2 Community License | exploratory backbone comparison; checkpoints fine-tuned from it keep that license |

Licenses of the base models were read from their Hugging Face model cards.

## Data

The WOFOST weather pools are derived from NASA POWER daily weather through the
WOFOSTGym/PCSE weather provider; see the NASA POWER data access terms. Cycles
weather and soil inputs come with the CyclesGym/Cycles distributions. Details:
`docs/reference/dataset_and_generator_sources.md`.
