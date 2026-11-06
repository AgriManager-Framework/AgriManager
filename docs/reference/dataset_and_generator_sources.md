# Upstream sources, versions and licenses

This page records where AgriManager's data and simulators come from, which
versions the paper used, and their licenses. It separates the three hosted
WOFOST weather-pool datasets from the Cycles and DSSAT simulators, which
generate scenarios from configs at run time. Citations are in
[upstream_citations.md](upstream_citations.md).

## Hosted WOFOST Datasets

These three Hugging Face datasets are the static dataset artifacts used by the
WOFOST experiments:

| Dataset | Main code use | Materials on the hub |
| --- | --- | --- |
| `AgriManager-Framework/wofost-weather-regime-pool` | T1.1 weather-regime shift for `chickpea` and `potato`. | Dataset card, parquet files, `meteo_cache.tar.gz`, Croissant JSON with RAI fields. |
| `AgriManager-Framework/wofost-weather-pool` | T1.2 cross-crop trait shift and WOFOST crop-support experiments. | Dataset card, parquet files, `meteo_cache.tar.gz`, Croissant JSON with RAI fields. |
| `AgriManager-Framework/weather_pool_maize` | T2/T3 maize WOFOST schema, action, reward, and cross-simulator experiments. | Dataset card, parquet files, `meteo_cache.tar.gz`, Croissant JSON with RAI fields. |

The hosted WOFOST rows are simulator-derived benchmark scenarios. The
meteorological inputs come from NASA POWER via the WOFOST-Gym/PCSE weather data
provider, but the released rows are curated and sampled by a deterministic
WOFOST-based pipeline. For Croissant RAI metadata, these datasets should set
`rai:hasSyntheticData` to `true` and describe the deterministic simulator-derived
scenario construction.

The runtime cache is part of the dataset artifact because it is needed to run
the exact same weather scenarios without re-querying NASA POWER. AgriManager's
WOFOST loader downloads parquet files and `meteo_cache.tar.gz`, extracts the
cache, and passes the extracted cache directory into the WOFOST environment.

Code references:

| Purpose | File |
| --- | --- |
| WOFOST weather-pool download/extraction/loading | `agrimanager/env/wofost_gym/weather_pool.py` |
| WOFOST artifact-first dataset builder | `agrimanager/env/wofost_gym/create_dataset.py` |
| Weather-pool generation configs | `integrations/wofost_gym/dataset_tools/weather_pool_configs/pool_crop_*.yaml` |
| Weather-pool build script | `integrations/wofost_gym/dataset_tools/build_weather_pool.py` |
| Weather-cache packaging | `integrations/wofost_gym/dataset_tools/package_weather_pool.py` |
| Hugging Face upload helper | `integrations/wofost_gym/dataset_tools/upload_weather_pool.py` |
| WOFOST dataset contract | `docs/reference/wofost_dataset_contract.md` |
| WOFOST pool definitions | `experiments/wofost_gym_datasets.md` |

## WOFOST Pool Construction Summary

The 20-crop weather pool is generated from per-crop configs under
`integrations/wofost_gym/dataset_tools/weather_pool_configs/`. Each crop config
uses `generation_seed: 42`, `year_range: [1984, 2019]`, and
`min_dvs_threshold: 1.5`. The generated split sizes are `train=3200`,
`val=128`, and `test=512` per crop. A candidate scenario is a
`(crop, year, latitude, longitude)` tuple and is retained only if a full
no-action WOFOST rollout has complete weather, reaches `max_DVS >= 1.5`, and
produces `max_WSO > 0`.

The weather-regime pool is a materialized benchmark source for `chickpea` and
`potato`. Its named validation regimes are crop-specific 20 percent buckets:
`drought` is the lowest 20 percent by growing-season cumulative `RAIN`, `wet`
is the highest 20 percent by growing-season cumulative `RAIN`, `cold` is the
lowest 20 percent by growing-season mean daily `TEMP`, and `hot` is the highest
20 percent by growing-season mean daily `TEMP`. The `normal` group excludes all
four named extremes. This rule does not reproduce the pool's labels exactly;
the held-out test suite therefore regroups unused scenarios with an explicit
rule (`docs/benchmark/test_suite.md`).

The maize weather pool is the maize-only WOFOST pool used by maize schema,
action, reward, and cross-simulator experiments. It uses the same WOFOST
viability filter as the 20-crop pool. Its source has `train=3200` and
`val=640`. The benchmark's maize test scenarios are drawn from scenarios
outside the submitted protocol's training and validation sets and are part of
the held-out test suite (`docs/benchmark/test_suite.md`).

## External Simulator Sources

AgriManager does not create CycleGym or DSSAT-Gym. It contributes configs,
adapters, prompt/rendering logic, dataset generators, training entrypoints, and
evaluation glue that use those external simulators.

| External source | AgriManager role | Code references |
| --- | --- | --- |
| WOFOST-Gym and PCSE/WOFOST | External crop simulator used by the three hosted WOFOST weather-pool datasets and WOFOST experiment rows. AgriManager wraps it through a text/numeric adapter and builds fixed parquet rows from hosted weather pools. | `agrimanager/env/wofost_gym/`, `integrations/wofost_gym/` |
| CycleGym | External Cycles-based RL simulator. AgriManager uses CycleGym to generate deterministic crop-planning and maize/corn rows from config-defined environments, seeds, year windows, locations, and price regimes. | `agrimanager/env/cycles_gym/`, `experiments/t1_3_cycles_gym_price_regime/`, `experiments/t3_1_cross_simulator_maize_transfer/config/t31_cross_sim_cycles_gym.yaml` |
| DSSAT-Gym / DSSAT-PDI / DSSAT-CSM | External DSSAT simulator interface. AgriManager uses DSSAT-Gym configs to generate deterministic maize, rice, and cotton rows from seeds, crop/task settings, decision intervals, and DSSAT environment parameters. | `agrimanager/env/gym_dssat/`, `smoke_tests/gym_dssat/`, `experiments/t3_1_cross_simulator_maize_transfer/config/t31_cross_sim_gym_dssat.yaml` |
| VERL | External RL training framework used by AgriManager's training adapter. It is code infrastructure, not a dataset source. | `verl/`, `agrimanager/adapter/`, `entrypoints/train/` |

Cite the upstream WOFOST-Gym, PCSE/WOFOST, CycleGym, DSSAT-Gym/DSSAT-PDI,
DSSAT-CSM, NASA POWER, and VERL projects separately from AgriManager. Do not
claim that AgriManager created those simulators or their underlying simulator
assets.

## Upstream Versions And Licenses

Upstream attribution is recorded separately from the install source.
`install.sh` installs WOFOST-Gym and CycleGym from forks used by AgriManager
at pinned commits: `github.com/AgriManager-Framework/WOFOSTGym` and
`github.com/AgriManager-Framework/cyclesgym`. Each fork is the upstream history
plus one commit whose tree is identical to the working fork the paper used;
`THIRD_PARTY_NOTICES.md` at the repository root lists what the patches change.

| Component | Upstream source to cite | Exact source used here | License | Notes |
| --- | --- | --- | --- | --- |
| WOFOST-Gym | `https://github.com/Intelligent-Reliable-Autonomous-Systems/WOFOSTGym` | `AgriManager-Framework/WOFOSTGym` at `253cf546f9b39a9efdb225049a1a48341a665c87` (tree-identical to the paper's `071f8a61cf81c8621304139cf36ae9b2665da923`). | MIT | WOFOST-Gym package metadata reports version `1.1.0`. AgriManager uses WOFOST-Gym through configs/adapters and hosted weather pools; it does not create WOFOST-Gym. |
| PCSE / WOFOST | `https://github.com/ajwdewit/pcse` and the WOFOST crop-model literature cited by WOFOST-Gym | Shipped inside the pinned WOFOST-Gym source above, under `pcse/` (modified PCSE 5.5.6) and `pcse_gym/`. | `pcse/`: European Union Public Licence (header of `pcse/pcse/__init__.py`; upstream PCSE is EUPL 1.1 or later), although WOFOST-Gym's `pcse/LICENSE.md` carries an MIT text; `pcse_gym/`: MIT | AgriManager uses PCSE/WOFOST through WOFOST-Gym. The WOFOST weather inputs are derived from NASA POWER through the WOFOST-Gym/PCSE weather provider. |
| CycleGym | `https://github.com/kora-labs/cyclesgym` | `AgriManager-Framework/cyclesgym` at `2eb875834b71bb4e3a1ae59d0468551b41882331` (tree-identical to the paper's `0c0a73986ad291f27564ec3a49b865af8268b4dd`). | BSD-3-Clause | CycleGym package metadata reports version `0.1.0`. AgriManager uses CycleGym as an external executable generator. |
| Cycles crop simulator | `https://github.com/PSUmodeling/Cycles` and `https://plantscience.psu.edu/research/labs/kemanian/models-and-tools/cycles` | Installed by CycleGym's `install_cycles.py` from the `v0.12.9-alpha` release archive: `Cycles_debian_0.12.9-alpha.zip`, `Cycles_macos_0.12.9-alpha.zip`, or `Cycles_win_0.12.9-alpha.zip`. | Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (`CC-BY-NC-ND-4.0`) | Native Cycles simulator copyright Armen Kemanian, 2023. The native simulator license is more restrictive than the BSD-3-Clause CycleGym wrapper license. AgriManager does not distribute or claim the Cycles simulator as an AgriManager-created dataset. |
| DSSAT-Gym / gym-DSSAT-PDI | `https://gitlab.inria.fr/rgautron/gym_dssat_pdi.git` | Optional DSSAT setup observed at commit `63f2c529e0bd339b4553beb9aa56d56af83b5e2b`; the upstream remote reports this as `HEAD` and `refs/heads/stable`. The Spack package recipe records `gym-dssat` version `0.0.8`. | BSD-3-Clause | DSSAT-Gym is optional and is not installed by `install.sh`; see `docs/getting-started/install_dssat.md`. AgriManager uses it as an external executable generator. |
| DSSAT-PDI / DSSAT-CSM | `https://gitlab.inria.fr/rgautron/dssat-csm-os.git` and `https://github.com/DSSAT/dssat-csm-os` | Optional Spack runtime records `dssat-pdi` version `4.8.0.24_2`; the DSSAT-Gym source also records `dssat-csm-data` tag `v4.8.0.28`. | BSD-3-Clause for DSSAT-CSM/DSSAT-PDI sources | The DSSAT native runtime is an external simulator runtime. AgriManager only supplies configs/adapters/generation/training/evaluation glue. |
| VERL | `https://github.com/verl-project/verl.git` | Git submodule commit `7522bef0eb5c5761500fa8652e7ed45936f5323d`; `git describe` reports `v0.7.0-1-g7522bef0`. | Apache-2.0 | VERL is the external RL training framework used by the AgriManager training adapter. |

### DSSAT Spack Runtime Details

The optional DSSAT runtime is not installed by the standard `install.sh` path.
When DSSAT-backed experiments are enabled, the local Spack recipe used by the
AgriManager DSSAT setup records the following source artifacts:

| Package/resource | Source artifact | Version/tag | Checksum or ref |
| --- | --- | --- | --- |
| `gym-dssat` | `https://gitlab.inria.fr/rgautron/gym_dssat_pdi/-/archive/v0.0.8/gym_dssat_pdi-v0.0.8.tar.bz2` | `0.0.8` | SHA256 `a97276f43ce0b6ea9e543367fde9cc1f74c20a6155cc652048593c3376e6b9f6` |
| `dssat-pdi` | `https://gitlab.inria.fr/rgautron/dssat-csm-os/-/archive/4.8.0.24_2/dssat-csm-os-4.8.0.24_2.tar.bz2` | `4.8.0.24_2` | SHA256 `410193834540d41831c37b847a3219f6f70386135647389e8429c0c62bfb3a27` |
| `dssat-csm-data` | `https://github.com/DSSAT/dssat-csm-data.git` | tag `v4.8.0.28` | Resolved commit `c1a31af16fc82659e0b024d58e40b021981b182b`. |

The same recipe records earlier `dssat-pdi` versions, but AgriManager's
DSSAT-backed setup should cite the `4.8.0.24_2` runtime unless a different
runtime is explicitly installed and used.

## CycleGym and DSSAT-Gym Generators

CycleGym and DSSAT-Gym rows are generated by executable configs rather than
hosted as standalone HF datasets. They become static dataset artifacts only if
the fixed generated parquet rows are released and claimed as datasets.
Otherwise, the codebase must provide the configs, pinned environment setup, and
reproduction commands needed to regenerate them.

CycleGym generator behavior:

| Item | Documentation |
| --- | --- |
| Generator | `agrimanager/env/cycles_gym/create_dataset.py` |
| Environment wrapper | `agrimanager/env/cycles_gym/env.py` and `agrimanager/env/cycles_gym/env_config.py` |
| Price-regime experiment | `experiments/t1_3_cycles_gym_price_regime/config/price_regime_llm_no_think.yaml` and paired think/NN configs |
| Cross-simulator maize config | `experiments/t3_1_cross_simulator_maize_transfer/config/t31_cross_sim_cycles_gym.yaml` |
| Main controls | `env_id`, `year_windows`, `seeds_per_combo`, `seed_start`, `rotation_crops`, `crop_price_sampling`, paired price-regime overrides, objective and reward parameters |

DSSAT-Gym generator behavior:

| Item | Documentation |
| --- | --- |
| Generator | `agrimanager/env/gym_dssat/create_dataset.py` |
| Environment wrapper | `agrimanager/env/gym_dssat/env.py` and `agrimanager/env/gym_dssat/env_config.py` |
| Smoke configs | `smoke_tests/gym_dssat/config/` |
| Cross-simulator maize config | `experiments/t3_1_cross_simulator_maize_transfer/config/t31_cross_sim_gym_dssat.yaml` |
| Main controls | `env_id`, `crop_name`, `train_seeds`, `val_seeds`, `decision_interval`, `num_seasons`, `env_params`, `dssat_params`, optional `soil_variation`, objective and reward parameters |
