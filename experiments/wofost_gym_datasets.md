# WOFOST-Gym Dataset Sources

This note centralizes the WOFOST-Gym weather-pool definitions used by the
experiment folders. The experiment configs consume materialized Hugging Face
sources; they do not recompute the source-level filtering logic at training time.

## Canonical Pools

| Pool | Hugging Face source | Main use |
| --- | --- | --- |
| Weather-regime pool | `AgriManager-Framework/wofost-weather-regime-pool` | T1.1 weather-regime shift. |
| 20-crop weather pool | `AgriManager-Framework/wofost-weather-pool` | T1.2 cross-crop trait shift and WOFOST crop-support runs. |
| Maize weather pool | `AgriManager-Framework/weather_pool_maize` | WOFOST maize schema, action, reward, and cross-simulator experiments. |

## Public Namespace Verification

The pools were first published on 2026-07-24, replacing earlier private copies of the
same pools. Equivalence was checked from Hub file metadata without downloading
the large weather archives: every Parquet and `meteo_cache.tar.gz` LFS object
had the same path, byte size, and SHA-256 in the corresponding repository.

| Release source | Release revision | Verified revision (2026-07-24) | Matching core files | Matching bytes |
| --- | --- | --- | ---: | ---: |
| `AgriManager-Framework/wofost-weather-pool` | `268da7d08a0f6e544469e56bfefd1dbb2b08ea64` | `0837df98e3d8c9989a4f4fb5195ac2e24b267cb9` | 61 | 17,564,531,383 |
| `AgriManager-Framework/wofost-weather-regime-pool` | `32b799473b82e708b308f6b0654c02f39f3434b1` | `cda48157a32b9f4e069f573b32e3fc402b11516e` | 15 | 289,811,027 |
| `AgriManager-Framework/weather_pool_maize` | `3389bc2d75acddbd6740f30198167e8d3e4559c3` | `11d9558c981fa8cf43a69ee8f586134da94ef5e6` | 3 | 250,207,408 |

On 2026-10-25 the pools were copied into the `AgriManager-Framework` organization. Every data
file of a release revision is byte-identical (same LFS SHA-256) to the verified
revision; the release revisions add a dataset card and Croissant metadata. The
`prepare_*_eval_pool.py` scripts pin the release revisions.

There were no missing or mismatched core files. As a second check, the small
maize train and validation Parquet files were downloaded from both namespaces
and compared byte-for-byte. The public repositories add documentation and
sample metadata; the maize `pool_manifest.json` changes only `source_pool` from
the private 20-crop repo ID to its public replacement.

## Generation Seed

All active WOFOST-Gym experiment configs that consume these pools use
`generation_seed: 42` for deterministic train/validation sampling.

At the source-pool level:

- The 20-crop weather pool per-crop configs all use `generation_seed: 42`.
- The maize weather pool is derived from the same deterministic maize source:
  its `train` split matches the 20-crop `maize/train` shard, and its `val`
  split is the union of the 20-crop `maize/val` and `maize/test` shards.
- The weather-regime pool is consumed as a materialized Hugging Face source; the
  local T1.1 configs use `generation_seed: 42` when building train and named
  validation artifacts from that source.

## Weather-Regime Pool

The weather-regime pool is a materialized validation source for `chickpea` and
`potato`. Each crop has:

- `train`: `1600` normal weather scenarios.
- `val`: `640` combined validation scenarios.
- `val_id`, `val_drought`, `val_wet`, `val_hot`, `val_cold`: `128` scenarios each.

The named validation regimes are equal-size, crop-specific `20%` buckets. For a
crop-specific candidate weather set `S_c`, define:

- `season_rain(s)`: cumulative `RAIN` over the WOFOST growing season.
- `season_temp(s)`: mean daily `TEMP` over the WOFOST growing season.

With `p = 0.20`, the intended filtering predicates are:

```text
drought(s) = season_rain(s) is in the lowest  p fraction of S_c
wet(s)     = season_rain(s) is in the highest p fraction of S_c
hot(s)     = season_temp(s) is in the highest p fraction of S_c
cold(s)    = season_temp(s) is in the lowest  p fraction of S_c
normal(s)  = not (drought(s) or wet(s) or hot(s) or cold(s))
```

The local T1.1 dataset builder reads the already materialized `val_*` splits and
propagates their `weather_regime` labels into generated parquet artifacts.

## 20-Crop Weather Pool

The 20-crop weather pool is generated from the per-crop configs under
`integrations/wofost_gym/dataset_tools/weather_pool_configs/pool_crop_*.yaml`.
Each crop config uses:

- `generation_seed: 42`
- `year_range: [1984, 2019]`
- `min_dvs_threshold: 1.5`
- split sizes: `train=3200`, `val=128`, `test=512`

The crop set is:

```text
barley, chickpea, cotton, cowpea, fababean, groundnut, maize, millet,
mungbean, pigeonpea, potato, rapeseed, rice, seed_onion, sorghum, soybean,
sugarbeet, sunflower, sweetpotato, wheat
```

Each candidate scenario is a `(crop, year, latitude, longitude)` tuple. It is
kept only if a full no-action WOFOST rollout succeeds and satisfies all viability
checks:

```text
weather_complete(s) = rollout has no missing-weather/cache errors
phenology_ok(s)     = max_DVS(s) >= 1.5
yield_ok(s)         = max_WSO(s) > 0
keep(s)             = weather_complete(s) and phenology_ok(s) and yield_ok(s)
```

Failed scenarios are discarded and replaced before writing the per-crop pool
parquets.

## Maize Weather Pool

The maize weather pool uses the same WOFOST viability filter as the 20-crop
weather pool:

```text
weather_complete(s) and max_DVS(s) >= 1.5 and max_WSO(s) > 0
```

It is a maize-only materialized source with:

- `train`: `3200` maize scenarios.
- `val`: `640` maize scenarios.
- no separate materialized `test` split in the current source.

WOFOST maize experiments draw from this pool and then render the same base
weather rows under the experiment-specific observation schema, action menu,
reward formulation, or simulator-transfer setting.
