# Baseline Evaluation

This directory contains the shared evaluation entrypoint for scripted baselines.
The main runner is `entrypoints/eval/baseline_eval.sh`, which launches
`agrimanager.rollout.inference.random_rollout` with a Hydra config from
`entrypoints/eval/config`.

Always run these commands in the `agrimanager` Conda environment:

```bash
conda run -n agrimanager bash entrypoints/eval/baseline_eval.sh \
  --config-name random \
  data.inference_file=<validation.parquet> \
  data.validation_axis=<axis_name> \
  output.dir=<output_dir> \
  wandb.name=<wandb_run_name>
```

The runner writes local artifacts and, when `wandb.enabled=true`, logs aggregate
metrics to Weights & Biases:

- `results.json`: per-environment, per-turn rollout records.
- `metrics.json`: aggregate metrics for the full validation file.
- `validation_metrics.json`: training-style validation metrics split by
  `data.validation_axis` or by `validation_set` metadata.
- `wandb/`: local W&B files when W&B is enabled.

For example, with `data.validation_axis=weather_regime`, metrics are logged with
keys such as:

```text
val-env-weather_regime/all/profit_ge_kg_ha/mean
val-env-weather_regime/id/profit_ge_kg_ha/mean
val-env-weather_regime/cold/profit_ge_kg_ha/mean
val-env-weather_regime/drought/profit_ge_kg_ha/mean
```

If the dataset contains five weather validation regimes, each regime is logged
as its own W&B metric key in the same run.

## Baselines

| Baseline | Config | Runtime mode | Intended role |
| --- | --- | --- | --- |
| Random | `random.yaml` | `random` | Stochastic lower-quality action reference. |
| No Action | `no_action.yaml` | `no_action` | Stable do-nothing reference for profit and risk. |
| Scripted agronomic reference | `ag_heuristic.yaml` | `ag_heuristic` | Current version of the scripted agronomic reference; not an upper bound. The paper's anchors use earlier versions (see below). |
| PPV0 | `ppv0.yaml` | `ppv0` | WOFOST potential-production yield diagnostic; not a profit policy. |

### Random

Random samples the environment action space at every decision turn:

```bash
conda run -n agrimanager bash entrypoints/eval/baseline_eval.sh \
  --config-name random \
  data.inference_file=<validation.parquet> \
  data.validation_axis=weather_regime
```

The random action spaces are seeded from `runtime.seed`, so repeated runs are
reproducible when the dataset and environment are unchanged. Use
`runtime.num_trials>1` when estimating variance across random seeds.

### No Action

No Action sends the native zero action at every decision turn:

```bash
conda run -n agrimanager bash entrypoints/eval/baseline_eval.sh \
  --config-name no_action \
  data.inference_file=<validation.parquet> \
  data.validation_axis=weather_regime
```

This baseline is important for profit experiments because any costed management
policy can be worse than doing nothing. It should be reported directly, and
paper figures should include the percentage of splits where each method is below
No Action.

### Scripted agronomic reference

The scripted agronomic reference is implemented as the `ag_heuristic` mode:

```bash
conda run -n agrimanager bash entrypoints/eval/baseline_eval.sh \
  --config-name ag_heuristic \
  data.inference_file=<validation.parquet> \
  data.validation_axis=weather_regime
```

The implementation lives in
`agrimanager/rollout/inference/ag_heuristic_rollout.py`, and the runner
dispatch is wired through `agrimanager/rollout/inference/random_rollout.py`.

Call it the "scripted agronomic reference". It is a reference point; do not
describe it as an expert policy or an upper bound.

Which version the paper uses: on WOFOST the paper's reference is the per-crop
policy of the paper era, and on Cycles crop growth the first implementation;
both are kept in `experiments/rebuttal_confirmatory_eval/paper_references/`.
DSSAT uses a code reconstruction of the fixed schedule.
`experiments/rebuttal_confirmatory_eval/run_e4_anchors.sh` runs these versions
on the held-out test suite. On the Cycles
crop-growth task the current `ag_heuristic` mode has no nitrogen rule and
scores exactly as no-action. Details and the related disclosures:
[docs/benchmark/baselines.md](../../docs/benchmark/baselines.md).

The reason it is not an upper bound is practical: published extension recommendations are usually
designed around crop production, soil testing, nutrient stewardship, and water
management. They are not guaranteed to maximize the simulator's profit reward,
especially in low-yield or distribution-shifted scenarios where input costs can
exceed added revenue.

### PPV0

PPV0 switches WOFOST-Gym to the native `pp-v0` environment:

```bash
conda run -n agrimanager bash entrypoints/eval/baseline_eval.sh \
  --config-name ppv0 \
  data.inference_file=<validation.parquet> \
  data.validation_axis=weather_regime
```

In PPV0, WOFOST simulates potential production without N/P/K/water limitation.
The runner uses zero management actions because the environment itself removes
those limitations. PPV0 is useful for yield-potential and yield-gap diagnostics,
but it is not a fair profit baseline: it does not pay the same input costs as a
costed `lnpkw-v0` management policy.

## Scripted agronomic reference: details

The current scripted agronomic reference is a frozen scripted policy. Its source manifest and policy
definition are documented in
[docs/benchmark/baselines.md](../../docs/benchmark/baselines.md).

### Where It Comes From

The policy is based on these agronomic frameworks:

- 4R Nutrient Stewardship: right source, right rate, right time, and right
  place for fertilizer decisions.
- NDSU fertilizer recommendation tables: soil-test and crop-specific N/P/K
  recommendations.
- Pulse-legume guidance from Saskatchewan and NDSU: chickpea and related pulse
  crops should not receive heavy nitrogen when properly inoculated; P/K are
  soil-test driven.
- Potato fertilizer guidance from Minnesota and NDSU: nutrient programs depend
  on soil tests, variety, harvest timing, yield goal, previous crop, and staged
  N timing.
- Corn/maize guidance from Colorado State: N rates depend on expected yield and
  soil/management credits; P/K are soil-test driven.
- FAO CROPWAT / crop-water guidance: irrigation should follow a soil-water
  balance, crop requirements, and stage-dependent water stress.

The checked-in constants are intentionally conservative because the simulator
reward is profit, not raw yield. They are crop or crop-family caps, not an
oracle tuned per validation scenario.

### Inputs Used

At each WOFOST decision turn, the policy reads the current observation and
environment metadata:

- Crop name and objective ID.
- Development stage `DVS`.
- Temperature `TEMP`, rainfall `RAIN`, and soil moisture `SM`.
- Available nutrient pools `NAVAIL`, `PAVAIL`, and `KAVAIL`.
- Cumulative applied inputs `TOTN`, `TOTP`, `TOTK`, and `TOTIRRIG`.
- The discrete WOFOST action menu exposed by the prompt generator.

If the environment is not WOFOST-Gym, the policy falls back to No Action.

### Decision Order

For WOFOST-Gym, one discrete action can be taken per turn. The policy applies
this priority order:

1. Do nothing before crop emergence (`DVS < 0`).
2. Irrigate for acute water stress when soil moisture is below a crop trigger,
   rainfall is low, and the seasonal irrigation cap has not been reached.
3. Apply early P or K only when the available P/K pool is below the crop
   threshold and the seasonal cap has not been reached.
4. Apply N for non-legume crops using stage targets: early vegetative,
   vegetative, and, for yield-oriented settings only, limited later N.
5. Suppress heavy N for pulse legumes such as chickpea, soybean, and groundnut;
   only a starter-N allowance is possible when the crop record permits it.
6. Use regular irrigation when rain is low, soil moisture is below the crop
   trigger, and water budget remains.
7. Otherwise take No Action.

Whenever an action is due, the policy chooses the largest available discrete
action that does not exceed the remaining seasonal cap. It never selects an
action amount above the cap.

### Known Limitation

The scripted agronomic reference is a codified reference point and has no
claim to be a profit upper bound. Small local probes showed that it can improve average profit
relative to No Action while still losing to No Action on some low-yield
scenarios. This is expected for a costed profit reward: a real agronomic
recommendation can still be net-negative if the simulated crop response is weak.
