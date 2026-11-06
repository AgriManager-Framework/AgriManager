# Dataset Config Conventions

This document defines how WOFOST OOD experiment dataset configs should be
written, and how those configs connect to training-time validation and
Weights & Biases logging.

Core principles:

- `experiments/<exp>/config/*.yaml` describes dataset generation only. It must
  not contain training hyperparameters.
- One dataset config may define `train`, `validation`, and `test` splits for
  the same experiment family.
- A validation set is a logging and analysis abstraction; it does not mean
  validation rollout runs one set at a time.
- Training scripts map generated parquet files into `data.train_files` and
  `data.val_sets`.

## 1. Config Boundary

The dataset config answers one question: which episode rows should be
materialized for this experiment?

```yaml
env_name: wofost_gym
dataset_id: weather_regime_chickpea_llm_without_traits_no_think
validation_axis: weather_regime

source:
  kind: weather_pool
  path: AgriManager-Framework/wofost-weather-regime-pool

sampling:
  generation_seed: 42
  splits:
    train:
      role: train
      crops: [chickpea]
      num_samples: 1600
      labels:
        weather_regime: id

env:
  env_id: lnpkw-v0
  llm_mode: true
  include_crop_traits: false
```

Do not put model names, batch size, checkpoint paths, W&B run names, or GPU
settings here. Those belong in `run_*.sh` and the train entrypoint overrides.

## 2. Top-Level Fields

- `env_name` — environment adapter name. WOFOST experiments use `wofost_gym`.
- `dataset_id` — generated dataset directory name. Output path is
  `data/<dataset_id>/` under the experiment directory; each split becomes one
  parquet file.
- `validation_axis` — the primary OOD categorization axis. It controls the
  compact validation namespace in W&B (see §7). Examples:
  - T1.1 weather regime shift: `validation_axis: weather_regime`
  - T1.2 cross-crop trait shift: `validation_axis: crop_regime`
  - T2.2 action menu shift: `validation_axis: action_menu`
  - T2.3 reward formulation shift: `validation_axis: reward_formulation`
  - T3.1 cross-simulator transfer: `validation_axis: simulator`

  With it set, W&B prioritizes metrics such as
  `val-env-weather_regime/all/...`, `val-env-weather_regime/drought/...`
  instead of expanding every available label into `val-env-by-*` namespaces.
- `source` — where generation data is sampled from, such as a weather pool. It
  does not describe how training consumes the data.
- `sampling.generation_seed` — default seed used to sample rows from the source
  pool. Keep it fixed so the config rebuilds the same rows. It is not
  incremented by split; a split or scenario set that omits its own `seed` uses
  this value. Use the underscore spelling `generation_seed`.
- `sampling.scenario_sets` — optional layer for paired designs (see below).
- `sampling.splits` — the core of the config. Each key is a split name and also
  the output file name: `val_cold -> data/<dataset_id>/val_cold.parquet`.
- `env` — default environment config used to generate episodes. Individual
  splits may override part of it with their own `env` block.

### scenario_sets

A scenario set defines one shared group of source scenarios; multiple split
files can render that same group with different labels or environment
overrides.

Use `scenario_sets` when the source scenarios should stay the same and the
experiment varies only the rendering or environment interface (action menu,
prompt schema, reward formulation). Do not use it to hide the target axis: if
the axis itself is weather regime, crop, or another source-scenario shift, each
split should sample its own source distribution directly.

```yaml
sampling:
  generation_seed: 2201
  scenario_sets:
    train_base:
      source_split: train
      crops: [maize]
      num_samples: 1600

    val_base:
      source_split: val
      crops: [maize]
      num_samples: 128

  splits:
    train_lnpkw:
      role: train
      scenario_set: train_base
      labels:
        action_menu: lnpkw
      env:
        env_id: lnpkw-v0

    train_lnpk:
      role: train
      scenario_set: train_base
      labels:
        action_menu: lnpk
      env:
        env_id: lnpk-v0
```

Here `train_lnpkw` and `train_lnpk` use the same training weather rows. Use the
same pattern for validation action-menu, prompt-schema, and reward-formulation
comparisons when the intended OOD axis should be isolated from weather
variation.

## 3. Split Fields

A validation split:

```yaml
val_cold:
  role: validation
  validation_set: cold
  crops: [chickpea]
  num_samples: 128
```

- `role` — semantic split role: `train`, `validation`, or `test`. May be
  explicit or inferred from the split name: `train`/`train_*` -> `train`;
  `val`/`val_*` -> `validation`; `test`/`test_*` -> `test`.
- `validation_set` — validation splits only. Names the validation set this file
  belongs to in the training script. `val_cold` infers `cold`, but writing it
  explicitly aids review.
- `crops`, `num_samples` — which crops the split samples and how many episodes.
  Cross-crop designs may use `crop_budgets` or `crop_variety_budgets`, but
  should not introduce a second overlapping concept such as `distribution` or
  `axes`.
- `scenario_set` — optional; points to `sampling.scenario_sets.<name>`. Use it
  when several split files share source rows while differing in
  prompt/action/reward config. Then put `crops`, `num_samples`, `crop_budgets`,
  `crop_variety_budgets`, and `source_split` on the scenario set; the split
  carries only role, labels, validation metadata, optional seed override, and env
  overrides. Referenced sets use `sampling.generation_seed` unless a
  split-local or set-local `seed` is given.
- `labels` — a key-value mapping (not a list) of semantic labels for analysis.
  For simple OOD splits, validation rows do not need to repeat the primary axis:
  if `validation_axis: weather_regime` and a split has `validation_set: cold`,
  the row builder fills `trajectory_group_labels.weather_regime: cold` when that
  label is missing. Write `labels.<validation_axis>` only when the axis value
  differs from `validation_set`, or when the split needs extra analysis labels
  such as `action_schema`, `prompt_condition`, or `reward_formulation_family`.
  Explicit and row-level labels always win over the fallback (e.g.
  `validation_axis: crop` keeps the true generated row crop, so
  `validation_set: heldout_crops` does not become `crop: heldout_crops`).
- `env` — optional split-level environment override. T2.2 action menu uses it to
  run different splits with different action menus:

  ```yaml
  val_ln:
    role: validation
    validation_set: ln
    crops: [maize]
    num_samples: 128
    labels:
      action_schema: ln
      prompt_condition: clean
    env:
      env_id: ln-v0
  ```

  Here `action_menu: ln` is inferred from `validation_set: ln` because
  `validation_axis: action_menu` is set at the top level.

## 4. Automatically Injected Labels

The builder writes normalized split metadata into each row's `env_config` and
`trajectory_group_labels`. Common automatic fields:

- `dataset_split` — physical split name, e.g. `val_cold`; maps to the generated
  file and manifest entry.
- `dataset_role` — `train` / `validation` / `test`.
- `validation_set` — validation set name, e.g. `cold`; maps to the named
  validation suite in the training script.
- `trajectory_group_labels.<validation_axis>` — for validation rows, defaults to
  `validation_set` when the generator has not already written that axis; maps to
  the primary OOD split shown in W&B.
- `simulator` — e.g. `wofost_gym`.
- `crop` — e.g. `chickpea`.
- `objective_id` — e.g. `profit_max`.
- `env_id` — e.g. `lnpkw-v0`.

`validation_set` and `trajectory_group_labels.<validation_axis>` may carry the
same value but mean different things: `validation_set: cold` says "this parquet
belongs to the `cold` validation set," while `weather_regime: cold` says "this
episode is in the `cold` category of the `weather_regime` axis." The fallback
can fill the repeated axis value, but never overrides more specific row labels
such as the true crop.

## 5. Manifest

Every dataset build writes `data/<dataset_id>/manifest.json`. It is for build
verification and offline analysis, not a new experiment launcher. Typical shape:

```json
{
  "schema_version": 1,
  "dataset_id": "weather_regime_chickpea_llm_without_traits_no_think",
  "env_name": "wofost_gym",
  "validation_axis": "weather_regime",
  "splits": {
    "val_cold": {
      "path": "val_cold.parquet",
      "role": "validation",
      "validation_set": "cold",
      "num_rows": 128,
      "labels": {
        "weather_regime": "cold"
      }
    }
  },
  "validation_sets": {
    "cold": {
      "files": ["val_cold.parquet"],
      "labels": {
        "weather_regime": "cold"
      }
    }
  }
}
```

`splits` records physical files; `validation_sets` records the training
validation suite. Both layers are kept because one validation set can contain
multiple files. Manifest `labels` include the primary validation-axis label only
when every successfully generated row in that split has the same value. If the
axis has multiple real row values (e.g. potato and rice in one held-out-crop
split), the manifest leaves that axis out instead of collapsing it to the
validation-set name; the parquet row metadata remains authoritative for
multi-valued axes.

## 6. How Training Uses Validation

Training scripts should not hand-write anonymous `val_files=[...]` for OOD
experiments. Use named validation sets:

```bash
"data.train_files=${DATASET_DIR}/train.parquet" \
"data.val_files=null" \
"data.validation_axis=weather_regime" \
"data.val_sets.id=${DATASET_DIR}/val_id.parquet" \
"data.val_sets.drought=${DATASET_DIR}/val_drought.parquet" \
"data.val_sets.wet=${DATASET_DIR}/val_wet.parquet" \
"data.val_sets.hot=${DATASET_DIR}/val_hot.parquet" \
"data.val_sets.cold=${DATASET_DIR}/val_cold.parquet"
```

Runtime behavior:

1. The trainer reads `data.val_sets`.
2. All validation files are flattened into one validation dataset.
3. Rollout runs as one merged validation pass, preserving parallelism.
4. Metrics are split after rollout using `validation_axis` and row labels.

A named validation suite therefore does not turn validation into five serial
rollout jobs.

## 7. W&B Logging

When `data.validation_axis=weather_regime` is set, the main W&B metrics:

```text
val-core-weather_regime/all/...
val-core-weather_regime/id/...
val-core-weather_regime/drought/...

val-env-weather_regime/all/target_yield/mean
val-env-weather_regime/id/target_yield/mean
val-env-weather_regime/drought/target_yield/mean
```

`val-core-*` contains trainer/reward-level aggregation such as reward, pass@N,
best@N, and majority@N. It should not contain duplicated environment trajectory
metrics such as `target_yield` or `total_irrig_mm`; those belong in `val-env-*`.
The implementation splits by axis before computing these metrics, so uid/repeat
grouping does not mix different validation sets.

`val-env-*` contains environment metric aggregation, such as `target_yield`,
`fertilizer_cost`, `irrigation_amount`, and other trajectory-level numeric
metrics.

Training rows also carry group labels, but W&B should not expand every label.
Training env metrics should log the aggregate `train-env/...` namespace and,
when `data.validation_axis` is set, the matching compact `train-env-<axis>/...`
namespace. Do not create high-cardinality training namespaces such as
`train-env-by-scenario_id/...`, or structural one-value namespaces such as
`train-env-by-dataset_role/train/...`. For paper figures and main tables, prefer
the compact validation-side OOD axis metrics.

## 8. End-to-End Example: T1.1 Weather Regime Shift

T1.1 sets `validation_axis: weather_regime` (see the §1 config) with a `train`
split and one validation split per regime, e.g.:

```yaml
sampling:
  splits:
    train: { role: train, crops: [chickpea], num_samples: 1600, labels: { weather_regime: id } }
    val_id:      { role: validation, validation_set: id,      crops: [chickpea], num_samples: 128 }
    val_drought: { role: validation, validation_set: drought, crops: [chickpea], num_samples: 128 }
    # ...val_wet, val_hot, val_cold likewise
```

Validation rows get `weather_regime: id`, `weather_regime: drought`, etc. from
the `validation_set` fallback. (Older configs that write these labels explicitly
still work.) Build produces `train.parquet`, `val_{id,drought,wet,hot,cold}.parquet`,
and `manifest.json`. The training script wires these via `data.val_sets.*` (§6),
and the primary namespace is `val-env-weather_regime/<regime>/...` (§7) — shorter
than `val-env-by-weather_regime/...` and closer to the experimental question.

### Scenario-set variant: T2.2 Action Menu Shift

When the OOD axis is the interface rather than the weather, share scenario sets
so action-menu variants render on the same weather rows. A validation split then
needs only `validation_set` plus its `env` override:

```yaml
val_lw:
  role: validation
  validation_set: lw
  scenario_set: val_base
  labels:
    action_schema: lw
    prompt_condition: clean
  env:
    env_id: lw-v0
```

Because `validation_axis: action_menu`, `validation_set: lw` is enough to
produce `trajectory_group_labels.action_menu: lw`; the extra labels are kept
only as auxiliary analysis dimensions. The training script selects the family A
mixture, e.g.:

```bash
"data.train_files=[${DATASET_DIR}/train_lnpkw.parquet,${DATASET_DIR}/train_lnpk.parquet]" \
"data.validation_axis=action_menu" \
"data.val_sets.lw=${DATASET_DIR}/val_lw.parquet"
```

## 9. Checklist

- One config generates one clear dataset artifact: `data/<dataset_id>/`.
- The config contains dataset generation logic only, not training logic.
- Every validation split has `role: validation` and a clear `validation_set`.
- `labels` is a key-value mapping. For simple validation OOD splits, omit
  `labels.<validation_axis>` and let it default from `validation_set`; write it
  explicitly only when the axis value differs or extra analysis labels are
  needed.
- Multiple validation parquet files are named through `data.val_sets` during
  training, not through an anonymous list.
- Validation is executed as one merged pass, then split by label for logging.
- W&B should not expand every label redundantly. The main experiment namespace
  should expand only `validation_axis`.
- Trainer/reward aggregation (`val-core-*`) and environment aggregation
  (`val-env-*`) should not duplicate each other's metrics.
- Training env metrics should not expand structural or high-cardinality labels
  such as `dataset_split`, `env_id`, or `scenario_id`.
