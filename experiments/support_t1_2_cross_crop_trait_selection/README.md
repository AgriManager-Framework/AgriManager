# WOFOST Cross-Crop Strategy Validation

Support material for the T1.2 cross-crop trait claim (see
[docs/benchmark/README.md](../../docs/benchmark/README.md#t12-cross-crop-trait-shift)).
It is separate from `experiments/t1_2_cross_crop_trait_shift/`, which holds the
main policy-training experiments.

The validation question is:

1. Do different crops require measurably different strategies?
2. Can crop traits predict those strategy differences?

## Data Source

The specialist-transfer evaluation reuses the main cross-crop no-traits test set:

```bash
experiments/t1_2_cross_crop_trait_shift/data/cross_crop_16id_nn_without_traits/test.parquet
```

This keeps the 20-crop evaluation scenarios aligned with the main experiment.

## Specialist Training

The eight specialist crops are `maize`, `wheat`, `cotton`, `rice`, `potato`,
`sugarbeet`, `barley`, and `seed_onion`. Existing maize/wheat no-traits weather
specialists serve as anchors; the six additional no-traits specialists are
trained with the fixed run scripts:

```bash
bash experiments/support_t1_2_cross_crop_trait_selection/run_wofost_specialist_transfer_cotton_nn_train_n1.sh
bash experiments/support_t1_2_cross_crop_trait_selection/run_wofost_specialist_transfer_rice_nn_train_n1.sh
bash experiments/support_t1_2_cross_crop_trait_selection/run_wofost_specialist_transfer_potato_nn_train_n1.sh
bash experiments/support_t1_2_cross_crop_trait_selection/run_wofost_specialist_transfer_sugarbeet_nn_train_n1.sh
bash experiments/support_t1_2_cross_crop_trait_selection/run_wofost_specialist_transfer_barley_nn_train_n1.sh
bash experiments/support_t1_2_cross_crop_trait_selection/run_wofost_specialist_transfer_seed_onion_nn_train_n1.sh
```

These can also be submitted together as a local CPU sbatch array.

The trained `agent.zip` / `vecnormalize.pkl` bundles are the per-crop specialists
used by the 8x20 cross-crop transfer evaluation that backs the T1.2 trait
analysis. The downstream diagnostic and trait-selection analysis scripts are not
shipped with this release; this directory provides the specialist-training runs
that produce their inputs.
