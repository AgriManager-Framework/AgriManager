# Evaluation on the held-out test suite

This page shows how to score a policy on the frozen
[held-out test suite](../benchmark/test_suite.md): a released checkpoint, your
own model, the untrained backbone, a scripted reference, or a neural baseline.

## Protocol

These rules hold for every number in the paper. Follow them to get comparable
numbers.

| Rule | Detail |
|---|---|
| One suite per setting, never tuned on | No training, checkpoint selection, prompt tuning or decoding change uses test scenarios |
| Terminal checkpoint | Score the last checkpoint of each run (the early-stop step for a run that stopped early); do not pick the best validation checkpoint |
| Response format of training | A think-trained checkpoint is scored in think format and a no-think one in no-think format; the evaluator reads the format from the checkpoint and asserts it |
| Greedy decoding | `runtime.temperature=0`, one pass (`runtime.num_trials=1`) |
| No turn cap | Never set `runtime.max_turns` for a reported run |
| Every split recorded | A failed split is written into the manifest with its status, never silently dropped |
| Report raw and references | Report the raw return with both reference returns and the gap; normalize only where the gap is usable ([Baselines](../benchmark/baselines.md#normalized-return)) |
| Seeds | Three training seeds per arm; report mean and SD over seeds, and differences between arms in propagated SD |

## Score one checkpoint

`entrypoints/eval/e5_evaluate_checkpoint.sh` evaluates one checkpoint
revision on every suite group of a setting and writes one result bundle.

```bash
bash entrypoints/eval/e5_evaluate_checkpoint.sh \
    --checkpoint-repo AgriManager-Framework/schema-t32-qwen-think-r2 \
    --revision step-800 \
    --suite-repo AgriManager-Framework/agrimanager-test-suite \
    --results-repo <your-hf-user>/<results-dataset> \
    --tier t32 \
    --no-upload
```

To score suites that are already on disk (for example a local copy of the
test-suite dataset), keep `--suite-repo` as the name recorded in the bundle and
add `--suite-dir`:

```bash
bash entrypoints/eval/e5_evaluate_checkpoint.sh \
    --checkpoint-repo AgriManager-Framework/schema-t13-price-think-r1 --revision step-141 \
    --suite-repo AgriManager-Framework/agrimanager-test-suite --suite-dir <dir-with-e4_t13_price_regime> \
    --results-repo local/none --tier t13 --no-upload --work-dir runs/t13-think-r1
```

| Argument | Meaning |
|---|---|
| `--checkpoint-repo`, `--revision` | Hugging Face model repository and branch (`step-N`); the terminal revision is the highest `step-N` |
| `--checkpoint-dir` | Score weights already on disk; the two options above still name the run in the bundle. Use this for the untrained backbone |
| `--suite-repo` | Test-suite dataset repository (required). With `--suite-dir` it is not downloaded and only names the suite in the bundle |
| `--suite-dir` | Score audited suites already on disk (a directory holding the suite-group folders, e.g. `e4_t13_price_regime/`) |
| `--results-repo` | Results dataset the bundle is uploaded to (required, also with `--no-upload`, where it is not used) |
| `--tier` | `t11`, `t11_chickpea`, `t12`, `t13`, `t21`, `t22`, `t23`, `t31`, `t32`; selects the suite groups of that setting |
| `--suite-group` | Override the groups (repeatable), for example to score a T3.1 specialist on the T3.2 suites |
| `--require-think`, `--think-tag` | Only for checkpoints whose manifest predates the format field; the bundle records that the value was supplied by hand |
| `--max-samples N` | Score only the first N rows of each split, for a pipeline check |
| `--no-upload` | Keep the bundle local, in `<work-dir>/bundle` (`--work-dir`, default `./.e5_work`); without it, `HF_TOKEN` is required and the work directory is deleted after upload unless `--keep-work-dir` |
| `--forbid` | Extra terms that must not appear in the bundle (also `PUBLISH_FORBID_TERMS`) |

The suite groups per tier are defined in `SUITE_GROUPS` in
`agrimanager/checkpoint/evaluate.py`. The bundle holds the per-scenario
outputs, the metrics of every split, the failures, the environment overrides
that were applied, and `eval_manifest.json`, which ties every number to the
checkpoint revision and suite revision that produced it.

Hardware: one CUDA GPU. Simulator paths follow the same environment variables
as training (`WOFOST_GYM_PATH`, `DSSAT_GYM_PATH`).

WOFOST scenarios read their weather from the bundled NASA POWER cache of a
public weather pool, and the evaluators download that pool into
`.cache/wofost_gym/` the first time a suite needs it: the maize pool (0.25 GB)
for T2.1, T2.2, T2.3 and the WOFOST parts of T3.1 and T3.2, and the 20-crop pool
(17.6 GB download) for T1.1 and T1.2. CycleGym and DSSAT suites need no download.

*Checked on 2026-10-25 from a fresh `install.sh --gpu` on one A40: single T2.1
splits and a T1.3 prefix were scored with the command above (results under
[Reproducibility across hardware](#reproducibility-across-hardware)); a full
setting was not re-scored for this revision.*

### The untrained model

The untrained backbone is scored on every split under both response formats,
as a reference point next to each trained arm. Pass the base model directory
with `--checkpoint-dir` and set the format explicitly with `--require-think`
and `--think-tag tool_call`.

## Score your own model

Any model served through the model interface can be scored on a suite split
with the generic evaluator. Apply the response format with env-config
overrides and assert it:

```bash
bash entrypoints/eval/eval.sh \
  data.inference_file=<suite_dir>/<suite>/<split>.parquet \
  data.validation_axis=<axis of the setting> \
  model.config=agrimanager/model_interface/configs/vllm_offline/default.yaml \
  model.path=<hf_repo_or_local_dir> \
  +data.env_config_overrides.require_think=True \
  +data.env_config_overrides.think_tag=tool_call \
  +data.assert_env_config.require_think=True \
  output.dir=<out_dir>
```

Hosted models work through `--config-name openrouter` and an
`OPENROUTER_API_KEY`; use a fixed model slug for comparable numbers. Report
results on the whole split.

By default an answer that names more amounts than its action takes (for
example `Apply 5.0, 10.0, or 15.0 cm irrigation depth.`) is an invalid action;
the env-config key `action_parser: legacy` (or `AGRI_ACTION_PARSER=legacy`)
restores the earlier first-amount reading, which `e5_evaluate_checkpoint.sh`
(`--action-parser`) and the `experiments/` run scripts pin.

## Scripted references and neural baselines

| What | Command | Hardware |
|---|---|---|
| No-action, the paper's scripted reference and the current one, on every split | `bash experiments/rebuttal_confirmatory_eval/run_e4_anchors.sh` | CPU; DSSAT runtime for DSSAT splits |
| One scripted policy on one split | `bash entrypoints/eval/baseline_eval.sh --config-name {no_action,ag_heuristic} data.inference_file=<split> output.dir=<out> wandb.enabled=false` | CPU |
| Neural baselines on the `_nn` suites | `experiments/t1_1_weather_regime_shift/run_eval_t11_{potato,chickpea}_nn_e4.sh`, `experiments/t1_2_cross_crop_trait_shift/run_eval_t12_nn_e4.sh`, `experiments/t1_3_cycles_gym_price_regime/run_eval_t13_nn_e4.sh`, `experiments/t2_2_action_menu_shift/run_eval_t22_nn_{mask,target}_e4.sh` | CPU |
| One neural checkpoint on one split | `bash entrypoints/eval/nn_eval.sh --config-name nn data.inference_file=<split> agent.path=<agent.zip> output.dir=<out>` | CPU |

## Scoring many checkpoints

`results/scoring_manifest.tsv` lists every revision scored for the paper.
`entrypoints/tools/build_e4_score_queue.sh` lists every terminal revision on
the hub (`HF_OWNER`, default `agrimanager`) that has no result yet.
`entrypoints/tools/e4_score_worker.sh --queue <file>` works through such a
queue inside one long batch allocation, skips runs whose bundle already exists
and stops before the allocation ends; `e4_score_fanout.sh` runs one worker per
GPU on a node. Results go to `--result-root` (default
`$AGRI_WORK_DIR/agri_e4_scoring`, with `AGRI_WORK_DIR` defaulting to
`~/agrimanager_work`); suites are read from `--suite-dir` (default
`experiments/rebuttal_confirmatory_eval/data`). The untrained floors need the
directories made by `entrypoints/tools/prepare_untrained_base.py --out <dir>`;
replace `<untrained-base>` in the manifest with that directory.

## Reproducibility across hardware

Greedy decoding with vLLM is close to deterministic: where two tokens are
nearly tied, a different GPU model, driver or even a rerun on the same GPU can
flip the choice, and that episode then diverges. Re-scoring the paper's
terminal checkpoints on the WOFOST and DSSAT targets in a later pass left 68 of
159 trained no-think seed means bit-identical; trained seed means moved by at
most 0.16 (no-think) and 0.29 (think), at most 1.7 scenario-level standard
errors, and the untrained model's by up to 0.62 (paper, Appendix D). Two H200
machines with different drivers differed on 6 of 128 WOFOST episodes of one
checkpoint.
In a fresh install on an A40 (2026-10-25), T2.1 split `s1` of replicate r2
scored 1.0854 (no-think) and 6.0038 (think) against 1.0878 and 5.9592 in the
paper's table; against an A100 re-score of the same checkpoints, 2 of 128
no-think episodes and 63 of 128 think episodes differed (long reasoning traces
diverge more), and the split means differed by 0.3% and 0.6%.
A DSSAT example from a fresh install on one RTX 4090 (2026-10-25): T3.1
DSSAT-trained no-think r2 (`step-114`) on the T3.1 DSSAT split scored
8.00614849805832, equal to the table to all digits, with all 128 episodes
identical to the paper's bundle down to the raw model responses; a repeat on
the same GPU diverged in one episode and scored 7.9922.
Expect differences of that size when you re-score on other hardware.

**CycleGym price-regime suite (T1.3).** Its environment
(`CropPlanningRockSpringsRW-v1`) draws each episode's weather sequence when the
episode is reset, from NumPy's global random state. The evaluator builds all
environments of a split first and resets them afterwards, so the weather an
episode gets depends on the whole batch, not only on its own `seed`. Scoring
the complete split reproduces the paper's numbers (a re-scoring on other
hardware returned the same T1.3 means to all printed digits); scoring a subset
(`--max-samples`) or replaying one scenario alone gives other weather draws. In
a 64-scenario prefix check, all 1,216 decisions matched the paper run, while
gross revenue differed by 0.3% because of the weather.
