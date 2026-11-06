# Model card template (draft)

Template for the README of each released language-model checkpoint
repository. Fill the fields in `{braces}` from the repository's
`run_manifest.json` and the release tables. Every number in a card must be
copied from a row of the released CSV tables (name the file and row); never
type a number from memory. Keep the card identity-neutral: no personal names,
user names, e-mail addresses, cluster paths or tracker entities.

The front matter is Hugging Face card metadata.

````markdown
---
license: {apache-2.0 for Qwen3-based checkpoints; llama3.2 for Llama-3.2-3B-AgriManager-t32-unified-*; other (qwen-research) for t32-diffmodels-qwen25-3b-*}
base_model: Qwen/Qwen3-4B-Instruct-2507
library_name: transformers
pipeline_tag: text-generation
tags:
  - agriculture
  - crop-management
  - reinforcement-learning
  - grpo
  - agrimanager
datasets:
  - {test-suite dataset repository}
---

# {repository name}

A crop-management policy from the AgriManager benchmark (NeurIPS 2026,
Evaluations and Datasets track): `Qwen3-4B-Instruct-2507` fine-tuned with GRPO
on the **{setting name} ({setting id})** setting, **{think | no-think}**
response format, training seed **{seed}** (replicate {rK}).

- Paper: TODO(release-url)
- Code: https://github.com/AgriManager-Framework/AgriManager
- Held-out test suite: TODO(release-url)

## What this checkpoint is

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen3-4B-Instruct-2507` |
| Algorithm | GRPO on VeRL, 4 rollouts per prompt, actor learning rate 1e-6 |
| Setting | {setting id and name}; tier {1, 2 or 3} |
| Simulator(s) | {WOFOST / DSSAT / Cycles} |
| Training data | {N} scenarios, {train schema(s), menus or objectives} |
| Response format | {no-think: `<answer>…</answer>` / think: `<tool_call>…</tool_call> <answer>…</answer>`} |
| Training seed | {seed} (`data.seed` and `actor.data_loader_seed`) |
| Revisions | {list of `step-N` branches}; **terminal revision: `{step-N}`** (the one scored in the paper) |
| Early stop | {none / stopped at step N by the rollout filter} |
| Run script | `experiments/{setting dir}/{script}.sh` |
| Paper tables | {e.g. Tier 1 figure, raw-return table, per-seed table} |
| Evidence status | {confirmatory: one of three scored seeds of this arm / exploratory: single seed} |

## How to use it

The model expects the AgriManager prompt of its simulator and replies with one
action in the format above. Use it through the AgriManager environment loop;
it is not a general-purpose agronomy assistant.

```bash
bash entrypoints/eval/e5_evaluate_checkpoint.sh \
    --checkpoint-repo {org}/{repository name} --revision {terminal step} \
    --suite-repo {test-suite dataset repository} \
    --results-repo {your results repository} --tier {tier key} --no-upload
```

The evaluator reads the response format from `run_manifest.json` and refuses
to score the checkpoint in a different format.

## Results on the held-out test suite

Raw return of this seed on each test group, the untrained backbone in the same
format, no-action and the scripted agronomic reference. Source:
`e4_by_seed.csv` rows with `cell={cell}`, `arm={arm}`, `seed={seed}`;
references from `e4_anchors.csv`.

| Test group | This seed | Arm mean ± SD (3 seeds) | Untrained | No-action | Scripted reference |
|---|---|---|---|---|---|
| {group} | {value} | {mean ± sd} | {value} | {value} | {value} |

{For T1.1 only:} The scripted reference is at or below no-action on these
splits, so this setting is reported as gain over no-action and is not
normalized.

## Scope of the paper's claim for this setting

Copy the row for this checkpoint's setting. These are the forms the final
paper supports; do not state a stronger version.

| Setting | Supported statement |
|---|---|
| T1.1 weather-regime | With reasoning, RL lifts the policy above the untrained model on every split and crop, and above the fixed-interface PPO baseline trained with the same interaction budget. Without reasoning, training stalls near no-action. Scores are gains over no-action; the setting cannot be normalized. |
| T1.2 crop-variety | With reasoning, RL lifts the policy above the untrained model and above PPO with the same budget (`n1`) on seen and held-out crops at every coverage level. The eight-pass PPO (`n8`, appendix) matches it on this setting. The per-crop breakdown is exploratory. |
| T1.3 price-regime | RL lifts both formats far above the untrained model, and the think policy is above PPO with the same budget. Think and no-think are level, so no advantage from reasoning was found here. |
| T2.1 observation-schema | With reasoning, the trained policy is above the untrained model at every seed on the seven variants other than S2b (renamed, anonymized, extended, and three single-component removals). Removing the resource-state fields (S2b) is the worst variant, and RL training does not lift the policy above the untrained model there: information the decision needs cannot be recovered by training. |
| T2.2 action-menu | Results are consistent with joint action-menu coverage. Family A (trained on `lnpkw`, `lnpk`) is above the untrained model on all four test menus; Family B (trained on `lnpk`, `lw`) only on its own training menu `lw`. A plain "Family A above Family B" statement does not hold on every menu. |
| T2.3 reward-formulation | With reasoning, a policy trained on yield alone transfers to the three objectives it was not trained on; multi-objective training adds a little. Multi-objective training without reasoning is unstable (two of three seeds collapse). |
| T3.1 cross-simulator | Training in-domain improves every simulator over the untrained model. Single-source training mostly hurts zero-shot transfer: four of six transfers score below the untrained model; only the transfers into DSSAT gain. This is consistent with a limit set by transition dynamics the prompt does not describe. |
| T3.2 unified policy | The unified policy learns DSSAT and approaches its specialist there; the per-simulator specialists stay ahead on the WOFOST `lnw` profit and Cycles profit callback targets. Matching the specialists with one policy remains open. |
| Backbone comparisons (Llama-3.2, other sizes and algorithms) | Exploratory, one seed. On Llama-3.2 the no-think arm leads on most WOFOST tuples, so no general claim about reasoning is made from these runs. |

## Limitations

- Trained and evaluated in simulation only (WOFOST, DSSAT, Cycles); simulator
  scores are not field outcomes, and no recommendation from this model should
  be applied to a real farm.
- The scripted agronomic reference is a conservative codified schedule,
  uncalibrated against experts or optimal control. A normalized score above 1
  means the policy beats that schedule, nothing more.
- Scores are greedy-decoding scores; re-scoring on other GPU types changes a
  few episodes per split.
- {Setting-specific caveats, e.g. Cycles targets: 13 physical scenarios per
  suite; T1.1 `id` group is a residual group.}

## Citation

```bibtex
{the paper's BibTeX, TODO(release-url)}
```
````

## Filling checklist

- [ ] Front matter license fields set (code and weights license decided).
- [ ] Setting, arm, seed, terminal revision copied from `run_manifest.json`
      and the hub branches.
- [ ] Every number traced to a CSV row; the card names the file.
- [ ] The claim row matches the final paper text for that setting.
- [ ] No identity hazards (run `tools/release_identity_scan.py` on the card
      text).
- [ ] Internal codes such as `E2-Q-T-R2` removed or explained.
