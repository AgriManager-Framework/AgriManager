# Small demo: train and score a policy on one consumer GPU, or on a CPU

This demo runs the whole AgriManager loop at a size that fits one 24 GB GPU
(for example an RTX 4090; about 2 hours, no-think) or a CPU (about 12 minutes):

1. build a small WOFOST maize task from the public maize weather pool;
2. score three scripted references on held-out scenarios: no action, a random
   valid action, and the scripted agronomic reference (`ag_heuristic`);
3. **`--gpu`**: score the untrained language model `Qwen/Qwen3-0.6B`, train it
   with GRPO for 120 steps, and score it again (`--think` for the reasoning
   format);
   **`--cpu-nn`**: train the numeric PPO baseline (Stable-Baselines3) and score it;
4. print a score table and save a training curve.

> **This is a demo, not a paper result.** It uses 128 training scenarios and
> 64 test scenarios drawn for the demo, a 0.6B model, and hours of training on
> one GPU. The paper's numbers come from `Qwen/Qwen3-4B-Instruct-2507`, 1,600
> training scenarios, 200 or more GRPO steps on H200 GPUs and the frozen test
> suites ([Evaluation](../../docs/guides/evaluation.md)).

## The task

The environment is the one in the CPU quickstart
(`smoke_tests/wofost_gym/config/quickstart_maize_cpu.yaml`): WOFOST maize, the
`lnpkw-v0` action menu (no-op, or one of four levels of N, P, K or irrigation),
one decision every 10 days for up to 24 decisions, and the `profit_max`
objective (yield minus input costs, in units of 1,000 kg/ha WSO-equivalent;
higher is better). By default the language model answers without a reasoning
block (no-think, 64-token answers); `--think` turns Qwen3's thinking on and
requires `<think>...</think><answer>...</answer>` (512-token answers). Answers
are parsed strictly: an answer that lists several amounts, such as
`Apply 5.0, 10.0, or 15.0 cm irrigation depth.`, is an invalid action (no-op).

Data (`config/small_demo_maize.yaml`, `generation_seed: 42`):

| Split | Scenarios | Drawn from | Used for |
|---|---|---|---|
| `train` | 128 | the pool's training sites | GRPO / PPO training |
| `val` | 16 | the pool's held-out sites | the training curve |
| `test` | 64 | other held-out site-years | the score table |

`val` and `test` are disjoint slices of one draw from the pool's held-out
split, whose sites never occur in training.

## Requirements

| | `--gpu` | `--cpu-nn` |
|---|---|---|
| Hardware | one NVIDIA GPU with 24 GB, otherwise idle; 8 or more CPU cores; 32 GB RAM (27 GB peak measured) | any x86-64 Linux machine, 8 or more cores |
| Install | `bash install.sh --gpu` | `bash install.sh --cpu` (or `--gpu`) |
| Disk | about 20 GB (13 GB environment, 1.5 GB model, 3 GB run outputs) plus up to 17 GB of resumable checkpoints while training | about 3 GB |
| Downloads on first run | `Qwen/Qwen3-0.6B` (1.5 GB) and the maize weather pool (0.25 GB) from Hugging Face | the maize weather pool |

## One command

From a fresh clone (see [Installation](../../docs/getting-started/install.md)):

```bash
uv venv --seed -p 3.12 .venv && . .venv/bin/activate
bash install.sh --gpu
bash examples/small_demo/run_small_demo.sh --gpu
```

CPU only:

```bash
uv venv --seed -p 3.12 .venv && . .venv/bin/activate
bash install.sh --cpu
bash examples/small_demo/run_small_demo.sh --cpu-nn
```

Outputs go to `examples/small_demo/runs/<mode>/`: the table in `summary.md`
(also printed), the curve in `training_curve.png`, all numbers in
`summary.json`, per-episode results under `eval/`, logs under `logs/`, and the
trained model under `grpo/checkpoints/global_step_<N>/actor/huggingface`
(GPU) or `nn_train/agent.zip` (CPU). Rerunning with the same command reuses
the dataset and the scores that already exist and resumes GRPO from its last
checkpoint (saved every 20 steps), so a run can be split across time-limited
jobs; `--fresh` starts over. `--help` lists the options.

## What to expect

Measured on 2026-10-26. The numbers will differ on your machine: GRPO and vLLM
are not bit-for-bit reproducible across GPUs, and a 0.6B model lands in
different places from run to run (see below).

### Time and memory

| Phase | `--gpu`, one RTX 4090 (24 GB) | `--cpu-nn`, 32-thread Linux server |
|---|---|---|
| `install.sh` (fresh venv) | 4.7 min (8 to 10 min with an empty pip cache on a cluster node) | 1 min (`--cpu`, pip cache filled) |
| Build the dataset | 7 s (plus about 10 s to download the pool once) | 10 s |
| Scripted references (CPU; beside the GPU phases in `--gpu`) | 2.5 min | 2.3 min |
| Score the untrained model, greedy and sampled | about 4 min (mostly vLLM start-up) | |
| Training | GRPO: 59 s per step, so about 2 h for 120 steps, plus a few minutes of start-up, validation and checkpoints | PPO, 10 passes (21k environment steps): 7 to 9 min |
| Score the trained policy | about 4 min | 1.2 min |
| **Total after install** | **about 2 h 15 min** | **about 12 min** |
| Peak GPU memory (whole GPU, `nvidia-smi`) | 17.2 GiB during GRPO, 13 GiB while scoring | |

The GRPO step time is the mean of 22 steps on an otherwise idle RTX 4090
(about 11 s rollouts, 37 s policy update, 10 s log-probabilities); the
120-step total multiplies it out. The 120-step runs below ran on one A100
(40 GB) with vLLM held to the same 8.4 GB it gets on a 24 GB card: 61 to 69 s
per step (the host CPU, which runs the simulator, is slower), GRPO memory peak
15.7 to 17.0 GiB, in one-hour jobs that resumed from checkpoints.

`--think` (512-token answers, measured on an A100 and an H200 with the same
8.4 GB vLLM share; no RTX 4090 was free): 224 s per step on the A100 (mean of
10 steps, 185 to 244 s; about 7.5 h for 120 steps; 77 s on the H200), GRPO
memory peak 17.3 GiB on the A100, which should fit a 24 GB card; this was not
run on one.

### Scores on the 64 test scenarios

Objective = profit (yield revenue minus input costs) in 1,000 kg/ha
WSO-equivalent. Scripted references, the same in every run:

| Policy | Mean | SD |
|---|---|---|
| No action | 0.290 | 0.366 |
| Random valid action | 0.586 | 2.020 |
| Scripted agronomic reference | 1.263 | 1.479 |

`--gpu` with the defaults (`Qwen/Qwen3-0.6B`, 120 steps, lr 3e-6, 8
scenarios × 4 rollouts per step), three seeds. *Greedy* is temperature 0, the
paper's evaluation protocol; *sampled* is temperature 1, the policy GRPO
actually trains:

| Run | Untrained, greedy | Trained, greedy | Untrained, sampled | Trained, sampled |
|---|---|---|---|---|
| no-think, seed 1 (collapsed; stopped at step 21) | -1.010 | 0.290 | -0.728 | 0.290 |
| no-think, seed 2 | -1.007 | **0.689** | -0.882 | 0.177 |
| no-think, seed 3 | -1.004 | **0.616** | -0.853 | **1.719** |
| `--think`, seed 1 (H200) | 0.138 | 0.290 | 0.037 | 0.290 |

Paired over the 64 scenarios (mean difference, 95% interval), trained greedy
against no action: seed 2 +0.40 [-0.01, +0.81], seed 3 +0.33 [+0.23, +0.43];
against the scripted reference: -0.57 [-1.13, -0.02] and -0.65 [-1.02, -0.28].
Seed 3 sampled against the scripted reference: +0.46 [-0.07, +0.98].

What this shows:

- **The untrained model is worse than doing nothing.** With thinking switched
  off, Qwen3-0.6B copies the example answer in the prompt ("Apply 20.0 kg/ha
  nitrogen fertilizer") at every decision; nitrogen alone, every ten days, costs
  more than it adds. Sampled at temperature 1 it copies the whole menu line
  ("Apply nitrogen fertilizer (20.0, 40.0, 60.0, or 80.0 kg/ha)") in about a
  quarter of its answers, which the strict parser counts as invalid.
- **GRPO improves the policy, but at this size not past the scripted
  reference.** In the two seeds that kept training, the greedy policy ended
  above no action (+0.33 and +0.40) and below the scripted reference (-0.57 and
  -0.65); it settles on one or two fixed answers ("Apply 5.0 cm of water" at
  every decision in seed 3; 20 kg/ha of nitrogen or of potassium in seed 2).
  The sampled policy of seed 3 is level with the scripted reference.
- **One of three seeds collapsed to "Do nothing".** In seed 1 the policy
  answered "Do nothing" at every decision by step 8; every group of rollouts
  then had the same reward, the rollout filter skipped five updates in a row
  and stopped the run at step 21. The last checkpoint is scored, so the result
  equals no action. The paper's no-think runs stop the same way
  ([Training](../../docs/guides/training.md)). Rerun with another `--seed` if it
  happens.
- **`--think` does not help a 0.6B model here.** Untrained, it runs past the
  512-token limit or breaks the format in 70% of its answers (invalid, so
  no-op); after 120 steps it reasons and then answers "Do nothing" at every
  decision, which scores exactly as no action.
- **The strict parser matters.** Under the earlier lenient parser (first amount
  in the answer, still available as `action_parser: legacy`), a 120-step
  no-think run (seed 2) learned to answer with option lists such as
  `Apply 5.0, 10.0, or 15.0 cm irrigation depth.` and scored 4.82 greedy, but
  every one of those answers is invalid under the strict parser, and the same
  checkpoint scores 0.290 (all no-op). The strict parser is the default for
  that reason.
- With `--lr 1e-5` the model drifted into long answers that are cut off at 64
  tokens and count as invalid (no-op); after 10 steps every test decision was
  invalid.

`--cpu-nn` (PPO, 10 passes over the 128 training scenarios, 8 parallel
environments). The evaluator samples actions by default, as the paper's neural
baselines do; the most likely action is scored as well:

| Seed | PPO, sampled actions | PPO, most likely action |
|---|---|---|
| 1 | 1.401 | 3.778 |
| 2 | 0.983 | 1.954 |
| 3 | 0.522 | 3.323 |

The numeric policy learns this small task in minutes: its most likely action
beats the scripted reference in all three seeds (+2.51 [+1.94, +3.09],
+0.69 [+0.03, +1.35], +2.06 [+1.56, +2.56], paired). Its sampled actions,
the paper's evaluation setting for the neural baselines, score +0.14
[-0.51, +0.79], -0.28 [-0.89, +0.33] and -0.74 [-1.35, -0.13] against it. The
language model has to learn the same agronomy from text, one sampled answer at
a time.

Example output of `--gpu` (seed 3 above):

```text
| Policy                         | Objective mean |      SD | Invalid actions |
|--------------------------------|----------------|---------|-----------------|
| No action                      |          0.290 |   0.366 |            0.0% |
| Random valid action            |          0.586 |   2.020 |            0.0% |
| Scripted agronomic reference   |          1.263 |   1.479 |            0.0% |
| LLM untrained, greedy          |         -1.004 |   0.368 |            0.0% |
| LLM after GRPO, greedy         |          0.616 |   0.301 |            0.0% |
| LLM untrained, sampled (T=1)   |         -0.853 |   1.325 |           24.0% |
| LLM after GRPO, sampled (T=1)  |          1.719 |   1.394 |            7.0% |
```

The training curve (`training_curve.png`) plots the greedy validation score
on 16 held-out scenarios every 5 steps, the sampled score of the training
episodes at every step, and the scripted references on the same 16 scenarios.

## Larger models on 24 GB

Measured on 2026-10-26 on Delta A100 (40 GB) and H200 nodes, with vLLM held to
the share it gets on a 24 GB card, strict parser, no-think, lr 1e-5 for LoRA:

| Model and training | Fits 24 GB? | Time per GRPO step | Result |
|---|---|---|---|
| `Qwen/Qwen3-1.7B`, all weights | no: 35.9 GiB peak (whole GPU), actor 41 GiB reserved | 57 to 66 s (A100) | not run further |
| `Qwen/Qwen3-1.7B`, LoRA rank 32 (`--lora-rank 32 --lr 1e-5`) | likely: 15.9 GiB peak (whole GPU, sampled); actor 21.0 GiB reserved, as in the 0.6B run that fit an RTX 4090 | about 47 s (A100) | untrained it answers "Do nothing" everywhere (0.290, no action); no group of rollouts differed, so the rollout filter stopped the run at step 18; trained score 0.290 |
| `Qwen/Qwen3-4B-Instruct-2507` (the paper's backbone), LoRA rank 32 | borderline: actor peak 21.6 GiB allocated, 22.6 GiB reserved, plus vLLM's sleeping residue; with base-weight offload the peak was 26.0 GiB | 330 to 440 s (A100; 250 s of it is the policy update), so about 13 h for 120 steps | after 5 steps greedy -0.32 to 1.00 and 1.04 in two probes (A100, H200); a 120-step run is in progress |

Neither larger model gives a 24 GB run of about two hours, so the default stays
at `Qwen/Qwen3-0.6B`. `--lora-rank` is available for cards with more memory
(the trained adapter is merged into the base by `merge_lora.py` before
scoring).

## Scaling up toward the paper's settings

| Setting | Demo | Paper (`experiments/*/run_*_train.sh`) |
|---|---|---|
| Model | `Qwen/Qwen3-0.6B`, full fine-tuning | `Qwen/Qwen3-4B-Instruct-2507` |
| Training scenarios | 128 | 1,600 (3,200 for T3.2) |
| Scenarios per step × rollouts | 8 × 4 | 16 × 4 |
| Steps | 120 | 200 (400 for T2.3, 800 for T3.2) |
| Learning rate | 3e-6 | 1e-6 |
| Response format and length | no-think, 64 tokens (`--think`: 512) | think and no-think arms, 512 tokens (1,024 in some T1.2 runs) |
| Rollout filter | on (paper default) | on |
| GPUs | one 24 GB | 2 (some settings 4) H200 per run |
| Test set | 64 demo scenarios | frozen test suites, 128 to 384 scenarios |
| Action parser | strict | legacy (`action_parser: legacy`, pinned by the paper's scripts) |

To move toward the paper: use `--model`, `--steps`, `--gen-batch`,
`--rollout-n` and `--lr`. Anything after `--` is passed to the trainer last,
so it overrides the script's settings, for example
`-- trainer.test_freq=10 trainer.rollout_filter.enable=False`. A 4B model does not fit next to its optimizer state on 24 GB; use the
paper's run scripts on larger GPUs, or LoRA (`actor_rollout_ref.model.lora_rank`,
see VeRL's LoRA guide; not tested here) and expect a much longer run. For a
different crop, objective or action menu, copy `config/small_demo_maize.yaml`
and change its `env` block
([dataset config conventions](../../docs/conventions/dataset_config_conventions.md)).

## If something goes wrong

| Symptom | Fix |
|---|---|
| `CUDA out of memory` during GRPO | the GPU is not idle, or it has less than 24 GB: lower `--rollout-mem` (vLLM's share, default 0.35), or pass `-- actor_rollout_ref.actor.ppo_max_token_len_per_gpu=1024` |
| `CUDA out of memory` while scoring | lower `gpu_memory_utilization` in `config/vllm_no_think.yaml` (or `vllm_think.yaml`) |
| Hugging Face download errors | check the network; `huggingface-cli login` if the weather pool asks for a token |
| A step fails half way, or a cluster job hits its time limit | rerun the same command; finished phases are kept and GRPO resumes from its last checkpoint |
