# AgriManager

![AgriManager overview](assets/main_diagram.png)

AgriManager is a framework and benchmark for training and evaluating
prompt-conditioned language-model policies for crop management. It puts three
crop simulators (DSSAT, WOFOST and Cycles) behind one text interface, trains
policies with reinforcement learning (GRPO on VeRL), and scores them on a
frozen held-out test suite.

The benchmark asks whether such policies generalize under two kinds of shift:

- **distributional shift** inside a fixed task interface (new weather, crops,
  prices), and
- **schema shift**, where the policy-facing schema changes between training
  and deployment: which variables are observed, which actions are available,
  which objective is rewarded, and, across simulators, the dynamics.

Eight settings arranged in three tiers test this. Every setting has a frozen
test suite with manifests, generation configs, seeds and disjointness audits.

This repository accompanies the NeurIPS 2026 Evaluations and Datasets track
paper *AgriManager: A Framework and Benchmark for LLM-RL Generalization in
Agricultural Management* (`TODO(paper-url)`: proceedings and arXiv links).

## Where to start

| I want to | Read |
|---|---|
| Install the code and simulators | [docs/getting-started/install.md](docs/getting-started/install.md) |
| Run something in five minutes on a CPU | [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md) |
| Understand the eight settings | [docs/benchmark/README.md](docs/benchmark/README.md) |
| Score my own policy on the held-out test suite | [docs/guides/evaluation.md](docs/guides/evaluation.md) |
| Reproduce a table or figure of the paper | [docs/guides/reproduce.md](docs/guides/reproduce.md) |
| Know how the test suite was built and audited | [docs/benchmark/test_suite.md](docs/benchmark/test_suite.md) |
| Download released checkpoints and data | [docs/release/models_and_data.md](docs/release/models_and_data.md) |
| Read the known limitations | [docs/faq.md](docs/faq.md) |

The full documentation index is [docs/README.md](docs/README.md).

## Install in short

```bash
git clone --recursive https://github.com/AgriManager-Framework/AgriManager.git
cd AgriManager
conda create -n agrimanager python=3.12 -y     # or: uv venv --seed -p 3.12 && source .venv/bin/activate
conda activate agrimanager
bash install.sh --cpu    # CPU only: WOFOST + Cycles, baselines, dataset building, tests
bash install.sh --gpu    # full stack for GRPO training and model scoring (CUDA 12.8)
```

Both modes pin every dependency: the VeRL submodule, WOFOST-Gym and CycleGym at
fixed commits (cloned into `../AgriManagerExternal/`), and the Python packages
(`requirements/cpu.txt` with `requirements/locks/agrimanager-py312-cpu.lock.txt`,
or `requirements/locks/agrimanager-py312-cu128.lock.txt`).
`install.sh` does not build the DSSAT runtime; follow
[docs/getting-started/install_dssat.md](docs/getting-started/install_dssat.md)
for DSSAT-backed settings. Hardware needs and environment variables are listed
in [docs/getting-started/install.md](docs/getting-started/install.md).

## Quickstart (CPU only)

```bash
bash smoke_tests/wofost_gym/run_cpu_quickstart.sh
```

This builds eight maize scenarios from the public maize weather pool and scores
two scripted policies on them (no-action and the scripted agronomic reference).
It needs `install.sh --cpu` but no GPU and no model download. See
[docs/getting-started/quickstart.md](docs/getting-started/quickstart.md) for
the expected output and the next steps.

## Repository layout

```text
agrimanager/            Python package
├── adapter/            VeRL adapter: agent loop, interactions, trainer, reward
├── checkpoint/         publishing checkpoints and scoring them on the test suite
├── env/                environment adapters (wofost_gym, gym_dssat, cycles_gym)
├── model_interface/    provider configs, models, server helpers
├── nn_ppo/             numeric baselines (PPO, MaskablePPO, A2C, DQN)
└── rollout/            batched rollouts and scripted baselines
entrypoints/            stable commands: dataset build, train, eval, tools
experiments/            one directory per benchmark setting, with fixed run scripts
smoke_tests/            small end-to-end checks per simulator
integrations/           AgriManager-side tools for external simulators
analysis/test_suite/    scripts that turn scored test-suite runs into the paper's tables
results/                the paper's tables (CSV), figures and the list of scored checkpoints
tests/                  unit tests (pytest)
docs/                   documentation
requirements/           pinned dependencies (CPU list, CPU and CUDA lock files)
install.sh              environment bootstrap (--cpu | --gpu)
```

## Citation

If you use AgriManager, please cite the paper. A machine-readable entry is in
[CITATION.cff](CITATION.cff).

```bibtex
@inproceedings{agrimanager2026,
  title     = {AgriManager: A Framework and Benchmark for {LLM-RL} Generalization
               in Agricultural Management},
  author    = {TODO(authors)},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Evaluations and Datasets Track},
  year      = {2026},
}
```

Please also cite the simulators and frameworks AgriManager builds on; see
[docs/reference/upstream_citations.md](docs/reference/upstream_citations.md).

## License

The code is licensed under the [Apache License 2.0](LICENSE) (see also
[NOTICE](NOTICE)). The datasets we publish are licensed under CC BY 4.0, and the
released model weights under Apache-2.0, except checkpoints fine-tuned from a
base model whose license says otherwise (Llama 3.2, Qwen2.5-3B). Simulators and
wrappers are not vendored; each keeps its own license, listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The native Cycles simulator
binary that CycleGym downloads at install time is distributed under
CC BY-NC-ND 4.0.

## Contributing

Developer conventions (experiment layout, run scripts, configs) are in
[CONTRIBUTING.md](CONTRIBUTING.md).
