# Developer notes

These pages describe how the repository is organized and the conventions for
adding experiments, configs, prompts and environments. They are for people
changing the code; users of the benchmark can skip them.

| Topic | Page |
|---|---|
| Repository boundaries: `entrypoints/`, `experiments/`, batch launch files; overrides flow downward; pluggable environments | [architecture.md](architecture.md) |
| Experiment and smoke-test layout, fixed `run_*.sh` scripts, batch launch files | [conventions/experiment_conventions.md](conventions/experiment_conventions.md) |
| What belongs in git; ownership between AgriManager and the external simulator repositories | [conventions/repository_rules.md](conventions/repository_rules.md) |
| Dataset config schema and validation conventions | [conventions/dataset_config_conventions.md](conventions/dataset_config_conventions.md) |
| WOFOST prompt blocks and where to edit them | [conventions/prompt_design_conventions.md](conventions/prompt_design_conventions.md) |
| Overrides in GRPO and NN run scripts | [conventions/training_script_conventions.md](conventions/training_script_conventions.md) |
| Environment adapter contract; adding a simulator | [reference/environment_adapter_contract.md](reference/environment_adapter_contract.md) |
| WOFOST dataset contract | [reference/wofost_dataset_contract.md](reference/wofost_dataset_contract.md) |
| Reward-variance filter used during GRPO | [reference/reward_variance_filter.md](reference/reward_variance_filter.md) |
| Experiment-tracker metric names | [reference/wandb_logging_overview.md](reference/wandb_logging_overview.md) |

## Rules in short

- Each `experiments/<setting>/run_*.sh` is one complete, fixed run: it
  hardcodes paths, model and Hydra overrides and accepts no free-form flags.
  A new variant gets a new script.
- Overrides are Hydra `key=value`. Experiments override entrypoints;
  entrypoints never read from experiments.
- An environment is a module `agrimanager/env/<env_name>/` discovered by name;
  the core code has no per-simulator branches.
- Generated data, results, logs and simulator runtimes stay out of git
  (`data/`, `results/`, `logs/`, `.cache/`, `spack/` are ignored).
- Before publishing anything (checkpoints, result bundles, cards, the
  repository), run `python tools/release_identity_scan.py` and pass the names
  that must not appear through `RELEASE_FORBID_TERMS`; the names are never
  stored in the repository.
