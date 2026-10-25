# AgriManager documentation

AgriManager trains and evaluates prompt-conditioned language-model policies for
crop management on three simulators (DSSAT, WOFOST, Cycles) and measures how
they generalize under distributional shift and schema shift. These pages are
ordered by what a reader usually needs first.

## Get started

1. [Installation](getting-started/install.md): requirements, `install.sh`,
   environment variables, a CPU-only install.
2. [DSSAT runtime](getting-started/install_dssat.md): the optional Spack build
   for DSSAT-backed settings.
3. [Quickstart](getting-started/quickstart.md): score two scripted policies on
   eight maize scenarios in a few minutes on a CPU, and print a real prompt.

## The benchmark

4. [Benchmark overview](benchmark/README.md): the three tiers and eight
   settings, with their training, validation and test splits.
5. [Environments and simulators](benchmark/environments.md): what each
   simulator exposes and how scenarios are defined.
6. [Schema and prompts](benchmark/schema_and_prompts.md): the schema
   `Σ = (Σ_O, Σ_A, Σ_R)`, prompt blocks, think and no-think formats, and how each
   setting shifts the schema.
7. [Reward objectives](benchmark/reward_objectives.md): objective formulas.
8. [Baselines and reference policies](benchmark/baselines.md): no-action, the
   scripted agronomic reference (with its disclosures), neural baselines,
   normalized return.
9. [The held-out test suite](benchmark/test_suite.md): construction, timeline,
   seeds, manifests, disjointness audits, the Cycles rebuild.

## Use it

10. [Training](guides/training.md): GRPO and neural-baseline training, the
    paper's run scripts and seeds.
11. [Evaluation on the held-out test suite](guides/evaluation.md): scoring a
    checkpoint, your own model, references and baselines.
12. [Reproducing the paper](guides/reproduce.md): from checkpoints to CSV tables
    to figures, table by table.

## Release

13. [Models and data](release/models_and_data.md): datasets, checkpoints,
    revisions, licenses.
14. [Model card template](release/model_card_template.md) and
    [test-suite dataset card](release/test_suite_dataset_card.md) (drafts).
15. [FAQ and limitations](faq.md).

## Reference

- [Upstream citations](reference/upstream_citations.md) and
  [upstream sources, versions and licenses](reference/dataset_and_generator_sources.md).
- [Developer notes](contributing.md): repository architecture and conventions,
  environment adapter contract, dataset contract.
- Command reference for the stable scripts: `entrypoints/README.md` and
  `entrypoints/eval/README.md` in the repository.

## Which page answers my question?

| Question | Page |
|---|---|
| How do I install it, and what hardware do I need? | [Installation](getting-started/install.md) |
| Can I try it without a GPU? | [Quickstart](getting-started/quickstart.md) |
| What exactly changes between train and test in setting T2.2? | [Benchmark overview](benchmark/README.md#t22-action-menu-shift), [Schema and prompts](benchmark/schema_and_prompts.md) |
| How were the test scenarios chosen, and are they disjoint from training? | [Test suite](benchmark/test_suite.md) |
| Which seeds and checkpoints are behind a paper number? | [Training](guides/training.md#paper-runs-and-seeds), [Models and data](release/models_and_data.md) |
| How do I recompute a table of the paper? | [Reproducing the paper](guides/reproduce.md) |
| What is the "scripted agronomic reference", and why is T1.1 not normalized? | [Baselines](benchmark/baselines.md) |
| Why does the neural baseline not appear on Tier 2? | [Schema and prompts](benchmark/schema_and_prompts.md#what-the-neural-baselines-can-and-cannot-do) |
| How do I add a new simulator? | [Environment adapter contract](reference/environment_adapter_contract.md) |
