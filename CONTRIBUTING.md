# Contributing

Thank you for improving AgriManager. Repository conventions (experiment
layout, fixed run scripts, configs, prompts, environment adapters) are
collected in [docs/contributing.md](docs/contributing.md).

Before opening a pull request:

- keep each experiment run as a fixed `experiments/<setting>/run_*.sh` script;
- keep generated data, results, logs and simulator runtimes out of git;
- run `python tools/release_identity_scan.py` on your change.

Bug reports are most useful with the failing command, the simulator and its
path variable (`WOFOST_GYM_PATH`, `CYCLES_GYM_PATH` or `DSSAT_GYM_PATH`), and the
log file the command printed.
