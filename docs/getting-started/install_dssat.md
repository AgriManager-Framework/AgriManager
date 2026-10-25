# DSSAT runtime (optional)

The DSSAT parts of the cross-simulator (T3.1) and unified-policy (T3.2)
settings run on the DSSAT crop model through Gym-DSSAT (`gym_dssat_pdi`). Its
native runtime is built with Spack and is not part of `install.sh`. You can
skip this page if you only use WOFOST or Cycles.

AgriManager does not ship DSSAT. It contains the adapter
(`agrimanager/env/gym_dssat/`), dataset generators, configs and run scripts
that drive the upstream runtime. Versions and licenses of the upstream
components are listed in
[dataset_and_generator_sources.md](../reference/dataset_and_generator_sources.md).

Upstream installation guide:
<https://rgautron.gitlabpages.inria.fr/gym-dssat-docs/Installation/spack.html>

## Platform notes

- Use Linux. The Spack recipe is not reliable on macOS ARM64 (`zlib` and
  `libffi` can fail against the Apple SDK).
- Run Spack v0.19 under Python 3.9 to 3.12; it does not bootstrap under
  Python 3.13.
- Use GCC 11.x. Newer compilers (for example GCC 13) fail on older
  dependencies such as `qhull@2020.1`. The validated setup used GCC 11.4.
- Run the simulator from a normal shell or a batch job where local ZeroMQ
  sockets are allowed. Restricted sandboxes fail with
  `Operation not permitted (src/ip_resolver.cpp:542)` even when the install
  is fine.
- Expect 1 to 2 hours for the build.

## 1. Install AgriManager first

Follow [install.md](install.md) so that the `agrimanager` Conda environment,
VeRL, WOFOST-Gym and CycleGym are in place. All commands below run from the
AgriManager repository root.

## 2. Build Gym-DSSAT with Spack

These commands follow the upstream Spack flow and keep Spack inside the
AgriManager checkout (the directories `spack/`, `.spack-user-cache/` and
`.spack-user-config/` are ignored by git).

```bash
# Spack itself
git clone --branch v0.19.0 -c feature.manyFiles=true https://github.com/spack/spack.git
source spack/share/spack/setup-env.sh

# PDI and Gym-DSSAT package repositories
git clone https://github.com/pdidev/spack.git spack/var/spack/repos/pdi
spack repo add spack/var/spack/repos/pdi
git clone https://gitlab.inria.fr/rgautron/gym_dssat_pdi-spack.git spack/var/spack/repos/gym-dssat-pdi
spack repo add spack/var/spack/repos/gym-dssat-pdi

# Build
spack install gym-dssat
```

Clone the Python bridge as well. AgriManager prefers this checkout because it
carries the maize, rice and cotton configurations used by the smoke tests;
some Spack-built `gym-dssat-0.0.8` bridges are maize-only.

```bash
mkdir -p ../AgriManagerExternal
git clone https://gitlab.inria.fr/rgautron/gym_dssat_pdi.git ../AgriManagerExternal/gym_dssat_pdi
```

The paper's runs used `gym-dssat` 0.0.8, `gym_dssat_pdi` commit
`63f2c529e0bd339b4553beb9aa56d56af83b5e2b` and `dssat-pdi` 4.8.0.24_2.

Create and concretize the Spack environment from the recipe in the Gym-DSSAT
Spack repository, then regenerate its view:

```bash
spack env create gym-dssat-pdi spack/var/spack/repos/gym-dssat-pdi/env/spack.yaml
spack env activate gym-dssat-pdi
spack concretize
spack env view regenerate
spack env deactivate
```

Find the view directory that contains `bin/run_dssat`. AgriManager looks in
`spack/gym-dssat-pdi` by default; the upstream recipe may create a
`gymdssat_spackenv`-style directory elsewhere.

```bash
test -x spack/gym-dssat-pdi/bin/run_dssat || test -x "$HOME/gymdssat_spackenv/bin/run_dssat"
```

## 3. Connect AgriManager to the runtime

```bash
conda activate agrimanager
export DSSAT_GYM_PATH="$PWD/spack/gym-dssat-pdi"     # or the view path you found above
export DSSAT_PDI_BRIDGE_DIR="$PWD/../AgriManagerExternal/gym_dssat_pdi/gym-dssat-pdi"   # only if not found automatically
source smoke_tests/gym_dssat/_activate_spack.sh
```

The helper collects the DSSAT library paths from the runtime and restores the
Conda Python, so AgriManager keeps running under Python 3.12. Set
`AGRI_DSSAT_ENABLE_SPACK_ACTIVATE=1` if you want it to activate the Spack
environment first.

> **Release note.** `smoke_tests/gym_dssat/_activate_spack.sh` and
> `agrimanager/env/gym_dssat/env.py` still list one cluster-specific fallback
> path. It is only tried after the paths above and will be removed before
> release.

## 4. Verify

```bash
test -x "$DSSAT_GYM_PATH/bin/run_dssat" && echo "runtime found"
python -c "from agrimanager.env.gym_dssat.env_config import DEFAULT_DSSAT_GYM_PATH; print(DEFAULT_DSSAT_GYM_PATH)"

# Live runtime check: imports the bridge, builds DSSATEnv and calls reset() with a timeout
bash smoke_tests/gym_dssat/run_runtime_smoke.sh --crop maize --timeout 240

# Smallest DSSAT dataset build (CPU)
bash smoke_tests/gym_dssat/run_build_datasets.sh maize
```

If the runtime smoke test times out at `construct_start` or `reset_start`, the
runtime is not working yet even if the bridge imports. Check
`smoke_tests/gym_dssat/logs/gym_dssat_runtime_smoke_<crop>.log` and the DSSAT
log directory printed by `DSSATEnv` (under the runtime view, for example
`spack/gym-dssat-pdi/dssat_logs/`).

GPU smoke tests (short GRPO run and model evaluation) are listed in
`smoke_tests/gym_dssat/README.md`.

## Common problems

| Symptom | Fix |
|---|---|
| `run_dssat` missing | Re-run `spack install` and `spack env view regenerate` inside the `gym-dssat-pdi` environment, then check `DSSAT_GYM_PATH` |
| `import gym_dssat_pdi` fails | `source smoke_tests/gym_dssat/_activate_spack.sh` before the command; set `DSSAT_PDI_BRIDGE_DIR` |
| Ray reports a socket path that is too long | `export RAY_LOG_DIR=$HOME/ray` |
| LLM evaluation says required arguments are missing | Run the wrapper script, for example `bash smoke_tests/gym_dssat/run_llm_eval_qwen3_4b_instruct.sh maize` |

When reporting a DSSAT problem, include the failing command, the value of
`DSSAT_GYM_PATH`, and whether `bin/run_dssat` exists there.

*Not executed for this documentation revision: the steps above are merged from
the two earlier DSSAT guides of the project and were not re-run on a clean
machine.*
