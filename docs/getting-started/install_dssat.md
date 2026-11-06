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

## Requirements

- **Linux.** The Spack recipe is not reliable on macOS ARM64 (`zlib` and
  `libffi` can fail against the Apple SDK).
- **Python 3.9 to 3.12 as the system `python3`** to run Spack v0.19 (it does
  not bootstrap under Python 3.13). Spack builds its own Python 3.10 for
  DSSAT; AgriManager itself keeps running in its Python 3.12 environment.
- **GCC 11 with C, C++ and Fortran** (`gcc`, `g++`, `gfortran`). DSSAT is
  Fortran. Newer compilers (for example GCC 13) fail on older dependencies
  such as `qhull@2020.1`. No root access is needed: if your system has no
  GCC 11 `gfortran`, step 3 builds one with Spack.
- **A short path for Spack, at most 37 characters, in a directory named
  `spack`** (for example `$HOME/gd/spack` or `/opt/dssat/spack`). DSSAT reads
  its profile `DSSATPRO.L48` in 80-column records, and each record holds the
  runtime path plus a model name. With a longer path the model name is cut off
  and every episode fails (see [Common problems](#common-problems)). The
  recipe also rewrites paths only under a directory named `spack`.
- **A shell or batch job where local ZeroMQ sockets are allowed.** Restricted
  sandboxes fail with `Operation not permitted (src/ip_resolver.cpp:542)` even
  when the install is fine.
- **Time and disk.** Measured on 16 cores (2026-10-25): `spack install
  gym-dssat` took 21 minutes on a local SSD and 40 minutes on a cluster's
  NVMe-backed parallel file system; building GCC 11.3 first (step 3) adds
  about 14 minutes. The Spack directory takes 2 to 4 GB, plus Spack's build
  area in `$TMPDIR`.

## 1. Install AgriManager first

Follow [install.md](install.md) (`install.sh --cpu` is enough for scripted
baselines; use `--gpu` to score language models). All commands below run from
the AgriManager repository root, with its Python environment active.

## 2. Get Spack and the package repositories

Choose the Spack directory. Inside the checkout is the default that
AgriManager finds without configuration, but only if the path is short
enough:

```bash
export SPACK_DIR="$PWD/spack"            # or a short path such as "$HOME/gd/spack"
echo "${#SPACK_DIR} characters"          # must be 37 or fewer
# Keep Spack's user configuration next to it instead of in ~/.spack
export SPACK_USER_CONFIG_PATH="$(dirname "$SPACK_DIR")/.spack-user-config"
export SPACK_USER_CACHE_PATH="$(dirname "$SPACK_DIR")/.spack-user-cache"
```

Set the same three variables again in every new shell that runs `spack`.

```bash
git clone --branch v0.19.0 -c feature.manyFiles=true https://github.com/spack/spack.git "$SPACK_DIR"
source "$SPACK_DIR/share/spack/setup-env.sh"

# PDI and Gym-DSSAT package repositories, at the commits used for the paper
git clone https://github.com/pdidev/spack.git "$SPACK_DIR/var/spack/repos/pdi"
git -C "$SPACK_DIR/var/spack/repos/pdi" checkout 5bf8f6035c03bb0970d45e2931659dada3fba17e
git clone https://gitlab.inria.fr/rgautron/gym_dssat_pdi-spack.git "$SPACK_DIR/var/spack/repos/gym-dssat-pdi"
git -C "$SPACK_DIR/var/spack/repos/gym-dssat-pdi" checkout 2bab78bd627707469581412cf3b51b270a19ae2b
spack repo add "$SPACK_DIR/var/spack/repos/pdi"
spack repo add "$SPACK_DIR/var/spack/repos/gym-dssat-pdi"
```

Pin the PDI repository. Its current head builds PDI 1.12, which does not
compile with GCC 11.3 (`use of deleted function
std::hash<std::filesystem::path>`); the pinned commit gives PDI 1.11.0, the
version the paper's runtime used.

## 3. Choose the compiler

```bash
spack compiler find
spack compilers
grep -E "spec:|cxx:|fc:" "$SPACK_USER_CONFIG_PATH"/*/compilers.yaml
```

Spack picks the newest GCC it finds unless told otherwise, and a GCC without
`g++` or `gfortran` fails on the first package (`Compiler 'gcc@12.3.0' does
not have a C++ compiler configured`). Pin a GCC 11 that has both `cxx` and
`fc`:

```bash
spack config --scope user add "packages:all:compiler:[gcc@11.5.0]"   # use your version
```

If no GCC 11 entry has an `fc` (Fortran) path, build one with any GCC that has
`g++`, register it and pin it:

```bash
spack install -j16 gcc@11.3.0 %gcc@11.4.0          # %gcc@<a version with cxx>; about 15 min on 16 cores
spack compiler find "$(spack location -i gcc@11.3.0)"
spack config --scope user add "packages:all:compiler:[gcc@11.3.0]"
```

## 4. Build Gym-DSSAT and its environment view

```bash
spack install -j16 gym-dssat

spack env create gym-dssat-pdi "$SPACK_DIR/var/spack/repos/gym-dssat-pdi/env/spack.yaml"
spack env activate gym-dssat-pdi
spack concretize
spack env view regenerate
spack env deactivate

test -x "$SPACK_DIR/gym-dssat-pdi/bin/run_dssat" && echo "runtime found"
```

`spack concretize` reuses the packages from `spack install` and builds
nothing new. The view `$SPACK_DIR/gym-dssat-pdi` is the runtime directory
AgriManager needs.

## 5. Get the Python bridge

AgriManager imports the `gym_dssat_pdi` bridge from a source checkout, which
carries the maize, rice and cotton configurations used by the smoke tests
(some Spack-built `gym-dssat-0.0.8` bridges are maize-only).

```bash
mkdir -p ../AgriManagerExternal
git clone https://gitlab.inria.fr/rgautron/gym_dssat_pdi.git ../AgriManagerExternal/gym_dssat_pdi
git -C ../AgriManagerExternal/gym_dssat_pdi checkout 63f2c529e0bd339b4553beb9aa56d56af83b5e2b
```

The bridge talks to the simulator over ZeroMQ. `install.sh --gpu` already
installs `pyzmq`; after `install.sh --cpu`, add it to the AgriManager
environment:

```bash
pip install pyzmq==27.1.0
```

The paper's runs used `gym-dssat` 0.0.8, `gym_dssat_pdi` commit
`63f2c529e0bd339b4553beb9aa56d56af83b5e2b`, `dssat-pdi` 4.8.0.24_2 and PDI
1.11.0.

## 6. Connect AgriManager to the runtime

In each shell that runs DSSAT settings, with the AgriManager environment
active:

```bash
export DSSAT_GYM_PATH="$SPACK_DIR/gym-dssat-pdi"   # can be left unset if SPACK_DIR is ./spack
source smoke_tests/gym_dssat/_activate_spack.sh
```

The helper finds the bridge in `../AgriManagerExternal/gym_dssat_pdi` (set
`DSSAT_PDI_BRIDGE_DIR` to `<checkout>/gym-dssat-pdi` if it lives elsewhere),
adds the DSSAT library paths and keeps your Python 3.12 interpreter first.
Set `AGRI_DSSAT_ENABLE_SPACK_ACTIVATE=1` if you want it to activate the Spack
environment as well (this needs Spack in `./spack`).

## 7. Verify

```bash
test -x "$DSSAT_GYM_PATH/bin/run_dssat" && echo "runtime found"

# Live runtime check: imports the bridge, builds DSSATEnv and calls reset() with a timeout
bash smoke_tests/gym_dssat/run_runtime_smoke.sh --crop maize --timeout 240

# Smallest DSSAT dataset build (CPU)
bash smoke_tests/gym_dssat/run_build_datasets.sh maize
```

A working runtime prints `DSSAT_RUNTIME_SMOKE_STAGE reset_ok` and
`DSSAT_RUNTIME_SMOKE_OK` within a few seconds. `run_runtime_smoke.sh` also
accepts `--crop rice` and `--crop cotton`.

To check the runtime against the paper, build the two T3.2 DSSAT test-suite
splits (building needs no DSSAT runtime) and score the scripted anchors on
them:

```bash
bash experiments/rebuttal_confirmatory_eval/run_build_t32_dssat_eval_suite.sh
bash experiments/rebuttal_confirmatory_eval/run_build_t32_dssat_yield_eval_suite.sh
OUT_ROOT=$PWD/anchors_dssat JOBS=3 bash experiments/rebuttal_confirmatory_eval/run_e4_anchors.sh
```

`run_e4_anchors.sh` scores every suite under
`experiments/rebuttal_confirmatory_eval/data/`; with only the DSSAT suites
built it takes about four minutes on three cores. The means in
`anchors_dssat/<mode>/<suite>/test/metrics.json` should match the DSSAT rows
of `results/tables/e4_anchors.csv`: on `e4_t32_dssat_profit`, no-action
1.07375 and the paper's scripted reference (`paper_sim_heuristic`) 7.35868;
the current `ag_heuristic` gives 4.12322. The T3.1 DSSAT split holds the same
128 scenarios (`run_build_t31_eval_suite.sh` builds it together with the
WOFOST split, which needs the maize weather pool).

DSSAT is deterministic for a given build, but different compilers and CPUs
can round its single-precision state differently. Two fresh builds on
2026-10-25, each scoring the three scripted policies on all three DSSAT splits
(3 × 3 × 128 scenario returns):

| Build | Bit-identical to the paper's runs | Largest difference |
|---|---|---|
| RHEL 9, system GCC 11.5 | 1,152 of 1,152 | none |
| Ubuntu 22.04, GCC 11.3 built by Spack | 1,140 of 1,152 | 1.2e-7 relative (the same 4 scenarios on each split); means unchanged to eight significant digits |

GPU smoke tests (short GRPO run and model evaluation) are listed in
`smoke_tests/gym_dssat/README.md`.

## Common problems

| Symptom | Fix |
|---|---|
| `SbangPathError: Install tree root is too long` while Spack bootstraps `clingo` | The Spack directory path is too long; use a short `SPACK_DIR` (step 2) |
| `Compiler 'gcc@12.3.0' does not have a C++ compiler configured`, or a Fortran compiler is missing | Pin a GCC 11 with `g++` and `gfortran`, or build one (step 3) |
| PDI fails with `use of deleted function 'std::hash<std::filesystem::__cxx11::path>::hash()'` | The PDI repository is not pinned; check out the commit in step 2 and run `spack install` again |
| The runtime smoke test times out at `construct_start` or `reset_start`, and the newest file in `$DSSAT_GYM_PATH/dssat_logs/` says `Crop code incompatible with model specified ... DSSATPRO.L48 ... Error key: MODELN` | The Spack path is longer than 37 characters, so the model names in `DSSATPRO.L48` are cut off at column 80. Rebuild in a shorter `SPACK_DIR` (Spack installs cannot be moved) |
| `ModuleNotFoundError: No module named 'zmq'` | `pip install pyzmq==27.1.0` in the AgriManager environment (step 5) |
| `run_dssat` missing | Re-run `spack install` and `spack env view regenerate` inside the `gym-dssat-pdi` environment, then check `DSSAT_GYM_PATH` |
| `import gym_dssat_pdi` fails | `source smoke_tests/gym_dssat/_activate_spack.sh` before the command; set `DSSAT_PDI_BRIDGE_DIR` |
| Ray reports a socket path that is too long | `export RAY_LOG_DIR=$HOME/ray` |
| LLM evaluation says required arguments are missing | Run the wrapper script, for example `bash smoke_tests/gym_dssat/run_llm_eval_qwen3_4b_instruct.sh maize` |

When the runtime smoke test times out, read the newest file in
`$DSSAT_GYM_PATH/dssat_logs/` first: DSSAT stops at an error prompt and the
Python side waits, so the error is only in that log.

When reporting a DSSAT problem, include the failing command, the value of
`DSSAT_GYM_PATH`, whether `bin/run_dssat` exists there, and the last lines of
that log.

*Tested 2026-10-25 by following this page from a fresh release snapshot on
Ubuntu 22.04 (system GCC 11.4 without Fortran, so step 3 built GCC 11.3) and
on RHEL 9 (system GCC 11.5 with Fortran): runtime smoke tests for maize, rice
and cotton, the maize dataset build and the anchor check above all passed.*
