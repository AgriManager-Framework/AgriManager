# AgriManager Runtime Deployment

This directory contains the packaging, unpacking, and validation helpers for
moving an AgriManager runtime snapshot to another Linux server.

The recommended workflow is runtime-only deployment:

- package AgriManager code plus local runtimes, including DSSAT/Spack;
- do not package a Conda environment;
- create a fresh Conda environment on the target machine;
- unpack the runtime snapshot into a short path such as `/workspace/am`;
- run `install.sh` in the target Conda environment.

Using a short deployment path matters for DSSAT. Some DSSAT/Fortran paths have
strict length limits, and long nested paths can cause confusing errors such as
`MAKEFW 18`, `UFGA.CLI`, or truncated `MODEL.ERR` paths.
The deployment helper rejects paths where the deployed DSSAT `Climate`
directory would exceed 70 characters. Known-safe examples include `/tmp/am`,
`/root/am`, and `/workspace/am`.

## What Gets Packed

Runtime-only archives created by `pack_deploy.sh --runtime-only` include:

- the AgriManager repository snapshot;
- the packed `verl/` snapshot;
- external runtime snapshots under `AgriManagerExternal/`, when present;
- the local DSSAT/Spack runtime under `AgriManager/spack/`.

They do not include the target Conda environment. By default, use
`AGRI_DEPLOY_INCLUDE_GIT=0` for transfer packages so the archive is a pure
runtime snapshot without `.git` metadata.

Because this is a runtime snapshot, `git status` inside the unpacked
`AgriManager` directory may say `not a git repository`. That is expected. The
source commit is recorded in the manifest:

```bash
cat /workspace/agrimanager_pack/agrimanager-deploy-<timestamp>.manifest
```

For the current package, the recorded source commit is:

```text
8945dc9957e87e3c846276d37bef932351e414a1
```

If the target server should be used for development rather than just running
experiments, clone the repository separately with Git and then copy or unpack
the runtime assets into that checkout. Do not turn a slim runtime snapshot into
a development checkout unless you understand which files were intentionally
excluded from the archive.

Example package command:

```bash
AGRI_DEPLOY_INCLUDE_GIT=0 tools/deploy/pack_deploy.sh --runtime-only <pack_dir>
```

The current package layout is:

```text
agrimanager-code-runtimes-<timestamp>.tar.gz
agrimanager-deploy-<timestamp>.manifest
unpack_deploy.sh
validate_deploy.sh
README.deploy.md
```

## Transfer To A Server

Prefer `/workspace` on rented GPU instances. It is usually the intended working
area and avoids filling small root volumes.

Example transfer:

```bash
ssh -p <port> root@<host> 'mkdir -p /workspace/agrimanager_pack'

scp -P <port> \
  <pack_dir>/agrimanager-code-runtimes-<timestamp>.tar.gz \
  <pack_dir>/agrimanager-deploy-<timestamp>.manifest \
  <pack_dir>/unpack_deploy.sh \
  <pack_dir>/validate_deploy.sh \
  <pack_dir>/README.deploy.md \
  root@<host>:/workspace/agrimanager_pack/
```

If a previous transfer used `/root/agrimanager_pack`, keep compatibility with a
symlink:

```bash
mkdir -p /workspace/agrimanager_pack
mv /root/agrimanager_pack/* /workspace/agrimanager_pack/
rmdir /root/agrimanager_pack
ln -s /workspace/agrimanager_pack /root/agrimanager_pack
```

## Install On The Target Server

Create the Conda environment on the target server. Prefer an explicit prefix
under `/workspace`:

```bash
conda create -p /workspace/conda_envs/agrimanager python=3.12 -y
conda activate /workspace/conda_envs/agrimanager
```

Unpack the runtime snapshot:

```bash
/workspace/agrimanager_pack/unpack_deploy.sh --runtime-only \
  /workspace/agrimanager_pack/agrimanager-code-runtimes-<timestamp>.tar.gz \
  /workspace/am \
  "$(which python)"
```

Then install Python packages and native runtime hooks:

```bash
cd /workspace/am/AgriManager
bash install.sh
```

After installation, activate the deployment for later sessions with:

```bash
conda activate /workspace/conda_envs/agrimanager
source /workspace/am/activate_agrimanager.sh
```

The activation helper exports:

```bash
AGRIMANAGER_ROOT=/workspace/am/AgriManager
WOFOST_GYM_DIR=/workspace/am/AgriManagerExternal/WOFOSTGym
CYCLES_GYM_DIR=/workspace/am/AgriManagerExternal/CyclesGym
DSSAT_GYM_PATH=/workspace/am/AgriManager/spack/gym-dssat-pdi
```

## Validate The Package And Deployment

`validate_deploy.sh` can check the archive structure before unpacking. On the
source machine, run:

```bash
bash tools/deploy/validate_deploy.sh \
  --archive <pack_dir>/agrimanager-code-runtimes-<timestamp>.tar.gz
```

On the target server, after transfer but before unpacking, run:

```bash
bash /workspace/agrimanager_pack/validate_deploy.sh \
  --archive /workspace/agrimanager_pack/agrimanager-code-runtimes-<timestamp>.tar.gz
```

On the target server, after unpacking and running `install.sh`, validate the
deployed runtime:

```bash
conda activate /workspace/conda_envs/agrimanager

bash /workspace/agrimanager_pack/validate_deploy.sh \
  --archive /workspace/agrimanager_pack/agrimanager-code-runtimes-<timestamp>.tar.gz \
  --deploy-dir /workspace/am \
  --python "$(which python)"
```

The deployment validation checks:

- the runtime archive contains the expected AgriManager, VERL, WOFOST-Gym,
  CyclesGym, and DSSAT/Spack paths;
- DSSAT `run_dssat` exists and points inside the deployed runtime;
- `DSSATPRO.L48` points to the deployed path, not the source machine path;
- the DSSAT climate file `UFGA.CLI` is reachable from the deployed `CLD` path;
- the active Python imports AgriManager from the deployed repo;
- core modules import: `agrimanager`, `verl`, `pcse`, `pcse_gym`, `cyclesgym`;
- a minimal `DSSATEnv.reset()` smoke test succeeds.

If the DSSAT smoke test is too slow for a quick check, skip it:

```bash
bash /workspace/agrimanager_pack/validate_deploy.sh \
  --deploy-dir /workspace/am \
  --python "$(which python)" \
  --skip-dssat-smoke
```

## Common Transfer Problems

`Permission denied (publickey)` means the source machine's public key is not in
the target server's authorized keys. Add the public key on the target server and
retry `ssh` or `scp`.

`MAKEFW 18` with `UFGA.CLI` usually means DSSAT can see the profile file but the
profile still points to an old or invalid climate-data path. Re-run
`unpack_deploy.sh` with the target Conda Python, or validate with
`validate_deploy.sh` to inspect the rewritten `DSSATPRO.L48` paths.

Truncated paths such as `.../bin/MODE` instead of `.../bin/MODEL.ERR`, or
`MAKEFW` failures with an existing `UFGA.CLI`, usually mean the deployment path
is still too long for DSSAT. Move the deployment to a shorter path, for example
`/workspace/am` or `/tmp/am`, and re-run `unpack_deploy.sh`.
