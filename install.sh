#!/usr/bin/env bash
# Install AgriManager and its simulator dependencies into the active Python
# environment (Python 3.12; a fresh conda env or venv is recommended).
#
#   bash install.sh --cpu     # CPU only: WOFOST + Cycles environments, scripted and
#                             # numerical-RL baselines, dataset building, tests
#   bash install.sh --gpu     # full stack: adds CUDA 12.8 PyTorch, vLLM, FlashAttention
#                             # and the pinned package lock used for GRPO training
#                             # (default when no mode is given)
#
# Options:
#   --external-dir DIR   where the simulator wrappers are cloned
#                        (default: ../AgriManagerExternal next to this repository;
#                        the code looks there when WOFOST_GYM_PATH/CYCLES_GYM_PATH
#                        are unset)
#   --skip-cycles        do not install CyclesGym and the Cycles binary
#
# External code, each pinned to an exact commit (see THIRD_PARTY_NOTICES.md):
#   verl        git submodule, upstream verl-project/verl
#   WOFOSTGym   AgriManager fork of Intelligent-Reliable-Autonomous-Systems/WOFOSTGym
#   CyclesGym   AgriManager fork of kora-labs/cyclesgym; its installer downloads
#               the Cycles simulator binary from the PSUmodeling/Cycles releases
# DSSAT is optional and installed separately: docs/getting-started/install_dssat.md.
#
# Environment overrides: WOFOST_GYM_REMOTE, WOFOST_GYM_REF, CYCLES_GYM_REMOTE,
# CYCLES_GYM_REF, AGRI_PIP_CONSTRAINT (GPU: replace the lock file),
# AGRI_USE_LOCKFILE=0 (let pip resolve), FLASH_ATTN_WHEEL_URL, TORCH_CPU_INDEX_URL.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

MODE="gpu"
EXTERNAL_DIR="${AGRI_EXTERNAL_DIR:-$REPO_ROOT/../AgriManagerExternal}"
SKIP_CYCLES=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu) MODE="cpu"; shift ;;
        --gpu) MODE="gpu"; shift ;;
        --external-dir) EXTERNAL_DIR="$2"; shift 2 ;;
        --skip-cycles) SKIP_CYCLES=1; shift ;;
        -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
    esac
done
mkdir -p "$EXTERNAL_DIR"
EXTERNAL_DIR="$(cd "$EXTERNAL_DIR" && pwd)"

# Pinned external sources. The default remotes are the public forks; the refs are
# the exact code the paper used (tree-identical to the original working forks).
WOFOST_GYM_REMOTE="${WOFOST_GYM_REMOTE:-https://github.com/AgriManager-Framework/WOFOSTGym.git}"
WOFOST_GYM_REF="${WOFOST_GYM_REF:-253cf546f9b39a9efdb225049a1a48341a665c87}"
CYCLES_GYM_REMOTE="${CYCLES_GYM_REMOTE:-https://github.com/AgriManager-Framework/cyclesgym.git}"
CYCLES_GYM_REF="${CYCLES_GYM_REF:-2eb875834b71bb4e3a1ae59d0468551b41882331}"
CYCLES_GYM_LEGACY_GYM_VERSION="0.26.2"
GPU_LOCK_FILE="$REPO_ROOT/requirements/locks/agrimanager-py312-cu128.lock.txt"
CPU_LOCK_FILE="$REPO_ROOT/requirements/locks/agrimanager-py312-cpu.lock.txt"
CPU_REQUIREMENTS="$REPO_ROOT/requirements/cpu.txt"
TORCH_CPU_INDEX_URL="${TORCH_CPU_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl}"
AGRI_USE_LOCKFILE="${AGRI_USE_LOCKFILE:-1}"

WOFOST_GYM_DIR="$EXTERNAL_DIR/WOFOSTGym"
CYCLES_GYM_DIR="$EXTERNAL_DIR/CyclesGym"
PIP=(python -m pip)

step() { echo; echo "=== $* ==="; }

python - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    sys.exit(f"AgriManager is pinned for Python 3.12; this interpreter is {sys.version.split()[0]}.")
PY

checkout_pinned() {  # remote dir ref
    local remote="$1" dir="$2" ref="$3"
    if [[ ! -d "$dir/.git" ]]; then
        git clone "$remote" "$dir"
    fi
    if ! git -C "$dir" cat-file -e "$ref^{commit}" 2>/dev/null; then
        git -C "$dir" fetch --tags "$remote" "$ref" 2>/dev/null || git -C "$dir" fetch --tags "$remote"
    fi
    git -C "$dir" -c advice.detachedHead=false checkout --detach "$ref"
    echo "$(basename "$dir") HEAD: $(git -C "$dir" rev-parse HEAD)"
}

step "Mode: $MODE; simulator wrappers under $EXTERNAL_DIR"
"${PIP[@]}" install --upgrade pip setuptools wheel

step "Python packages"
NO_DEPS=(--no-deps)
if [[ "$MODE" == "cpu" ]]; then
    CONSTRAINT=()
    [[ -f "$CPU_LOCK_FILE" && "$AGRI_USE_LOCKFILE" != "0" ]] && CONSTRAINT=(-c "$CPU_LOCK_FILE")
    "${PIP[@]}" install --extra-index-url "$TORCH_CPU_INDEX_URL" -r "$CPU_REQUIREMENTS" "${CONSTRAINT[@]}"
    "${PIP[@]}" install "gym==$CYCLES_GYM_LEGACY_GYM_VERSION" "${CONSTRAINT[@]}"
else
    LOCK="${AGRI_PIP_CONSTRAINT:-}"
    [[ -z "$LOCK" && "$AGRI_USE_LOCKFILE" != "0" ]] && LOCK="$GPU_LOCK_FILE"
    if [[ -n "$LOCK" ]]; then
        [[ -f "$LOCK" ]] || { echo "lock file not found: $LOCK" >&2; exit 1; }
        "${PIP[@]}" install --no-deps -r "$LOCK"
        "${PIP[@]}" install --no-deps "$FLASH_ATTN_WHEEL_URL"
        "${PIP[@]}" install --no-deps "gym==$CYCLES_GYM_LEGACY_GYM_VERSION" cloudpickle
    else
        NO_DEPS=()
        "${PIP[@]}" install "gym==$CYCLES_GYM_LEGACY_GYM_VERSION"
    fi
fi

step "VeRL (git submodule)"
if [[ -e "$REPO_ROOT/.git" ]]; then
    git submodule update --init verl
fi
[[ -f "$REPO_ROOT/verl/setup.py" || -f "$REPO_ROOT/verl/pyproject.toml" ]] \
    || { echo "verl/ is empty: clone with --recursive or run 'git submodule update --init verl'" >&2; exit 1; }
if [[ "$MODE" == "gpu" && ${#NO_DEPS[@]} -eq 0 ]]; then
    (cd verl && USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh)
fi
"${PIP[@]}" install --no-deps -e verl

step "WOFOST-Gym"
checkout_pinned "$WOFOST_GYM_REMOTE" "$WOFOST_GYM_DIR" "$WOFOST_GYM_REF"
"${PIP[@]}" install "${NO_DEPS[@]}" -e "$WOFOST_GYM_DIR/pcse" -e "$WOFOST_GYM_DIR/pcse_gym"
"${PIP[@]}" install "${NO_DEPS[@]}" -e "$WOFOST_GYM_DIR/imitation" -e "$WOFOST_GYM_DIR/stable-baselines3"

if [[ "$SKIP_CYCLES" != "1" ]]; then
    step "CyclesGym and the Cycles binary"
    checkout_pinned "$CYCLES_GYM_REMOTE" "$CYCLES_GYM_DIR" "$CYCLES_GYM_REF"
    # setup.py's post-install hook is skipped; the native bootstrap runs explicitly
    # below so that a failed download is visible.
    (cd "$CYCLES_GYM_DIR" && CYCLES_GYM_SKIP_INSTALL=1 "${PIP[@]}" install --no-deps -e . && python install_cycles.py)
    for f in cycles/Cycles cycles/input/ContinuousCorn.operation cycles/input/GenericCrops.crop \
             cycles/input/GenericHagerstown.soil cycles/input/RockSprings.weather; do
        [[ -f "$CYCLES_GYM_DIR/$f" ]] || { echo "Cycles bootstrap incomplete: missing $CYCLES_GYM_DIR/$f" >&2; exit 1; }
    done
fi

step "AgriManager"
"${PIP[@]}" install --no-deps -e "$REPO_ROOT"

step "Check"
python - <<'PY'
import importlib
mods = ["agrimanager", "verl", "pcse", "pcse_gym", "stable_baselines3"]
try:
    importlib.import_module("cyclesgym")
    mods.append("cyclesgym")
except ImportError:
    pass
for m in mods:
    importlib.import_module(m)
print("imports ok:", ", ".join(mods))
PY

cat <<EOF

Done ($MODE). Simulator wrappers: $EXTERNAL_DIR
If this is not ../AgriManagerExternal next to the repository, export
  WOFOST_GYM_PATH=$WOFOST_GYM_DIR
  CYCLES_GYM_PATH=$CYCLES_GYM_DIR
Next: bash smoke_tests/wofost_gym/run_cpu_quickstart.sh
EOF
