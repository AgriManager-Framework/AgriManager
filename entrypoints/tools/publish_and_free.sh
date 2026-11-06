#!/usr/bin/env bash
# Publish every local checkpoint of a set of finished runs to the hub, verify
# each one against its own manifest, and only then delete the local shards.
#
# Usage:
#   bash entrypoints/tools/publish_and_free.sh --queue <file.tsv> [--owner agrimanager]
#                                              [--work-dir <dir>] [--public]
#                                              [--dry-run]
#
# The queue is a tab-separated file, one finished run per line, comments and
# blank lines ignored:
#
#   run_name <TAB> run_dir <TAB> seed <TAB> wandb_id <TAB> run_definition <TAB> cell_id <TAB> dataset_dir [<TAB> total_training_steps]
#
# The eighth column is optional and defaults to 200. The think/no-think arm is
# NOT a column: it is derived from the run name, because a manifest that claims
# the wrong arm is worse than one that omits it.
#
# Every `global_step_N` under run_dir is published to revision `step-N` of the
# single repository <owner>/<run_name>, which is the layout the earlier batches
# already use.
#
# Why this exists as its own tool rather than a loop in a shell:
#
#   Local disk on a shared cluster is often staging, not storage, so freeing it is a step
#   of running the next batch and not a cleanup at the end. That makes deletion
#   routine, and routine deletion is where an unverified upload turns into a
#   lost checkpoint. A branch can be created and left holding nothing but
#   .gitattributes -- a storage-quota failure produced exactly that here once
#   already -- so "the upload command exited 0" is not evidence. Nothing is
#   deleted here until verify_published has matched every file in the revision's
#   run_manifest.json against the hub by size and, for the weights, by SHA-256.
#
#   Runs are processed one at a time, and each merge directory is discarded
#   before the next begins, so the peak extra space this needs is one merged
#   model (~8 GB) and not the size of the queue.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

QUEUE=""
OWNER="AgriManager-Framework"
WORK_DIR="${AGRI_WORK_DIR:-$HOME/agrimanager_work}/agri_publish_work"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
PUBLIC=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --queue)      QUEUE="$2";      shift 2 ;;
        --owner)      OWNER="$2";      shift 2 ;;
        --work-dir)   WORK_DIR="$2";   shift 2 ;;
        --base-model) BASE_MODEL="$2"; shift 2 ;;
        --public)     PUBLIC=1;        shift ;;
        --dry-run)    DRY_RUN=1;       shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$QUEUE" && -f "$QUEUE" ]] || { echo "--queue <file.tsv> is required" >&2; exit 2; }

# The token is read from the environment the caller already prepared. It is
# never accepted as an argument, because arguments are visible in the process
# table on a shared machine.
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set in the environment; refusing to start." >&2
    exit 2
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/verl:${PYTHONPATH:-}"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  $*"; }

# Checkpoints go under $AGRI_WORK_DIR (default ~/agrimanager_work), outside the
# repository: each retained checkpoint is 17.6 GB.
mkdir -p "$WORK_DIR"
if [[ -n "${AGRI_LUSTRE_POOL:-}" ]] && command -v lfs >/dev/null 2>&1; then
    lfs getstripe -d "$WORK_DIR" 2>/dev/null | grep -q "pool:[[:space:]]*$AGRI_LUSTRE_POOL" \
        || lfs setstripe --pool "$AGRI_LUSTRE_POOL" -c 4 "$WORK_DIR" 2>/dev/null || true
fi

free_gb() {
    df -Pk "$WORK_DIR" 2>/dev/null | awk 'NR==2 {print int($4/1048576)}'
}

verify() {  # repo revision
    python3 -m agrimanager.checkpoint.verify_published \
        --repo-id "$1" --revision "$2" >/dev/null 2>&1
}

# The W&B run id a revision already on the hub claims to have come from.
# Empty if the revision has no manifest or cannot be read.
hub_wandb_id() {  # repo revision
    python3 - "$1" "$2" <<'PYEOF' 2>/dev/null || true
import json, os, sys
from huggingface_hub import hf_hub_download
try:
    p = hf_hub_download(sys.argv[1], "run_manifest.json", revision=sys.argv[2],
                        token=os.environ.get("HF_TOKEN"))
    print((json.load(open(p)).get("provenance") or {}).get("wandb_run_id") or "")
except Exception:
    pass
PYEOF
}

n_published=0 n_verified=0 n_freed=0 n_kept=0
freed_gb_total=0

while IFS=$'\t' read -r run_name run_dir seed wandb_id run_def cell_id dataset_dir total_steps; do
    [[ -z "${run_name// }" || "${run_name:0:1}" == "#" ]] && continue

    repo="$OWNER/$run_name"

    # The arm is derived from the run name rather than hardcoded. Publishing a
    # no-think run with --require-think True would put a false claim in its
    # run_manifest.json, and the manifest is the thing a reader trusts.
    case "$run_name" in
        *no-think*|*no_think*) require_think=False ;;
        *)                     require_think=True  ;;
    esac
    # Optional eighth column; the T2.3 cells run to 400 rather than 200.
    [[ "${total_steps:-}" =~ ^[0-9]+$ ]] || total_steps=200

    if [[ ! -d "$run_dir" ]]; then
        log "SKIP $run_name: $run_dir does not exist"
        continue
    fi

    # Refuse a run that is still being written. Publishing then deleting the
    # shards of a live job destroys it, and the queue is built from names rather
    # than from job state, so nothing else here would notice. A finished run's
    # directory is static; anything touched in the last fifteen minutes is
    # assumed to still be training.
    if [[ -n "$(find "$run_dir" -maxdepth 2 -newermt '-15 minutes' -print -quit 2>/dev/null)" ]]; then
        log "SKIP $run_name: written to within the last 15 minutes; it looks like it is still running"
        n_kept=$(( n_kept + 1 ))
        continue
    fi

    log "=== $run_name  ($cell_id, seed $seed)  free=$(free_gb)G"

    for ckpt in $(ls -d "$run_dir"/global_step_* 2>/dev/null | sort -t_ -k3 -n); do
        step="${ckpt##*global_step_}"
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        rev="step-${step}"

        if [[ ! -f "$ckpt/actor/fsdp_config.json" ]]; then
            # `max_actor_ckpt_to_keep` removes a superseded checkpoint's weights
            # but leaves the directory holding a ~12 KB data.pt. That is a
            # retention artefact, not a checkpoint held back from publishing, so
            # it must not count towards n_kept -- counting it made the tool exit
            # non-zero and every monitor read that as "shards left unverified".
            sz_kb="$(du -sk "$ckpt" 2>/dev/null | cut -f1)"
            if [[ "${sz_kb:-0}" -lt 1024 ]]; then
                log "  $rev: superseded by max_actor_ckpt_to_keep; only a stub remains"
            else
                log "  $rev: no actor/fsdp_config.json but ${sz_kb} KB on disk; incomplete, keeping it"
                n_kept=$(( n_kept + 1 ))
            fi
            continue
        fi

        if verify "$repo" "$rev"; then
            # A complete revision on the hub does NOT by itself mean it is the
            # revision these local shards would produce. verify_published checks
            # a revision against its *own* manifest, so a repository that was
            # filled by an earlier training run of the same name passes while
            # describing different weights. Freeing on that basis deletes a run
            # whose output was never uploaded. This happened on 2026-08-29:
            # schema-t32-qwen-think-r3 had been published on 2026-07-26 from
            # W&B run 1viq5awn, a fresh run y0k2ghyk was trained under the same
            # name, and all four of its checkpoints were deleted against the
            # older copy without either being compared.
            hub_id="$(hub_wandb_id "$repo" "$rev")"
            if [[ -n "$hub_id" && -n "$wandb_id" && "$hub_id" != "$wandb_id" ]]; then
                log "  $rev: on the hub already, but from W&B run $hub_id, not $wandb_id"
                log "        these shards are a DIFFERENT run under the same name; keeping them"
                n_kept=$(( n_kept + 1 ))
                continue
            fi
            log "  $rev: already complete on the hub"
        elif (( DRY_RUN )); then
            log "  $rev: would publish (dry run)"
            continue
        else
            log "  $rev: publishing"
            pub_args=(
                --checkpoint-dir "$ckpt"
                --repo-id "$repo"
                --run-name "$run_name"
                --experiment-id "$cell_id"
                --seed "$seed"
                --base-model "$BASE_MODEL"
                --checkpoint-step "$step"
                --total-training-steps "$total_steps"
                --save-freq 100
                --require-think "$require_think"
                --thinking-mode think
                --think-tag tool_call
                --run-definition "$run_def"
                --wandb-run-id "$wandb_id"
                --dataset-dir "$dataset_dir"
                --revision "$rev"
                --work-dir "$WORK_DIR"
                --no-verify-load
            )
            (( PUBLIC )) && pub_args+=(--public)

            if ! bash entrypoints/tools/publish_checkpoint.sh "${pub_args[@]}"; then
                log "  $rev: publish FAILED; keeping the local shards and moving on"
                n_kept=$(( n_kept + 1 ))
                rm -rf "$WORK_DIR/${run_name}-gs${step}" 2>/dev/null || true
                continue
            fi
            n_published=$(( n_published + 1 ))

            # The upload exiting 0 is not the evidence; the manifest match is.
            if ! verify "$repo" "$rev"; then
                log "  $rev: uploaded but does NOT verify against its manifest; keeping the local shards"
                python3 -m agrimanager.checkpoint.verify_published \
                    --repo-id "$repo" --revision "$rev" 2>&1 | sed 's/^/      /' || true
                n_kept=$(( n_kept + 1 ))
                continue
            fi
        fi

        n_verified=$(( n_verified + 1 ))
        if (( DRY_RUN )); then
            log "  $rev: would free $(du -sh "$ckpt" 2>/dev/null | cut -f1) (dry run)"
            continue
        fi
        sz_gb="$(du -s --block-size=1G "$ckpt" 2>/dev/null | cut -f1)"
        rm -rf "$ckpt"
        n_freed=$(( n_freed + 1 ))
        freed_gb_total=$(( freed_gb_total + ${sz_gb:-0} ))
        log "  $rev: verified on the hub; freed ${sz_gb}G locally"
    done

    # Keep latest_checkpointed_iteration.txt. It is a few bytes and it is the
    # only local record of where the run actually stopped, which on the
    # early-terminating arms is itself the measurement.
    rm -rf "$WORK_DIR/${run_name}"-gs* 2>/dev/null || true
done < "$QUEUE"

log "=== done: $n_published published, $n_verified verified, $n_freed freed (~${freed_gb_total}G), $n_kept kept back"
log "    free now $(free_gb)G"
(( n_kept == 0 ))
