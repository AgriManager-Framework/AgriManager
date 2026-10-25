#!/usr/bin/env bash
# Watch one training run and publish each periodic checkpoint as it completes.
#
# Checkpoints live only on the machine that produced them, so a run that
# finishes unattended strands its artifacts until someone reconnects. This polls
# the run directory and publishes every new checkpoint, which lets a host be
# released as soon as training ends.
#
# Usage:
#   bash entrypoints/tools/watch_and_publish.sh \
#       --result-dir <run_dir> \
#       --repo-id <owner>/<neutral-name> \
#       --run-name <neutral-name> \
#       --experiment-id E2-Q-T-R2 \
#       --seed 43 \
#       --base-model Qwen/Qwen3-4B-Instruct-2507 \
#       --total-training-steps 800 \
#       --save-freq 200 \
#       [--run-definition <repo-relative path>] [--wandb-run-id <id>] \
#       [--interval 300] [--work-dir <dir>] [--state-file <path>]
#
# Requires HF_TOKEN. Publishing failures are logged and retried on the next
# cycle; only a successful publish is recorded, so nothing is silently skipped.
#
# The trigger is `latest_checkpointed_iteration.txt`, which the trainer writes
# only after every shard is on disk (ray_trainer.py calls it the atomic marker).
# Watching for the directory instead would race the write and merge a partial
# checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

RESULT_DIR=""
INTERVAL=300
STATE_FILE=""
WORK_DIR=""
TOTAL_STEPS=""
REPO_ID=""
MIN_FREE_GB=0
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --result-dir) RESULT_DIR="$2"; shift 2 ;;
        --interval)   INTERVAL="$2";   shift 2 ;;
        --state-file) STATE_FILE="$2"; shift 2 ;;
        --work-dir)   WORK_DIR="$2";   shift 2 ;;
        --min-free-gb) MIN_FREE_GB="$2"; shift 2 ;;
        --total-training-steps)
            TOTAL_STEPS="$2"
            PASSTHROUGH+=("--total-training-steps" "$2")
            shift 2 ;;
        # Needed here to index the repository, and still forwarded to publish.
        --repo-id)
            REPO_ID="$2"
            PASSTHROUGH+=("--repo-id" "$2")
            shift 2 ;;
        *) PASSTHROUGH+=("$1"); shift ;;
    esac
done

if [[ -z "$RESULT_DIR" ]]; then
    echo "--result-dir is required" >&2
    exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set" >&2
    exit 2
fi

STATE_FILE="${STATE_FILE:-$RESULT_DIR/.published_steps}"
WORK_DIR="${WORK_DIR:-$RESULT_DIR/.publish_work}"
TRACKER="$RESULT_DIR/latest_checkpointed_iteration.txt"
PID_FILE="$RESULT_DIR/.watcher.pid"
mkdir -p "$(dirname "$STATE_FILE")"
touch "$STATE_FILE"

# Stop an earlier watcher by recorded PID. Never by `pkill -f watch_and_publish`:
# that pattern also matches the very shell issuing it, so the command kills its
# own session and everything after it silently never runs.
if [[ -f "$PID_FILE" ]]; then
    old="$(tr -dc '0-9' < "$PID_FILE")"
    if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
        kill "$old" 2>/dev/null || true
        sleep 2
    fi
fi
echo "$$" > "$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

log() { echo "[$(date -u +%H:%M:%S)] $*"; }

already_published() {
    grep -qx "$1" "$STATE_FILE"
}

free_gb() {
    df -BG --output=avail "$RESULT_DIR" 2>/dev/null | tail -1 | tr -dc '0-9'
}

# Reclaim space by dropping sharded checkpoints that are already on the hub.
#
# Two runs sharing a host each keep their own retained checkpoints, so the disk
# has to hold the sum. Running out during a save loses both runs, not just the
# one that overflowed. Only an already-published step is ever removed: its
# merged weights are on the hub, verified by a full load and a per-file SHA-256,
# and the protocol selects a terminal or last-valid checkpoint and never resumes
# into a trajectory, so the shards have no remaining use here.
#
# verl removes its own retired checkpoints with ignore_errors=True, so deleting
# one it still counts cannot fault the trainer.
reclaim_published_checkpoints() {
    [[ "$MIN_FREE_GB" -gt 0 ]] || return 0
    local avail
    avail="$(free_gb)"
    [[ -n "$avail" ]] || return 0
    (( avail >= MIN_FREE_GB )) && return 0

    log "free space ${avail}G is under the ${MIN_FREE_GB}G floor; reclaiming published checkpoints"
    for dir in $(ls -d "$RESULT_DIR"/global_step_* 2>/dev/null | sort -t_ -k3 -n); do
        local step="${dir##*global_step_}"
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        already_published "$step" || continue
        # The local record says the publish returned success. Confirm against
        # the hub anyway before destroying the only other copy: the check costs
        # two API calls and compares every file in the revision's manifest,
        # SHA-256 included for the weights.
        if [[ -n "$REPO_ID" ]] && ! python3 -m agrimanager.checkpoint.verify_published \
                --repo-id "$REPO_ID" --revision "step-${step}" >/dev/null 2>&1; then
            log "  step $step is recorded as published but does not verify on the hub; keeping it"
            continue
        fi
        rm -rf "$dir"
        avail="$(free_gb)"
        log "  removed local shards for step $step, verified complete on the hub; free space now ${avail}G"
        (( avail >= MIN_FREE_GB )) && break
    done
}

log "watching $RESULT_DIR every ${INTERVAL}s; target ${TOTAL_STEPS:-unbounded}"
log "already published: $(tr '\n' ' ' < "$STATE_FILE")"

while true; do
    # Before publishing, not only after: merging a checkpoint needs room for the
    # converted weights on top of the shards, so the disk is tightest during the
    # very step that would otherwise have to fit in whatever the last save left.
    reclaim_published_checkpoints

    if [[ -f "$TRACKER" ]]; then
        latest="$(tr -dc '0-9' < "$TRACKER")"
    else
        latest=""
    fi

    if [[ -n "$latest" ]]; then
        # Ascending order, so a crash mid-sweep leaves a contiguous published set.
        for dir in $(ls -d "$RESULT_DIR"/global_step_* 2>/dev/null | sort -t_ -k3 -n); do
            step="${dir##*global_step_}"
            [[ "$step" =~ ^[0-9]+$ ]] || continue
            (( step > latest )) && continue          # still being written
            already_published "$step" && continue

            log "publishing step $step"
            if bash "$SCRIPT_DIR/publish_checkpoint.sh" \
                    --checkpoint-dir "$dir" \
                    --checkpoint-step "$step" \
                    --revision "step-${step}" \
                    --work-dir "$WORK_DIR" \
                    "${PASSTHROUGH[@]}"; then
                echo "$step" >> "$STATE_FILE"
                log "step $step published"
                # Checkpoints live on step-N revisions, so main stays empty and
                # the repository front page reads as though nothing was
                # uploaded. Refresh the index that says otherwise. A failure
                # here is cosmetic: the checkpoint is already published, so it
                # must not mark the step unpublished or stop the watcher.
                if [[ -n "$REPO_ID" ]] \
                    && python3 -m agrimanager.checkpoint.reindex --repo-id "$REPO_ID"; then
                    log "main index refreshed"
                else
                    log "index refresh failed; checkpoint itself is published"
                fi
                reclaim_published_checkpoints
            else
                log "step $step failed; will retry next cycle"
            fi
        done
    fi

    if [[ -n "$TOTAL_STEPS" && -n "$latest" ]] \
        && (( latest >= TOTAL_STEPS )) && already_published "$TOTAL_STEPS"; then
        log "terminal step $TOTAL_STEPS published; watcher exiting"
        break
    fi

    sleep "$INTERVAL"
done
