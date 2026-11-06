#!/usr/bin/env bash
# Run a queue of training jobs on one GPU group, unattended.
#
# A rented host is idle from the moment a run ends until someone notices. This
# drives the next job itself: launch, wait, make sure every checkpoint reached
# the hub, drop the local shards, start the next one.
#
# Usage:
#   run_queue.sh --gpus 0,1 --queue /workspace/queue_01.tsv [--tag q01]
#
# To take a host over by hand: `touch /workspace/HUMAN_HOLD` (write the reason
# into it). The runner finishes what it is doing, then waits before starting
# anything else. Delete the file to give the host back.
#
# Queue file: one job per tab-separated line, blank lines and # comments ignored.
#   run_script <TAB> run_name <TAB> experiment_id <TAB> seed <TAB> require_think
#                <TAB> total_steps <TAB> save_freq <TAB> base_model
#
# The run_name doubles as the HF repository name under $HF_OWNER.
#
# Everything is tracked by PID file. Pattern matching is unusable here: an ssh
# command line carrying this script contains the strings a search would match,
# so `pkill -f` kills the session issuing it.
set -uo pipefail

GPUS=""
QUEUE=""
TAG=""
WAIT_FREE=0
FREE_THRESHOLD="${FREE_THRESHOLD:-2000}"
PUBLISH_INTERVAL="${PUBLISH_INTERVAL:-180}"
# A checkpoint merge and 8-9 GB upload takes minutes, not hours. Past this the
# run is left alone with its shards intact rather than deleted on a guess.
PUBLISH_TIMEOUT="${PUBLISH_TIMEOUT:-5400}"
HF_OWNER="${HF_OWNER:-AgriManager-Framework}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)  GPUS="$2";  shift 2 ;;
        --queue) QUEUE="$2"; shift 2 ;;
        --tag)   TAG="$2";   shift 2 ;;
        # Arm a runner on GPUs a current run still holds, so a queue can be set
        # up while today's runs are still going.
        --wait-free) WAIT_FREE=1; shift ;;
        # Free memory below this counts as idle, in MiB. A trainer holds tens of
        # GB; anything this small is a stray context, not a run.
        --free-threshold-mib) FREE_THRESHOLD="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "$GPUS"  ]] || { echo "--gpus is required" >&2; exit 2; }
[[ -f "$QUEUE" ]] || { echo "queue file not found: $QUEUE" >&2; exit 2; }
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN is not set" >&2; exit 2; }
TAG="${TAG:-gpu${GPUS//,/}}"

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/AgriManager}"
cd "$PROJECT_ROOT" || exit 2

STATE_DIR="/workspace/queue_${TAG}"
mkdir -p "$STATE_DIR"
echo "$$" > "$STATE_DIR/runner.pid"
trap 'rm -f "$STATE_DIR/runner.pid"' EXIT

log() { echo "[$(date -u +%H:%M:%S)] [$TAG] $*"; }

# --------------------------------------------------------------------------- #

wait_for_publication() {  # result_dir
    local rdir="$1" waited=0 pending
    while (( waited < PUBLISH_TIMEOUT )); do
        pending=""
        for dir in $(ls -d "$rdir"/global_step_* 2>/dev/null); do
            local step="${dir##*global_step_}"
            [[ "$step" =~ ^[0-9]+$ ]] || continue
            grep -qx "$step" "$rdir/.published_steps" 2>/dev/null || pending+="$step "
        done
        [[ -z "$pending" ]] && return 0
        log "  waiting on publication of step(s): $pending"
        sleep "$PUBLISH_INTERVAL"
        waited=$(( waited + PUBLISH_INTERVAL ))
    done
    log "  TIMEOUT waiting for publication; leaving shards in place"
    return 1
}

drop_verified() {  # result_dir repo
    local rdir="$1" repo="$2"
    for dir in $(ls -d "$rdir"/global_step_* 2>/dev/null); do
        local step="${dir##*global_step_}"
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        grep -qx "$step" "$rdir/.published_steps" 2>/dev/null || continue
        if python3 -m agrimanager.checkpoint.verify_published \
                --repo-id "$repo" --revision "step-${step}" >/dev/null 2>&1; then
            rm -rf "$dir"
            log "  dropped local shards for step $step (verified on the hub)"
        else
            log "  step $step does not verify on the hub; KEEPING its shards"
        fi
    done
}

# --------------------------------------------------------------------------- #

log "queue runner started on GPU $GPUS, reading $QUEUE"

gpus_busy() {
    # Ask the driver, not the process table. nvidia-smi reports compute-app PIDs
    # from the host namespace, which do not exist inside this container, so
    # `kill -0` on one always fails and reads as "already exited" -- that is
    # exactly how an earlier version launched four runs straight onto occupied
    # GPUs. Memory in use is namespace-independent and cannot lie this way.
    local used
    while read -r used; do
        [[ -n "$used" ]] || continue
        (( used > FREE_THRESHOLD )) && return 0
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" 2>/dev/null)
    return 1
}

if (( WAIT_FREE )); then
    if gpus_busy; then
        log "GPU $GPUS is in use; waiting for it to drain"
        while gpus_busy; do sleep 60; done
        log "GPU $GPUS is free"
        # Let the previous occupant's watcher publish and reclaim before
        # competing with it for disk and for the same HF token.
        sleep 180
    else
        log "GPU $GPUS is already free; starting now"
    fi
fi

# A person taking over this host needs a way to stop the automation that does
# not require finding and killing it. `touch /workspace/HUMAN_HOLD` is checked
# before every launch and before every pause decision, so a teammate can claim a
# machine without racing the queue for its GPUs. Remove the file to hand it back.
HOLD_FILE="${HOLD_FILE:-/workspace/HUMAN_HOLD}"

wait_out_human_hold() {
    [[ -f "$HOLD_FILE" ]] || return 0
    log "HUMAN_HOLD present; standing down until it is removed"
    log "  reason: $(head -c 300 "$HOLD_FILE" 2>/dev/null)"
    while [[ -f "$HOLD_FILE" ]]; do sleep 60; done
    log "HUMAN_HOLD cleared; resuming"
    # The person may have left a run of their own on these GPUs.
    while gpus_busy; do sleep 60; done
}

while IFS=$'\t' read -r script run_name exp_id seed require_think total_steps save_freq base_model; do
    [[ -z "${script// }" || "${script:0:1}" == "#" ]] && continue

    if grep -qx "$run_name" "$STATE_DIR/.done" 2>/dev/null; then
        log "skipping $run_name; already recorded done"
        continue
    fi

    wait_out_human_hold

    rdir="$(dirname "$script")/results/llm_train/$run_name"
    repo="$HF_OWNER/$run_name"
    mkdir -p "$rdir"

    log "=== starting $run_name ($exp_id, seed $seed, require_think=$require_think)"

    CUDA_VISIBLE_DEVICES="$GPUS" nohup bash "$script" \
        > "/workspace/launch_logs/${run_name}.launch.log" 2>&1 &
    train_pid=$!
    echo "$train_pid" > "$STATE_DIR/train.pid"
    log "  trainer pid=$train_pid"

    nohup bash entrypoints/tools/watch_and_publish.sh \
        --result-dir "$rdir" \
        --repo-id "$repo" \
        --run-name "$run_name" \
        --experiment-id "$exp_id" \
        --seed "$seed" \
        --base-model "$base_model" \
        --total-training-steps "$total_steps" \
        --save-freq "$save_freq" \
        --run-definition "$script" \
        --require-think "$require_think" \
        --thinking-mode think --think-tag tool_call \
        --public --interval 300 --min-free-gb 30 \
        > "/workspace/watcher_${run_name}.log" 2>&1 &
    log "  watcher started"

    wait "$train_pid"
    train_status=$?
    log "  trainer exited with status $train_status"

    # The watcher exits on its own only when the terminal step is published. A
    # run stopped early by rollout filtering never reaches that step, so wait on
    # the checkpoints that exist rather than on the watcher.
    wait_for_publication "$rdir"
    drop_verified "$rdir" "$repo"

    if [[ -f "$rdir/.watcher.pid" ]]; then
        wpid="$(tr -dc '0-9' < "$rdir/.watcher.pid")"
        [[ -n "$wpid" ]] && kill "$wpid" 2>/dev/null
    fi

    # A failed run must not be recorded as done and silently skipped on the next
    # arming; and if it produced nothing, the next job on this host would most
    # likely fail the same way, so stop rather than burn the queue.
    published=0
    [[ -s "$rdir/.published_steps" ]] && published=$(grep -c . "$rdir/.published_steps")
    if (( train_status != 0 && published == 0 )); then
        log "=== $run_name FAILED (status $train_status) with no checkpoint; stopping the queue"
        log "    see /workspace/launch_logs/${run_name}.launch.log"
        break
    fi
    if (( train_status != 0 )); then
        log "=== $run_name exited $train_status but published $published checkpoint(s); continuing"
    fi
    echo "$run_name" >> "$STATE_DIR/.done"
    log "=== finished $run_name; free space $(df -BG /workspace | tail -1 | awk '{print $4}')"
done < "$QUEUE"

log "queue drained"
