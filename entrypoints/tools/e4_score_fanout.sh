#!/usr/bin/env bash
# Run one E4 scoring worker per GPU inside a single allocation.
#
#   bash entrypoints/tools/e4_score_fanout.sh --queue <file.tsv> [--gpus 4]
#
# One allocation holding four GPUs beats four one-GPU allocations for the reason
# that actually costs time here: it queues once. On this partition, with a few
# hundred jobs pending against half the nodes draining, the wait dominates
# everything else. It is billed the same either way -- four GPU-hours is four
# GPU-hours -- so the only question is how many times you pay the wait.
#
# Each worker gets its own CUDA_VISIBLE_DEVICES and its own WORKER_SLOT. The slot
# matters: workers in one allocation share SLURM_JOB_ID, and the row-claiming in
# e4_score_worker.sh treats a claim owned by "me" as re-enterable, so without a
# distinguishing slot all four would adopt the same row and score it four times
# in parallel.
#
# Workers share one queue and one result root and coordinate only through the
# atomic claim directories, so they need no communication and one dying does not
# disturb the others.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

QUEUE=""
GPUS=0
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --queue) QUEUE="$2"; shift 2 ;;
        --gpus)  GPUS="$2";  shift 2 ;;
        *)       PASSTHROUGH+=("$1"); shift ;;
    esac
done

[[ -n "$QUEUE" ]] || { echo "--queue <file.tsv> is required" >&2; exit 2; }

# Default to however many GPUs SLURM actually gave us rather than to a guess.
if (( GPUS == 0 )); then
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
        GPUS="$SLURM_GPUS_ON_NODE"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        GPUS="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"
    else
        GPUS=1
    fi
fi

echo "fanning out $GPUS worker(s) over queue $QUEUE"

pids=()
for (( i = 0; i < GPUS; i++ )); do
    log_file="${E4_LOG_DIR:-/tmp}/e4-worker-${SLURM_JOB_ID:-local}-w${i}.log"
    CUDA_VISIBLE_DEVICES="$i" WORKER_SLOT="$i" \
        bash "$SCRIPT_DIR/e4_score_worker.sh" --queue "$QUEUE" "${PASSTHROUGH[@]}" \
        > >(tee -a "$log_file") 2>&1 &
    pids+=($!)
    echo "  worker $i -> GPU $i, log $log_file"
    # Stagger the starts. Four workers hitting the hub at the same instant race
    # on the shared HF cache directory, and four vLLM servers initialising
    # together spike host memory well above what they need in steady state.
    sleep 20
done

rc=0
for pid in "${pids[@]}"; do
    # Wait for every worker even after one fails: the others are still doing
    # useful work, and killing them would strand their claims.
    wait "$pid" || rc=1
done

echo "all workers exited (rc=$rc)"
exit "$rc"
