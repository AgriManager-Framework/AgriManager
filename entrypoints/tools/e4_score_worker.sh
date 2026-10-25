#!/usr/bin/env bash
# Score a queue of published checkpoints against the frozen E4 suites, inside a
# single long-lived allocation.
#
#   bash entrypoints/tools/e4_score_worker.sh --queue results/scoring_manifest.tsv [--result-root DIR]
#
# Why one allocation rather than one job per checkpoint. Each job charges its
# whole reservation against the account whether or not it uses it, and each one
# queues separately -- on a busy cluster that can mean anywhere from minutes to
# two days per job. Forty-one jobs is forty-one queue waits and forty-one
# rounding-ups. One allocation pays one wait and is billed for the time it
# actually holds the GPU, and the work inside it costs nothing extra to
# schedule. The queue file is read fresh each pass, so rows can be appended
# while this is running and it will pick them up.
#
# The three things that make a long worker safe to leave alone:
#
#   Idempotent. A run whose bundle already exists is skipped, so restarting
#   after a crash, a timeout, or a manual kill resumes rather than repeats.
#
#   It stops before the wall rather than being killed at it. Each pass checks
#   the time SLURM has left against what a task actually took, measured here
#   rather than assumed, and exits cleanly when the next one would not fit. A
#   task killed mid-evaluation leaves a partial bundle that looks finished.
#
#   It yields the machine. If /workspace/HUMAN_HOLD or a local HOLD file
#   appears, it finishes nothing new and exits, because a teammate has claimed
#   the host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

QUEUE=""
RESULT_ROOT="${AGRI_WORK_DIR:-$HOME/agrimanager_work}/agri_e4_scoring"
SUITE_DIR="$PROJECT_ROOT/experiments/rebuttal_confirmatory_eval/data"
SUITE_REPO="local/schema-shift-confirmatory-eval-v1"
# Minutes to assume a task needs until one has actually been measured. The first
# task is the expensive unknown; after that the worker uses the real number.
DEFAULT_TASK_MIN=75
SAFETY_MIN=15
MAX_TASKS=0            # 0 = no limit
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --queue)        QUEUE="$2";           shift 2 ;;
        --result-root)  RESULT_ROOT="$2";     shift 2 ;;
        --suite-dir)    SUITE_DIR="$2";       shift 2 ;;
        --task-minutes) DEFAULT_TASK_MIN="$2"; shift 2 ;;
        --safety-minutes) SAFETY_MIN="$2";    shift 2 ;;
        --max-tasks)    MAX_TASKS="$2";       shift 2 ;;
        --dry-run)      DRY_RUN=1;            shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$QUEUE" && -f "$QUEUE" ]] || { echo "--queue <file.tsv> is required" >&2; exit 2; }
[[ -d "$SUITE_DIR" ]] || { echo "suite directory not found: $SUITE_DIR" >&2; exit 2; }

cd "$PROJECT_ROOT"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

# Identity of this worker. WORKER_SLOT distinguishes workers sharing one
# allocation; without it every worker in a 4-GPU job reports the same
# SLURM_JOB_ID, claim() sees its own id on a row a sibling is working, treats it
# as "already mine" and runs the row a second time on another GPU.
WORKER_SLOT="${WORKER_SLOT:-0}"
WORKER_JOB="${SLURM_JOB_ID:-local-$$}"
WORKER_ID="${WORKER_JOB}.${WORKER_SLOT}"

log() { echo "$(date '+%Y-%m-%d %H:%M:%S')  [w${WORKER_SLOT}] $*"; }

mkdir -p "$RESULT_ROOT"
STATE_DIR="$RESULT_ROOT/.worker"
mkdir -p "$STATE_DIR"
DURATIONS="$STATE_DIR/task_seconds"

held() {
    [[ -e /workspace/HUMAN_HOLD || -e "$RESULT_ROOT/HOLD" ]]
}

# Claim a row so several workers can share one queue. mkdir is atomic, so
# exactly one worker wins the race; the loser moves on to the next row rather
# than duplicating an hour of GPU.
#
# A claim outlives the worker that made it, so it records the SLURM job id and a
# claim whose job is no longer in the queue is taken over. Without that, one
# worker dying mid-task would strand its row forever -- a bundle never appears,
# so the skip check never fires, and no other worker will touch it.
claim() {  # claim_dir
    local dir="$1" owner owner_job
    if mkdir "$dir" 2>/dev/null; then
        echo "$WORKER_ID" > "$dir/owner"
        return 0
    fi
    owner="$(cat "$dir/owner" 2>/dev/null || true)"
    [[ -n "$owner" ]] || return 1
    # Only this exact worker may re-enter its own claim. Comparing on the job id
    # alone would let a sibling in the same allocation adopt the row.
    [[ "$owner" == "$WORKER_ID" ]] && return 0
    owner_job="${owner%%.*}"
    [[ "$owner_job" == "$WORKER_JOB" ]] && return 1   # a live sibling holds it
    if [[ "$owner_job" == local-* ]]; then
        return 1                       # cannot test a non-SLURM owner; leave it
    fi
    # Take over only on positive evidence that the owner reached a terminal
    # state. An empty `squeue -j` is not that evidence: it looks identical
    # whether the job finished, the id is unknown, or squeue itself failed, and
    # treating a failed query as "owner is gone" lets two workers run the same
    # hour of GPU. sacct naming a terminal state is evidence; anything else
    # leaves the claim alone.
    local state
    state="$(sacct -j "$owner_job" -X -n -o State 2>/dev/null | head -1 | awk '{print $1}')"
    case "$state" in
        COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|PREEMPTED|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE) ;;
        *) return 1 ;;
    esac
    log "  taking over a claim left by $owner, whose job ended $state"
    echo "$WORKER_ID" > "$dir/owner"
    return 0
}

# Minutes SLURM has left on this allocation. Prints a very large number outside
# SLURM so the worker is usable interactively.
remaining_min() {
    local left
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then echo 100000; return; fi
    left="$(squeue -h -j "$SLURM_JOB_ID" -o '%L' 2>/dev/null | tr -d ' ')" || true
    [[ -n "$left" && "$left" != "UNLIMITED" ]] || { echo 100000; return; }
    # %L is [DD-]HH:MM:SS or MM:SS
    python3 - "$left" <<'PYEOF'
import sys
t = sys.argv[1]
days = 0
if "-" in t:
    d, t = t.split("-", 1)
    days = int(d)
parts = [int(x) for x in t.split(":")]
while len(parts) < 3:
    parts.insert(0, 0)
h, m, s = parts
print(days * 1440 + h * 60 + m + (1 if s else 0))
PYEOF
}

# What a task has actually cost, as the slowest seen rather than the mean: the
# budget check must survive the worst case, not the typical one.
budget_min() {
    if [[ ! -s "$DURATIONS" ]]; then echo "$DEFAULT_TASK_MIN"; return; fi
    awk 'BEGIN{m=0} {if ($1+0 > m) m = $1+0} END{printf "%d", (m/60) + 1}' "$DURATIONS"
}

n_done=0 n_skip=0 n_fail=0 n_started=0 n_claimed=0

# A row is run_name, repo, revision, tier and an optional fifth column of extra
# evaluator arguments, space-separated. The fifth column is what lets one queue
# carry the controls as well as the published checkpoints: the untrained model
# is a local directory with an operator-supplied reasoning format
# (--checkpoint-dir, --require-think), and a specialist scored on another
# tier's suite names that suite (--suite-group).
while IFS=$'\t' read -r run_name repo revision tier extra; do
    [[ -z "${run_name// }" || "${run_name:0:1}" == "#" ]] && continue
    [[ -n "$tier" ]] || { log "SKIP $run_name: malformed row"; n_fail=$((n_fail+1)); continue; }

    work_dir="$RESULT_ROOT/$run_name"
    bundle="$work_dir/bundle"
    # Completion is eval_manifest.json, not the bundle directory. The evaluator
    # creates bundle/ when it starts and fills it split by split, writing the
    # manifest only after every split has finished, so the directory exists for
    # the whole run. Keying the skip on the directory meant an interrupted task
    # left something that looked finished: the claim is released on failure, but
    # the directory stays, and every worker afterwards skips that row as done
    # and the checkpoint silently never gets scored.
    done_marker="$bundle/eval_manifest.json"

    if [[ -f "$done_marker" ]]; then
        n_skip=$((n_skip+1))
        continue
    fi

    if held; then
        log "HOLD file present; stopping without starting $run_name"
        break
    fi

    if (( MAX_TASKS > 0 && n_started >= MAX_TASKS )); then
        log "reached --max-tasks $MAX_TASKS; stopping"
        break
    fi

    need="$(budget_min)"
    left="$(remaining_min)"
    if (( left < need + SAFETY_MIN )); then
        log "STOP: ${left}m left, a task needs ~${need}m plus ${SAFETY_MIN}m safety."
        log "      $run_name and everything after it stay queued for the next allocation."
        break
    fi

    if (( DRY_RUN )); then
        log "=== $run_name  ($tier, $revision)  ${left}m left, budget ~${need}m"
        log "  would score (dry run)"
        n_started=$((n_started+1))
        continue
    fi

    if ! claim "$work_dir.claim"; then
        n_claimed=$((n_claimed+1))
        continue
    fi

    log "=== $run_name  ($tier, $revision)  ${left}m left, budget ~${need}m${extra:+  [$extra]}"

    # A partial directory from a killed task would otherwise be reused.
    rm -rf "$work_dir"
    mkdir -p "$work_dir"

    started=$(date +%s)
    n_started=$((n_started+1))
    if bash entrypoints/eval/e5_evaluate_checkpoint.sh \
            --checkpoint-repo "$repo" \
            --revision "$revision" \
            --suite-repo "$SUITE_REPO" \
            --suite-dir "$SUITE_DIR" \
            --results-repo "local/schema-results-${tier}" \
            --tier "$tier" \
            --work-dir "$work_dir" \
            --no-upload \
            --keep-work-dir \
            ${extra:-} \
            > "$work_dir/score.log" 2>&1
    then
        elapsed=$(( $(date +%s) - started ))
        if [[ -f "$done_marker" ]]; then
            echo "$elapsed" >> "$DURATIONS"
            n_done=$((n_done+1))
            log "  scored in $((elapsed/60))m$((elapsed%60))s -> $bundle"
        else
            # Exit 0 without a bundle is worse than a failure, because the skip
            # check keys on the bundle and this would silently repeat forever.
            n_fail=$((n_fail+1))
            rm -rf "$work_dir.claim"
            log "  FAILED: exited 0 but wrote no eval_manifest.json; see $work_dir/score.log"
        fi
    else
        elapsed=$(( $(date +%s) - started ))
        n_fail=$((n_fail+1))
        # The evaluator writes eval_manifest.json even when every split failed, and the skip check keys on it,
        # so a failed row would read as done on the next pass (seen 2026-10-24). Remove it with the claim.
        rm -f "$done_marker"
        rm -rf "$work_dir.claim"
        log "  FAILED after $((elapsed/60))m$((elapsed%60))s; see $work_dir/score.log"
        tail -5 "$work_dir/score.log" 2>/dev/null | sed 's/^/      /'
    fi

    # The downloaded weights are ~8 GB per checkpoint and are not the product;
    # the bundle is. Keeping them would fill the pool inside a dozen tasks.
    rm -rf "$work_dir/checkpoint" "$work_dir/merged" 2>/dev/null || true
done < "$QUEUE"

remaining=$(awk -F'\t' '!/^#/ && NF>=4' "$QUEUE" | wc -l)
log "=== done this pass: $n_done scored, $n_skip already had bundles, $n_claimed held by another worker, $n_fail failed"
log "    queue holds $remaining rows; rerun to continue where this stopped"
(( n_fail == 0 ))
