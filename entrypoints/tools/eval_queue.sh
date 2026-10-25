#!/usr/bin/env bash
# Evaluate published checkpoints as they appear, unattended.
#
# E5 is the long pole once training finishes, and every revision is independent,
# so there is no reason for it to wait for a person. This polls the checkpoint
# repositories and evaluates any step revision it has not evaluated yet.
#
# Usage:
#   eval_queue.sh --targets /workspace/eval_targets.tsv [--interval 600]
#
# Targets file: one repository per tab-separated line, # comments ignored.
#   repo_name <TAB> tier <TAB> require_think <TAB> results_repo
#
# Result bundles are uploaded as they are produced, addressed by path so that
# two runs in one tier reaching the same step cannot overwrite each other.
#
# Only a revision that verifies against its own manifest is evaluated: a branch
# whose upload did not land would otherwise be scored as a failure and recorded
# as done, hiding it. Unverified revisions are retried on the next pass.
set -uo pipefail

TARGETS=""
INTERVAL=600
GPU=""
WORKER=""
HF_OWNER="${HF_OWNER:-agrimanager}"
SUITE_REPO="${SUITE_REPO:-agrimanager/schema-shift-confirmatory-eval-v1}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --targets)  TARGETS="$2";  shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        # Pin this worker to one GPU so several can share a host. Evaluation is
        # single-GPU, so a freed 4-GPU training box is four workers, and at
        # $2.95 per GPU-hour it is cheaper per evaluation than the dedicated
        # single-GPU host as well as four times the throughput.
        --gpu)      GPU="$2";      shift 2 ;;
        --worker)   WORKER="$2";   shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -f "$TARGETS" ]] || { echo "targets file not found: $TARGETS" >&2; exit 2; }
[[ -n "${HF_TOKEN:-}" ]] || { echo "HF_TOKEN is not set" >&2; exit 2; }

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/AgriManager}"
cd "$PROJECT_ROOT" || exit 2

STATE_DIR=/workspace/eval_queue
CLAIMS="$STATE_DIR/.claims"
mkdir -p "$STATE_DIR" "$CLAIMS"
DONE="$STATE_DIR/.evaluated"
touch "$DONE"
WORKER="${WORKER:-${GPU:-solo}}"
PIDFILE="$STATE_DIR/runner${WORKER:+.$WORKER}.pid"
echo "$$" > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT

[[ -n "$GPU" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

log() { echo "[$(date -u +%H:%M:%S)] [eval:$WORKER] $*"; }

# Claim a revision atomically. `mkdir` either creates the directory or fails, in
# one step, so two workers polling the same list cannot both take the same
# revision -- without this they would each download the checkpoint and evaluate
# it, doubling the work and the disk for an identical result.
claim() {
    mkdir "$CLAIMS/$1" 2>/dev/null
}

list_step_revisions() {  # repo
    python3 - "$1" <<'PY'
import json, os, sys, urllib.request
repo = sys.argv[1]
req = urllib.request.Request(
    f"https://huggingface.co/api/models/{repo}/refs",
    headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
)
try:
    refs = json.load(urllib.request.urlopen(req))
except Exception:
    raise SystemExit(0)
steps = []
for b in refs.get("branches", []):
    name = b["name"]
    if name.startswith("step-") and name[5:].isdigit():
        steps.append((int(name[5:]), name))
for _, name in sorted(steps):
    print(name)
PY
}

log "eval queue started; polling every ${INTERVAL}s${GPU:+, pinned to GPU $GPU}"
log "already evaluated: $(wc -l < "$DONE") revision(s)"

while true; do
    while IFS=$'\t' read -r repo_name tier require_think results_repo; do
        [[ -z "${repo_name// }" || "${repo_name:0:1}" == "#" ]] && continue
        repo="$HF_OWNER/$repo_name"

        for rev in $(list_step_revisions "$repo"); do
            key="${repo_name}@${rev}"
            grep -qx "$key" "$DONE" && continue
            claim "${key//\//_}" || continue

            if ! python3 -m agrimanager.checkpoint.verify_published \
                    --repo-id "$repo" --revision "$rev" >/dev/null 2>&1; then
                # Release the claim: this revision is meant to be retried once
                # its upload lands, and a claim held past the check would keep
                # every worker off it permanently.
                rmdir "$CLAIMS/${key//\//_}" 2>/dev/null
                log "$key not complete on the hub yet; will retry"
                continue
            fi

            tag="${repo_name}_${rev}"
            log "evaluating $key (tier=$tier, require_think=$require_think)"
            bash entrypoints/eval/e5_evaluate_checkpoint.sh \
                --checkpoint-repo "$repo" --revision "$rev" \
                --suite-repo "$SUITE_REPO" \
                --results-repo "$HF_OWNER/$results_repo" \
                --tier "$tier" --require-think "$require_think" --think-tag tool_call \
                --work-dir "/workspace/e5/$tag" --keep-work-dir \
                > "/workspace/e5/$tag.log" 2>&1
            code=$?

            inval=$(python3 - "$tag" <<'PY' 2>/dev/null
import glob, json, sys
rates = []
for p in glob.glob(f"/workspace/e5/{sys.argv[1]}/bundle/*/*/metrics.json"):
    try:
        r = json.load(open(p)).get("invalid_action_rate_mean")
        if r is not None:
            rates.append(r)
    except Exception:
        pass
print(f"{max(rates):.3f}" if rates else "n/a")
PY
)
            log "  $key exit=$code max_invalid_action_rate=$inval"
            # A non-zero exit is recorded too: retrying a deterministic failure
            # every cycle would spin forever and starve the revisions behind it.
            echo "$key" >> "$DONE"
            rm -rf "/workspace/e5/$tag/checkpoint" "/workspace/e5/$tag/suites"
        done
    done < "$TARGETS"

    sleep "$INTERVAL"
done
