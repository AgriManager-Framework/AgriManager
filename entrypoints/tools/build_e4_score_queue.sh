#!/usr/bin/env bash
# Build a test-suite ("E4") scoring queue from the hub, which is the only
# complete inventory of what exists.
#
#   HF_OWNER=agrimanager bash entrypoints/tools/build_e4_score_queue.sh > my_score_queue.tsv
#
# HF_OWNER is the account or organization holding the schema-* checkpoint
# repositories and schema-results-* datasets (default: agrimanager).
# AGRI_E4_RESULT_ROOTS is a colon-separated list of local result roots to check
# for finished evaluations (default: $AGRI_WORK_DIR/agri_e4_scoring).
#
# Why the hub rather than local disk: local shards are staging and are deleted
# once an upload verifies, so the filesystem holds only whatever has not been
# published yet. Reconstructing the work list from it would silently omit every
# run already dealt with, which is most of them.
#
# Two things this refuses to guess.
#
# The revision. Each row names the terminal revision -- the highest step-N
# branch in the repository -- because that is the selection rule the protocol
# fixed: select the terminal checkpoint, or the last-valid checkpoint after a
# documented interruption; never the best validation checkpoint. For a run that stopped early the terminal revision
# is the early-stop step, and that is correct: the run ended there, so those are
# its final weights. Nothing here compares a run against its declared 200.
#
# Whether it is already scored. A row is emitted only if no result for that run
# exists yet, checked in both places results can live -- the hub results dataset
# for its tier, and a local bundle directory from an earlier on-disk run. GPU
# time is the scarce thing here; re-scoring something already scored spends it
# for nothing.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN is not set in the environment; refusing to start." >&2
    exit 2
fi
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

python3 - "$PROJECT_ROOT" <<'PY'
import os, re, sys
from pathlib import Path
from huggingface_hub import HfApi

project_root = Path(sys.argv[1])
api = HfApi(token=os.environ.get("HF_TOKEN"))
OWNER = os.environ.get("HF_OWNER", "agrimanager")

# Tier is derived from the run name and must match SUITE_GROUPS in
# agrimanager/checkpoint/evaluate.py. Chickpea is its own tier there rather than
# a second group under t11, because scoring one crop's checkpoint against the
# other crop's suite would be meaningless; the order below matters for that
# reason -- chickpea has to be tested before the generic t11 prefix.
TIER_RULES = [
    ("schema-t11-chickpea-", "t11_chickpea"),
    ("schema-t11-potato-",   "t11"),
    ("schema-t12-",          "t12"),
    ("schema-t13-",          "t13"),
    ("schema-t21-",          "t21"),
    ("schema-t22-",          "t22"),
    ("schema-t23-",          "t23"),
    ("schema-t31-",          "t31"),
    ("schema-t32-qwen-",     "t32"),
]

# Out of scope by decision, 2026-08-01: the T3.2 llama-3.2-3B cells stay at one
# seed and remain exploratory.
SKIP_PREFIXES = ("schema-t32-llama32-",)

# Where a finished result would already be, per tier.
RESULTS_DATASETS = {
    "t11":          f"{OWNER}/schema-results-t11-potato",
    "t11_chickpea": f"{OWNER}/schema-results-t11-chickpea",
    "t12":          f"{OWNER}/schema-results-t12-crosscrop",
    "t13":          f"{OWNER}/schema-results-t13-price",
    "t21":          f"{OWNER}/schema-results-t21-observation",
    "t22":          f"{OWNER}/schema-results-t22-llm",
    "t23":          f"{OWNER}/schema-results-t23-reward",
    "t31":          f"{OWNER}/schema-results-t31",
    "t32":          f"{OWNER}/schema-results-t32-qwen",
}

# Bundle directories written by the earlier on-disk evaluation wrappers.
# Where a finished evaluation can be found on disk. The first entry is the
# persistent worker's result root and holds almost all of them; the other two
# predate it and are kept because a few early evaluations were written there.
_work = os.environ.get("AGRI_WORK_DIR", os.path.expanduser("~/agrimanager_work"))
LOCAL_BUNDLE_ROOTS = [
    Path(p) for p in os.environ.get("AGRI_E4_RESULT_ROOTS", f"{_work}/agri_e4_scoring").split(":") if p
]

def tier_of(name):
    for prefix, tier in TIER_RULES:
        if name.startswith(prefix):
            return tier
    return None

def scored_on_hub(dataset):
    """Run names that already have at least one file in this results dataset."""
    try:
        files = api.list_repo_files(dataset, repo_type="dataset")
    except Exception:
        return set()          # absent dataset means nothing scored yet
    out = set()
    for f in files:
        m = re.search(r"(schema-[a-z0-9-]+-[rs]\d+)", f)
        if m:
            out.add(m.group(1))
    return out

def scored_locally():
    out = set()
    for root in LOCAL_BUNDLE_ROOTS:
        if root.is_dir():
            for d in root.iterdir():
                # eval_manifest.json, not the bundle directory. The
                # evaluator creates bundle/ when it starts and writes the
                # manifest only once every split has finished, so the directory
                # exists for the whole run and for any run that died partway.
                # Keying on it would mark an interrupted evaluation as scored
                # and drop it from the queue for good.
                if (d / "bundle" / "eval_manifest.json").is_file():
                    out.add(d.name)
    return out

hub_scored = {t: scored_on_hub(ds) for t, ds in RESULTS_DATASETS.items()}
local_scored = scored_locally()

rows, skipped = [], []
for model in api.list_models(author=OWNER):
    name = model.id.split("/", 1)[1]
    if not name.startswith("schema-") or any(name.startswith(p) for p in SKIP_PREFIXES):
        continue
    tier = tier_of(name)
    if tier is None:
        skipped.append((name, "no tier rule matches this run name"))
        continue
    try:
        refs = api.list_repo_refs(model.id)
        steps = sorted(int(b.name[5:]) for b in refs.branches
                       if re.fullmatch(r"step-\d+", b.name))
    except Exception as e:
        skipped.append((name, f"cannot list refs: {type(e).__name__}"))
        continue
    if not steps:
        skipped.append((name, "no step-N revision published"))
        continue
    if name in local_scored:
        skipped.append((name, "already scored: local bundle on disk"))
        continue
    if name in hub_scored.get(tier, set()):
        skipped.append((name, f"already scored: present in {RESULTS_DATASETS[tier]}"))
        continue
    rows.append((name, model.id, f"step-{steps[-1]}", tier))

rows.sort()
print("# E4 scoring queue, generated by entrypoints/tools/build_e4_score_queue.sh")
print("# Terminal revision per run, per the protocol's selection rule. A run that")
print("# stopped early is scored at the step it stopped at; that is where it ended,")
print("# so those are its final weights.")
print("#")
print("# run_name\trepo\trevision\ttier")
for r in rows:
    print("\t".join(r))

print(f"\n# {len(rows)} to score", file=sys.stderr)
by_tier = {}
for _, _, _, t in rows:
    by_tier[t] = by_tier.get(t, 0) + 1
for t in sorted(by_tier):
    print(f"#   {t:14s} {by_tier[t]}", file=sys.stderr)
print(f"# {len(skipped)} skipped", file=sys.stderr)
for name, why in sorted(skipped):
    print(f"#   {name:40s} {why}", file=sys.stderr)
PY
