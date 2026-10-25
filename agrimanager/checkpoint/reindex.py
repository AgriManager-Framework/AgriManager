"""Write a landing page on a checkpoint repository's ``main`` branch.

Each periodic checkpoint is published to its own revision (``step-200``,
``step-400``, ...) so that peak-to-final analysis can address them individually.
Nothing is ever committed to ``main``, so the repository's front page shows only
the ``.gitattributes`` that repository creation put there and reads as empty --
the weights are one branch-selector click away, but nothing on the page says so.

This builds an index from the revisions that actually exist, reading each one's
``run_manifest.json`` rather than trusting the branch name, and commits it to
``main``. It never touches a step revision, so a published checkpoint cannot be
altered by re-indexing.

    python -m agrimanager.checkpoint.reindex --repo-id owner/schema-t32-qwen-think-r2

The same identity guard that gates a checkpoint upload gates this page.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from agrimanager.checkpoint.publish import (
    PublishError,
    collect_forbid_terms,
    _guard_text,
)

STEP_REVISION = re.compile(r"^step-(\d+)$")


def _step_of(branch: str) -> int | None:
    match = STEP_REVISION.match(branch)
    return int(match.group(1)) if match else None


def collect_revisions(api: Any, repo_id: str) -> list[dict[str, Any]]:
    """Describe every ``step-N`` revision, newest step first.

    A revision that carries no manifest is reported rather than skipped: an
    empty branch is exactly the failure this page needs to make visible, and
    silently dropping it would hide it a second time.
    """
    refs = api.list_repo_refs(repo_id=repo_id, repo_type="model")
    revisions: list[dict[str, Any]] = []

    for branch in refs.branches:
        step = _step_of(branch.name)
        if step is None:
            continue
        entry: dict[str, Any] = {"revision": branch.name, "step": step}

        files = api.list_repo_files(repo_id=repo_id, repo_type="model", revision=branch.name)
        entry["file_count"] = len(files)

        if "run_manifest.json" in files:
            path = api.hf_hub_download(
                repo_id=repo_id,
                filename="run_manifest.json",
                repo_type="model",
                revision=branch.name,
            )
            manifest = json.loads(Path(path).read_text())
            entry["manifest"] = manifest
        else:
            entry["manifest"] = None
        revisions.append(entry)

    revisions.sort(key=lambda item: item["step"], reverse=True)
    return revisions


def build_index_card(repo_id: str, revisions: Sequence[dict[str, Any]], public: bool) -> str:
    """Render the landing page. Facts come from the manifests, not the caller."""
    name = repo_id.split("/")[-1]
    complete = [r for r in revisions if r["manifest"]]

    training: dict[str, Any] = {}
    experiment_id = None
    for rev in complete:
        training = rev["manifest"].get("training") or {}
        experiment_id = (rev["manifest"].get("experiment") or {}).get("cell")
        if training:
            break

    lines = [
        "---",
        "library_name: transformers",
        "tags:",
        "  - reinforcement-learning",
        "  - agriculture",
        "---",
        "",
        f"# {name}",
        "",
        "Checkpoints from one training run. **Every checkpoint lives on its own",
        "revision, not on `main`** -- this branch carries only this index.",
        "",
    ]

    if training:
        lines += [
            f"- Base model: `{training.get('base_model')}`",
            f"- Algorithm: {training.get('algorithm')}",
            f"- Seed: {training.get('seed')}",
        ]
        if experiment_id:
            lines.append(f"- Cell: `{experiment_id}`")
        total = training.get("total_training_steps")
        if total:
            lines.append(f"- Schedule: {total} steps, saving every {training.get('save_freq')}")
        reasoning = training.get("reasoning") or {}
        if reasoning:
            lines.append(
                f"- Reasoning format: `require_think={reasoning.get('require_think')}`, "
                f"`thinking_mode={reasoning.get('thinking_mode')}`, "
                f"`think_tag={reasoning.get('think_tag')}`"
            )
        lines.append("")

    lines += ["## Revisions", "", "| Revision | Step | Files |", "|---|---:|---|"]
    for rev in revisions:
        state = f"{rev['file_count']}" if rev["manifest"] else f"{rev['file_count']} — **no manifest, incomplete**"
        lines.append(f"| `{rev['revision']}` | {rev['step']} | {state} |")
    if not revisions:
        lines.append("| — | — | no checkpoint published yet |")
    lines.append("")

    target = complete[0]["revision"] if complete else "step-200"
    lines += [
        "## Loading a checkpoint",
        "",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f'repo = "{repo_id}"',
        f'rev  = "{target}"',
        "",
        "model = AutoModelForCausalLM.from_pretrained(repo, revision=rev)",
        "tokenizer = AutoTokenizer.from_pretrained(repo, revision=rev)",
        "```",
        "",
        "Omitting `revision` resolves to `main`, which holds no weights.",
        "",
        "## Provenance",
        "",
        "Each revision carries a `run_manifest.json` recording the base model, the",
        "seed, the training schedule, the reasoning format the run trained under,",
        "and a SHA-256 for every file.",
        "",
        "Checkpoint selection is terminal or last-valid. No checkpoint here was",
        "chosen by validation score.",
        "",
    ]
    if not public:
        lines += ["Private research artifact. Not for redistribution.", ""]
    return "\n".join(lines)


def reindex(
    repo_id: str,
    token: str | None,
    forbid_terms: Sequence[str] = (),
    dry_run: bool = False,
) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    # Ask the hub what the repository actually is instead of taking it on the
    # command line: this page states the visibility, and a wrong claim there is
    # worse than no page.
    info = api.repo_info(repo_id=repo_id, repo_type="model")
    revisions = collect_revisions(api, repo_id)
    card = build_index_card(repo_id, revisions, public=not info.private)
    _guard_text(f"{repo_id} README.md", card, forbid_terms)

    if dry_run:
        return card

    with tempfile.TemporaryDirectory() as work:
        readme = Path(work) / "README.md"
        readme.write_text(card)
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            revision="main",
            commit_message=f"index {len(revisions)} revision(s)",
        )
    return card


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m agrimanager.checkpoint.reindex",
        description="Write a revision index onto a checkpoint repository's main branch.",
    )
    parser.add_argument("--repo-id", required=True, action="append", dest="repo_ids",
                        help="repository to index; repeatable")
    parser.add_argument("--forbid", action="append", default=[],
                        help="extra term to reject; repeatable. Also read from PUBLISH_FORBID_TERMS")
    parser.add_argument("--dry-run", action="store_true", help="print the page, upload nothing")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("HF_TOKEN is not set", file=sys.stderr)
        return 2

    terms = collect_forbid_terms(args.forbid)
    failures = 0
    for repo_id in args.repo_ids:
        try:
            card = reindex(repo_id, token, terms, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001 - one bad repo must not stop the rest
            print(f"{repo_id}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1
            continue
        if args.dry_run:
            print(f"===== {repo_id}\n{card}")
        else:
            print(f"indexed https://huggingface.co/{repo_id}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
