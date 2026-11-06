"""Upload E5 result bundles to their tier's results repository.

One results repository holds every run in a tier, so the bundle cannot be
addressed by revision alone: `schema-results-t32-llama32` covers both the think
and no-think arms, and both produce a `step-200`. Publishing each bundle on a
branch named for its step would make the second arm silently overwrite the
first, and the loss would be invisible — the branch would exist and look right.

Bundles are therefore addressed by path inside one branch:

    <results-repo>/<run-name>/<revision>/eval_manifest.json
                                        /<suite-group>/<split>/...

which cannot collide, and shows the whole tier in one tree.

New bundles go to a private repository first. The per-run records are released
publicly with the paper (AgriManager-Framework/schema-results-*) after review; this tool
refuses to add bundles to a repository that is already public.

    python -m agrimanager.checkpoint.upload_results --bundle <dir> --results-repo <owner>/<name>
    python -m agrimanager.checkpoint.upload_results --scan /workspace/e5 --map <tsv>

`--scan` walks a work directory of finished evaluations and uploads every bundle
it has not uploaded before, reading each one's own manifest for its identity.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from agrimanager.checkpoint.publish import (
    PublishError,
    collect_forbid_terms,
    guard_folder,
)


def bundle_identity(bundle: Path) -> tuple[str, str]:
    """Return (run_name, revision) from the bundle's own manifest.

    Read from the manifest rather than parsed out of the directory name: the
    directory is a convenience, the manifest is the record.
    """
    manifest_path = bundle / "eval_manifest.json"
    if not manifest_path.is_file():
        raise PublishError(f"{bundle} has no eval_manifest.json; it is not a finished bundle")
    manifest = json.loads(manifest_path.read_text())
    checkpoint = manifest.get("checkpoint") or {}
    run_name = checkpoint.get("run_name")
    revision = checkpoint.get("revision")
    if not run_name or not revision:
        raise PublishError(f"{manifest_path} does not name a run and revision")
    return run_name, revision


def upload_bundle(
    bundle: Path,
    results_repo: str,
    token: str | None,
    forbid_terms: Sequence[str] = (),
    dry_run: bool = False,
) -> str:
    run_name, revision = bundle_identity(bundle)
    path_in_repo = f"{run_name}/{revision}"

    # The evaluator already guarded this bundle before writing it. Guard again:
    # this is the step that makes it world-visible to anyone the repository is
    # shared with, and the cost is a scan of a few dozen JSON files.
    guard_folder(bundle, forbid_terms)

    if dry_run:
        return f"(dry run) {results_repo}/{path_in_repo}"

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=results_repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo_id=results_repo, repo_type="dataset")
    if not info.private:
        raise PublishError(
            f"{results_repo} is public. New result bundles go to a private "
            f"repository and are published after review; refusing to upload."
        )

    api.upload_folder(
        folder_path=str(bundle),
        path_in_repo=path_in_repo,
        repo_id=results_repo,
        repo_type="dataset",
        commit_message=f"{run_name} {revision}",
    )
    return f"https://huggingface.co/datasets/{results_repo}/tree/main/{path_in_repo}"


def load_map(path: Path) -> dict[str, str]:
    """run_name -> results_repo, from the same TSV the eval queue reads."""
    mapping: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) >= 4:
            mapping[parts[0].strip()] = parts[3].strip()
    return mapping


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m agrimanager.checkpoint.upload_results",
        description="Upload E5 result bundles, addressed by path so runs cannot collide.",
    )
    parser.add_argument("--bundle", help="a single bundle directory")
    parser.add_argument("--results-repo", help="target repository for --bundle")
    parser.add_argument("--scan", help="work directory holding <tag>/bundle/ subdirectories")
    parser.add_argument("--map", help="TSV mapping run_name to results repo (the eval targets file)")
    parser.add_argument("--owner", default="AgriManager-Framework")
    parser.add_argument("--state", default="/workspace/eval_queue/.uploaded",
                        help="record of what has already been uploaded")
    parser.add_argument("--forbid", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    if not token and not args.dry_run:
        print("HF_TOKEN is not set", file=sys.stderr)
        return 2
    terms = collect_forbid_terms(args.forbid)

    if args.bundle:
        if not args.results_repo:
            print("--results-repo is required with --bundle", file=sys.stderr)
            return 2
        print(upload_bundle(Path(args.bundle), args.results_repo, token, terms, args.dry_run))
        return 0

    if not args.scan or not args.map:
        print("give either --bundle with --results-repo, or --scan with --map", file=sys.stderr)
        return 2

    mapping = load_map(Path(args.map))
    state = Path(args.state)
    done = set(state.read_text().split()) if state.is_file() else set()

    failures = 0
    uploaded = 0
    for bundle in sorted(Path(args.scan).glob("*/bundle")):
        try:
            run_name, revision = bundle_identity(bundle)
        except PublishError as exc:
            print(f"skip {bundle}: {exc}", file=sys.stderr)
            continue
        key = f"{run_name}@{revision}"
        if key in done:
            continue
        repo = mapping.get(run_name)
        if not repo:
            print(f"skip {key}: no results repo mapped for {run_name}", file=sys.stderr)
            failures += 1
            continue
        target = repo if "/" in repo else f"{args.owner}/{repo}"
        try:
            url = upload_bundle(bundle, target, token, terms, args.dry_run)
        except Exception as exc:  # noqa: BLE001 - one bad bundle must not stop the rest
            print(f"FAILED {key}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1
            continue
        print(f"uploaded {key} -> {url}", flush=True)
        uploaded += 1
        if not args.dry_run:
            state.parent.mkdir(parents=True, exist_ok=True)
            with state.open("a") as fh:
                fh.write(key + "\n")

    print(f"\n{uploaded} uploaded, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
