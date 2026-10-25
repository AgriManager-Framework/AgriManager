"""Confirm a published checkpoint is complete on the hub before deleting it locally.

Deleting local shards on the strength of "the branch exists" is not safe: a
branch can be created and left holding nothing but `.gitattributes`, which is
exactly what a storage-quota failure produced here once already. It also is not
enough to see files of roughly the right size.

Each revision carries a `run_manifest.json` listing every file it was published
with and that file's SHA-256. The hub reports the SHA-256 of every LFS object as
its `lfs.oid`, so the weights can be checked against the manifest without
downloading them. Small files are not LFS-tracked and expose only a git blob
SHA-1, so those are checked on exact byte size.

A revision passes only if the manifest is present and parseable, every file it
names exists, no size disagrees, and no available SHA-256 disagrees.

    python -m agrimanager.checkpoint.verify_published --repo-id owner/name
    python -m agrimanager.checkpoint.verify_published --repo-id owner/name --json

Exit status is 0 only when every revision inspected passed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Sequence

API = "https://huggingface.co/api/models"


def _get(url: str, token: str | None) -> Any:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
        return json.load(response)


def _get_raw(url: str, token: str | None) -> bytes:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
        return response.read()


def verify_revision(repo_id: str, revision: str, token: str | None) -> dict[str, Any]:
    """Check one revision against its own manifest."""
    result: dict[str, Any] = {
        "repo_id": repo_id,
        "revision": revision,
        "ok": False,
        "problems": [],
        "checked_sha256": 0,
        "checked_size": 0,
    }

    try:
        tree = _get(f"{API}/{repo_id}/tree/{revision}", token)
    except urllib.error.HTTPError as exc:
        result["problems"].append(f"cannot list revision: HTTP {exc.code}")
        return result

    on_hub = {entry["path"]: entry for entry in tree if entry.get("type") == "file"}
    result["file_count"] = len(on_hub)
    result["total_bytes"] = sum(entry.get("size", 0) for entry in on_hub.values())

    if "run_manifest.json" not in on_hub:
        result["problems"].append(
            "no run_manifest.json — the revision was created but its contents "
            "never landed"
        )
        return result

    try:
        raw = _get_raw(
            f"https://huggingface.co/{repo_id}/resolve/{revision}/run_manifest.json", token
        )
        manifest = json.loads(raw)
    except (urllib.error.HTTPError, json.JSONDecodeError) as exc:
        result["problems"].append(f"run_manifest.json unreadable: {exc}")
        return result

    declared = manifest.get("files") or {}
    if not declared:
        result["problems"].append("run_manifest.json declares no files")
        return result

    result["step"] = (manifest.get("training") or {}).get("checkpoint_step")
    result["artifact_id"] = manifest.get("artifact_id")

    for name, spec in sorted(declared.items()):
        entry = on_hub.get(name)
        if entry is None:
            result["problems"].append(f"{name}: declared in the manifest, absent from the hub")
            continue

        expected_bytes = spec.get("bytes")
        if expected_bytes is not None and entry.get("size") != expected_bytes:
            result["problems"].append(
                f"{name}: {entry.get('size')} bytes on the hub, manifest says {expected_bytes}"
            )
            continue
        result["checked_size"] += 1

        expected_sha = spec.get("sha256")
        actual_sha = (entry.get("lfs") or {}).get("oid")
        if expected_sha and actual_sha:
            if actual_sha != expected_sha:
                result["problems"].append(
                    f"{name}: sha256 {actual_sha[:16]}… on the hub, manifest says "
                    f"{expected_sha[:16]}…"
                )
                continue
            result["checked_sha256"] += 1

    # The weights themselves must be among the SHA-256-verified files; passing on
    # size alone for a shard would defeat the point of the check.
    weights = [n for n in declared if n.endswith(".safetensors")]
    if not weights:
        result["problems"].append("the manifest declares no .safetensors weights")
    else:
        unverified = [
            n for n in weights if not (on_hub.get(n, {}).get("lfs") or {}).get("oid")
        ]
        if unverified:
            result["problems"].append(
                f"weights not SHA-256-verifiable on the hub: {', '.join(sorted(unverified))}"
            )

    result["ok"] = not result["problems"]
    return result


def verify_repo(repo_id: str, token: str | None) -> list[dict[str, Any]]:
    refs = _get(f"{API}/{repo_id}/refs", token)
    names = [
        b["name"] for b in refs.get("branches", []) if b["name"].startswith("step-")
    ]
    names.sort(key=lambda n: int(n.split("-")[1]) if n.split("-")[1].isdigit() else 0)
    return [verify_revision(repo_id, name, token) for name in names]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m agrimanager.checkpoint.verify_published",
        description="Verify published checkpoints against their manifests.",
    )
    parser.add_argument("--repo-id", required=True, action="append", dest="repo_ids")
    parser.add_argument("--revision", action="append", default=[],
                        help="limit to these revisions; default is every step-N branch")
    parser.add_argument("--json", action="store_true", help="emit machine-readable results")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    token = os.environ.get("HF_TOKEN")

    results: list[dict[str, Any]] = []
    for repo_id in args.repo_ids:
        if args.revision:
            results.extend(verify_revision(repo_id, r, token) for r in args.revision)
        else:
            results.extend(verify_repo(repo_id, token))

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            mark = "OK  " if r["ok"] else "FAIL"
            size = r.get("total_bytes", 0) / 1e9
            print(
                f"{mark} {r['repo_id'].split('/')[-1]:34s} {r['revision']:9s} "
                f"{r.get('file_count', 0):2d} files {size:5.2f} GB  "
                f"sha256-verified={r['checked_sha256']} size-verified={r['checked_size']}"
            )
            for problem in r["problems"]:
                print(f"       - {problem}")

    failed = [r for r in results if not r["ok"]]
    if not args.json:
        print(f"\n{len(results) - len(failed)}/{len(results)} revisions verified complete")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
