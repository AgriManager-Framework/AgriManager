"""Merge an FSDP training checkpoint and publish it as a private HF repository.

The trainer writes sharded tensors (``model_world_size_N_rank_K.pt``) that no
evaluation script can load directly. This module performs the four steps between
that output and a usable artifact:

1. merge the shards into Transformers layout via ``verl.model_merger``
2. strip host-specific metadata the merge carries over
3. record provenance in ``run_manifest.json``
4. upload to a **private** repository, refusing anything that leaks identity

Step 4 is why this exists rather than ``verl.model_merger``'s own
``--hf_upload_path``: that path creates a public repository and writes no
manifest.

Identity terms are never hardcoded here. Writing a username into this file to
detect that username would embed it in the repository permanently. Structural
patterns (absolute paths, e-mail addresses, tracker URLs) are built in; specific
names are supplied at call time through ``--forbid`` or ``PUBLISH_FORBID_TERMS``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

# Metadata keys that Transformers populates with wherever the model happened to
# be loaded from. On a training host that is an absolute path.
_HOST_PATH_KEYS = ("_name_or_path", "name_or_path", "tokenizer_file", "vocab_file", "merges_file")

# Files whose text is scanned before upload. Weights are exempt: they are
# tensors, and hashing them is handled separately.
_SCANNED_SUFFIXES = (".json", ".md", ".txt", ".py", ".yaml", ".yml")

_STRUCTURAL_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"(?<![\w/])/(?:work|projects|home|scratch|workspace|mnt|data)/[\w.\-/]+", "absolute filesystem path"),
    (r"(?<![\w/])/u/[\w.\-/]+", "absolute filesystem path"),
    # No Windows-path rule. Everything here runs on Linux, and `X:\\...` matches
    # the escaped backslashes inside Jinja chat templates, failing every real
    # tokenizer_config.json.
    (r"[\w.\-+]+@[\w\-]+\.[\w.\-]+", "e-mail address"),
    (r"wandb\.ai/[\w\-]+", "W&B entity"),
    # Deliberately not word-bounded: `\b` does not match between a letter and an
    # underscore, so `\bagrimanager\b` would miss `agrimanager_t32_unified`,
    # which is exactly the form W&B project names take.
    (r"agrimanager", "project name"),
)


class PublishError(RuntimeError):
    """Raised when the artifact is unfit to publish."""


@dataclass
class PublishSpec:
    """Everything needed to describe one checkpoint as a published artifact."""

    checkpoint_dir: Path
    repo_id: str
    run_name: str
    experiment_id: str
    seed: int
    base_model: str
    checkpoint_step: int
    total_training_steps: int | None = None
    save_freq: int | None = None
    algorithm: str = "GRPO"
    require_think: bool | None = None
    thinking_mode: str | None = None
    think_tag: str | None = None
    run_definition: str | None = None
    wandb_run_id: str | None = None
    dataset_dir: Path | None = None
    notes: str | None = None
    revision: str | None = None
    forbid_terms: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# identity guard
# --------------------------------------------------------------------------- #

def find_identity_leaks(text: str, extra_terms: Iterable[str] = ()) -> list[str]:
    """Return human-readable descriptions of identity leaks found in ``text``."""
    leaks: list[str] = []
    for pattern, label in _STRUCTURAL_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            leaks.append(f"{label}: {match.group(0)!r}")
    for term in extra_terms:
        term = term.strip()
        if term and term.lower() in text.lower():
            leaks.append(f"forbidden term: {term!r}")
    return leaks


def collect_forbid_terms(cli_terms: Sequence[str]) -> tuple[str, ...]:
    """Merge ``--forbid`` values with ``PUBLISH_FORBID_TERMS`` from the env."""
    terms = list(cli_terms)
    env = os.environ.get("PUBLISH_FORBID_TERMS", "")
    terms.extend(part for part in env.split(",") if part.strip())
    return tuple(dict.fromkeys(t.strip() for t in terms if t.strip()))


def _guard_text(label: str, text: str, terms: Iterable[str]) -> None:
    leaks = find_identity_leaks(text, terms)
    if leaks:
        joined = "\n  ".join(sorted(set(leaks)))
        raise PublishError(f"{label} would leak identifying information:\n  {joined}")


def guard_folder(folder: Path, terms: Iterable[str]) -> None:
    """Scan every text-like file in ``folder`` for identity leaks."""
    terms = tuple(terms)
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _SCANNED_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        _guard_text(str(path.relative_to(folder)), text, terms)


# --------------------------------------------------------------------------- #
# source inspection
# --------------------------------------------------------------------------- #

def resolve_actor_dir(checkpoint_dir: Path) -> Path:
    """Accept either ``global_step_N`` or ``global_step_N/actor``."""
    if (checkpoint_dir / "fsdp_config.json").is_file():
        return checkpoint_dir
    actor = checkpoint_dir / "actor"
    if (actor / "fsdp_config.json").is_file():
        return actor
    raise PublishError(
        f"no FSDP checkpoint under {checkpoint_dir}: expected fsdp_config.json "
        f"there or in an actor/ subdirectory"
    )


def inspect_source(actor_dir: Path) -> dict[str, Any]:
    """Verify the shard set is complete and report what it contains."""
    config = json.loads((actor_dir / "fsdp_config.json").read_text())
    world_size = config.get("world_size")
    if not isinstance(world_size, int) or world_size < 1:
        raise PublishError(f"fsdp_config.json has no usable world_size: {config!r}")

    missing = [
        f"model_world_size_{world_size}_rank_{rank}.pt"
        for rank in range(world_size)
        if not (actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt").is_file()
    ]
    if missing:
        raise PublishError(
            "incomplete checkpoint, missing model shards: " + ", ".join(missing)
        )

    if not (actor_dir / "huggingface" / "config.json").is_file():
        raise PublishError(
            f"{actor_dir}/huggingface/config.json is absent, so the merge has no "
            f"model config to rebuild from"
        )

    present = {p.name.split("_world_size_")[0] for p in actor_dir.glob("*_world_size_*.pt")}
    return {
        "world_size": world_size,
        "save_contents": sorted(present),
        "shard_bytes": sum(p.stat().st_size for p in actor_dir.glob("*_world_size_*.pt")),
    }


# --------------------------------------------------------------------------- #
# merge and sanitize
# --------------------------------------------------------------------------- #

def merge_checkpoint(actor_dir: Path, target_dir: Path, project_root: Path) -> None:
    """Run ``verl.model_merger`` to produce Transformers layout in ``target_dir``."""
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    verl_path = str(project_root / "verl")
    env["PYTHONPATH"] = f"{verl_path}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)

    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", str(actor_dir),
        "--target_dir", str(target_dir),
    ]
    print("merging:", " ".join(cmd), flush=True)
    result = subprocess.run(cmd, env=env, cwd=str(project_root), check=False)
    if result.returncode != 0:
        raise PublishError(f"verl.model_merger failed with exit code {result.returncode}")

    weights = list(target_dir.glob("*.safetensors")) + list(target_dir.glob("*.bin"))
    if not weights:
        raise PublishError(f"merge produced no weight files in {target_dir}")


def sanitize_metadata(target_dir: Path) -> list[str]:
    """Blank out metadata keys carrying the training host's paths.

    Transformers writes ``_name_or_path`` from wherever the base model was
    loaded. On a training host that is an absolute path, which the artifact
    rules forbid. Returns the list of changed ``file:key`` pairs.
    """
    changed: list[str] = []
    for path in sorted(target_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        dirty = False
        for key in _HOST_PATH_KEYS:
            value = data.get(key)
            if isinstance(value, str) and value and find_identity_leaks(value):
                data[key] = ""
                changed.append(f"{path.name}:{key}")
                dirty = True
        if dirty:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return changed


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_merged(target_dir: Path, load_model: bool) -> dict[str, Any]:
    """Confirm the merged directory is loadable and describe it."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(target_dir), local_files_only=True)
    info: dict[str, Any] = {
        "architecture": (config.architectures or ["unknown"])[0],
        "dtype": str(getattr(config, "torch_dtype", None) or getattr(config, "dtype", "unknown")),
    }

    index = target_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        info["tensor_count"] = len(weight_map)

    if load_model:
        import torch
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            str(target_dir), local_files_only=True, torch_dtype=torch.bfloat16
        )
        info["param_count"] = sum(p.numel() for p in model.parameters())
        del model
    return info


def hash_folder(folder: Path) -> dict[str, dict[str, Any]]:
    """Checksum every file that will be uploaded."""
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(folder))] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    return out


# --------------------------------------------------------------------------- #
# manifest and card
# --------------------------------------------------------------------------- #

def _git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        # GIT_OPTIONAL_LOCKS=0 makes git skip anything that would take a lock.
        # `git status` otherwise grabs .git/index.lock to write back a
        # refreshed index, and this runs once per checkpoint from a compute
        # node against a Lustre working tree. A slow or interrupted refresh
        # leaves the lock behind, and every later commit in the repository
        # then fails with "Another git process seems to be running" until
        # someone removes it by hand. That happened twice here. We only read
        # the status, so the refresh is worth nothing to us.
        env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
        try:
            out = subprocess.run(
                ["git", *args], cwd=str(project_root), check=True,
                capture_output=True, text=True, timeout=30, env=env,
            )
            return out.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit, "dirty": bool(status) if status is not None else None}


def _run_definition_digest(spec: PublishSpec, project_root: Path) -> str | None:
    """Hash the run script so the definition is pinned even without git."""
    if not spec.run_definition:
        return None
    path = project_root / spec.run_definition
    return _sha256(path) if path.is_file() else None


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("torch", "transformers", "verl"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except Exception:  # noqa: BLE001 - a missing version is not fatal
            out[name] = "unavailable"
    return out


def build_manifest(
    spec: PublishSpec,
    source_info: dict[str, Any],
    merged_info: dict[str, Any],
    files: dict[str, dict[str, Any]],
    sanitized: list[str],
    project_root: Path,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "manifest_version": "1",
        "artifact_id": f"{spec.run_name}-gs{spec.checkpoint_step}",
        "revision": spec.revision or "main",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "experiment": {
            "cell": spec.experiment_id,
            "run_name": spec.run_name,
        },
        "training": {
            "base_model": spec.base_model,
            "algorithm": spec.algorithm,
            "seed": spec.seed,
            "data_seed": spec.seed,
            "data_loader_seed": spec.seed,
            "checkpoint_step": spec.checkpoint_step,
            "total_training_steps": spec.total_training_steps,
            "save_freq": spec.save_freq,
            "selection_rule": "terminal or last-valid checkpoint, never best-validation",
            # The reasoning format the policy was trained under. Frozen
            # evaluation suites are policy-neutral, so evaluation has to
            # reapply this; recording it here is what lets the evaluator
            # assert the condition instead of being told it.
            "reasoning": {
                "require_think": spec.require_think,
                "thinking_mode": spec.thinking_mode,
                "think_tag": spec.think_tag,
            },
        },
        "provenance": {
            "run_definition": spec.run_definition,
            # A rented host holds a copy of the tree, not a clone, so git state
            # is often unavailable. Hashing the run definition pins the exact
            # experiment description regardless.
            "run_definition_sha256": _run_definition_digest(spec, project_root),
            "wandb_run_id": spec.wandb_run_id,
            "git": _git_state(project_root),
            "versions": _versions(),
        },
        "checkpoint": {
            "source_layout": "fsdp-sharded",
            "world_size": source_info["world_size"],
            "save_contents": source_info["save_contents"],
            "sanitized_metadata_keys": sanitized,
            **merged_info,
        },
        "files": files,
    }
    if spec.dataset_dir is not None:
        manifest["training"]["dataset_files"] = {
            p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)}
            for p in sorted(spec.dataset_dir.glob("*.parquet"))
        }
    if spec.notes:
        manifest["notes"] = spec.notes
    return manifest


def build_model_card(spec: PublishSpec, manifest: dict[str, Any], public: bool = False) -> str:
    training = manifest["training"]
    # State the visibility that actually applies. A card that calls a
    # world-readable repository private misdescribes it to everyone who reads it.
    standing = (
        "Research artifact released for review."
        if public
        else "Private research artifact. Not for redistribution."
    )
    return "\n".join([
        "---",
        "library_name: transformers",
        "tags:",
        "  - reinforcement-learning",
        "  - agriculture",
        "---",
        "",
        f"# {spec.run_name}",
        "",
        standing,
        "",
        f"- Cell: `{spec.experiment_id}`",
        f"- Base model: `{training['base_model']}`",
        f"- Algorithm: {training['algorithm']}",
        f"- Seed: {training['seed']}",
        f"- Checkpoint step: {training['checkpoint_step']}"
        + (f" of {training['total_training_steps']}" if training["total_training_steps"] else ""),
        "",
        "Selection rule: terminal or last-valid checkpoint. This checkpoint was",
        "not chosen by validation score.",
        "",
        "See `run_manifest.json` for full provenance and file checksums.",
        "",
    ])


# --------------------------------------------------------------------------- #
# upload
# --------------------------------------------------------------------------- #

def upload_folder(
    folder: Path,
    repo_id: str,
    token: str | None,
    commit_message: str,
    revision: str | None = None,
    public: bool = False,
) -> str:
    """Upload to a repository, optionally on its own branch.

    One run owns one repository, so the delivery board keeps a single link per
    cell. Periodic checkpoints go to separate revisions rather than separate
    repositories, which is what peak-to-final analysis needs.

    Visibility is never inherited. ``public`` creates a new repository public and
    permits uploading into an existing public one; without it, an existing public
    repository is refused, so weights cannot be published because someone flipped
    a setting elsewhere.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=not public, exist_ok=True)

    info = api.repo_info(repo_id=repo_id, repo_type="model")
    if not info.private and not public:
        raise PublishError(
            f"{repo_id} is public. Uploading would publish these weights. Pass "
            f"--public if that is intended, or make the repository private."
        )
    if not info.private:
        print(f"note: {repo_id} is public; these weights are world-readable", flush=True)

    if revision:
        api.create_branch(repo_id=repo_id, branch=revision, repo_type="model", exist_ok=True)

    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        commit_message=commit_message,
    )
    suffix = f"/tree/{revision}" if revision else ""
    return f"https://huggingface.co/{repo_id}{suffix}"


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def publish_checkpoint(
    spec: PublishSpec,
    work_dir: Path,
    project_root: Path,
    *,
    upload: bool = True,
    verify_load: bool = True,
    keep_work_dir: bool = False,
    token: str | None = None,
    public: bool = False,
) -> dict[str, Any]:
    """Merge, verify, describe and optionally upload one checkpoint."""
    _guard_text("repository id", spec.repo_id, spec.forbid_terms)
    _guard_text("run name", spec.run_name, spec.forbid_terms)

    actor_dir = resolve_actor_dir(spec.checkpoint_dir)
    source_info = inspect_source(actor_dir)
    print(
        f"source: world_size={source_info['world_size']} "
        f"contents={source_info['save_contents']} "
        f"{source_info['shard_bytes'] / 1e9:.1f} GB",
        flush=True,
    )

    target_dir = work_dir / f"{spec.run_name}-gs{spec.checkpoint_step}"
    merge_checkpoint(actor_dir, target_dir, project_root)

    sanitized = sanitize_metadata(target_dir)
    if sanitized:
        print(f"sanitized host paths from: {', '.join(sanitized)}", flush=True)

    merged_info = verify_merged(target_dir, load_model=verify_load)
    print(f"merged: {merged_info}", flush=True)

    files = hash_folder(target_dir)
    manifest = build_manifest(spec, source_info, merged_info, files, sanitized, project_root)

    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    _guard_text("run_manifest.json", manifest_text, spec.forbid_terms)
    (target_dir / "run_manifest.json").write_text(manifest_text)

    card = build_model_card(spec, manifest, public=public)
    _guard_text("README.md", card, spec.forbid_terms)
    (target_dir / "README.md").write_text(card)

    guard_folder(target_dir, spec.forbid_terms)
    print(f"identity guard passed over {len(files) + 2} files", flush=True)

    result: dict[str, Any] = {
        "local_dir": str(target_dir),
        "manifest": manifest,
        "uploaded": False,
    }

    if upload:
        url = upload_folder(
            target_dir,
            spec.repo_id,
            token,
            commit_message=f"{spec.experiment_id} step {spec.checkpoint_step}",
            revision=spec.revision,
            public=public,
        )
        result["uploaded"] = True
        result["url"] = url
        # Say which visibility actually applied. A hardcoded "private" here
        # once made a public upload read as private in the watcher log.
        print(f"uploaded ({'public' if public else 'private'}): {url}", flush=True)
        if not keep_work_dir:
            shutil.rmtree(target_dir)
            result["local_dir"] = None
    else:
        print(f"upload skipped; artifact left at {target_dir}", flush=True)

    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m agrimanager.checkpoint.publish",
        description="Merge an FSDP checkpoint and publish it as a private HF repository.",
    )
    parser.add_argument("--checkpoint-dir", required=True, type=Path,
                        help="global_step_N directory, or its actor/ subdirectory")
    parser.add_argument("--repo-id", required=True,
                        help="target repository, e.g. owner/schema-t32-qwen-think-r2")
    parser.add_argument("--run-name", required=True, help="identity-neutral run name")
    parser.add_argument("--experiment-id", required=True, help="delivery-board cell, e.g. E2-Q-T-R2")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint-step", required=True, type=int)
    parser.add_argument("--total-training-steps", type=int)
    parser.add_argument("--save-freq", type=int)
    parser.add_argument("--algorithm", default="GRPO")
    parser.add_argument("--require-think", choices=["True", "False"],
                        help="reasoning format the run trained under; recorded for the evaluator")
    parser.add_argument("--thinking-mode")
    parser.add_argument("--think-tag")
    parser.add_argument("--run-definition", help="repository-relative path of the run script")
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--dataset-dir", type=Path, help="hash these parquets into the manifest")
    parser.add_argument("--notes")
    parser.add_argument("--revision",
                        help="branch to publish on, e.g. step-200; default is main")
    parser.add_argument("--forbid", action="append", default=[],
                        help="extra term to reject; repeatable. Also read from PUBLISH_FORBID_TERMS")
    parser.add_argument("--work-dir", type=Path, default=Path("./.publish_work"))
    parser.add_argument("--project-root", type=Path,
                        default=Path(__file__).resolve().parents[2])
    parser.add_argument("--no-upload", action="store_true", help="merge and verify only")
    parser.add_argument("--no-verify-load", action="store_true",
                        help="skip the full model load (saves memory, weakens the check)")
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--public", action="store_true",
                        help="publish publicly: create a new repository public and permit "
                             "uploading into an existing public one")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    spec = PublishSpec(
        checkpoint_dir=args.checkpoint_dir,
        repo_id=args.repo_id,
        run_name=args.run_name,
        experiment_id=args.experiment_id,
        seed=args.seed,
        base_model=args.base_model,
        checkpoint_step=args.checkpoint_step,
        total_training_steps=args.total_training_steps,
        save_freq=args.save_freq,
        algorithm=args.algorithm,
        require_think=None if args.require_think is None else args.require_think == "True",
        thinking_mode=args.thinking_mode,
        think_tag=args.think_tag,
        run_definition=args.run_definition,
        wandb_run_id=args.wandb_run_id,
        dataset_dir=args.dataset_dir,
        notes=args.notes,
        revision=args.revision,
        forbid_terms=collect_forbid_terms(args.forbid),
    )
    try:
        publish_checkpoint(
            spec,
            work_dir=args.work_dir,
            project_root=args.project_root,
            upload=not args.no_upload,
            verify_load=not args.no_verify_load,
            keep_work_dir=args.keep_work_dir,
            token=os.environ.get("HF_TOKEN"),
            public=args.public,
        )
    except PublishError as exc:
        print(f"publish failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
