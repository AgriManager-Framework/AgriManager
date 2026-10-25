#!/usr/bin/env python3
"""Upload the frozen held-out test suites ("E4" suites) to a Hugging Face dataset.

Uploads every suite directory under data/ (manifests, audits, test parquet
files) and the frozen source selections they were drawn from, in one commit.
The dataset is created private unless --public is given; publishing it is a
deliberate, separate step.

Requires a write token. Provide it as HF_TOKEN, or in a file passed with
--token-file; do not pass it on the command line, which would expose it in the
process list on a shared node.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi


EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"
REPO_ROOT = EXPERIMENT_DIR.parents[1]

# Suites plus the frozen source selections that prove where they came from.
INCLUDE_DIRS = (
    "e4_t11_potato_weather_regime",
    "e4_t11_potato_weather_regime_nn",
    "e4_t11_chickpea_weather_regime",
    "e4_t11_chickpea_weather_regime_nn",
    "e4_t12_cross_crop",
    "e4_t12_cross_crop_nn",
    "e4_t13_price_regime",
    "e4_t13_price_regime_nn",
    "e4_t21_observation_schema",
    "e4_t22_action_menu",
    "e4_t22_action_menu_nn",
    "e4_t23_reward_formulation",
    "e4_t31_cross_sim_wofost",
    "e4_t31_cross_sim_dssat",
    "e4_t31_cross_sim_cycles",
    "e4_t32_wofost_seen",
    "e4_t32_wofost_heldout",
    "e4_t32_dssat_yield",
    "e4_t32_dssat_profit",
    "e4_t32_cycles_yield",
    "e4_t32_cycles_profit",
    # The Cycles suites rebuilt on unused weather years; they replace the three
    # first-build Cycles suites above in every reported table.
    "e4_t31_cross_sim_cycles_v2",
    "e4_t32_cycles_yield_v2",
    "e4_t32_cycles_profit_v2",
    "source_pools/maize_e4",
    "source_pools/t11_potato_e4",
    "source_pools/t11_chickpea_e4",
    "source_pools/t12_cross_crop_e4",
)

# rejected_explorations is an internal provenance record, not evaluation data.
# meteo_cache placeholders are empty. Neither belongs in the transfer bundle.
EXCLUDE_NAMES = {"meteo_cache", "rejected_explorations"}


def collect() -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for rel in INCLUDE_DIRS:
        root = DATA_DIR / rel
        if not root.is_dir():
            raise SystemExit(f"missing suite directory: {root}")
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in EXCLUDE_NAMES for part in path.relative_to(DATA_DIR).parts):
                continue
            files.append((path, str(path.relative_to(DATA_DIR))))
    return files


def guard_identity(files: list[tuple[Path, str]]) -> None:
    """Refuse to upload anything still carrying a machine path or account name."""
    import pandas as pd

    markers = ("/work/", "/u/", "/projects/", "/taiga/", "/home/", "/scratch/")
    offenders: list[str] = []
    for path, rel in files:
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            if "extra_info" not in frame.columns:
                continue
            blob = str(frame["extra_info"].tolist())
        else:
            blob = path.read_text(encoding="utf-8", errors="ignore")
        if any(m in blob for m in markers):
            offenders.append(rel)
    if offenders:
        raise SystemExit(
            "refusing to upload; machine paths or account names remain in:\n  "
            + "\n  ".join(offenders)
            + "\nrun normalize_eval_suite_paths.py --apply first"
        )


def ensure_writable_hf_home() -> None:
    """Point HF_HOME somewhere writable before the xet backend needs it.

    The xet uploader stages chunks under HF_HOME and, when it cannot, fails
    with a bare `Permission denied (os error 13)` that names no path -- so the
    cause is not discoverable from the error.

    This used to be an `os.environ.setdefault`, which does nothing at all when
    HF_HOME is already set. On this cluster the login profile sets it to a
    read-only project directory, so the guard silently did not fire and the
    upload failed exactly as it would have without it. Test writability rather
    than presence, and leave a deliberate writable HF_HOME alone.
    """
    fallback = REPO_ROOT / ".cache" / "huggingface"
    current = os.environ.get("HF_HOME")
    if current:
        probe = Path(current)
        try:
            probe.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=probe):
                pass
            return
        except OSError:
            print(f"HF_HOME is not writable, staging in {fallback} instead", file=sys.stderr)
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(fallback)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="e.g. <account>/<neutral-name>")
    parser.add_argument("--token-file", type=Path, default=None)
    parser.add_argument("--public", action="store_true", help="create the dataset public (default: private)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = collect()
    guard_identity(files)

    total = sum(p.stat().st_size for p, _ in files)
    print(f"{len(files)} files, {total/1024:.0f} KiB")
    for _, rel in files:
        print(f"  {rel}")

    if args.dry_run:
        print("\ndry run: nothing uploaded")
        return

    token = os.environ.get("HF_TOKEN")
    if args.token_file:
        token = args.token_file.read_text().strip()
    if not token:
        raise SystemExit("no token: set HF_TOKEN or pass --token-file")

    ensure_writable_hf_home()

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=not args.public,
        exist_ok=True,
    )
    # One commit built from the explicit file list. Per-file uploads are one
    # commit each, slow enough to be interrupted part way and leave the suite
    # half-published; folder upload relies on glob patterns that quietly missed
    # files sitting directly in a suite directory.
    from huggingface_hub import CommitOperationAdd

    operations = [
        CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(path))
        for path, rel in files
    ]
    api.create_commit(
        repo_id=args.repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message="Freeze the held-out test suites",
    )
    info = api.repo_info(repo_id=args.repo_id, repo_type="dataset")
    print(f"\nuploaded to {args.repo_id} (private={info.private})")


if __name__ == "__main__":
    main()
