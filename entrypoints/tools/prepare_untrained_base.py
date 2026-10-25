#!/usr/bin/env python3
"""Make the untrained-base checkpoint directories used as floors on the test suite.

    python entrypoints/tools/prepare_untrained_base.py --out <untrained-base> \
        [--model Qwen/Qwen3-4B-Instruct-2507]

Writes one directory per response format, ``<out>/<name>-think`` and
``<out>/<name>-no-think``. Each holds symlinks to the downloaded base-model files
and a ``run_manifest.json`` that records that no training took place. The two
directories have byte-identical weights; only the evaluation protocol differs
(``--require-think True --think-tag tool_call`` versus ``--require-think False``),
which is why the format is part of the name. ``results/scoring_manifest.tsv``
refers to them as ``<untrained-base>/qwen3-4b-instruct-2507-{think,no-think}``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True, help="parent directory for the floor checkpoints")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Hugging Face model id")
    ap.add_argument("--name", default=None, help="directory stem (default: lower-cased model name)")
    ap.add_argument("--revision", default=None, help="model revision to download (default: main)")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download

    snapshot = Path(snapshot_download(args.model, revision=args.revision))
    stem = args.name or args.model.split("/")[-1].lower()
    for arm in ("think", "no-think"):
        d = args.out / f"{stem}-{arm}"
        d.mkdir(parents=True, exist_ok=True)
        for f in snapshot.iterdir():
            link = d / f.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(os.path.realpath(f))
        note = (
            f"Untrained base model, the floor for the {arm} arms. Weights are byte-identical to the other "
            "baseline arm; only the evaluation protocol differs, which is why the arm is in the name. "
            "Records no training condition because there was none."
        )
        manifest = {
            "experiment": {"cell": f"control-untrained-base-{arm}", "run_name": f"baseline-{stem}-untrained-{arm}"},
            "artifact_id": None,
            "training": None,
            "provenance": {"source_repo": args.model, "source_revision": args.revision, "note": note},
        }
        (d / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(d)


if __name__ == "__main__":
    main()
