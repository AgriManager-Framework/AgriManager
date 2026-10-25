#!/usr/bin/env python3
"""Repair env-config list fields that an earlier normalization turned into text.

`normalize_eval_suite_paths.py` used to deep-copy each record with
`json.loads(json.dumps(record, default=str))`. `default=str` replaces anything
json cannot encode with its `str()`, so a numpy array of observation-field names
was stored as the literal text "['FIN' 'DVS' ...]". The environment then treated
that string as a sequence and iterated it character by character, and every
affected suite failed to construct at all:

    ValueError: prompt_observation_fields contains fields that are not
    available in the simulator observation: ['[', "'", 'F', 'I', 'N', ...]

The root cause is fixed in `normalize_eval_suite_paths.py`. This repairs the
files already written.

Every list-valued field is checked, not just the one that surfaced first: the
same flattening hit `output_vars` in one T2.1 split.

Values are not recovered by parsing the damaged text. Each suite's config
declares these fields outright, so the declaration is the authority, and the
check is that re-flattening it reproduces the stored text character for
character. A disagreement means the file holds something the config cannot
explain, and the repair stops rather than guessing.

    python repair_stringified_env_fields.py            # report only
    python repair_stringified_env_fields.py --apply
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_DIR = Path(__file__).resolve().parent
DATA_DIR = EXPERIMENT_DIR / "data"
CONFIG_DIR = EXPERIMENT_DIR / "config"


def looks_stringified(value: object) -> bool:
    """A list field that was flattened into its numpy repr."""
    return isinstance(value, str) and value.startswith("[") and value.endswith("]")


def declared_value(dataset_id: str, split: str, field: str) -> list | None:
    """Return what the suite config declares for this field on this split.

    A split-level declaration wins over the dataset-level one, which is how the
    T2.1 suite gives each observation schema its own field set.
    """
    config_path = CONFIG_DIR / f"{dataset_id}.yaml"
    if not config_path.is_file():
        return None
    config = yaml.safe_load(config_path.read_text()) or {}

    split_env = (
        ((config.get("sampling") or {}).get("splits") or {}).get(split) or {}
    ).get("env") or {}
    if field in split_env:
        return list(split_env[field])

    dataset_env = config.get("env") or {}
    if field in dataset_env:
        return list(dataset_env[field])
    return None


def reproduces(declared: list, stored: str) -> bool:
    """Confirm the declaration is what was flattened.

    The damage was exactly `str(numpy_array)`, so re-flattening the declared
    value must reproduce the stored text character for character. This is
    stronger than parsing tokens out of the text and works for any dtype.
    """
    return str(np.array(declared)) == stored


def inspect(path: Path) -> dict[str, Any]:
    dataset_id, split = path.parent.name, path.stem
    frame = pd.read_parquet(path)
    if "extra_info" not in frame.columns:
        return {"path": path, "status": "skipped", "detail": "no extra_info"}

    # A healthy field round-trips out of parquet as a numpy array, which is
    # unhashable, so only the damaged string values may reach a set.
    # Any list-valued field could have been flattened, not just the one that
    # happened to surface first. Collect every damaged key in the file.
    damaged: dict[str, set[str]] = {}
    for info in frame["extra_info"]:
        for field, value in info["interaction_kwargs"]["env_config"].items():
            if looks_stringified(value):
                damaged.setdefault(field, set()).add(value)
    if not damaged:
        return {"path": path, "status": "clean"}

    repairs: dict[str, list] = {}
    for field, values in sorted(damaged.items()):
        if len(values) > 1:
            return {
                "path": path,
                "status": "error",
                "detail": f"{field}: {len(values)} distinct damaged values in one file",
            }
        declared = declared_value(dataset_id, split, field)
        if declared is None:
            return {
                "path": path,
                "status": "error",
                "detail": f"config/{dataset_id}.yaml declares no {field}",
            }
        stored = next(iter(values))
        if not reproduces(declared, stored):
            return {
                "path": path,
                "status": "error",
                "detail": (
                    f"{field}: the config declaration does not reproduce the stored "
                    f"text, so the file holds something the config cannot explain"
                ),
            }
        repairs[field] = declared

    return {
        "path": path,
        "status": "repairable",
        "frame": frame,
        "repairs": repairs,
        "rows": len(frame),
    }


def repair(report: dict[str, Any]) -> None:
    path, frame, repairs = report["path"], report["frame"], report["repairs"]

    records = []
    for info in frame["extra_info"]:
        record = dict(info)
        interaction = dict(record["interaction_kwargs"])
        env_config = dict(interaction["env_config"])
        for field, declared in repairs.items():
            env_config[field] = list(declared)
        interaction["env_config"] = env_config
        record["interaction_kwargs"] = interaction
        records.append(record)

    frame = frame.copy()
    frame["extra_info"] = records
    frame.to_parquet(path, index=False)

    after = pd.read_parquet(path)
    if len(after) != report["rows"]:
        raise SystemExit(f"row count changed while repairing {path}")
    for field, declared in repairs.items():
        values = {
            tuple(info["interaction_kwargs"]["env_config"][field])
            for info in after["extra_info"]
        }
        if values != {tuple(declared)}:
            raise SystemExit(f"repair of {field} did not take effect in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        action="store_true",
        help="rewrite the files; without it the script only reports",
    )
    args = parser.parse_args()

    reports = [inspect(path) for path in sorted(DATA_DIR.glob("e4_*/*.parquet"))]
    errors = [r for r in reports if r["status"] == "error"]
    repairable = [r for r in reports if r["status"] == "repairable"]
    clean = [r for r in reports if r["status"] == "clean"]

    for report in errors:
        print(f"  ERROR {report['path'].relative_to(DATA_DIR)}: {report['detail']}")
    for report in repairable:
        rel = report["path"].relative_to(DATA_DIR)
        fields = ", ".join(f"{k}({len(v)})" for k, v in report["repairs"].items())
        print(f"  {'repairing' if args.apply else 'damaged'} {str(rel):52s} "
              f"rows={report['rows']:4d} {fields}")
        if args.apply:
            repair(report)

    print(
        f"\n{len(clean)} clean, {len(repairable)} damaged, {len(errors)} unresolved"
    )
    if errors:
        raise SystemExit("refusing to repair while any file is unresolved")
    if repairable and not args.apply:
        print("re-run with --apply, then re-run every audit script and re-upload")


if __name__ == "__main__":
    main()
