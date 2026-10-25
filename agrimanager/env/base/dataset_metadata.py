"""Dataset split metadata helpers.

These helpers keep dataset generation and validation logging aligned without
making every experiment script re-encode split semantics by hand.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


_RESERVED_LABEL_KEYS = {
    "dataset_split",
    "dataset_role",
    "split",
    "validation_set",
}


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def clean_label_mapping(raw: Any) -> dict[str, str]:
    """Return non-empty string labels from a user-provided mapping."""
    if not isinstance(raw, dict):
        return {}
    labels: dict[str, str] = {}
    for key, value in raw.items():
        key_str = _clean_str(key)
        value_str = _clean_str(value)
        if key_str and value_str:
            labels[key_str] = value_str
    return labels


def infer_split_role(split: str, explicit_role: Any = None) -> str:
    """Infer the semantic role for a dataset split name."""
    role = _clean_str(explicit_role).lower().replace("-", "_")
    aliases = {
        "valid": "validation",
        "validate": "validation",
        "validation": "validation",
        "val": "validation",
        "eval": "validation",
    }
    role = aliases.get(role, role)
    if role in {"train", "validation", "test"}:
        return role

    split_norm = _clean_str(split).lower().replace("-", "_")
    if split_norm == "train" or split_norm.startswith("train_"):
        return "train"
    if split_norm == "val" or split_norm.startswith("val_"):
        return "validation"
    if split_norm == "validation" or split_norm.startswith("validation_"):
        return "validation"
    if split_norm == "test" or split_norm.startswith("test_"):
        return "test"
    return split_norm or "unknown"


def infer_validation_set(split: str, role: str, explicit_validation_set: Any = None) -> str | None:
    """Infer a validation-set name for validation splits."""
    if role != "validation":
        return None
    explicit = _clean_str(explicit_validation_set)
    if explicit:
        return explicit

    split_norm = _clean_str(split).replace("-", "_")
    for prefix in ("val_", "validation_"):
        if split_norm.startswith(prefix):
            suffix = split_norm[len(prefix):].strip("_")
            if suffix:
                return suffix
    return split_norm or "val"


def normalize_split_metadata(
    split: str,
    split_config: Any = None,
    *,
    dataset_labels: Any = None,
    inferred_labels: Any = None,
) -> dict[str, Any]:
    """Normalize split role, validation set, and effective labels.

    Inferred labels are written first, then dataset-level labels, then
    split-level labels. User labels therefore override inferred labels for the
    same key while reserved structural keys stay canonical.
    """
    split_cfg = split_config if isinstance(split_config, dict) else {}
    role = infer_split_role(split, split_cfg.get("role"))

    labels: dict[str, str] = {}
    labels.update(clean_label_mapping(inferred_labels))
    labels.update(clean_label_mapping(dataset_labels))
    labels.update(clean_label_mapping(split_cfg.get("labels")))

    validation_set = (
        split_cfg.get("validation_set")
        or labels.pop("validation_set", None)
    )
    validation_set = infer_validation_set(split, role, validation_set)

    user_labels = {
        key: value
        for key, value in labels.items()
        if key not in _RESERVED_LABEL_KEYS
    }
    group_labels = {
        "split": split,
        "dataset_split": split,
        "dataset_role": role,
    }
    if validation_set:
        group_labels["validation_set"] = validation_set
    group_labels.update(user_labels)

    return {
        "split": split,
        "role": role,
        "validation_set": validation_set,
        "labels": user_labels,
        "group_labels": group_labels,
    }


def apply_split_metadata_to_env_config(
    env_config: dict[str, Any],
    split: str,
    split_config: Any = None,
    *,
    dataset_labels: Any = None,
    inferred_labels: Any = None,
) -> dict[str, Any]:
    """Inject normalized split metadata into an environment config."""
    metadata = normalize_split_metadata(
        split,
        split_config,
        dataset_labels=dataset_labels,
        inferred_labels=inferred_labels,
    )
    env_config["dataset_split"] = metadata["split"]
    env_config["dataset_role"] = metadata["role"]
    if metadata["validation_set"]:
        env_config["validation_set"] = metadata["validation_set"]
    else:
        env_config.pop("validation_set", None)

    group_labels = dict(env_config.get("trajectory_group_labels") or {})
    group_labels.update(metadata["group_labels"])
    env_config["trajectory_group_labels"] = group_labels
    return env_config


def apply_validation_axis_fallback(
    env_config: dict[str, Any],
    validation_axis: Any = None,
) -> dict[str, Any]:
    """Fill a missing validation-axis label from ``validation_set``.

    This is deliberately applied to final row-level env configs, not to split
    metadata, so row-level labels such as the true crop or simulator are never
    overwritten by suite names like ``heldout_crops``.
    """
    axis = _clean_str(validation_axis)
    if not axis:
        return env_config
    if _clean_str(env_config.get("dataset_role")) != "validation":
        return env_config

    group_labels = dict(env_config.get("trajectory_group_labels") or {})
    if _clean_str(group_labels.get(axis)):
        return env_config

    validation_set = _clean_str(
        env_config.get("validation_set") or group_labels.get("validation_set")
    )
    if not validation_set:
        return env_config

    group_labels[axis] = validation_set
    env_config["trajectory_group_labels"] = group_labels
    return env_config


def summarize_constant_validation_axis_label(
    env_configs: list[dict[str, Any]],
    validation_axis: Any = None,
) -> dict[str, str]:
    """Return the validation axis when every generated row has one value.

    Manifests describe split-level metadata, so a label is only safe to record
    when it is present and identical on every row. Multi-valued axes such as a
    held-out crop split remain represented in the parquet rows rather than
    being collapsed into a misleading manifest label.
    """
    axis = _clean_str(validation_axis)
    if not axis or not env_configs:
        return {}

    values: set[str] = set()
    for env_config in env_configs:
        group_labels = env_config.get("trajectory_group_labels") or {}
        value = _clean_str(group_labels.get(axis))
        if not value:
            return {}
        values.add(value)
        if len(values) > 1:
            return {}

    return {axis: values.pop()}


def _config_sha256(config: dict[str, Any]) -> str:
    config_path = config.get("_config_path")
    if config_path:
        path = Path(str(config_path))
        if path.is_file():
            return hashlib.sha256(path.read_bytes()).hexdigest()
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_dataset_manifest(
    save_dir: str | Path,
    config: dict[str, Any],
    summary: dict[str, int],
    *,
    split_metadata: dict[str, dict[str, Any]] | None = None,
    split_env_configs: dict[str, list[dict[str, Any]]] | None = None,
) -> Path:
    """Write a manifest describing generated dataset splits."""
    save_path = Path(save_dir)
    metadata_by_split = split_metadata or {}
    env_configs_by_split = split_env_configs or {}
    validation_axis = config.get("validation_axis")

    splits: dict[str, dict[str, Any]] = {}
    validation_sets: dict[str, dict[str, Any]] = {}
    for split, count in summary.items():
        metadata = metadata_by_split.get(split) or normalize_split_metadata(
            split,
            ((config.get("sampling") or {}).get("splits") or {}).get(split),
            dataset_labels=config.get("labels"),
        )
        labels = dict(metadata.get("labels") or {})
        labels.update(
            summarize_constant_validation_axis_label(
                env_configs_by_split.get(split, []),
                validation_axis,
            )
        )
        entry: dict[str, Any] = {
            "path": f"{split}.parquet",
            "role": metadata["role"],
            "num_rows": int(count),
            "labels": labels,
        }
        validation_set = metadata.get("validation_set")
        if validation_set:
            entry["validation_set"] = validation_set
            val_entry = validation_sets.setdefault(
                validation_set,
                {"files": [], "labels": {}},
            )
            val_entry["files"].append(entry["path"])
            val_entry["labels"].update(entry["labels"])
        splits[split] = entry

    manifest = {
        "schema_version": 1,
        "dataset_id": config.get("dataset_id"),
        "env_name": config.get("env_name"),
        "validation_axis": validation_axis,
        "config_sha256": _config_sha256(config),
        "splits": splits,
        "validation_sets": validation_sets,
    }

    manifest_path = save_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path
