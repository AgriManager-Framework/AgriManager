import json
from pathlib import Path

from agrimanager.env.base.dataset_metadata import (
    apply_split_metadata_to_env_config,
    apply_validation_axis_fallback,
    normalize_split_metadata,
    summarize_constant_validation_axis_label,
    write_dataset_manifest,
)
from agrimanager.env.cycles_gym.create_dataset import CyclesDatasetGenerator
from agrimanager.env.gym_dssat.create_dataset import DSSATDatasetGenerator


def test_legacy_validation_split_infers_role_and_validation_set():
    metadata = normalize_split_metadata(
        "val_drought",
        {},
        inferred_labels={"weather_regime": "drought"},
    )

    assert metadata["role"] == "validation"
    assert metadata["validation_set"] == "drought"
    assert metadata["group_labels"]["dataset_split"] == "val_drought"
    assert metadata["group_labels"]["dataset_role"] == "validation"
    assert metadata["group_labels"]["validation_set"] == "drought"
    assert metadata["group_labels"]["weather_regime"] == "drought"


def test_explicit_labels_override_inferred_labels():
    metadata = normalize_split_metadata(
        "val_drought",
        {"labels": {"weather_regime": "dry_spell"}},
        inferred_labels={"weather_regime": "drought"},
    )

    assert metadata["labels"]["weather_regime"] == "dry_spell"
    assert metadata["group_labels"]["weather_regime"] == "dry_spell"


def test_train_split_never_gets_validation_set_by_default():
    metadata = normalize_split_metadata(
        "train",
        {},
        dataset_labels={"validation_set": "yield_max"},
    )

    assert metadata["role"] == "train"
    assert metadata["validation_set"] is None
    assert "validation_set" not in metadata["group_labels"]


def test_apply_split_metadata_to_env_config_sets_group_labels():
    env_config = {"trajectory_group_labels": {"simulator": "wofost_gym"}}

    updated = apply_split_metadata_to_env_config(
        env_config,
        "val_id",
        {"labels": {"weather_regime": "id"}},
    )

    assert updated["dataset_split"] == "val_id"
    assert updated["dataset_role"] == "validation"
    assert updated["validation_set"] == "id"
    assert updated["trajectory_group_labels"]["simulator"] == "wofost_gym"
    assert updated["trajectory_group_labels"]["validation_set"] == "id"
    assert updated["trajectory_group_labels"]["weather_regime"] == "id"


def test_validation_axis_fallback_uses_validation_set_when_axis_missing():
    env_config = {"trajectory_group_labels": {}}

    updated = apply_split_metadata_to_env_config(
        env_config,
        "val_lw",
        {"validation_set": "lw"},
    )
    updated = apply_validation_axis_fallback(updated, "action_menu")

    assert updated["trajectory_group_labels"]["action_menu"] == "lw"


def test_validation_axis_fallback_does_not_override_explicit_axis_label():
    env_config = {"trajectory_group_labels": {}}

    updated = apply_split_metadata_to_env_config(
        env_config,
        "val_lw",
        {"validation_set": "lw", "labels": {"action_menu": "ln"}},
    )
    updated = apply_validation_axis_fallback(updated, "action_menu")

    assert updated["trajectory_group_labels"]["action_menu"] == "ln"


def test_validation_axis_fallback_only_applies_to_validation_rows():
    env_config = {"trajectory_group_labels": {}}

    updated = apply_split_metadata_to_env_config(
        env_config,
        "train_lw",
        {"role": "train", "validation_set": "lw"},
    )
    updated = apply_validation_axis_fallback(updated, "action_menu")

    assert "action_menu" not in updated["trajectory_group_labels"]


def test_validation_axis_fallback_ignores_empty_axis():
    env_config = {"trajectory_group_labels": {}}

    updated = apply_split_metadata_to_env_config(
        env_config,
        "val_lw",
        {"validation_set": "lw"},
    )
    updated = apply_validation_axis_fallback(updated, "")

    assert "action_menu" not in updated["trajectory_group_labels"]


def test_validation_axis_fallback_can_read_group_label_validation_set():
    env_config = {
        "dataset_role": "validation",
        "trajectory_group_labels": {"validation_set": "yield_max"},
    }

    updated = apply_validation_axis_fallback(env_config, "reward_formulation")

    assert updated["trajectory_group_labels"]["reward_formulation"] == "yield_max"


def test_validation_axis_fallback_does_not_override_row_level_axis():
    env_config = {"trajectory_group_labels": {"crop": "potato"}}

    updated = apply_split_metadata_to_env_config(
        env_config,
        "val_heldout_crops",
        {"validation_set": "heldout_crops"},
    )
    updated = apply_validation_axis_fallback(updated, "crop")

    assert updated["trajectory_group_labels"]["crop"] == "potato"


def test_manifest_axis_summary_requires_one_value_on_every_row():
    constant = summarize_constant_validation_axis_label(
        [
            {"trajectory_group_labels": {"weather_regime": "cold"}},
            {"trajectory_group_labels": {"weather_regime": "cold"}},
        ],
        "weather_regime",
    )
    multiple = summarize_constant_validation_axis_label(
        [
            {"trajectory_group_labels": {"crop": "potato"}},
            {"trajectory_group_labels": {"crop": "rice"}},
        ],
        "crop",
    )
    missing = summarize_constant_validation_axis_label(
        [
            {"trajectory_group_labels": {"action_menu": "ln"}},
            {"trajectory_group_labels": {}},
        ],
        "action_menu",
    )

    assert constant == {"weather_regime": "cold"}
    assert multiple == {}
    assert missing == {}


def test_cycles_rows_get_validation_metadata_and_axis_fallback(tmp_path: Path):
    generator = CyclesDatasetGenerator(
        {
            "dataset_id": "cycles_axis_fallback",
            "validation_axis": "price_regime",
            "sampling": {
                "splits": {
                    "val": {
                        "role": "validation",
                        "validation_set": "low_price",
                    },
                },
            },
            "val_configs": [
                {"env_id": "CropPlanning-RockSprings-RW-v0"},
            ],
        },
        str(tmp_path),
    )

    record = generator.build_split_records("val")[0]

    assert record["dataset_role"] == "validation"
    assert record["validation_set"] == "low_price"
    assert record["trajectory_group_labels"]["validation_set"] == "low_price"
    assert record["trajectory_group_labels"]["price_regime"] == "low_price"


def test_gym_dssat_rows_get_validation_metadata_and_axis_fallback(tmp_path: Path):
    generator = DSSATDatasetGenerator(
        {
            "dataset_id": "dssat_axis_fallback",
            "validation_axis": "schema_tuple",
            "sampling": {
                "splits": {
                    "val": {
                        "role": "validation",
                        "validation_set": "dssat_schema",
                    },
                },
            },
            "val_seeds": [1],
        },
        str(tmp_path),
    )

    record = generator.build_split_records("val")[0]

    assert record["dataset_role"] == "validation"
    assert record["validation_set"] == "dssat_schema"
    assert record["trajectory_group_labels"]["validation_set"] == "dssat_schema"
    assert record["trajectory_group_labels"]["schema_tuple"] == "dssat_schema"


def test_write_dataset_manifest_records_validation_sets(tmp_path: Path):
    config_path = tmp_path / "dataset.yaml"
    config_path.write_text("env_name: wofost_gym\ndataset_id: demo\n", encoding="utf-8")
    save_dir = tmp_path / "demo"
    save_dir.mkdir()

    manifest_path = write_dataset_manifest(
        save_dir,
        {
            "dataset_id": "demo",
            "env_name": "wofost_gym",
            "validation_axis": "weather_regime",
            "_config_path": str(config_path),
        },
        {"train": 2, "val_drought": 3},
        split_metadata={
            "train": normalize_split_metadata("train", {}),
            "val_drought": normalize_split_metadata(
                "val_drought",
                {"validation_set": "drought"},
            ),
        },
        split_env_configs={
            "train": [
                {"trajectory_group_labels": {"weather_regime": "id"}},
            ],
            "val_drought": [
                {"trajectory_group_labels": {"weather_regime": "drought"}},
                {"trajectory_group_labels": {"weather_regime": "drought"}},
            ],
        },
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["validation_axis"] == "weather_regime"
    assert manifest["splits"]["val_drought"]["num_rows"] == 3
    assert manifest["splits"]["val_drought"]["validation_set"] == "drought"
    assert manifest["splits"]["val_drought"]["labels"]["weather_regime"] == "drought"
    assert (
        manifest["validation_sets"]["drought"]["labels"]["weather_regime"]
        == "drought"
    )


def test_write_dataset_manifest_does_not_collapse_multivalue_axis(tmp_path: Path):
    manifest_path = write_dataset_manifest(
        tmp_path,
        {
            "dataset_id": "cross_crop",
            "env_name": "wofost_gym",
            "validation_axis": "crop",
        },
        {"val_heldout_crops": 2},
        split_metadata={
            "val_heldout_crops": normalize_split_metadata(
                "val_heldout_crops",
                {"validation_set": "heldout_crops"},
            ),
        },
        split_env_configs={
            "val_heldout_crops": [
                {"trajectory_group_labels": {"crop": "potato"}},
                {"trajectory_group_labels": {"crop": "rice"}},
            ],
        },
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["splits"]["val_heldout_crops"]["labels"] == {}
    assert manifest["validation_sets"]["heldout_crops"]["labels"] == {}
