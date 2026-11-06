"""Strict (default) and legacy action parsing in the text environments."""

from __future__ import annotations

import numpy as np
import pytest

from agrimanager.env.base.action_parser import (
    ACTION_PARSER_ENV_VAR,
    answer_amounts,
    resolve_action_parser,
)
from agrimanager.env.cycles_gym.prompt import CornPromptGenerator, CropPlanningPromptGenerator
from agrimanager.env.gym_dssat.prompt import DSSATPromptGenerator
from agrimanager.env.wofost_gym.prompt import WOFOSTPromptGenerator


def _wofost(mode: str) -> WOFOSTPromptGenerator:
    generator = WOFOSTPromptGenerator(
        num_fert=4,
        num_irrig=4,
        fert_amount=20.0,
        irrig_amount=5.0,
        intervention_interval=10,
        output_vars=["DAYS", "DVS"],
    )
    generator.action_parser = mode
    return generator


def _dssat(mode: str) -> DSSATPromptGenerator:
    generator = DSSATPromptGenerator(crop_name="maize")
    generator.action_parser = mode
    return generator


def _corn(mode: str) -> CornPromptGenerator:
    generator = CornPromptGenerator(n_actions=11, max_n=150.0)
    generator.action_parser = mode
    return generator


def _planning(mode: str) -> CropPlanningPromptGenerator:
    generator = CropPlanningPromptGenerator(
        rotation_crops=["CornRM.100", "SoybeanMG.3"], action_nvec=[2, 14]
    )
    generator.action_parser = mode
    return generator


# --------------------------------------------------------------------------
# Mode resolution
# --------------------------------------------------------------------------


def test_default_is_strict(monkeypatch):
    monkeypatch.delenv(ACTION_PARSER_ENV_VAR, raising=False)
    assert resolve_action_parser(None) == "strict"
    assert resolve_action_parser({"env_id": "lnpkw-v0"}) == "strict"


def test_config_key_wins_over_env_var(monkeypatch):
    monkeypatch.setenv(ACTION_PARSER_ENV_VAR, "legacy")
    assert resolve_action_parser({}) == "legacy"
    assert resolve_action_parser({"action_parser": "strict"}) == "strict"

    class Config:
        action_parser = "LEGACY"

    monkeypatch.delenv(ACTION_PARSER_ENV_VAR)
    assert resolve_action_parser(Config()) == "legacy"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        resolve_action_parser({"action_parser": "lenient"})


def test_amounts_ignore_chemical_formulas_and_repeats():
    assert answer_amounts("apply 20 kg/ha p2o5 (20.0 kg)") == [20.0]
    assert answer_amounts("irrigate 5-20 cm") == [5.0, 20.0]


# --------------------------------------------------------------------------
# WOFOST
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["strict", "legacy"])
def test_wofost_single_amount_is_valid_in_both_modes(mode):
    generator = _wofost(mode)
    assert generator.parse_action_response("<answer>Apply 80 kg/ha nitrogen fertilizer.</answer>") == 4
    assert generator.parse_action_response("<answer>Irrigate with 5.0 cm of water.</answer>") is not None
    assert generator.parse_action_response("<answer>Take no action.</answer>") == 0


def test_wofost_candidate_list_is_invalid_in_strict_and_first_amount_in_legacy():
    menu_line = "<answer>Apply 5.0, 10.0, or 15.0 cm irrigation depth.</answer>"
    assert _wofost("strict").parse_action_response(menu_line) is None
    assert _wofost("legacy").parse_action_response(menu_line) == _wofost("legacy").parse_action_response(
        "<answer>Irrigate with 5.0 cm of water.</answer>"
    )


def test_wofost_range_in_answer_is_invalid_in_strict_only():
    answer = "<answer>Irrigate 5-20 cm of water.</answer>"
    assert _wofost("strict").parse_action_response(answer) is None
    assert _wofost("legacy").parse_action_response(answer) == _wofost("legacy").parse_action_response(
        "<answer>Irrigate with 5.0 cm of water.</answer>"
    )


def test_wofost_menu_ranges_in_the_prompt_are_not_read_as_the_answer():
    generator = _wofost("strict")
    prompt = generator.get_turn_prompt(np.array([1.0, 0.0]))
    # The action menu itself lists several amounts per action ...
    assert "Irrigate with 5.0, 10.0, 15.0, or 20.0 cm of water" in prompt
    # ... but only the answer block is parsed, so a single choice stays valid,
    for mode in ("strict", "legacy"):
        assert _wofost(mode).parse_action_response(
            "<answer>Irrigate with 10.0 cm of water.</answer>"
        ) == _wofost(mode).parse_action_response("<answer>Irrigate with 10 cm of water.</answer>")
    # and menu text outside the answer tags makes the response invalid in both modes.
    echoed = "Irrigate with 5.0, 10.0, 15.0, or 20.0 cm of water. <answer>Irrigate with 10.0 cm of water.</answer>"
    assert _wofost("strict").parse_action_response(echoed) is None
    assert _wofost("legacy").parse_action_response(echoed) is None


def test_wofost_no_action_with_an_amount_is_invalid_in_strict():
    answer = "<answer>Do nothing or apply 20 kg/ha nitrogen fertilizer.</answer>"
    assert _wofost("strict").parse_action_response(answer) is None
    assert _wofost("legacy").parse_action_response(answer) == 0


# --------------------------------------------------------------------------
# DSSAT
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["strict", "legacy"])
def test_dssat_single_and_combined_actions_are_valid(mode):
    generator = _dssat(mode)
    assert generator.parse_action_response("<answer>Apply 20 kg/ha of nitrogen fertilizer.</answer>") == 2
    combined = generator.parse_action_response(
        "<answer>Apply 20 kg/ha nitrogen fertilizer and irrigate 10 mm of water.</answer>"
    )
    assert combined == {"anfer": 20.0, "amir": 10.0}


def test_dssat_candidate_list_is_invalid_in_strict_only():
    answer = "<answer>Apply 10, 20, or 40 kg/ha nitrogen fertilizer.</answer>"
    assert _dssat("strict").parse_action_response(answer) is None
    assert _dssat("legacy").parse_action_response(answer) is not None


# --------------------------------------------------------------------------
# Cycles
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["strict", "legacy"])
def test_cycles_corn_single_amount_is_valid(mode):
    assert _corn(mode).parse_action_response("<answer>Apply 30 kg/ha nitrogen.</answer>") == 2


def test_cycles_corn_candidate_list_is_invalid_in_strict_only():
    answer = "<answer>Apply 30, 45, or 60 kg/ha nitrogen.</answer>"
    assert _corn("strict").parse_action_response(answer) is None
    # Legacy reads the amount written next to the unit (60 kg/ha -> level 4).
    assert _corn("legacy").parse_action_response(answer) == 4


@pytest.mark.parametrize("mode", ["strict", "legacy"])
def test_cycles_crop_planning_single_choice_is_valid(mode):
    assert _planning(mode).parse_action_response("<answer>1, 4</answer>") == [1, 4]


def test_cycles_crop_planning_extra_options_are_invalid_in_strict_only():
    answer = "<answer>0, 3 or 1, 4</answer>"
    assert _planning("strict").parse_action_response(answer) is None
    assert _planning("legacy").parse_action_response(answer) == [1, 4]


# --------------------------------------------------------------------------
# Environment wiring
# --------------------------------------------------------------------------


def test_eval_entry_point_pins_legacy_for_paper_scoring():
    from agrimanager.checkpoint.evaluate import Reasoning

    reasoning = Reasoning(require_think=False, thinking_mode=None, think_tag=None, source="test")
    assert reasoning.overrides()["action_parser"] == "legacy"
    assert reasoning.assertions()["action_parser"] == "legacy"
