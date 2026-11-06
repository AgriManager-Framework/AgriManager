"""How strictly the text environments read the action inside <answer> tags.

``strict`` (default): an answer that names more amounts than the action takes,
such as ``Apply 5.0, 10.0, or 15.0 cm irrigation depth`` or ``Irrigate 5-20 cm``,
is an invalid action, and so is a no-action answer that also names a non-zero
amount.

``legacy``: the parsers as used for the paper's runs, which read the first amount
(or, for Cycles crop planning, the last integers) and ignore the rest.

Set it per scenario with the env-config key ``action_parser`` (for example
``+data.env_config_overrides.action_parser=legacy``), or for a whole process with
the environment variable ``AGRI_ACTION_PARSER``. The env-config key wins.
"""

from __future__ import annotations

import os
import re
from typing import Any, List

STRICT = "strict"
LEGACY = "legacy"
ACTION_PARSER_MODES = (STRICT, LEGACY)
ACTION_PARSER_ENV_VAR = "AGRI_ACTION_PARSER"

# A number that is not part of a word such as "P2O5" or "NH4".
_AMOUNT_RE = re.compile(r"(?<![A-Za-z0-9.])\d+(?:\.\d+)?")


def resolve_action_parser(config: Any = None) -> str:
    """Return the parser mode for an env config (key, then env var, then strict)."""
    value = getattr(config, "action_parser", None) if config is not None else None
    if isinstance(config, dict):
        value = config.get("action_parser")
    if value is None or str(value).strip() == "":
        value = os.environ.get(ACTION_PARSER_ENV_VAR, "")
    mode = str(value).strip().lower() or STRICT
    if mode not in ACTION_PARSER_MODES:
        raise ValueError(
            f"action_parser must be one of {ACTION_PARSER_MODES}, got {value!r}"
        )
    return mode


def answer_amounts(text: str) -> List[float]:
    """Distinct numeric amounts in an answer, in order of appearance."""
    seen: List[float] = []
    for match in _AMOUNT_RE.finditer(text or ""):
        value = float(match.group(0))
        if value not in seen:
            seen.append(value)
    return seen


def names_too_many_amounts(text: str, allowed: int = 1) -> bool:
    """True when the answer names more distinct amounts than the action takes."""
    return len(answer_amounts(text)) > allowed


def no_action_with_amount(text: str) -> bool:
    """True when a no-action answer also names a non-zero amount."""
    return any(value != 0.0 for value in answer_amounts(text))
