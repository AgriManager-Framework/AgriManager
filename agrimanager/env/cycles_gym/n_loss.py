"""Nitrogen-loss accounting for the pollution-penalized CycleGym objective.

CycleGym runs write a daily ``N.dat`` table (``env.soil_n_manager.soil_n_state``)
whose flux columns are in kg N/ha per day. The crop-planning prompt already
shows several of them under the "Nitrogen losses" observation group; the
default T1.3 objective (cumulative gross revenue) does not score them.

``objective_id: pollution_penalized_revenue`` keeps the native CycleGym reward
(gross revenue ``yield x crop price`` on crop planning) and subtracts, per
environment step, a weighted sum of the nitrogen lost during the days that step
simulated. On crop planning one step is one season, so the per-step penalty is
a per-season penalty and the episode return sums it over the rotation horizon,
the same way gross revenue is summed.
"""

from __future__ import annotations

import datetime
import logging
import math
from typing import Any, Dict, Mapping, Tuple

POLLUTION_OBJECTIVE_ID = "pollution_penalized_revenue"

# Penalty terms and the CycleGym/Cycles ``N.dat`` columns each one sums: the
# four loss pathways named for this objective (nitrate and ammonium leaching,
# ammonia volatilization, N2O emission). The bypass columns are nitrogen leached
# through preferential (bypass) flow, so they count as leaching unless
# ``n_loss_include_bypass`` is false. ``NO3 DENITRIF`` is left out on purpose;
# the N2O reported for denitrification is already in ``n2o_emission``.
N_LOSS_TERMS: Dict[str, Tuple[str, ...]] = {
    "no3_leaching": ("NO3 LEACHING",),
    "nh4_leaching": ("NH4 LEACHING",),
    "nh3_volatilization": ("NH3 VOLATILIZ",),
    "n2o_emission": ("N2O FROM NITRIF", "N2O FROM DENIT"),
}
N_LOSS_BYPASS_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "no3_leaching": ("NO3 BYPASS",),
    "nh4_leaching": ("NH4 BYPASS",),
}

# Short metric suffixes; long metric keys have crashed downstream loggers.
N_LOSS_METRIC_SUFFIX: Dict[str, str] = {
    "no3_leaching": "no3_leach",
    "nh4_leaching": "nh4_leach",
    "nh3_volatilization": "nh3_volat",
    "n2o_emission": "n2o",
}

# Default weights in US dollars per kg N lost per hectare, the unit of the
# native crop-planning reward (dollars per hectare). They are illustrative
# placeholders chosen so the penalty is visible next to a gross revenue of
# roughly 1,000-2,000 $/ha/season; they are not damage-cost estimates and no
# source is claimed for them. Users should set their own weights in
# ``reward_params``; when a default is used, one info line is logged per process.
DEFAULT_N_LOSS_COSTS_USD_PER_KG_N: Dict[str, float] = {
    "no3_leaching": 10.0,
    "nh4_leaching": 10.0,
    "nh3_volatilization": 10.0,
    "n2o_emission": 10.0,
}

N_LOSS_COST_PARAM_NAMES: Dict[str, Tuple[str, ...]] = {
    term: (f"n_loss_cost_{term}_usd_per_kg_n", f"n_loss_cost_{term}")
    for term in N_LOSS_TERMS
}


def _finite_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def n_loss_penalty_params(params: Mapping[str, Any] | None = None) -> Dict[str, Any]:
    """Normalize the pollution-penalty weights from ``reward_params``.

    Recognized keys (all optional):

    - ``n_loss_cost_<term>_usd_per_kg_n`` (or ``n_loss_cost_<term>``) for each
      term in ``N_LOSS_TERMS``; non-negative, dollars per kg N per hectare.
    - ``n_loss_cost_usd_per_kg_n``: one weight for every term that has no
      term-specific key.
    - ``n_loss_include_bypass``: count bypass-flow leaching (default true).
    """

    params = dict(params or {})
    shared = params.get("n_loss_cost_usd_per_kg_n")
    costs: Dict[str, float] = {}
    defaulted = []
    for term, names in N_LOSS_COST_PARAM_NAMES.items():
        default = DEFAULT_N_LOSS_COSTS_USD_PER_KG_N[term]
        if shared is not None:
            default = _finite_float(shared, default)
        value = default
        explicit = shared is not None
        for name in names:
            if name in params and params[name] is not None:
                value = _finite_float(params[name], default)
                explicit = True
                break
        if not explicit:
            defaulted.append(term)
        if value < 0.0:
            raise ValueError(f"N-loss cost for {term} must be non-negative, got {value}")
        costs[term] = float(value)
    return {
        "costs": costs,
        "include_bypass": _as_bool(params.get("n_loss_include_bypass"), True),
        "defaulted_terms": defaulted,
    }


_DEFAULT_WEIGHT_NOTICE_SHOWN = False


def log_default_weights_once(n_loss_params: Mapping[str, Any]) -> None:
    """Log, once per process, which terms fell back to the illustrative default weight."""

    global _DEFAULT_WEIGHT_NOTICE_SHOWN
    terms = list(n_loss_params.get("defaulted_terms") or [])
    if not terms or _DEFAULT_WEIGHT_NOTICE_SHOWN:
        return
    _DEFAULT_WEIGHT_NOTICE_SHOWN = True
    logging.getLogger(__name__).info(
        "pollution_penalized_revenue: using the illustrative default N-loss weight "
        "(%s $/kg N, no cited source) for %s; set n_loss_cost_<term>_usd_per_kg_n in "
        "reward_params to use your own.",
        ", ".join(f"{DEFAULT_N_LOSS_COSTS_USD_PER_KG_N[t]:g}" for t in terms),
        ", ".join(terms),
    )


def n_loss_columns(include_bypass: bool = True) -> Dict[str, Tuple[str, ...]]:
    """Return the ``N.dat`` columns summed by each penalty term."""

    columns = {term: tuple(cols) for term, cols in N_LOSS_TERMS.items()}
    if include_bypass:
        for term, extra in N_LOSS_BYPASS_COLUMNS.items():
            columns[term] = columns[term] + tuple(extra)
    return columns


def _ordinal_days(soil_n_df: Any) -> list[int]:
    years = soil_n_df["YEAR"].astype(int).tolist()
    doys = soil_n_df["DOY"].astype(int).tolist()
    return [datetime.date(y, 1, 1).toordinal() + d - 1 for y, d in zip(years, doys)]


def n_loss_totals(
    soil_n_df: Any,
    start_date: datetime.date | None,
    end_date: datetime.date | None,
    *,
    include_bypass: bool = True,
) -> Dict[str, float]:
    """Sum daily N losses (kg N/ha) over ``start_date <= day < end_date``.

    The window is half open so that consecutive steps tile the simulation
    without counting a day twice: an action taken on ``start_date`` changes
    that day and later days, never earlier ones. Missing columns count as zero.
    """

    totals = {term: 0.0 for term in N_LOSS_TERMS}
    if soil_n_df is None or start_date is None or end_date is None:
        return totals
    try:
        if len(soil_n_df) == 0 or "YEAR" not in soil_n_df or "DOY" not in soil_n_df:
            return totals
    except TypeError:
        return totals

    start = start_date.toordinal()
    end = end_date.toordinal()
    in_window = [start <= day < end for day in _ordinal_days(soil_n_df)]
    if not any(in_window):
        return totals
    window = soil_n_df.loc[in_window]
    for term, columns in n_loss_columns(include_bypass).items():
        total = 0.0
        for column in columns:
            if column in window:
                total += float(window[column].astype(float).sum())
        totals[term] = total
    return totals


def n_loss_penalty(
    totals_kg_ha: Mapping[str, float],
    costs_usd_per_kg_n: Mapping[str, float],
) -> Tuple[float, Dict[str, float]]:
    """Return the total penalty and its per-term parts, in $/ha."""

    parts = {
        term: float(costs_usd_per_kg_n.get(term, 0.0)) * float(totals_kg_ha.get(term, 0.0))
        for term in N_LOSS_TERMS
    }
    return float(sum(parts.values())), parts


__all__ = [
    "DEFAULT_N_LOSS_COSTS_USD_PER_KG_N",
    "N_LOSS_METRIC_SUFFIX",
    "N_LOSS_TERMS",
    "POLLUTION_OBJECTIVE_ID",
    "n_loss_columns",
    "n_loss_penalty",
    "n_loss_penalty_params",
    "log_default_weights_once",
    "n_loss_totals",
]
