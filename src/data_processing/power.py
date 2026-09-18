"""Derived columns and range checks for battery power (the extracted
initialMw field, in MW - positive discharging, negative charging).

No sentinel masking here: unlike SOC's chargeLevel 999 (see soc.py), a
sample of initialMw across the dispatchSolution corpus found no equivalent
sentinel value - a full-corpus check confirmed it, the worst overshoot being
KWINANA_ESR1 at 1.9% over its rating.
"""

import logging

import pandas as pd

from tools.constants import battery_capacity_MW, battery_codes

logger = logging.getLogger(__name__)


def add_power_pct_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add a <code>_power_pct column for each battery, computed from its
    power column (MW) and rated capacity in battery_capacity_MW."""
    for code in battery_codes:
        if code not in df:
            logger.warning(f"{code}: no power column found, skipping power_pct")
            continue
        df[f"{code}_power_pct"] = df[code] / battery_capacity_MW[code] * 100

    return df


def clean_power_overshoot_df(df: pd.DataFrame, capacity: dict[str, float] | None = None) -> pd.DataFrame:
    """Clip each battery's power to +/- its rated capacity in MW. The clip is
    symmetric because initialMw is signed - positive discharging, negative
    charging - so both directions are capped at the rating."""
    rated_capacity = battery_capacity_MW if capacity is None else capacity
    df = df.copy()

    for code in battery_codes:
        if code not in df:
            continue
        rated = rated_capacity[code]
        n_clipped = df[code].abs().gt(rated).sum()
        if n_clipped:
            logger.info(f"{code}: clipping {n_clipped} readings above rated capacity ({rated} MW)")
        df[code] = df[code].clip(lower=-rated, upper=rated)

    return df


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Extracted power (MW) -> analysis-ready: readings clipped to rated
    capacity, plus a <code>_power_pct column per battery alongside the bare
    <code> MW columns. Clipping runs first so the pct columns are derived
    from the clipped values; it copies, so add_power_pct_columns writing into
    the frame it's given doesn't touch the caller's."""
    df = clean_power_overshoot_df(df)
    return add_power_pct_columns(df)
