"""Derived columns and range checks for battery power (the extracted
initialMw field, in MW - positive discharging, negative charging).

No sentinel masking here: unlike SOC's chargeLevel 999 (see soc.py), a
sample of initialMw across the dispatchSolution corpus found no equivalent
sentinel value. flag_out_of_range is the full-corpus check on that.
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


def flag_out_of_range(df: pd.DataFrame, capacity: dict[str, float], tolerance: float = 1.05) -> None:
    """Log (don't drop) any values whose magnitude exceeds a battery's rated
    capacity by more than `tolerance`. Used to check a full extraction for
    sentinel-like values (e.g. the chargeLevel 999 case) without silently
    dropping anything - a human should look at what's flagged."""
    for code in battery_codes:
        if code not in df:
            continue
        limit = capacity[code] * tolerance
        n_out = df[code].abs().gt(limit).sum()
        if n_out:
            logger.warning(f"{code}: {n_out} values exceed {limit:.1f} (rated {capacity[code]})")


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Extracted power (MW) -> analysis-ready: a <code>_power_pct column per
    battery, alongside the bare <code> MW columns. Copies first because
    add_power_pct_columns writes into the frame it's given."""
    df = df.copy()
    df = add_power_pct_columns(df)
    flag_out_of_range(df, battery_capacity_MW)
    return df
