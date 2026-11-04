"""Battery power (MW, + discharging, - charging): clipping and % columns. Non-battery columns pass through.
No sentinel value was found in power (unlike SOC's 999), so none is masked.
"""

import logging

import pandas as pd

from tools.constants import battery_capacity_MW, battery_codes

logger = logging.getLogger(__name__)


def add_power_pct_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add a <code>_power_pct column per battery: power as % of its rated MW."""
    for code in battery_codes:
        if code not in df:
            logger.warning(f"{code}: no power column found, skipping power_pct")
            continue
        df[f"{code}_power_pct"] = df[code] / battery_capacity_MW[code] * 100

    return df


def clean_power_overshoot_df(df: pd.DataFrame, capacity: dict[str, float] | None = None) -> pd.DataFrame:
    """Clip each battery's power to ± its rated MW.

    Args:
        df: Power (MW), one column per facility.
        capacity: {code: rated MW}; defaults to battery_capacity_MW.

    Returns:
        A clipped copy of df.
    """
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
    """Clean extracted power.

    Args:
        df: Extracted power (MW, + discharging, - charging), one column per facility.

    Returns:
        df with battery columns clipped to rated MW, plus a <code>_power_pct column per battery.
    """
    df = clean_power_overshoot_df(df)
    return add_power_pct_columns(df)
