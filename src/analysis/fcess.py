"""FCESS enablement: the MW each battery was enabled for in each frequency control market."""

from collections.abc import Mapping

import pandas as pd

from data_extraction.catalog import load

BATTERY_SERVICES = ["contingencyRaise", "contingencyLower", "regulationRaise", "regulationLower"]


def enabled_mw(codes: list[str]) -> pd.DataFrame:
    """MW enabled in each FCESS service, summed over batteries.

    Args:
        codes: Battery codes to sum.

    Returns:
        One row per interval, one column per BATTERY_SERVICES entry; 0 where a battery has no row.
    """
    schedule = load("fcess_schedule", filters=[("code", "in", codes)])
    enabled = schedule.pivot_table(index=schedule.index, columns="market_service", values="quantity", aggfunc="sum")
    return enabled.reindex(columns=BATTERY_SERVICES).fillna(0)


def mean_by_window(df: pd.DataFrame | pd.Series, windows: Mapping[str, tuple[str, str]]) -> pd.DataFrame:
    """Average of df over each window of the day (windows may wrap past midnight).

    Args:
        df: 5-min Series or frame.
        windows: {label: (start, end)}, e.g. energy.DAY_WINDOWS.

    Returns:
        One row per window, in the order given.
    """
    return pd.DataFrame({label: df.between_time(start, end, inclusive="left").mean()
                         for label, (start, end) in windows.items()}).T
