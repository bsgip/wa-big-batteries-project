"""FCESS enablement: how much frequency control (FCESS) each battery was enabled to provide.

From fcess_schedule: per dispatch interval, the MW each facility was enabled for in each FCESS market
(regulation raise/lower, contingency raise/lower, RoCoF - the batteries are never enabled for RoCoF).
"""

from collections.abc import Mapping

import pandas as pd

from data_extraction.catalog import load

BATTERY_SERVICES = ["contingencyRaise", "contingencyLower", "regulationRaise", "regulationLower"]


def enabled_mw(codes: list[str]) -> pd.DataFrame:
    """Per interval, MW enabled in each of BATTERY_SERVICES, summed over the given batteries.

    A battery with no row in an interval counts as 0 MW enabled.
    """
    schedule = load("fcess_schedule", filters=[("code", "in", codes)])
    enabled = schedule.pivot_table(index=schedule.index, columns="market_service", values="quantity", aggfunc="sum")
    return enabled.reindex(columns=BATTERY_SERVICES).fillna(0)


def mean_by_window(df: pd.DataFrame | pd.Series, windows: Mapping[str, tuple[str, str]]) -> pd.DataFrame:
    """Average of df over the intervals in each window of the day (windows may wrap past midnight).

    windows: {label: (start, end)}, e.g. energy.DAY_WINDOWS - one row each, in that order
    """
    return pd.DataFrame({label: df.between_time(start, end, inclusive="left").mean()
                         for label, (start, end) in windows.items()}).T
