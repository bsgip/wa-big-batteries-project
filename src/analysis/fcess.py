"""FCESS enablement: the MW each battery was enabled for in each frequency control market."""

from collections.abc import Mapping

import pandas as pd

from analysis.price_setting import OTHER_FACILITIES
from tools.io import load_clean

BATTERY_SERVICES = ["contingencyRaise", "contingencyLower", "regulationRaise", "regulationLower"]


def enabled_mw(codes: list[str]) -> pd.DataFrame:
    """MW enabled in each FCESS service, summed over batteries.

    Args:
        codes: Battery codes to sum.

    Returns:
        One row per interval, one column per BATTERY_SERVICES entry; 0 where a battery has no row.
    """
    schedule = load_clean("fcess_schedule", start_date=None, filters=[("code", "in", codes)])
    enabled = schedule.pivot_table(index=schedule.index, columns="market_service", values="quantity", aggfunc="sum")
    return enabled.reindex(columns=BATTERY_SERVICES).fillna(0)


def monthly_enabled_shares(codes: list[str]) -> dict[str, pd.DataFrame]:
    """Each month, the share of each FCESS service's enabled MW that each battery provided.

    Shares are of the MW summed over every interval in the month, across every facility enabled
    for FCESS (not only batteries).

    Args:
        codes: Battery codes, one column each.

    Returns:
        {service: frame}, one per BATTERY_SERVICES entry. Each frame has one row per month and %
        columns for codes and OTHER_FACILITIES (rows add up to 100).
    """
    all_codes = load_clean("fcess_schedule", start_date=None, columns=["code"])["code"].unique().tolist()
    total = enabled_mw(all_codes).resample("MS").sum()
    per_battery = {code: enabled_mw([code]).resample("MS").sum().reindex(total.index, fill_value=0) for code in codes}

    shares = {}
    for service in BATTERY_SERVICES:
        mw = pd.DataFrame({code: per_battery[code][service] for code in codes})
        mw[OTHER_FACILITIES] = total[service] - mw.sum(axis=1)
        shares[service] = mw.div(total[service], axis=0) * 100
    return shares


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
