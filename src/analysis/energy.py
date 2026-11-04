from collections.abc import Mapping

import pandas as pd

from tools.constants import battery_capacity_MWh, battery_codes, codes_in, commissioned_capacity

INTERVAL_HOURS = 5 / 60

def window_energy_mwh(mw: pd.DataFrame | pd.Series, start: str, end: str) -> pd.DataFrame | pd.Series:
    """Daily energy between two times of day, from 5-min MW.

    Args:
        mw: 5-min power (MW), Series or one column per unit.
        start: Start time, e.g. "17:30" (inclusive).
        end: End time (exclusive).

    Returns:
        Daily MWh, same shape as mw; NaN for days with no data.
    """
    mwh = (
        (mw.between_time(start, end, inclusive="left") * INTERVAL_HOURS)
        .resample("D").sum(min_count=1)
    )

    return mwh


def daily_energy_mwh(mw: pd.Series) -> pd.Series:
    """Daily energy (MWh) of a 5-min MW series over the whole day; NaN for days with no data."""
    return (mw * INTERVAL_HOURS).resample("D").sum(min_count=1)


# parts of the day for splitting a battery's discharge, in order; overnight wraps past midnight
DAY_WINDOWS = {
    "Overnight (22:00-05:00)": ("22:00", "05:00"),
    "Morning (05:00-09:00)": ("05:00", "09:00"),
    "Midday (09:00-17:00)": ("09:00", "17:00"),
    "Evening (17:00-22:00)": ("17:00", "22:00"),
}


def discharge_cycles_by_window(
    power: pd.DataFrame,
    capacity: Mapping[str, float],
    first_days: Mapping[str, pd.Timestamp],
    windows: Mapping[str, tuple[str, str]] = DAY_WINDOWS,
) -> pd.DataFrame:
    """Average equivalent full cycles per day each battery discharges in each window of the day.

    A cycle is energy discharged (MWh, positive power only) divided by rated capacity (MWh).
    Averages, so a battery's windows add up to its cycles per day.

    Args:
        power: Battery power (MW), one column per battery.
        capacity: {code: rated MWh}.
        first_days: {code: first day to count}, e.g. commissioning date or analysis start.
        windows: {label: (start, end)}; defaults to DAY_WINDOWS.

    Returns:
        One row per battery, one column per window.
    """
    rows = {}
    for code, first in first_days.items():
        discharging = power[code].clip(lower=0)
        days = discharging.resample("D").count()
        days = days.index[(days > 0) & (days.index >= first.normalize())]  # days with power readings

        row = {}
        for label, (start, end) in windows.items():
            # a day with readings but no discharge in this window counts as 0, not missing
            daily = window_energy_mwh(discharging, start, end).reindex(days).fillna(0)
            row[label] = daily.mean() / capacity[code]
        rows[code] = row
    return pd.DataFrame(rows).T


def fleet_net_energy_mwh(power: pd.DataFrame, start: str, end: str, codes: list[str] | None = None) -> pd.Series:
    """Daily fleet net energy between two times of day.

    Args:
        power: Power (MW), one column per unit (+ discharging, - charging).
        start: Start time, e.g. "09:00".
        end: End time (exclusive).
        codes: Units to sum; defaults to the batteries in power.

    Returns:
        Daily MWh, named fleet_net_energy_mwh.
    """
    if codes is None:
        codes = codes_in(power, battery_codes)
    daily_energy = window_energy_mwh(power[codes], start, end)
    return daily_energy.sum(axis=1, min_count=1).rename("fleet_net_energy_mwh")


def fleet_required_energy_mwh(rcoq: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Daily energy the fleet must be able to discharge between two times to meet its RCOQ.

    Args:
        rcoq: RCOQ (MW), one column per facility, from load_clean("rcoq").
        start: Start time, e.g. "17:30".
        end: End time (exclusive).

    Returns:
        Daily MWh, named fleet_required_energy_mwh.
    """
    return fleet_net_energy_mwh(rcoq, start, end).rename("fleet_required_energy_mwh")


def credit_equivalent_soc_mwh(
    rcoq: pd.DataFrame, start: str, end: str, codes: list[str] | None = None
) -> pd.DataFrame | pd.Series:
    """Daily charge each unit needs at start to deliver its RCOQ until end (RCOQ x window hours).

    Args:
        rcoq: RCOQ (MW), one column per facility, from load_clean("rcoq").
        start: Start time, e.g. "17:30".
        end: End time (exclusive).
        codes: Units to keep; defaults to the batteries in rcoq.

    Returns:
        Daily MWh, one column per unit; 0 on days with no obligation.
    """
    if codes is None:
        codes = codes_in(rcoq, battery_codes)
    return window_energy_mwh(rcoq[codes], start, end)


def typical_credit_equivalent_mwh(credit_soc: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Each unit's most common non-zero credit-equivalent SOC over a period.

    Args:
        credit_soc: Daily values from credit_equivalent_soc_mwh.
        start: First date of the period, e.g. the start of a capacity year.
        end: Last date of the period.

    Returns:
        MWh per unit, named credit_equivalent_mwh; NaN for a unit with no obligation.
    """
    period = credit_soc.loc[start:end]
    period = period.where(period > 0).round(1)  # round so float noise doesn't split one level in two
    return period.mode().iloc[0].rename("credit_equivalent_mwh")


def monthly_share_below(values: pd.DataFrame, threshold: float | pd.Series) -> pd.DataFrame:
    """Per month, the % of days each unit's value is below a threshold.

    Only days with a reading count.

    Args:
        values: One daily value per unit, e.g. SOC (MWh or %) at the window start.
        threshold: One number for every unit, or a Series with one per unit; units with a NaN
            threshold are left out.

    Returns:
        One row per month, one % column per unit plus "all" (pooled over every unit-day).
    """
    if not isinstance(threshold, pd.Series):
        threshold = pd.Series(threshold, index=values.columns)
    threshold = threshold.dropna()
    soc = values[threshold.index]
    below = (soc < threshold).astype(float).where(soc.notna())  # 1 below, 0 not, NaN no reading
    n_below = below.resample("MS").sum()
    n_days = below.resample("MS").count()

    share = n_below / n_days.where(n_days > 0) * 100
    share["all"] = n_below.sum(axis=1) / n_days.sum(axis=1) * 100
    return share


def fleet_net_energy_pct(
    power: pd.DataFrame,
    commissioning: dict[str, pd.Timestamp],
    start: str,
    end: str,
    codes: list[str] | None = None,
    capacity: Mapping[str, float] = battery_capacity_MWh,
) -> pd.Series:
    """Daily fleet net energy between two times, as % of commissioned fleet capacity.

    Args:
        power: Battery power (MW), one column per battery.
        commissioning: {code: commissioning timestamp}, from constants.commissioning_dates.
        start: Start time, e.g. "09:00".
        end: End time (exclusive).
        codes: Units to sum; defaults to the batteries in power.
        capacity: {code: MWh}; defaults to battery_capacity_MWh.

    Returns:
        Daily %, named fleet_net_energy_pct.
    """
    if codes is None:
        codes = codes_in(power, battery_codes)
    net_energy = fleet_net_energy_mwh(power, start, end, codes)

    dates = {code: commissioning[code] for code in codes if code in commissioning}
    commissioned = commissioned_capacity(net_energy.index, dates, capacity)

    return (net_energy / commissioned * 100).rename("fleet_net_energy_pct")


def fleet_soc_drop_pct(soc: pd.DataFrame, start: str, end: str, codes: list[str] | None = None) -> pd.Series:
    """Daily fleet SOC at start minus fleet SOC at end.

    NaN on days where the batteries reporting, or the capacity counted, differ between the two times.

    Args:
        soc: Clean SOC frame (with fleet columns), from load_clean("soc").
        start: First time, e.g. "17:30".
        end: Second time, e.g. "21:30".
        codes: The fleet's batteries; defaults to the batteries in soc.

    Returns:
        Daily % (positive = SOC used), named fleet_soc_drop_pct.
    """
    if codes is None:
        codes = codes_in(soc, battery_codes)

    soc_start = soc["fleet_soc_pct"].at_time(start)
    soc_end = soc["fleet_soc_pct"].at_time(end)
    reporting_start = soc[codes].at_time(start).notna()
    reporting_end = soc[codes].at_time(end).notna()
    counted_start = soc["fleet_capacity_counted_MWh"].at_time(start)
    counted_end = soc["fleet_capacity_counted_MWh"].at_time(end)

    # index by day so the two times line up
    for s in (soc_start, soc_end, reporting_start, reporting_end, counted_start, counted_end):
        s.index = s.index.normalize()

    same_fleet = (reporting_start == reporting_end).all(axis=1) & (counted_start == counted_end)
    return (soc_start - soc_end).where(same_fleet).rename("fleet_soc_drop_pct")


def low_soc_days_per_battery(soc: pd.DataFrame, at: str, threshold: float) -> pd.Series:
    """Count the days each battery's SOC at a time of day was at or below a threshold.

    Args:
        soc: Clean SOC frame with <code>_soc_pct columns.
        at: Time of day, e.g. "17:30".
        threshold: SOC (%).

    Returns:
        Day count per battery code, highest first.
    """
    pct = soc[[f"{code}_soc_pct" for code in battery_codes]].at_time(at)
    counts = (pct <= threshold).sum()
    counts.index = counts.index.str.removesuffix("_soc_pct")
    return counts.sort_values(ascending=False)
