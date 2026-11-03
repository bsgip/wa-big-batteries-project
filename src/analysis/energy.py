from collections.abc import Mapping

import pandas as pd

from tools.constants import battery_capacity_MWh, battery_codes, codes_in, commissioned_capacity

INTERVAL_HOURS = 5 / 60

def window_energy_mwh(mw: pd.DataFrame | pd.Series, start: str, end: str) -> pd.DataFrame | pd.Series:
    """Daily energy (MWh) of a 5-min MW series between start and end."""
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

    Equivalent full cycles = energy discharged (MWh, positive power only) / rated capacity (MWh),
    so 1.0 is one full battery's worth of discharge. Averages, not medians, so a battery's
    windows add up to its average cycles per day.

    power: battery power (MW), one column per battery
    capacity: {code: rated MWh}
    first_days: {code: first day to count} - e.g. commissioning date, or the analysis start if later
    windows: {label: (start, end)} (default: DAY_WINDOWS)

    Returns one row per battery, one column per window.
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
    """Daily fleet net energy (MWh) between start and end, summed over the codes columns.

    codes: units to sum (default: the batteries in power)
    """
    if codes is None:
        codes = codes_in(power, battery_codes)
    daily_energy = window_energy_mwh(power[codes], start, end)
    return daily_energy.sum(axis=1, min_count=1).rename("fleet_net_energy_mwh")


def fleet_required_energy_mwh(rcoq: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Daily energy (MWh) the fleet must be able to discharge between start and end to meet its RCOQ.

    rcoq: load_clean("rcoq"), one MW column per facility
    """
    return fleet_net_energy_mwh(rcoq, start, end).rename("fleet_required_energy_mwh")


def credit_equivalent_soc_mwh(
    rcoq: pd.DataFrame, start: str, end: str, codes: list[str] | None = None
) -> pd.DataFrame | pd.Series:
    """Daily charge (MWh) each unit must hold at start to deliver its RCOQ through to end.

    RCOQ (MW) integrated over the window, i.e. RCOQ x 4h for ESROI. 0 on days the
    unit has no obligation.

    rcoq: load_clean("rcoq"), one MW column per facility
    codes: units to keep (default: the batteries in rcoq)
    """
    if codes is None:
        codes = codes_in(rcoq, battery_codes)
    return window_energy_mwh(rcoq[codes], start, end)


def typical_credit_equivalent_mwh(credit_soc: pd.DataFrame, start: str, end: str) -> pd.Series:
    """One credit-equivalent SOC (MWh) per unit: its most common non-zero daily value between start and end.

    Reduced-obligation days (outages, changes partway through the window) are
    rare, so the most common value is the unit's usual level and leaves them out.
    NaN for a unit with no obligation in the period.

    credit_soc: from credit_equivalent_soc_mwh
    start, end: dates bounding the period, e.g. one capacity year
    """
    period = credit_soc.loc[start:end]
    period = period.where(period > 0).round(1)  # round so float noise doesn't split one level in two
    return period.mode().iloc[0].rename("credit_equivalent_mwh")


def monthly_share_below(values: pd.DataFrame, threshold: float | pd.Series) -> pd.DataFrame:
    """Per month, % of days each unit's value is below threshold, plus "all":
    the same % pooled over every unit-day that month.

    values: one daily value per unit, e.g. SOC (MWh or %) at the window start
    threshold: one number for every unit (e.g. 50 for SOC %), or one per unit
        (e.g. credit-equivalent SOC in MWh, from typical_credit_equivalent_mwh) -
        units with a NaN threshold are left out

    Only days with a reading count. "all" moves when units join or leave
    as well as when units change, so read it alongside the per-unit columns.
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
    """Daily fleet net energy between start and end, as % of commissioned fleet capacity (MWh).

    commissioning: from constants.commissioning_dates(...)
    codes: units to sum (default: the batteries in power)
    capacity: {code: MWh} (default: battery_capacity_MWh)
    """
    if codes is None:
        codes = codes_in(power, battery_codes)
    net_energy = fleet_net_energy_mwh(power, start, end, codes)

    dates = {code: commissioning[code] for code in codes if code in commissioning}
    commissioned = commissioned_capacity(net_energy.index, dates, capacity)

    return (net_energy / commissioned * 100).rename("fleet_net_energy_pct")


def fleet_soc_drop_pct(soc: pd.DataFrame, start: str, end: str, codes: list[str] | None = None) -> pd.Series:
    """Daily fleet SOC (%) at start minus at end. Positive = SOC used.

    NaN on days the two times aren't measured over the same fleet:
    - the batteries reporting differ: a battery dropping out between the two
      times would read as its whole charge being used (or gained), or
    - the capacity counted differs (fleet_capacity_counted_MWh): a battery with
      no SOC reading counted as offline at one time and as a data gap at the
      other changes the denominator, not the charge.

    codes: the fleet's batteries (default: the batteries in soc)
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
