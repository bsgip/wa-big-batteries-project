"""Cleaning and derived columns for battery SOC (chargeLevel, MWh). process() runs every step.

After cleaning, a missing reading (NaN) means the battery's charge is unknown, and 0 means the
battery really is empty. Each 5-min reading is judged on its own:

    1. Missing, or the 999 sentinel              -> unknown (NaN)
    2. 0, and the battery offered nothing        -> outage, unknown (NaN)
    3. 0, offering, but it got there by a drop
       bigger than the battery can discharge     -> telemetry fault, unknown (NaN)
    4. 0, offering, and it drained down to 0     -> really empty, stays 0

The fleet SOC % only counts batteries with a reading, so rules 1-3 leave it alone and rule 4
pulls it down.
"""

import logging

import pandas as pd

from tools.constants import (
    battery_capacity_MW,
    battery_capacity_MWh,
    battery_codes,
    codes_in,
    commissioned_capacity,
    commissioning_dates,
    esr_codes,
)

logger = logging.getLogger(__name__)

CHARGE_LEVEL_SENTINEL = 999
OFFERED_MIN_MW = 1  # offering less than this in total (charge + discharge) = not offering


def clean_charge_level_sentinel_df(df: pd.DataFrame) -> pd.DataFrame:
    """Set the 999 MWh sentinel (missing telemetry) to NaN. Rule 1."""
    return df.replace(CHARGE_LEVEL_SENTINEL, float("nan"))


def clean_charge_level_overshoot_df(df: pd.DataFrame, capacity: dict[str, float] | None = None) -> pd.DataFrame:
    """Clip each battery's SOC to its rated capacity (KWINANA_ESR2 reads up to ~121%).

    Run after clean_charge_level_sentinel_df, or the 999 sentinel becomes a full charge.

    Args:
        df: SOC (MWh), one column per battery.
        capacity: {code: rated MWh}; defaults to battery_capacity_MWh.

    Returns:
        A clipped copy of df.
    """
    capacity = capacity if capacity is not None else battery_capacity_MWh
    df = df.copy()

    for code in esr_codes:
        if code not in df:
            continue
        rated = capacity[code]
        n_clipped = (df[code] > rated).sum()

        if n_clipped:
            logger.info(
                f"{code}: clipping {n_clipped} readings above rated capacity "
                f"({rated} MWh, max observed {df[code].max():.1f} MWh)"
            )
        df[code] = df[code].clip(upper=rated)

    return df


def offered_mw(bidstack: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """Total MW each battery offered per interval, charge and discharge together.

    Args:
        bidstack: Clean bid stack, indexed by dispatch_interval (one row per tranche).
        codes: Batteries to total.

    Returns:
        One column per battery; 0 where it offered nothing. Intervals missing from the bid stack
        altogether are not in the index (no data, not "offered nothing").
    """
    batteries = bidstack[bidstack["code"].isin(codes)]
    total = batteries["quantity"].abs().groupby([batteries.index, batteries["code"]]).sum().unstack()
    return total.reindex(index=bidstack.index.unique(), columns=codes).fillna(0)


def zero_while_not_offering(df: pd.DataFrame, offered: pd.DataFrame) -> pd.DataFrame:
    """Readings of 0 while the battery offered nothing: an outage. Rule 2.

    Args:
        df: SOC (MWh), one column per battery.
        offered: From offered_mw.

    Returns:
        Boolean frame like df; False where the bid stack has no data.
    """
    not_offering = offered.reindex(index=df.index, columns=df.columns) < OFFERED_MIN_MW  # NaN -> False
    return (df == 0) & not_offering


def zero_after_impossible_drop(df: pd.DataFrame, max_mw: dict[str, float] | None = None) -> pd.DataFrame:
    """Runs of 0 that began with a drop bigger than the battery can discharge. Rule 3.

    A battery can lose at most (rated MW x hours since its last reading) MWh. The whole run of
    zeros is flagged if its first zero breaks that, or if there is no earlier reading to check.

    Args:
        df: SOC (MWh), one column per battery.
        max_mw: {code: rated MW}; defaults to battery_capacity_MW.

    Returns:
        Boolean frame like df.
    """
    max_mw = max_mw if max_mw is not None else battery_capacity_MW
    flags = pd.DataFrame(False, index=df.index, columns=df.columns)

    for code in codes_in(df, list(max_mw)):
        soc = df[code].dropna()  # judge each zero against the last real reading, across gaps
        is_zero = soc == 0
        run_start = is_zero & ~is_zero.shift(fill_value=False)

        hours = soc.index.to_series().diff().dt.total_seconds() / 3600
        max_drop = max_mw[code] * hours
        possible = soc.shift() <= max_drop  # NaN (no earlier reading) -> not provably possible

        run_id = run_start.cumsum()
        bad_start = run_start & ~possible
        flags[code] = (is_zero & bad_start.groupby(run_id).transform("any")).reindex(df.index, fill_value=False)

    return flags


def mask_unknown_zeros(df: pd.DataFrame, offered: pd.DataFrame) -> pd.DataFrame:
    """Set to NaN the zeros that are an outage (rule 2) or a telemetry fault (rule 3).

    The zeros left are rule 4: the battery really is empty.

    Args:
        df: SOC (MWh), one column per battery.
        offered: From offered_mw.

    Returns:
        A copy of df.
    """
    outage = zero_while_not_offering(df, offered)
    fault = zero_after_impossible_drop(df) & ~outage

    for code in codes_in(df, list(battery_capacity_MWh)):
        n_outage, n_fault = outage[code].sum(), fault[code].sum()
        n_empty = (df[code] == 0).sum() - n_outage - n_fault
        logger.info(
            f"{code}: zeros - {n_outage} outage, {n_fault} impossible drop (masked); {n_empty} really empty (kept)"
        )

    return df.mask(outage | fault)


def add_soc_pct_columns(df: pd.DataFrame, capacity: dict[str, float] | None = None) -> pd.DataFrame:
    """Add a <code>_soc_pct column per battery.

    Args:
        df: SOC (MWh), one column per battery.
        capacity: {code: rated MWh}; defaults to battery_capacity_MWh.

    Returns:
        df with the % columns added.
    """
    capacity = capacity if capacity is not None else battery_capacity_MWh
    for code in esr_codes:
        if code not in df:
            logger.warning(f"{code}: no SOC column found, skipping soc_pct")
            continue
        df[f"{code}_soc_pct"] = df[code] / capacity[code] * 100

    return df


def add_fleet_soc_columns(
    df: pd.DataFrame,
    capacity: dict[str, float] | None = None,
    commissioning: dict[str, pd.Timestamp] | None = None,
    codes: list[str] | None = None,
) -> pd.DataFrame:
    """Add fleet totals: fleet_capacity_MWh, fleet_capacity_counted_MWh, fleet_soc_MWh, fleet_soc_pct.

    fleet_capacity_MWh counts every battery from its commissioning date.
    fleet_capacity_counted_MWh counts only those with an SOC reading in the interval, so a
    battery whose charge is unknown (NaN, rules 1-3) drops out of both sides of fleet_soc_pct.

    Args:
        df: Clean SOC (MWh), one column per battery.
        capacity: {code: rated MWh}; defaults to battery_capacity_MWh.
        commissioning: {code: commissioning timestamp}; defaults to the first reading in df.
        codes: Batteries in the fleet; defaults to battery_codes.

    Returns:
        df with the fleet columns added (NaN where no battery reports).
    """
    rated = battery_capacity_MWh if capacity is None else capacity
    fleet = battery_codes if codes is None else codes
    codes_present = [code for code in fleet if code in df]
    if not codes_present:
        logger.warning("no battery SOC columns found, skipping fleet columns")
        return df

    if commissioning is None:
        commissioning = commissioning_dates(df, candidates=codes_present)
    dates = {code: commissioning[code] for code in codes_present if code in commissioning}
    in_fleet = list(dates)

    df["fleet_capacity_MWh"] = commissioned_capacity(df.index, dates, rated)

    reporting = df[in_fleet].notna()
    counted = reporting.mul(pd.Series(rated)[in_fleet]).sum(axis=1)
    df["fleet_capacity_counted_MWh"] = counted.where(counted > 0)

    df["fleet_soc_MWh"] = df[codes_present].sum(axis=1, min_count=1)
    df["fleet_soc_pct"] = df["fleet_soc_MWh"] / df["fleet_capacity_counted_MWh"] * 100

    return df


def process(df: pd.DataFrame, bidstack: pd.DataFrame) -> pd.DataFrame:
    """Clean extracted SOC and add the derived columns.

    Applies rules 1-3 (see the module docstring), clips to rated capacity, and adds the
    <code>_soc_pct and fleet columns.

    Args:
        df: Extracted SOC (MWh), one column per facility.
        bidstack: Clean bid stack, to tell outages (not offering) from real empties.

    Returns:
        The cleaned frame.
    """
    df = clean_charge_level_sentinel_df(df)
    df = clean_charge_level_overshoot_df(df)
    df = mask_unknown_zeros(df, offered_mw(bidstack, codes_in(df, list(battery_capacity_MWh))))
    df = add_soc_pct_columns(df)

    return add_fleet_soc_columns(df)
