"""Cleaning and derived columns for battery SOC (chargeLevel, MWh). process() runs every step."""

import logging

import pandas as pd

from tools.constants import (
    battery_capacity_MWh,
    battery_codes,
    commissioned_capacity,
    commissioning_dates,
    esr_codes,
)

logger = logging.getLogger(__name__)

CHARGE_LEVEL_SENTINEL = 999

# telling an SOC data gap from an outage (see soc_telemetry_gap)
POWER_ACTIVE_MW = 1  # |MW| above this = charging or discharging, not idle
DATA_GAP_WINDOW_MINUTES = 60  # how far either side of a missing SOC reading to look for power activity


def clean_charge_level_sentinel_df(df: pd.DataFrame) -> pd.DataFrame:
    """Set the 999 MWh sentinel (missing telemetry) to NaN."""
    return df.replace(CHARGE_LEVEL_SENTINEL, float("nan"))


def clean_charge_level_overshoot_df(
    df: pd.DataFrame, capacity: dict[str, float] | None = None
) -> pd.DataFrame:
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


def mask_sustained_zero_runs(df: pd.DataFrame, min_run_minutes: int = 60) -> pd.DataFrame:
    """Set SOC to NaN where it stays at exactly 0 for a long run (outages, commissioning).

    A heuristic, not checked against raw SCADA. Shorter runs of 0 are kept.

    Args:
        df: SOC (MWh), one column per battery.
        min_run_minutes: Shortest run of zeros to mask.

    Returns:
        A copy of df with those runs set to NaN.
    """
    df = df.copy()
    interval_minutes = df.index.to_series().diff().median().total_seconds() / 60
    min_run_length = max(1, round(min_run_minutes / interval_minutes))

    for code in esr_codes:
        if code not in df:
            continue
        is_zero = df[code] == 0
        run_id = (is_zero != is_zero.shift()).cumsum()
        run_length = is_zero.groupby(run_id).transform("size")
        sustained_zero = is_zero & (run_length >= min_run_length)
        n_masked = sustained_zero.sum()
        if n_masked:
            logger.info(f"{code}: masking {n_masked} sustained-zero readings as missing")
        df.loc[sustained_zero, code] = float("nan")

    return df


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


def soc_telemetry_gap(
    df: pd.DataFrame,
    power: pd.DataFrame,
    codes: list[str],
    window_minutes: int = DATA_GAP_WINDOW_MINUTES,
    active_mw: float = POWER_ACTIVE_MW,
) -> pd.DataFrame:
    """Flag missing SOC readings where the battery's charge is unknown, not 0.

    A gap is a missing SOC reading where, within window_minutes either side, the battery's
    power is active (|MW| > active_mw) or missing too. Missing SOC with idle power is not a gap.

    Args:
        df: SOC (MWh), one column per battery.
        power: Battery power (MW), same 5-min index.
        codes: Batteries to check.
        window_minutes: How far either side to look at power.
        active_mw: |MW| above which a battery counts as operating.

    Returns:
        Boolean frame, one column per battery in codes.
    """
    interval = df.index.to_series().diff().median()
    half = max(1, round(pd.Timedelta(minutes=window_minutes) / interval))

    def anywhere_nearby(flags: pd.DataFrame) -> pd.DataFrame:
        return flags.astype(float).rolling(2 * half + 1, center=True, min_periods=1).max().gt(0)

    mw = power.reindex(index=df.index, columns=codes)
    active_nearby = anywhere_nearby(mw.abs().gt(active_mw))  # NaN compares False, so never active
    readings_nearby = anywhere_nearby(mw.notna())

    return df[codes].isna() & (active_nearby | ~readings_nearby)


def add_fleet_soc_columns(
    df: pd.DataFrame,
    capacity: dict[str, float] | None = None,
    commissioning: dict[str, pd.Timestamp] | None = None,
    codes: list[str] | None = None,
    power: pd.DataFrame | None = None,
    sentinel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add fleet totals: fleet_capacity_MWh, fleet_capacity_counted_MWh, fleet_soc_MWh, fleet_soc_pct.

    fleet_capacity_MWh counts every battery from its commissioning date. Counted capacity
    also leaves out batteries whose charge is unknown: a 999 sentinel reading, or an SOC
    telemetry gap (see soc_telemetry_gap). Other missing readings still count, pulling
    fleet_soc_pct down. fleet_soc_pct is fleet_soc_MWh / fleet_capacity_counted_MWh.

    Args:
        df: SOC (MWh), one column per battery.
        capacity: {code: rated MWh}; defaults to battery_capacity_MWh.
        commissioning: {code: commissioning timestamp}; defaults to the first reading in df.
        codes: Batteries in the fleet; defaults to battery_codes.
        power: Battery power (MW), to find telemetry gaps; None leaves gaps in.
        sentinel: True where the raw reading was 999, from before cleaning; None for none.

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

    df["fleet_capacity_MWh"] = commissioned_capacity(df.index, dates, rated)

    in_fleet = list(dates)
    gaps = pd.DataFrame(False, index=df.index, columns=in_fleet)
    if sentinel is not None:
        gaps |= sentinel.reindex(index=df.index, columns=in_fleet, fill_value=False)
    if power is not None:
        gaps |= soc_telemetry_gap(df, power, in_fleet)

    # only a battery already in the capacity (commissioned) can be taken out of it
    commissioned = pd.DataFrame({code: df.index >= dates[code] for code in in_fleet}, index=df.index)
    gap_capacity = (gaps & commissioned).mul(pd.Series(rated)[in_fleet]).sum(axis=1)
    logger.info(f"fleet capacity: {int((gap_capacity > 0).sum())} intervals with an SOC telemetry gap taken out")
    counted = df["fleet_capacity_MWh"] - gap_capacity
    df["fleet_capacity_counted_MWh"] = counted

    df["fleet_soc_MWh"] = df[codes_present].sum(axis=1, min_count=1)
    df["fleet_soc_pct"] = df["fleet_soc_MWh"] / counted.where(counted > 0) * 100

    return df


def process(df: pd.DataFrame, power: pd.DataFrame | None = None) -> pd.DataFrame:
    """Clean extracted SOC and add the derived columns.

    Masks the 999 sentinel and long runs of zeros, clips to rated capacity, and adds the
    <code>_soc_pct and fleet columns.

    Args:
        df: Extracted SOC (MWh), one column per facility.
        power: Battery power (MW), to tell telemetry gaps from outages; None to skip.

    Returns:
        The cleaned frame.
    """
    # where the 999s were, before cleaning makes them indistinguishable from other NaNs
    sentinel = df == CHARGE_LEVEL_SENTINEL

    df = clean_charge_level_sentinel_df(df)
    df = clean_charge_level_overshoot_df(df)
    df = mask_sustained_zero_runs(df)
    df = add_soc_pct_columns(df)

    return add_fleet_soc_columns(df, power=power, sentinel=sentinel)
