"""Cleaning and derived columns for battery SOC (the extracted chargeLevel
field, in MWh).

`process()` is what data_processing/main.py runs; the individual steps are
public so callers with their own requirements (e.g. visualisation/plotting.py,
which does its own gap handling) can compose just the parts they want.
"""

import logging

import pandas as pd

from tools.constants import battery_capacity_MWh, esr_codes

logger = logging.getLogger(__name__)

CHARGE_LEVEL_SENTINEL = 999


def clean_charge_level_sentinel_df(df: pd.DataFrame) -> pd.DataFrame:
    """The chargeLevel SCADA tag reports a fixed 999 (MWh) sentinel when
    telemetry is missing/invalid (e.g. KWINANA_ESR2 sits at exactly 999 for
    long stretches, well above its 900 MWh capacity). Treat it as missing
    data rather than a real reading."""
    return df.replace(CHARGE_LEVEL_SENTINEL, float("nan"))


def clean_charge_level_overshoot_df(
    df: pd.DataFrame, capacity: dict[str, float] | None = None
) -> pd.DataFrame:
    """The chargeLevel data of KWINANA_ESR2 have values over its rated capacity
    of 900 MWh (peaking at ~121%). Treat these values as maximum, i.e. clip
    each battery's readings to its rated capacity rather than dropping them -
    the overshoot is measurement headroom, not a missing reading.

    Run this *after* clean_charge_level_sentinel_df: the 999 sentinel sits
    above some batteries' rated capacity (e.g. KWINANA_ESR2 at 900 MWh), and
    clipping first would silently turn it into a plausible-looking full-charge
    reading instead of NaN.
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
    """Treat a battery's chargeLevel reading as missing (NaN) rather than a
    real 0 whenever it holds at exactly 0 for at least `min_run_minutes`
    straight - this accounts for outages and readings during the battery's
    initial stage of commissioning.

    E.g. COLLIE_ESR1 has zero-runs up to 260h, and COLLIE_ESR4/COLLIE_ESR5
    share an identical ~335h zero run starting at the exact same 5-min
    timestamp as KWINANA_ESR1's own zero run - a shared-outage signature,
    not independent battery behaviour. Unlike CHARGE_LEVEL_SENTINEL (999),
    this hasn't been confirmed against raw SCADA JSON (see
    clean_charge_level_df) - it's a statistical heuristic, so re-check
    against raw data if/when it's available for these periods. A short
    isolated 0 (a real battery briefly fully discharging) is left alone.
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
    """Add a <code>_soc_pct column for each battery, computed from its
    charge_level column (MWh) and a rated capacity - battery_capacity_MWh by
    default, or an empirically-derived one (see derive_capacity_from_observed_max)
    if the caller passes one."""
    capacity = capacity if capacity is not None else battery_capacity_MWh
    for code in esr_codes:
        if code not in df:
            logger.warning(f"{code}: no SOC column found, skipping soc_pct")
            continue
        df[f"{code}_soc_pct"] = df[code] / capacity[code] * 100

    return df


def add_fleet_soc_columns(df: pd.DataFrame, capacity: dict[str, float] | None = None) -> pd.DataFrame:
    """Add fleet-wide aggregates across the esr_codes columns:

    - fleet_capacity_MWh  rated capacity of the commissioned fleet
    - fleet_soc_MWh       stored energy summed over the batteries reporting
    - fleet_soc_pct       the second as a percentage of the first

    fleet_capacity_MWh is a monotonic step function, not the constant ~5767
    MWh total: the fleet commissions in stages (KWINANA_ESR1 from 2023-09,
    COLLIE_ESR5 only from 2026-01), so a constant total would show the 2023
    fleet sitting at a meaningless ~2% SOC. A battery counts from its first
    non-NaN reading onwards and stays in the total through any later gap -
    the battery still exists during an outage.

    A commissioned battery with no reading therefore contributes its capacity
    but no stored energy, so an outage pulls fleet_soc_pct down - intended,
    since unavailable energy is unavailable to the system whatever the cause.
    Note this makes fleet_soc_pct a measure of usable fleet energy, not of
    how charged the reporting batteries are; 3-12% of each battery's
    post-commissioning readings are missing, so the dips are frequent.

    That only holds while *something* is reporting, though. A row where no
    battery reports at all is no observation rather than an empty fleet, so
    fleet_soc_MWh and fleet_soc_pct are NaN there (min_count=1) instead of 0
    - otherwise the 2023-24 record, when KWINANA_ESR1 was the only battery
    and any gap in it blacked out the whole fleet, reads as ~25k intervals of
    a stone-dead fleet. fleet_capacity_MWh keeps its latched value through
    those rows: the batteries still exist, they just aren't being seen.
    """
    rated = battery_capacity_MWh if capacity is None else capacity
    codes_present = [code for code in esr_codes if code in df]
    if not codes_present:
        logger.warning("no battery SOC columns found, skipping fleet columns")
        return df

    reporting = df[codes_present].notna()
    commissioned = reporting.cummax()  # latches True from each battery's first reading

    df["fleet_capacity_MWh"] = (
        commissioned.mul([rated[code] for code in codes_present]).sum(axis=1).where(commissioned.any(axis=1))
    )
    df["fleet_soc_MWh"] = df[codes_present].sum(axis=1, min_count=1)
    df["fleet_soc_pct"] = df["fleet_soc_MWh"] / df["fleet_capacity_MWh"] * 100

    return df


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Extracted SOC (MWh) -> analysis-ready: sentinel and sustained-zero
    readings masked as missing, plus a <code>_soc_pct column per battery.
    The bare <code> MWh columns are kept alongside the pct ones - plots use
    both."""
    df = clean_charge_level_sentinel_df(df)
    df = clean_charge_level_overshoot_df(df)
    df = mask_sustained_zero_runs(df)
    df = add_soc_pct_columns(df)

    return add_fleet_soc_columns(df)
