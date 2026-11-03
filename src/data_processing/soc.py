"""Cleaning and derived columns for battery SOC (the extracted chargeLevel
field, in MWh).

`process()` is what data_processing/main.py runs; the individual steps are
public so callers with their own requirements (e.g. visualisation/plotting.py,
which does its own gap handling) can compose just the parts they want.
"""

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


def soc_telemetry_gap(
    df: pd.DataFrame,
    power: pd.DataFrame,
    codes: list[str],
    window_minutes: int = DATA_GAP_WINDOW_MINUTES,
    active_mw: float = POWER_ACTIVE_MW,
) -> pd.DataFrame:
    """True where a battery has no SOC reading and, within window_minutes either side,
    its power either
      - shows it charging or discharging (|MW| > active_mw): it is operating, only the SOC data is missing, or
      - has no readings at all: its telemetry is missing altogether.
    Either way its charge is unknown, not 0.

    No SOC reading with power readings that are all idle is NOT a gap: that battery
    may be tripped or on outage, so it stays in the fleet's capacity and pulls
    fleet_soc_pct down.

    power: battery power (MW), one column per battery, same 5-min index as df
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
    """Add fleet_capacity_MWh, fleet_capacity_counted_MWh, fleet_soc_MWh and fleet_soc_pct
    over the fleet's batteries.

    codes: batteries in the fleet (default: battery_codes, i.e. without ALINTA_WGP_ESR1,
        so the fleet matches the power data)
    commissioning: from constants.commissioning_dates (default: from this frame)
    power: battery power (MW) - used to tell SOC data gaps from outages (see below)
    sentinel: True where the raw reading was the 999 sentinel, from before
        clean_charge_level_sentinel_df turned it into NaN

    fleet_capacity_MWh: every battery from its commissioning date.
    fleet_capacity_counted_MWh: the same, minus batteries with an SOC telemetry gap, whose
        charge is unknown, not 0:
        - the reading was the 999 sentinel - telemetry invalid by definition, whatever power does
        - no SOC reading while power shows them operating, or no power reading either
          (soc_telemetry_gap)
        Any other missing reading with idle power still counts, so a trip or outage pulls
        fleet_soc_pct down. Without power or sentinel, nothing is taken out.
    fleet_soc_pct: fleet_soc_MWh / fleet_capacity_counted_MWh. Rows where no battery reports are NaN, not 0.
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
    """Extracted SOC (MWh) -> analysis-ready: sentinel and sustained-zero
    readings masked as missing, plus a <code>_soc_pct column per battery.
    The bare <code> MWh columns are kept alongside the pct ones - plots use
    both.

    power: battery power (MW), e.g. load("power") - lets the fleet columns
    tell SOC data gaps from outages (see add_fleet_soc_columns)."""
    # where the 999s were, before cleaning makes them indistinguishable from other NaNs
    sentinel = df == CHARGE_LEVEL_SENTINEL

    df = clean_charge_level_sentinel_df(df)
    df = clean_charge_level_overshoot_df(df)
    df = mask_sustained_zero_runs(df)
    df = add_soc_pct_columns(df)

    return add_fleet_soc_columns(df, power=power, sentinel=sentinel)
