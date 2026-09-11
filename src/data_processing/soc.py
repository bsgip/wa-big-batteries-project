"""Cleaning and derived columns for battery SOC (the extracted chargeLevel
field, in MWh).

`process()` is what data_processing/main.py runs; the individual steps are
public so callers with their own requirements (e.g. visualisation/plotting.py,
which does its own gap handling) can compose just the parts they want.
"""

import logging

import pandas as pd

from tools.constants import battery_capacity_MWh, battery_codes

logger = logging.getLogger(__name__)

CHARGE_LEVEL_SENTINEL = 999


def clean_charge_level_df(df: pd.DataFrame) -> pd.DataFrame:
    """The chargeLevel SCADA tag reports a fixed 999 (MWh) sentinel when
    telemetry is missing/invalid (e.g. KWINANA_ESR2 sits at exactly 999 for
    long stretches, well above its 900 MWh capacity). Treat it as missing
    data rather than a real reading.

    Confirmed against the raw caseInputData JSON (not a parsing artifact):
    AEMO's own SCADA feed reports the literal value 999 with
    qualityFlag="good", dataSource="SCADA". Specific to chargeLevel - a
    sample of the equivalent power field (initialMw) across the
    dispatchSolution corpus found no equivalent sentinel; see
    power.flag_out_of_range for a full-corpus check of that assumption."""
    return df.replace(CHARGE_LEVEL_SENTINEL, float("nan"))


def mask_sustained_zero_runs(df: pd.DataFrame, min_run_minutes: int = 60) -> pd.DataFrame:
    """Treat a battery's chargeLevel reading as missing (NaN) rather than a
    real 0 whenever it holds at exactly 0 for at least `min_run_minutes`
    straight - a battery genuinely sitting at 0 MWh for that long while the
    market keeps dispatching is implausible.

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

    for code in battery_codes:
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


def derive_capacity_from_observed_max(df: pd.DataFrame) -> dict[str, float]:
    """Empirically derive a per-battery capacity dict from the observed max
    value in `df`'s columns, for use when a documented rated capacity looks
    stale/wrong. E.g. KWINANA_ESR2's documented battery_capacity_MWh (900)
    made its observed chargeLevel readings peak at 121% SOC, far more than
    every other battery's ~100-105% (plausible measurement headroom) - using
    the observed max instead pins that battery's own peak reading at exactly
    100% rather than guessing at a "corrected" documented value."""
    return {code: df[code].max() for code in battery_codes if code in df}


def add_soc_pct_columns(df: pd.DataFrame, capacity: dict[str, float] | None = None) -> pd.DataFrame:
    """Add a <code>_soc_pct column for each battery, computed from its
    charge_level column (MWh) and a rated capacity - battery_capacity_MWh by
    default, or an empirically-derived one (see derive_capacity_from_observed_max)
    if the caller passes one."""
    capacity = capacity if capacity is not None else battery_capacity_MWh
    for code in battery_codes:
        if code not in df:
            logger.warning(f"{code}: no SOC column found, skipping soc_pct")
            continue
        df[f"{code}_soc_pct"] = df[code] / capacity[code] * 100

    return df


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Extracted SOC (MWh) -> analysis-ready: sentinel and sustained-zero
    readings masked as missing, plus a <code>_soc_pct column per battery.
    The bare <code> MWh columns are kept alongside the pct ones - plots use
    both."""
    df = clean_charge_level_df(df)
    df = mask_sustained_zero_runs(df)

    observed_capacity_MWh = derive_capacity_from_observed_max(df)
    logger.info(f"observed SOC capacity (MWh): {observed_capacity_MWh}")

    return add_soc_pct_columns(df, observed_capacity_MWh)
