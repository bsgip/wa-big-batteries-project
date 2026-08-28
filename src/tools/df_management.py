import logging
from pathlib import Path

import pandas as pd

from tools.constants import battery_capacity_MW, battery_capacity_MWh, battery_codes

logger = logging.getLogger(__name__)

CHARGE_LEVEL_SENTINEL = 999


def save_df_to_csv(df: pd.DataFrame, filename: Path):
    df.to_csv(filename)
    print(f"csv successfully saved to {filename}")


def save_df_to_parquet(df: pd.DataFrame, filename: Path):
    df.to_parquet(filename)
    print(f"parquet successfully saved to {filename}")


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
    flag_out_of_range for a full-corpus check of that assumption."""
    return df.replace(CHARGE_LEVEL_SENTINEL, float("nan"))


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


def add_power_pct_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add a <code>_power_pct column for each battery, computed from its
    power column (MW) and rated capacity in battery_capacity_MW."""
    for code in battery_codes:
        if code not in df:
            logger.warning(f"{code}: no power column found, skipping power_pct")
            continue
        df[f"{code}_power_pct"] = df[code] / battery_capacity_MW[code] * 100

    return df


def flag_out_of_range(df: pd.DataFrame, capacity: dict[str, float], tolerance: float = 1.05) -> None:
    """Log (don't drop) any values whose magnitude exceeds a battery's rated
    capacity by more than `tolerance`. Used to check a full extraction for
    sentinel-like values (e.g. the chargeLevel 999 case) without silently
    dropping anything - a human should look at what's flagged."""
    for code in battery_codes:
        if code not in df:
            continue
        limit = capacity[code] * tolerance
        n_out = df[code].abs().gt(limit).sum()
        if n_out:
            logger.warning(f"{code}: {n_out} values exceed {limit:.1f} (rated {capacity[code]})")
