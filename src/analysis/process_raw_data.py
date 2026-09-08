"""Stage 2 of the pipeline: turn stage 1's raw SOC/power dataframes into
analysis-ready ones (cleaning, % columns, range checks), and build/save the
tidy stress-event table from them.

These are cheap derived computations, always redone fresh from the raw cache
produced by extract_raw_data.py - this is what makes constants.py fixes
(capacities, the 999 sentinel, etc.) not require re-running the expensive
extraction.
"""

import logging
from collections.abc import Callable

import pandas as pd

from tools.constants import battery_capacity_MW, battery_codes, system_stress_events
from tools.df_management import (
    add_power_pct_columns,
    add_soc_pct_columns,
    clean_charge_level_df,
    derive_capacity_from_observed_max,
    flag_out_of_range,
    mask_sustained_zero_runs,
    save_df_to_csv,
)
from tools.paths import repo_processed_data_dir

logger = logging.getLogger(__name__)


def _filter_to_event_dates(data: pd.DataFrame | pd.Series, event_dates: list[str]):
    return data[data.index.strftime("%Y-%m-%d").isin(event_dates)]


def _melt_raw_and_pct(df: pd.DataFrame, pct_suffix: str, raw_name: str, pct_name: str) -> pd.DataFrame:
    """Melt a wide <code> + <code><pct_suffix> dataframe into long format
    with both the raw and % columns side by side. Reuses whatever pct
    values are already on `df` (computed once, e.g. by add_soc_pct_columns)
    instead of re-deriving them from a capacity constant independently -
    the two must never be able to disagree."""
    raw_long = df[battery_codes].reset_index().melt(id_vars="dispatch_interval", var_name="code", value_name=raw_name)

    pct_cols = [f"{c}{pct_suffix}" for c in battery_codes if f"{c}{pct_suffix}" in df]
    pct_long = df[pct_cols].reset_index().melt(id_vars="dispatch_interval", var_name="code", value_name=pct_name)
    pct_long["code"] = pct_long["code"].str.removesuffix(pct_suffix)

    return raw_long.merge(pct_long, on=["dispatch_interval", "code"], how="left")


def build_stress_event_data(
    soc_df: pd.DataFrame, power_df: pd.DataFrame, demand: pd.Series, price: pd.Series
) -> pd.DataFrame:
    """Long/tidy table (one row per dispatch_interval x battery_code) for
    just the system_stress_events dates. Time, battery, and field don't fit
    a plain wide 2D table together - a tidy long table still is 2D, it just
    carries all three dimensions via composite key columns instead of one
    column per (battery, field) pair."""
    event_dates = [event["date"] for event in system_stress_events]

    soc = _filter_to_event_dates(soc_df, event_dates)
    power = _filter_to_event_dates(power_df, event_dates)

    soc_long = _melt_raw_and_pct(soc, "_soc_pct", "soc_mwh", "soc_pct")
    power_long = _melt_raw_and_pct(power, "_power_pct", "power_mw", "power_pct")

    long_df = soc_long.merge(power_long, on=["dispatch_interval", "code"], how="outer")

    demand_event = _filter_to_event_dates(demand, event_dates).rename("demand_mw")
    price_event = _filter_to_event_dates(price, event_dates).rename("energy_price")

    long_df = long_df.merge(demand_event, left_on="dispatch_interval", right_index=True, how="left")
    long_df = long_df.merge(price_event, left_on="dispatch_interval", right_index=True, how="left")

    return long_df.sort_values(["dispatch_interval", "code"]).reset_index(drop=True)


def process_raw_data(
    step: Callable[..., object],
    raw_soc_df: pd.DataFrame | None,
    raw_power_df: pd.DataFrame | None,
    demand: pd.Series | None,
    price: pd.Series | None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Run stage 2: clean/derive soc_df and power_df from stage 1's raw
    outputs, and build+save the stress-event table. Returns (soc_df,
    power_df)."""
    soc_df = None
    if raw_soc_df is not None:
        cleaned_soc_df = step("clean SOC (999 sentinel)", clean_charge_level_df, raw_soc_df)
        if cleaned_soc_df is not None:
            cleaned_soc_df = step(
                "mask sustained zero-SOC artifacts", mask_sustained_zero_runs, cleaned_soc_df
            )
        if cleaned_soc_df is not None:
            observed_capacity_MWh = step(
                "derive observed SOC capacity", derive_capacity_from_observed_max, cleaned_soc_df
            )
            if observed_capacity_MWh is not None:
                logger.info(f"observed SOC capacity (MWh): {observed_capacity_MWh}")
            soc_df = step(
                "add SOC pct columns", add_soc_pct_columns, cleaned_soc_df, observed_capacity_MWh
            )

    power_df = None
    if raw_power_df is not None:
        power_df = step("add power pct columns", add_power_pct_columns, raw_power_df)
        if power_df is not None:
            step("check power values against rated capacity", flag_out_of_range, power_df, battery_capacity_MW)

    if soc_df is not None and power_df is not None:
        stress_event_data = step(
            "build stress-event data", build_stress_event_data, soc_df, power_df, demand, price
        )
        if stress_event_data is not None:
            step(
                "save stress-event data",
                save_df_to_csv,
                stress_event_data,
                repo_processed_data_dir / "stress_event_data.csv",
            )
    else:
        logger.warning("skipping stress-event data - need both SOC and power")

    return soc_df, power_df
