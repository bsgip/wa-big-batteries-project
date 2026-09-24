"""Build the tidy stress-event table: one row per (dispatch_interval,
battery_code) for the system_stress_events dates only, carrying SOC, power,
demand and price side by side.

Run it when you want the table refreshed:
    python -m analysis.stress_events
"""

import logging

import pandas as pd

from data_extraction.catalog import load
from data_processing.store import load_clean
from tools.constants import battery_codes, system_stress_events
from tools.io import save_df_to_csv
from tools.paths import local_processed_data_dir

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


def main():
    df = build_stress_event_data(
        soc_df=load_clean("soc"),
        power_df=load_clean("power"),
        demand=load("demand")["dispatchCondition.demand"],
        price=load("price")["energy_price"],
    )
    save_df_to_csv(df, local_processed_data_dir / "stress_event_data.csv")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
