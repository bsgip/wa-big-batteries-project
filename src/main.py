import logging

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
    save_df_to_parquet,
)
from tools.paths import repo_processed_data_dir
from visualisation.plot_distributions import plot_distribution_grid
from visualisation.plot_heatmap import plot_heatmap_grid
from visualisation.plot_soc import plot_event_day_summaries
from visualisation.plot_time_of_day import plot_soc_timeofday_box_grid, plot_soc_timeofday_fan_grid
from wem_data.parse_case_input import build_charge_level_and_demand_df
from wem_data.parse_dispatch_solution import build_price_and_power_df

logger = logging.getLogger(__name__)

# One place to check at the end of a run: every step below reports here
# instead of letting an exception stop the rest of the pipeline.
_run_summary: list[str] = []


def _step(description: str, fn, *args, **kwargs):
    """Run one pipeline step. On failure, log it, record it in the
    end-of-run summary, and return None instead of raising - nothing here
    should ever be able to stop the rest of the pipeline from running."""
    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        logger.error(f"FAILED: {description} ({type(e).__name__}: {e})")
        _run_summary.append(f"FAILED: {description} - {type(e).__name__}: {e}")
        return None
    _run_summary.append(f"OK: {description}")
    return result


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


# --- raw extraction (expensive - the actual JSON walk) -----------------
#
# These two functions cache/load ONLY the raw extracted values (chargeLevel
# straight off SCADA, raw initialMw, raw energy_price) - no cleaning, no %
# conversion, no derived columns. That split matters: cleaning logic, rated
# capacities, and anything else derived from these raw values can have bugs
# (like the KWINANA_ESR1/ESR2 capacity swap, or a wrong sentinel value) that
# get fixed in tools/constants.py or tools/df_management.py without ever
# needing to redo the expensive walk - only a bug in the raw extraction
# itself (parsing, sign convention, row ordering) requires deleting the
# cached parquet and re-extracting.


def _extract_soc_and_demand():
    soc_path = repo_processed_data_dir / "soc.parquet"
    demand_path = repo_processed_data_dir / "demand.parquet"

    if soc_path.exists() and demand_path.exists():
        logger.info(f"{soc_path.name} and {demand_path.name} already exist, loading instead of re-extracting")
        return pd.read_parquet(soc_path), pd.read_parquet(demand_path)

    soc_df, demand_df = build_charge_level_and_demand_df(max_workers=12)
    save_df_to_parquet(soc_df, soc_path)
    save_df_to_parquet(demand_df, demand_path)
    return soc_df, demand_df


def _extract_power_and_price():
    power_path = repo_processed_data_dir / "power.parquet"
    price_path = repo_processed_data_dir / "price.parquet"

    if power_path.exists() and price_path.exists():
        logger.info(f"{power_path.name} and {price_path.name} already exist, loading instead of re-extracting")
        return pd.read_parquet(power_path), pd.read_parquet(price_path)

    price_df, power_df = build_price_and_power_df(dates=None, max_workers=12)
    save_df_to_parquet(power_df, power_path)
    save_df_to_parquet(price_df, price_path)
    return power_df, price_df


def main():
    repo_processed_data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Extracting/loading raw SOC (chargeLevel) and demand from caseInputData...")
    soc_demand = _step("extract/load raw SOC and demand", _extract_soc_and_demand)
    raw_soc_df, demand_df = soc_demand if soc_demand else (None, None)
    demand = demand_df["dispatchCondition.demand"] if demand_df is not None else None

    logger.info("Extracting/loading raw power and price from dispatchSolution (slow if not cached, ~45-60min)...")
    power_price = _step("extract/load raw power and price", _extract_power_and_price)
    raw_power_df, price_df = power_price if power_price else (None, None)
    price = price_df["energy_price"] if price_df is not None else None

    # cheap derived computations, always redone fresh from the raw cache
    # above - this is what makes constants.py fixes (capacities, the 999
    # sentinel, etc.) not require re-running the expensive extraction
    soc_df = None
    if raw_soc_df is not None:
        cleaned_soc_df = _step("clean SOC (999 sentinel)", clean_charge_level_df, raw_soc_df)
        if cleaned_soc_df is not None:
            cleaned_soc_df = _step(
                "mask sustained zero-SOC artifacts", mask_sustained_zero_runs, cleaned_soc_df
            )
        if cleaned_soc_df is not None:
            observed_capacity_MWh = _step(
                "derive observed SOC capacity", derive_capacity_from_observed_max, cleaned_soc_df
            )
            if observed_capacity_MWh is not None:
                logger.info(f"observed SOC capacity (MWh): {observed_capacity_MWh}")
            soc_df = _step(
                "add SOC pct columns", add_soc_pct_columns, cleaned_soc_df, observed_capacity_MWh
            )

    power_df = None
    if raw_power_df is not None:
        power_df = _step("add power pct columns", add_power_pct_columns, raw_power_df)
        if power_df is not None:
            _step("check power values against rated capacity", flag_out_of_range, power_df, battery_capacity_MW)

    if soc_df is not None and power_df is not None:
        stress_event_data = _step(
            "build stress-event data", build_stress_event_data, soc_df, power_df, demand, price
        )
        if stress_event_data is not None:
            _step(
                "save stress-event data",
                save_df_to_csv,
                stress_event_data,
                repo_processed_data_dir / "stress_event_data.csv",
            )
    else:
        logger.warning("skipping stress-event data - need both SOC and power")

    logger.info("Building plots...")

    if soc_df is not None:
        _step(
            "plot: SOC distribution grid",
            plot_distribution_grid,
            soc_df,
            "_soc_pct",
            "SOC (%)",
            "Battery SOC distribution — entire timeframe",
            "timeframe_soc_distribution_grid.png",
        )
        _step(
            "plot: SOC heatmap grid",
            plot_heatmap_grid,
            soc_df,
            "_soc_pct",
            cmap="viridis",
            center=None,
            colorbar_label="SOC (%)",
            title="Battery SOC heatmap — entire timeframe",
            filename="timeframe_soc_heatmap_grid.png",
        )
        _step(
            "plot: SOC distribution grid (MWh)",
            plot_distribution_grid,
            soc_df,
            "",
            "SOC (MWh)",
            "Battery SOC distribution (MWh) — entire timeframe",
            "timeframe_soc_distribution_grid_mwh.png",
        )
        _step(
            "plot: SOC heatmap grid (MWh)",
            plot_heatmap_grid,
            soc_df,
            "",
            cmap="viridis",
            center=None,
            colorbar_label="SOC (MWh)",
            title="Battery SOC heatmap (MWh) — entire timeframe",
            filename="timeframe_soc_heatmap_grid_mwh.png",
            vmax=soc_df[battery_codes].max().max(),
        )
        _step(
            "plot: SOC time-of-day fan grid (past year)",
            plot_soc_timeofday_fan_grid,
            soc_df,
            "_soc_pct",
            "SOC (%)",
            "Battery SOC by time of day — past year",
            "soc_timeofday_fan_grid.png",
        )
        _step(
            "plot: SOC time-of-day box grid (past year)",
            plot_soc_timeofday_box_grid,
            soc_df,
            "_soc_pct",
            "SOC (%)",
            "Battery SOC by time of day — past year",
            "soc_timeofday_box_grid.png",
        )
    else:
        logger.warning("skipping SOC plots - no SOC data")

    if power_df is not None:
        _step(
            "plot: power distribution grid",
            plot_distribution_grid,
            power_df,
            "_power_pct",
            "Power (% of rated capacity)",
            "Battery power distribution — entire timeframe",
            "timeframe_power_distribution_grid.png",
            bicolor=True,
        )
        _step(
            "plot: power heatmap grid",
            plot_heatmap_grid,
            power_df,
            "_power_pct",
            cmap="RdBu_r",
            center=0,
            colorbar_label="Power (% of rated, + discharge / - charge)",
            title="Battery power heatmap — entire timeframe",
            filename="timeframe_power_heatmap_grid.png",
        )
    else:
        logger.warning("skipping power plots - no power data")

    if soc_df is not None and power_df is not None:
        _step("plot: event-day summaries", plot_event_day_summaries, soc_df, power_df, demand, price)
    else:
        logger.warning("skipping event-day plots - need both SOC and power")

    logger.info("Done. Run summary:")
    for line in _run_summary:
        logger.info(f"  {line}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
