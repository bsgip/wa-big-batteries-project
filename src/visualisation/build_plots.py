"""Build the standard set of overview plots (SOC/power distribution, heatmap,
time-of-day, event-day summaries) from the cleaned datasets.

Run it directly - it loads what it needs:
    python -m visualisation.build_plots

Needs clean/soc and clean/power, so run data_processing/main.py first if
either is missing. For one-off interactive plots of a particular day, see
visualisation/plotting.py instead.
"""

import logging

import pandas as pd

from data_extraction.catalog import load
from data_processing.store import load_clean
from tools.constants import battery_codes
from visualisation.plot_distributions import plot_distribution_grid
from visualisation.plot_heatmap import plot_heatmap_grid
from visualisation.plot_soc import plot_event_day_summaries
from visualisation.plot_time_of_day import plot_soc_timeofday_box_grid, plot_soc_timeofday_fan_grid

logger = logging.getLogger(__name__)


def _try(label: str, fn, *args, **kwargs):
    """These plots are independent of each other, so one failing shouldn't
    cost you the other seven. Logs and carries on."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        logger.error(f"FAILED {label}: {type(e).__name__}: {e}")
        return None


def build_plots(soc_df: pd.DataFrame, power_df: pd.DataFrame, demand: pd.Series, price: pd.Series) -> None:
    logger.info("Building plots...")

    _try(
        "SOC distribution grid",
        plot_distribution_grid,
        soc_df,
        "_soc_pct",
        "SOC (%)",
        "Battery SOC distribution — entire timeframe",
        "timeframe_soc_distribution_grid.png",
    )
    _try(
        "SOC heatmap grid",
        plot_heatmap_grid,
        soc_df,
        "_soc_pct",
        cmap="viridis",
        center=None,
        colorbar_label="SOC (%)",
        title="Battery SOC heatmap — entire timeframe",
        filename="timeframe_soc_heatmap_grid.png",
    )
    _try(
        "SOC distribution grid (MWh)",
        plot_distribution_grid,
        soc_df,
        "",
        "SOC (MWh)",
        "Battery SOC distribution (MWh) — entire timeframe",
        "timeframe_soc_distribution_grid_mwh.png",
    )
    _try(
        "SOC heatmap grid (MWh)",
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
    _try(
        "SOC time-of-day fan grid (past year)",
        plot_soc_timeofday_fan_grid,
        soc_df,
        "_soc_pct",
        "SOC (%)",
        "Battery SOC by time of day — past year",
        "soc_timeofday_fan_grid.png",
    )
    _try(
        "SOC time-of-day box grid (past year)",
        plot_soc_timeofday_box_grid,
        soc_df,
        "_soc_pct",
        "SOC (%)",
        "Battery SOC by time of day — past year",
        "soc_timeofday_box_grid.png",
    )
    _try(
        "power distribution grid",
        plot_distribution_grid,
        power_df,
        "_power_pct",
        "Power (% of rated capacity)",
        "Battery power distribution — entire timeframe",
        "timeframe_power_distribution_grid.png",
        bicolor=True,
    )
    _try(
        "power heatmap grid",
        plot_heatmap_grid,
        power_df,
        "_power_pct",
        cmap="RdBu_r",
        center=0,
        colorbar_label="Power (% of rated, + discharge / - charge)",
        title="Battery power heatmap — entire timeframe",
        filename="timeframe_power_heatmap_grid.png",
    )
    _try("event-day summaries", plot_event_day_summaries, soc_df, power_df, demand, price)


def main():
    build_plots(
        soc_df=load_clean("soc"),
        power_df=load_clean("power"),
        demand=load("demand")["dispatchCondition.demand"],
        price=load("price")["energy_price"],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
