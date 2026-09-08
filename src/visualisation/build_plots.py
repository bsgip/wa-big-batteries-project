"""Stage 3 of the pipeline: run every pipeline plot (SOC/power distribution,
heatmap, time-of-day, event-day summaries) against stage 2's processed
dataframes."""

import logging
from collections.abc import Callable

import pandas as pd

from tools.constants import battery_codes
from visualisation.plot_distributions import plot_distribution_grid
from visualisation.plot_heatmap import plot_heatmap_grid
from visualisation.plot_soc import plot_event_day_summaries
from visualisation.plot_time_of_day import plot_soc_timeofday_box_grid, plot_soc_timeofday_fan_grid

logger = logging.getLogger(__name__)


def build_plots(
    step: Callable[..., object],
    soc_df: pd.DataFrame | None,
    power_df: pd.DataFrame | None,
    demand: pd.Series | None,
    price: pd.Series | None,
) -> None:
    """Run stage 3: build all pipeline plots from stage 2's soc_df/power_df."""
    logger.info("Building plots...")

    if soc_df is not None:
        step(
            "plot: SOC distribution grid",
            plot_distribution_grid,
            soc_df,
            "_soc_pct",
            "SOC (%)",
            "Battery SOC distribution — entire timeframe",
            "timeframe_soc_distribution_grid.png",
        )
        step(
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
        step(
            "plot: SOC distribution grid (MWh)",
            plot_distribution_grid,
            soc_df,
            "",
            "SOC (MWh)",
            "Battery SOC distribution (MWh) — entire timeframe",
            "timeframe_soc_distribution_grid_mwh.png",
        )
        step(
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
        step(
            "plot: SOC time-of-day fan grid (past year)",
            plot_soc_timeofday_fan_grid,
            soc_df,
            "_soc_pct",
            "SOC (%)",
            "Battery SOC by time of day — past year",
            "soc_timeofday_fan_grid.png",
        )
        step(
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
        step(
            "plot: power distribution grid",
            plot_distribution_grid,
            power_df,
            "_power_pct",
            "Power (% of rated capacity)",
            "Battery power distribution — entire timeframe",
            "timeframe_power_distribution_grid.png",
            bicolor=True,
        )
        step(
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
        step("plot: event-day summaries", plot_event_day_summaries, soc_df, power_df, demand, price)
    else:
        logger.warning("skipping event-day plots - need both SOC and power")
