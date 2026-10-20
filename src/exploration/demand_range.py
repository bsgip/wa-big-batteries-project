"""Daily demand range (max - min, MW) -> clean/demand_range.parquet, and SOC plotted against it.

    python -m exploration.demand_range
"""

import logging

import pandas as pd

from data_processing.demand import clean_demand
from tools.io import load_clean, save_clean
from visualisation.plot_soc_correlations import plot_soc_vs

logger = logging.getLogger(__name__)

# A day needs this many of its 288 5-min intervals to count; part days give a too-small range.
MIN_INTERVALS_PER_DAY = 240


def daily_demand_range(demand: pd.Series, min_intervals: int = MIN_INTERVALS_PER_DAY) -> pd.DataFrame:
    """One row per day: peak, trough and range (peak - trough, MW). Days under min_intervals are dropped."""
    grouped = clean_demand(demand).groupby(demand.index.normalize())

    df = pd.DataFrame({
        "peak": grouped.max(),
        "trough": grouped.min(),
        "n_intervals": grouped.count(),
    })
    df["range"] = df["peak"] - df["trough"]
    df.index.name = "date"

    incomplete = df["n_intervals"] < min_intervals
    if incomplete.any():
        logger.info(f"dropping {incomplete.sum()} days with <{min_intervals} intervals")

    return df[~incomplete]


def broadcast_to_intervals(daily: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Repeat each day's value across every interval of that day (missing days -> NaN)."""
    return daily.reindex(index.normalize()).set_axis(index)


def main():
    demand = load_clean("demand")["demand"]

    demand_range = daily_demand_range(demand)
    save_clean("demand_range", demand_range)

    soc = load_clean("soc")
    plot_soc_vs(
        soc_df=soc,
        x_series=broadcast_to_intervals(demand_range["range"], soc.index),
        x_label="Daily demand range (MW)",
        title="Battery SOC vs daily demand range - full history",
        filename="soc_vs_demand_range.png",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
