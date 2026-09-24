"""Daily peak-to-trough spread of system demand, and SOC plotted against it.

The delta is one number per day (that day's max demand minus its min), so
it's a "how hard did the system swing today" measure rather than a
per-interval one.

    python -m exploration.delta_peak_trough
"""

import logging

import pandas as pd

from data_extraction.catalog import load
from data_processing.demand import DEMAND_COLUMN, clean_demand
from data_processing.store import load_clean
from exploration.variables_correlations import plot_soc_vs
from tools.io import save_df_to_csv
from tools.paths import clean_data_dir

logger = logging.getLogger(__name__)

# A day needs this many of its 288 5-min intervals before its max/min are
# taken as that day's real peak and trough. 28 days in the corpus are part
# days (96 or 192 intervals - a third or two thirds of a day); their trough
# in particular is whatever the demand happened to be doing when the data
# stopped, not an overnight minimum.
MIN_INTERVALS_PER_DAY = 240


def daily_peak_trough(demand: pd.Series, min_intervals: int = MIN_INTERVALS_PER_DAY) -> pd.DataFrame:
    """One row per day: peak, trough, and the delta between them (MW).

    Days with fewer than `min_intervals` readings are dropped rather than
    reported with an understated delta - see MIN_INTERVALS_PER_DAY."""
    grouped = clean_demand(demand).groupby(demand.index.normalize())

    df = pd.DataFrame({
        "peak": grouped.max(),
        "trough": grouped.min(),
        "n_intervals": grouped.count(),
    })
    df["delta"] = df["peak"] - df["trough"]
    df.index.name = "date"

    incomplete = df["n_intervals"] < min_intervals
    if incomplete.any():
        logger.info(f"dropping {incomplete.sum()} days with <{min_intervals} intervals")

    return df[~incomplete]


def broadcast_to_intervals(daily: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Spread one value per day back across every interval of that day, so a
    daily measure can be joined onto 5-minutely data. Days missing from
    `daily` (the dropped part days) come back as NaN and get filtered out
    downstream."""
    return daily.reindex(index.normalize()).set_axis(index)


def main():
    demand = load("demand")[DEMAND_COLUMN]

    delta_df = daily_peak_trough(demand)
    save_df_to_csv(delta_df, clean_data_dir / "demand_delta_peak_trough.csv")

    soc = load_clean("soc")
    plot_soc_vs(
        soc_df=soc,
        x_series=broadcast_to_intervals(delta_df["delta"], soc.index),
        x_label="Daily demand peak - trough (MW)",
        title="Battery SOC vs daily demand peak-trough spread - full history",
        filename="soc_vs_demand_delta.png",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
