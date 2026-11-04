"""Daily demand measures from 5-min demand -> clean/demand_measures.parquet.

    python -m analysis.demand_measures
"""

import logging

import pandas as pd

from data_processing.demand import clean_demand
from tools.io import load_clean, save_clean

logger = logging.getLogger(__name__)

# A day needs this many of its 288 5-min intervals to count; part days give misleading stats.
MIN_INTERVALS_PER_DAY = 240


def daily_demand_measures(demand, start=None, end=None, min_intervals=MIN_INTERVALS_PER_DAY):
    """Daily demand statistics from 5-min demand.

    Args:
        demand: 5-min demand (MW), as a Series or one-column frame.
        start: Start of the time window used each day, e.g. "17:30"; None for the whole day.
        end: End of that window (exclusive).
        min_intervals: Days with fewer readings are dropped; None keeps every day.

    Returns:
        One row per day: peak, trough, range, mean, median, std, iqr (all MW) and n.
    """
    s = demand.squeeze()  # one-column DataFrame -> Series
    if start is not None and end is not None:
        s = s.between_time(start, end, inclusive="left")

    r = s.resample("D")

    out = pd.DataFrame({
        "peak": r.max(),
        "trough": r.min(),
        "range": r.max() - r.min(),
        "mean": r.mean(),
        "median": r.median(),
        "std": r.std(),
        "iqr": r.quantile(.75) - r.quantile(.25),
        "n": r.count(),
    })

    if min_intervals is not None:
        incomplete = out["n"] < min_intervals
        if incomplete.any():
            logger.info(f"demand measures: dropping {incomplete.sum()} days with <{min_intervals} intervals")
        out = out[~incomplete]

    return out


def daily_demand_range(demand: pd.Series, min_intervals: int = MIN_INTERVALS_PER_DAY) -> pd.DataFrame:
    """Daily peak, trough and range of demand, after clean_demand.

    Args:
        demand: 5-min demand (MW).
        min_intervals: Days with fewer readings are dropped.

    Returns:
        One row per date: peak, trough, n_intervals and range (peak - trough, MW).
    """
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
    """Repeat each day's value across every interval of that day.

    Args:
        daily: Values indexed by date.
        index: Intervals to fill.

    Returns:
        Series on index; NaN for days missing from daily.
    """
    return daily.reindex(index.normalize()).set_axis(index)


def main():
    measures = daily_demand_measures(load_clean("demand")["demand"])
    save_clean("demand_measures", measures)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
