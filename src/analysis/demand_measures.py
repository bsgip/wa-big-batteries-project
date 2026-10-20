"""Daily demand measures from 5-min demand -> clean/demand_measures.parquet.

    python -m analysis.demand_measures
"""

import logging

import pandas as pd

from tools.io import load_clean, save_clean

logger = logging.getLogger(__name__)

# A day needs this many of its 288 5-min intervals to count; part days give misleading stats.
MIN_INTERVALS_PER_DAY = 240


def daily_demand_measures(demand, start=None, end=None, min_intervals=MIN_INTERVALS_PER_DAY):
    """Daily demand measures (MW) from 5-min demand, optionally only between start and end each day.

    demand: 5-min demand series (or one-column frame), MW
    start, end: time window each day, e.g. "17:30", "21:30" (None = whole day)
    min_intervals: days with fewer readings are dropped (lower it for short windows, None keeps all)
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


def main():
    measures = daily_demand_measures(load_clean("demand")["demand"])
    save_clean("demand_measures", measures)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
