"""Daily price measures from 5-min prices -> clean/price_measures.parquet.

    python -m analysis.price_measures
"""
from numpy.testing import measure

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_extraction.catalog import load
from tools.io import save_clean
from visualisation.plot_trend_band import trend_band_event_plot

logger = logging.getLogger(__name__)

INTERVALS_PER_HOUR = 12  # 5-min prices

# A day needs this many of its 288 5-min intervals to count; part days give misleading stats.
MIN_INTERVALS_PER_DAY = 240


def _best_spread(day_prices, hours):
    """Highest minus lowest `hours`-long average price within one day."""
    n = hours * INTERVALS_PER_HOUR
    window_avg = day_prices.rolling(n, min_periods=n).mean()
    return window_avg.max() - window_avg.min()


def daily_price_measures(
    price,
    start=None,
    end=None,
    threshold=300,
    spread_hours=2,
    min_intervals=MIN_INTERVALS_PER_DAY,
):
    """Daily price measures from 5-min prices, optionally only between start and end each day.

    price: 5-min price series (or one-column frame), $/MWh
    start, end: time window each day, e.g. "00:00", "17:30" (None = whole day)
    threshold: price for pct_above_threshold ($/MWh)
    spread_hours: window length for the best-average spread
    min_intervals: days with fewer readings are dropped (lower it for short windows, None keeps all)
    """
    s = price.squeeze()  # one-column DataFrame -> Series
    if start is not None and end is not None:
        s = s.between_time(start, end, inclusive="left")

    day = s.index.normalize()
    r = s.resample("D")

    # 5-min changes within each day only, so a day's first change isn't from yesterday's last price
    change = s.groupby(day).diff()
    c = change.resample("D")
    abs_c = change.abs().resample("D")

    median = r.median()
    iqr = r.quantile(.75) - r.quantile(.25)

    out = pd.DataFrame({
        # price level and spread
        "max": r.max(),
        "min": r.min(),
        "mean": r.mean(),
        "median": median,
        "std": r.std(),
        "iqr": iqr,
        "log_iqr": np.log(iqr.where(iqr > 0)),  # NaN if a day's IQR is 0
        "range": r.max() - r.min(),
        "spike": r.max() - median,
        f"spread_{spread_hours}h": s.groupby(day).apply(_best_spread, hours=spread_hours),

        # volatility: 5-min price changes
        "mad": abs_c.mean(),  # mean absolute change
        "std_change": c.std(),
        "max_move": abs_c.max(),  # largest absolute change
        "tv": abs_c.sum(),  # total variation: sum of absolute changes

        # share of intervals (%)
        "pct_above_threshold": (s > threshold).resample("D").mean() * 100,
        "pct_negative": (s < 0).resample("D").mean() * 100,

        "n": r.count(),
    })

    if min_intervals is not None:
        incomplete = out["n"] < min_intervals
        if incomplete.any():
            logger.info(f"price measures: dropping {incomplete.sum()} days with <{min_intervals} intervals")
        out = out[~incomplete]

    return out


def plot_std_trend(measures):
    """Monthly trend band of the daily price standard deviation, with SSE events below."""
    trend_band_event_plot(
        df=measures[['std']],
        ylabel='Daily price standard deviation ($/MWh)',
        title='Intraday price variability: monthly spread of the daily price std',
        filename='price_daily_std_trend_events.png',
    )
    
    
def plot_price_median_trend(measures):
    """Monthly trend band of daily energy price median"""
    trend_band_event_plot(
        df=measures['median'],
        ylabel='Daily price median ($/MWh)',
        title='Intraday price median: monthly spread of the daily price median',
        filename='price_daily_median_trend_events.png',
    )



def main():
    pd.set_option("display.max_columns", None)
    
    measures = daily_price_measures(load("price"))
    
    print(measures.head())
    
    # save_clean("price_measures", measures)
    plot_std_trend(measures)
    plot_price_median_trend(measures)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
