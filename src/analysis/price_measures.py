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
from tools.paths import local_plots_dir
from tools.plot_style import save_figure
from visualisation.plot_trend_band import mark_period_medians, trend_band_plot

logger = logging.getLogger(__name__)

INTERVALS_PER_HOUR = 12  # 5-min prices

# A day needs this many of its 288 5-min intervals to count; part days give misleading stats.
MIN_INTERVALS_PER_DAY = 240

# year-on-year comparison: the same months each year (Dec-Sep, the full months every year has),
# so seasonality doesn't read as change
SAME_MONTHS = [("2023-12-01", "2024-09-30"), ("2024-12-01", "2025-09-30"), ("2025-12-01", "2026-09-30")]


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
    """Daily price statistics from 5-min prices.

    Args:
        price: 5-min energy price ($/MWh), as a Series or one-column frame.
        start: Start of the time window used each day, e.g. "00:00"; None for the whole day.
        end: End of that window (exclusive).
        threshold: Price ($/MWh) for pct_above_threshold.
        spread_hours: Window length (hours) for spread_<n>h.
        min_intervals: Days with fewer readings are dropped; None keeps every day.

    Returns:
        One row per day: price level and spread (max, min, mean, median, std, iqr, log_iqr,
        range, spike, spread_<n>h), 5-min change volatility (mad, std_change, max_move, tv),
        pct_above_threshold, pct_negative and n.
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
    """Plot the monthly trend band of the daily price standard deviation.

    Args:
        measures: Daily measures, from daily_price_measures.
    """
    trend_band_plot(
        df=measures[['std']],
        ylabel='Daily price standard deviation ($/MWh)',
        title='Daily price variability: monthly spread of the daily price std',
        filename='price_daily_std_trend.png',
    )


def plot_price_median_trend(measures):
    """Plot the monthly trend band of the daily median price, with SAME_MONTHS medians marked.

    Args:
        measures: Daily measures, from daily_price_measures.
    """
    filename = 'price_daily_median_trend.png'
    fig, ax, _ = trend_band_plot(
        df=measures['median'],
        ylabel='Daily price median ($/MWh)',
        title='Daily price median: monthly spread of the daily price median',
        filename=filename,
    )
    legend = ax.get_legend()  # mark_period_medians sets its own legend - keep the one above the plot too
    mark_period_medians(ax, measures['median'], SAME_MONTHS, value_format="{:,.0f} $/MWh")
    ax.add_artist(legend)
    save_figure(fig, local_plots_dir / filename)  # over the same file



def main():
    pd.set_option("display.max_columns", None)
    
    measures = daily_price_measures(load("price")["energy_price"])
    
    print(measures.head())
    
    # save_clean("price_measures", measures)
    plot_std_trend(measures)
    plot_price_median_trend(measures)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
