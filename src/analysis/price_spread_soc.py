"""How well does the daily price spread, actual or forecast, explain each battery's SOC at 17:30?

    python -m analysis.price_spread_soc
"""

import math

import pandas as pd

from tools.constants import PEAK_ESROI_START, START_DATE, battery_codes
from tools.io import load_clean
from visualisation.plot_price_spread_soc import plot_r2_heatmap, plot_spread_soc_grid
from visualisation.plot_variables_correlation import plot_variables_correlation_by_season

LEAD_TIMES_H: list[float] = [1, 2, 4, 8, 24]
MIN_TRADING_INTERVALS = 40  # of a day's 48; part days give a misleading spread
MIN_WINDOW_SHARE = 0.8  # a window needs this share of its trading intervals, as 40 of 48 for a day

# {slug: (label, start, end)}, in the order they happen before 17:30.
# A window whose start is after its end runs over midnight and belongs to the day it ends on.
TIME_WINDOWS = {
    "midnight": ("Midnight", "21:00", "06:00"),
    "morning": ("Morning", "06:00", "09:00"),
    "midday": ("Midday", "09:00", "15:00"),
    "pre_esroi": ("Pre-ESROI", "15:00", "17:00"),
    "peak": ("Peak", "17:00", "21:00"),
}


def hours_after_midnight(hhmm: str) -> float:
    """"HH:MM" as hours after midnight."""
    hour, minute = hhmm.split(":")
    return int(hour) + int(minute) / 60


def _spread_by_day(price: pd.Series, day: pd.Index, min_intervals: int) -> pd.Series:
    """Max minus min of the prices in each day group; NaN for days with too few trading intervals."""
    g = price.groupby(day)
    spread = g.max() - g.min()
    return spread.where(g.count() >= min_intervals)


def _forecast_by_lead(lead_times_h: list[float], start: str = START_DATE) -> pd.DataFrame:
    """Pre-dispatch price forecasts, one row per trading interval, one column per lead time ($/MWh)."""
    forecast = load_clean(
        "predispatch_price", start_date=None, columns=["energy_price", "dispatch_interval"]
    ).reset_index()
    forecast["lead_time_h"] = (
        forecast["dispatch_interval"] - forecast["primary_dispatch_interval"]
    ).dt.total_seconds() / 3600
    forecast = forecast.loc[forecast["lead_time_h"].isin(lead_times_h)]

    by_lead = forecast.pivot(index="dispatch_interval", columns="lead_time_h", values="energy_price")
    return by_lead.loc[by_lead.index >= pd.Timestamp(start, tz=by_lead.index.tz)]


def actual_daily_spread(start: str = START_DATE) -> pd.Series:
    """Daily spread of the actual price, each trading interval being the mean of its six 5-min prices.

    Args:
        start: First day to keep, e.g. "2024-12-01".

    Returns:
        Spread ($/MWh) per day (midnight timestamps).
    """
    price = load_clean("price", start_date=start)["energy_price"].resample("30min").mean().dropna()
    return _spread_by_day(price, price.index.normalize(), MIN_TRADING_INTERVALS)


def forecast_daily_spread(lead_times_h: list[float] = LEAD_TIMES_H, start: str = START_DATE) -> pd.DataFrame:
    """Daily spread of the pre-dispatch price forecast, at fixed lead times.

    Each trading interval's price is the one forecast lead_time_h hours before it, so the
    spread at 1h is built from 48 forecasts each made an hour ahead of its own interval.

    Args:
        lead_times_h: Lead times in hours (multiples of 0.5).
        start: First day to keep, e.g. "2024-12-01".

    Returns:
        One row per day (midnight timestamps), one column per lead time: forecast_spread_<n>h ($/MWh).
    """
    by_lead = _forecast_by_lead(lead_times_h, start)
    out = {}
    for lead in lead_times_h:
        price = by_lead[lead].dropna()
        out[f"forecast_spread_{lead:g}h"] = _spread_by_day(price, price.index.normalize(), MIN_TRADING_INTERVALS)
    return pd.DataFrame(out)


def window_forecast_spread(
    lead_time_h: float = 4, windows: dict = TIME_WINDOWS, start: str = START_DATE
) -> pd.DataFrame:
    """Spread of the pre-dispatch price forecast within each time window of the day, at one lead time.

    Args:
        lead_time_h: Lead time in hours (a multiple of 0.5).
        windows: {slug: (label, "HH:MM" start, "HH:MM" end)}; see TIME_WINDOWS.
        start: First day to keep, e.g. "2024-12-01".

    Returns:
        One row per day (midnight timestamps), one column per window:
        forecast_spread_<n>h_<slug> ($/MWh). A window over midnight is on the day it ends.
    """
    price = _forecast_by_lead([lead_time_h], start)[lead_time_h].dropna()

    out = {}
    for slug, (_, w_start, w_end) in windows.items():
        p = price.between_time(w_start, w_end, inclusive="left")
        start_h, end_h = hours_after_midnight(w_start), hours_after_midnight(w_end)
        if start_h >= end_h:  # over midnight: shift so the evening part lands on the next day
            day = (p.index + pd.Timedelta(hours=24 - start_h)).normalize()
        else:
            day = p.index.normalize()

        n_intervals = ((end_h - start_h) % 24) * 2  # trading intervals in the window
        min_intervals = math.ceil(MIN_WINDOW_SHARE * n_intervals)
        out[f"forecast_spread_{lead_time_h:g}h_{slug}"] = _spread_by_day(p, day, min_intervals)
    return pd.DataFrame(out)


def daily_spread_soc(at: str = PEAK_ESROI_START, lead_times_h: list[float] = LEAD_TIMES_H) -> pd.DataFrame:
    """Each battery's SOC at a time of day, beside that day's actual and forecast price spread.

    The spread covers the whole day (00:00-24:00), including the hours after `at`.

    Args:
        at: Time of day the SOC is taken at, e.g. "17:30".
        lead_times_h: Forecast lead times in hours.

    Returns:
        One row per day: <code>_soc_pct per battery, actual_spread and forecast_spread_<n>h.
    """
    soc = load_clean("soc")[[f"{code}_soc_pct" for code in battery_codes]].at_time(at)
    soc.index = soc.index.normalize()

    spreads = forecast_daily_spread(lead_times_h)
    spreads.insert(0, "actual_spread", actual_daily_spread())
    return soc.join(spreads, how="inner")


def spread_soc_r2(daily: pd.DataFrame) -> pd.DataFrame:
    """r^2 between each spread column and each battery's SOC.

    Each r^2 uses the days where both values exist, so batteries commissioned later use fewer days.

    Args:
        daily: From daily_spread_soc, optionally with more spread columns joined on.

    Returns:
        One row per battery, one column per spread column: Pearson r^2.
    """
    soc_cols = [f"{code}_soc_pct" for code in battery_codes]
    spread_cols = [c for c in daily.columns if c not in soc_cols]
    r = daily.corr().loc[soc_cols, spread_cols]  # pairwise: NaNs dropped per pair
    r.index = battery_codes
    r.index.name = "battery"
    return r ** 2


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)
    daily = daily_spread_soc()
    r2 = spread_soc_r2(daily)
    print(r2.round(3))

    plot_r2_heatmap(r2, filename="correlations/price_spread_soc/r2_heatmap.png")
    for code in battery_codes:
        plot_spread_soc_grid(daily, code, filename=f"correlations/price_spread_soc/{code}.png")

    # 4h forecast spread by season group
    for code in battery_codes:
        plot_variables_correlation_by_season(
            daily, "forecast_spread_4h", f"{code}_soc_pct",
            x_label="Daily price spread, forecast 4h ahead ($/MWh)",
            y_label=f"SOC (%) at {PEAK_ESROI_START}",
            title=f"{code}: 4h forecast price spread vs SOC at {PEAK_ESROI_START}, by season",
            filename=f"correlations/price_spread_soc/{code}_forecast_4h_by_season.png",
        )

    # 4h forecast spread by time window
    soc_cols = [f"{code}_soc_pct" for code in battery_codes]
    by_window = daily[soc_cols + ["forecast_spread_4h"]].join(window_forecast_spread(4))
    print(spread_soc_r2(by_window).round(3))

    panels = {"forecast_spread_4h": "Whole day (00:00-24:00)"}
    for slug, (label, start, end) in TIME_WINDOWS.items():
        panels[f"forecast_spread_4h_{slug}"] = f"{label} ({start}-{end})"
    panels["forecast_spread_4h_midnight"] = "Midnight (21:00 day before - 06:00)"

    for code in battery_codes:
        plot_spread_soc_grid(
            by_window, code,
            filename=f"correlations/price_spread_soc/{code}_forecast_4h_by_window.png",
            panels=panels,
            x_label="Price spread in the window, forecast 4h ahead ($/MWh)",
            title=f"{code}: 4h forecast price spread by time window vs SOC at {PEAK_ESROI_START}",
        )


if __name__ == "__main__":
    main()
