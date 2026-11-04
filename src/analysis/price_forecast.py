"""How accurate are pre-dispatch price forecasts, by lead time?
Each forecast half-hour is compared with the mean of its six 5-min prices, both labelled by interval start.
"""

import pandas as pd

from data_extraction.catalog import load
from tools.constants import START_DATE

PERCENTILES = {"p10": 0.10, "p25": 0.25, "median": 0.50, "p75": 0.75, "p90": 0.90}


def price_forecast_errors(start: str = START_DATE) -> pd.DataFrame:
    """Error of every pre-dispatch price forecast against the actual price.

    The actual price is the mean of the half-hour's six 5-min prices. Half-hours with no
    actual price are dropped.

    Args:
        start: First half-hour to keep, e.g. "2024-12-01".

    Returns:
        One row per (forecast run, half-hour): dispatch_interval, lead_time_h, forecast,
        actual and error (forecast - actual, $/MWh).
    """
    forecast = load("predispatch_price").rename(columns={"energy_price": "forecast"})
    actual = load("price")["energy_price"].resample("30min").mean().rename("actual")

    df = forecast.join(actual, on="dispatch_interval").dropna(subset=["actual"])
    df = df.loc[df["dispatch_interval"] >= pd.Timestamp(start, tz=df["dispatch_interval"].dt.tz)]

    df["lead_time_h"] = (df["dispatch_interval"] - df["primary_dispatch_interval"]).dt.total_seconds() / 3600
    df["error"] = df["forecast"] - df["actual"]
    return df[["dispatch_interval", "lead_time_h", "forecast", "actual", "error"]]


def error_by_lead_time(errors: pd.DataFrame, window: tuple[str, str], lead_range: tuple[float, float]) -> pd.DataFrame:
    """Spread of the forecast error by lead time, for half-hours inside a window of the day.

    Args:
        errors: From price_forecast_errors.
        window: ("HH:MM", "HH:MM"); half-hours starting at or after the first and before the second.
        lead_range: (first, last) lead time in hours, both included.

    Returns:
        One row per lead time: p10, p25, median, p75, p90, median_abs and forecasts (count).
    """
    start, end = (pd.Timedelta(f"{t}:00") for t in window)
    time_of_day = errors["dispatch_interval"] - errors["dispatch_interval"].dt.normalize()
    keep = (time_of_day >= start) & (time_of_day < end) & errors["lead_time_h"].between(*lead_range)

    by_lead = errors.loc[keep].groupby("lead_time_h")["error"]
    out = pd.DataFrame({name: by_lead.quantile(q) for name, q in PERCENTILES.items()})
    out["median_abs"] = by_lead.apply(lambda e: e.abs().median())
    out["forecasts"] = by_lead.size()
    return out
