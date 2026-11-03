"""How accurate are pre-dispatch price forecasts, by how far ahead they were made (lead time)?

Pre-dispatch runs every 30 minutes and forecasts the price of each half-hour up to 48 hours ahead,
starting with the half-hour it runs in. Each forecast half-hour is compared with the actual price:
the mean of the six 5-min prices in that half-hour. Both are labelled by the interval start - this
alignment gives the smallest error at lead time 0 (median absolute error 3.9 $/MWh, vs 7.2 if the
half-hour is taken as ending at its label).
"""

import pandas as pd

from data_extraction.catalog import load
from tools.constants import START_DATE

PERCENTILES = {"p10": 0.10, "p25": 0.25, "median": 0.50, "p75": 0.75, "p90": 0.90}


def price_forecast_errors(start: str = START_DATE) -> pd.DataFrame:
    """One row per (forecast run, forecast half-hour), for half-hours from start on:

    dispatch_interval   the half-hour forecast (its start)
    lead_time_h         hours between the run and that half-hour (0, 0.5, ... 47.5)
    forecast, actual    price ($/MWh) - actual is the mean of the half-hour's 5-min prices
    error               forecast − actual ($/MWh): + forecast too high, − too low

    Half-hours with no actual price are dropped.
    """
    forecast = load("predispatch_price").rename(columns={"energy_price": "forecast"})
    actual = load("price")["energy_price"].resample("30min").mean().rename("actual")

    df = forecast.join(actual, on="dispatch_interval").dropna(subset=["actual"])
    df = df.loc[df["dispatch_interval"] >= pd.Timestamp(start, tz=df["dispatch_interval"].dt.tz)]

    df["lead_time_h"] = (df["dispatch_interval"] - df["primary_dispatch_interval"]).dt.total_seconds() / 3600
    df["error"] = df["forecast"] - df["actual"]
    return df[["dispatch_interval", "lead_time_h", "forecast", "actual", "error"]]


def error_by_lead_time(errors: pd.DataFrame, window: tuple[str, str], lead_range: tuple[float, float]) -> pd.DataFrame:
    """Per lead time, the spread of the forecast error for half-hours starting inside a window of the day.

    errors: from price_forecast_errors
    window: ("HH:MM", "HH:MM") - half-hours starting at or after the start and before the end
    lead_range: (first, last) lead time in hours, both included

    Returns one row per lead time: error percentiles (p10, p25, median, p75, p90), median absolute
    error, and the number of forecasts.
    """
    start, end = (pd.Timedelta(f"{t}:00") for t in window)
    time_of_day = errors["dispatch_interval"] - errors["dispatch_interval"].dt.normalize()
    keep = (time_of_day >= start) & (time_of_day < end) & errors["lead_time_h"].between(*lead_range)

    by_lead = errors.loc[keep].groupby("lead_time_h")["error"]
    out = pd.DataFrame({name: by_lead.quantile(q) for name, q in PERCENTILES.items()})
    out["median_abs"] = by_lead.apply(lambda e: e.abs().median())
    out["forecasts"] = by_lead.size()
    return out
