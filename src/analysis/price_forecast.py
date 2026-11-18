"""How accurate are pre-dispatch price forecasts, by lead time?
Each forecast half-hour is compared with the mean of its six 5-min prices, both labelled by interval start.
"""

import pandas as pd

from tools.constants import START_DATE
from tools.io import load_clean

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
    forecast = load_clean("predispatch_price", start_date=None).reset_index().rename(columns={"energy_price": "forecast"})
    actual = load_clean("price", start_date=None)["energy_price"].resample("30min").mean().rename("actual")

    df = forecast.join(actual, on="dispatch_interval").dropna(subset=["actual"])
    df = df.loc[df["dispatch_interval"] >= pd.Timestamp(start, tz=df["dispatch_interval"].dt.tz)]

    df["lead_time_h"] = (df["dispatch_interval"] - df["primary_dispatch_interval"]).dt.total_seconds() / 3600
    df["error"] = df["forecast"] - df["actual"]
    return df[["dispatch_interval", "lead_time_h", "forecast", "actual", "error"]]


def errors_between(errors: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Rows of price_forecast_errors whose half-hour is in [start, end), e.g. "2024-10-01"."""
    t = errors["dispatch_interval"]
    return errors.loc[(t >= pd.Timestamp(start, tz=t.dt.tz)) & (t < pd.Timestamp(end, tz=t.dt.tz))]


def highest_days(daily: pd.Series, start: str, end: str, share: float) -> pd.Index:
    """Days in [start, end) whose value is in the top share of that period (0.1 = top 10%).

    Args:
        daily: One value per day, e.g. the mean ESROI price.
        start: First day, e.g. "2024-10-01".
        end: Day after the last.
        share: Fraction of days to keep.

    Returns:
        The days kept (midnight timestamps).
    """
    tz = daily.index.tz
    period = daily.loc[(daily.index >= pd.Timestamp(start, tz=tz)) & (daily.index < pd.Timestamp(end, tz=tz))].dropna()
    return period.index[period >= period.quantile(1 - share)]


def errors_on_days(errors: pd.DataFrame, days: pd.Index) -> pd.DataFrame:
    """Rows of price_forecast_errors whose half-hour falls on one of days (midnight timestamps)."""
    return errors.loc[errors["dispatch_interval"].dt.normalize().isin(days)]


def error_by_lead_time(errors: pd.DataFrame, window: tuple[str, str], lead_range: tuple[float, float]) -> pd.DataFrame:
    """Spread of the forecast error by lead time, for half-hours inside a window of the day.

    Args:
        errors: From price_forecast_errors.
        window: ("HH:MM", "HH:MM"); half-hours starting at or after the first and before the second.
        lead_range: (first, last) lead time in hours, both included.

    Returns:
        One row per lead time: p10, p25, median, p75, p90, p10_p90_width, median_abs and
        forecasts (count).
    """
    start, end = (pd.Timedelta(f"{t}:00") for t in window)
    time_of_day = errors["dispatch_interval"] - errors["dispatch_interval"].dt.normalize()
    keep = (time_of_day >= start) & (time_of_day < end) & errors["lead_time_h"].between(*lead_range)

    by_lead = errors.loc[keep].groupby("lead_time_h")["error"]
    out = pd.DataFrame({name: by_lead.quantile(q) for name, q in PERCENTILES.items()})
    out["p10_p90_width"] = out["p90"] - out["p10"]
    out["median_abs"] = by_lead.apply(lambda e: e.abs().median())
    out["forecasts"] = by_lead.size()
    return out
