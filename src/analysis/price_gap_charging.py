"""Does a battery charge more in a window when the evening peak is forecast to be dearer than that window?

    python -m analysis.price_gap_charging
"""

import math

import pandas as pd

from analysis.energy import fleet_net_energy_pct
from analysis.price_spread_soc import MIN_WINDOW_SHARE, TIME_WINDOWS, hours_after_midnight
from tools.constants import START_DATE, battery_codes, commissioning_dates
from tools.io import load_clean
from visualisation.plot_price_spread_soc import plot_scatter_grid

PEAK = "peak"
CHARGING_WINDOWS = {slug: w for slug, w in TIME_WINDOWS.items() if slug != PEAK}


def _window_hours(start: str, end: str) -> tuple[float, float]:
    """(start hour, length in hours) of a "HH:MM" window; the length wraps past midnight."""
    start_h = hours_after_midnight(start)
    return start_h, (hours_after_midnight(end) - start_h) % 24


def _hhmm(hours: float) -> str:
    """Hours after midnight as "HH:MM"."""
    return f"{int(hours):02d}:{round(hours % 1 * 60):02d}"


def _mean_per_run(forecast: pd.DataFrame, rows: pd.Series, n_intervals: float) -> pd.Series:
    """Mean forecast price of the selected rows per run; NaN for runs covering too few trading intervals."""
    g = forecast.loc[rows].groupby("primary_dispatch_interval")["energy_price"]
    return g.mean().where(g.count() >= math.ceil(MIN_WINDOW_SHARE * n_intervals))


def forecast_peak_gap(
    windows: dict = CHARGING_WINDOWS, peak: tuple = TIME_WINDOWS[PEAK], start: str = START_DATE
) -> pd.DataFrame:
    """Forecast peak price minus forecast window price, from the run issued at each window's start.

    Both means come from the same pre-dispatch run, so the gap is what the battery could see
    when the window began.

    Args:
        windows: {slug: (label, "HH:MM" start, "HH:MM" end)}; see TIME_WINDOWS.
        peak: (label, "HH:MM" start, "HH:MM" end) of the peak.
        start: First day to keep, e.g. "2024-12-01".

    Returns:
        One row per day (midnight timestamps of the peak's day, so a window over midnight is on
        the day it ends), one column per window: gap_<slug> ($/MWh). NaN where the run is
        missing or covers under MIN_WINDOW_SHARE of either window.
    """
    forecast = load_clean(
        "predispatch_price", start_date=None, columns=["energy_price", "dispatch_interval"]
    ).reset_index()
    run = forecast["primary_dispatch_interval"]
    ahead_h = (forecast["dispatch_interval"] - run).dt.total_seconds() / 3600
    run_time = run.dt.strftime("%H:%M")
    peak_start_h, peak_length = _window_hours(peak[1], peak[2])

    out = {}
    for slug, (_, w_start, w_end) in windows.items():
        start_h, length = _window_hours(w_start, w_end)
        to_peak = (peak_start_h - start_h) % 24  # hours from the run to that day's peak start
        from_run = run_time == w_start
        in_window = from_run & (ahead_h < length)
        in_peak = from_run & (ahead_h >= to_peak) & (ahead_h < to_peak + peak_length)

        gap = _mean_per_run(forecast, in_peak, peak_length * 2) - _mean_per_run(forecast, in_window, length * 2)
        gap.index = (gap.index + pd.Timedelta(hours=to_peak)).normalize()
        out[f"gap_{slug}"] = gap

    df = pd.DataFrame(out)
    return df.loc[df.index >= pd.Timestamp(start, tz=df.index.tz)]


def _in_one_day(data: pd.DataFrame | pd.Series, w_start: str, w_end: str):
    """(data, start, end) with a window over midnight shifted to sit in the day it ends; others unchanged."""
    start_h, length = _window_hours(w_start, w_end)
    if start_h + length <= 24:
        return data, w_start, w_end
    return data.shift(freq=pd.Timedelta(hours=24 - start_h)), "00:00", _hhmm(length)


def window_net_charge_pct(
    power: pd.DataFrame,
    soc: pd.DataFrame,
    commissioning: dict[str, pd.Timestamp],
    code: str,
    windows: dict = CHARGING_WINDOWS,
) -> pd.DataFrame:
    """Net energy one battery charged in each window per day, as % of its capacity.

    Window-days where the battery's clean SOC is NaN throughout are dropped: those are outages,
    where power reads 0 but the battery isn't running.

    Args:
        power: Battery power (MW), one column per battery (+ discharging, - charging).
        soc: Clean SOC, one column per battery, from load_clean("soc").
        commissioning: {code: commissioning timestamp}, from constants.commissioning_dates.
        code: Battery code.
        windows: {slug: (label, "HH:MM" start, "HH:MM" end)}; see TIME_WINDOWS.

    Returns:
        One row per day, one column per window: net_charge_<slug> (%, + = charged). A window over
        midnight is on the day it ends.
    """
    out = {}
    for slug, (_, w_start, w_end) in windows.items():
        p, start, end = _in_one_day(power, w_start, w_end)
        net = fleet_net_energy_pct(p, commissioning, start, end, codes=[code])

        known, _, _ = _in_one_day(soc[code].notna(), w_start, w_end)
        running = known.between_time(start, end, inclusive="left").resample("D").sum() > 0
        out[f"net_charge_{slug}"] = (-net).where(running)
    return pd.DataFrame(out)


def main():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 250)

    soc = load_clean("soc")
    power = load_clean("power")[battery_codes]
    commissioning = commissioning_dates(soc, power)
    gap = forecast_peak_gap()

    peak_label, peak_start, peak_end = TIME_WINDOWS[PEAK]
    panels = {
        f"{label} ({start}-{end})": (f"gap_{slug}", f"net_charge_{slug}")
        for slug, (label, start, end) in CHARGING_WINDOWS.items()
    }
    panels = {k.replace("Midnight (21:00", "Midnight (21:00 day before"): v for k, v in panels.items()}

    r = {}
    for code in battery_codes:
        df = gap.join(window_net_charge_pct(power, soc, commissioning, code), how="inner")
        r[code] = {slug: df[f"gap_{slug}"].corr(df[f"net_charge_{slug}"]) for slug in CHARGING_WINDOWS}
        plot_scatter_grid(
            df, panels,
            x_label=f"Forecast {peak_label.lower()} price minus window price ($/MWh)",
            y_label="Net charge in the window (% of capacity)",
            title=f"{code}: net charge vs forecast price gap to the {peak_label.lower()} ({peak_start}-{peak_end})",
            filename=f"correlations/price_gap_charging/{code}.png",
            n_cols=2,
        )

    r = pd.DataFrame(r).T
    print("r\n", r.round(2))
    print("r^2\n", (r ** 2).round(3))


if __name__ == "__main__":
    main()
