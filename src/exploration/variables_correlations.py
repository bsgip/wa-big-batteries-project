"""SOC against an exogenous variable - temperature to start with - as one
scatter panel per battery, with a binned-median line and the correlation
coefficient per panel.

    python -m exploration.variables_correlations
"""

import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_extraction.catalog import load
from data_processing.demand import DEMAND_COLUMN, clean_demand
from data_processing.store import load_clean
from tools.constants import battery_codes
from tools.paths import clean_data_dir, local_plots_dir
from tools.plot_style import UNIT_COLORS, save_figure


def load_temperature() -> pd.Series:
    """5-minutely Perth temperature, converted to the same timezone as the
    SOC index so the two align on a plain index join. The csv is written in
    UTC by exploration/weather.py, so it has to be parsed as dates first -
    tz_convert on a plain string index just raises."""
    path = clean_data_dir / "5min_temperature.csv"
    temp = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")["temperature"]
    return temp.tz_convert("Australia/Perth")


def load_energy_price() -> pd.Series:
    """5-minute energy market clearing price ($/MWh), from the extracted
    dispatchSolution data (not case input - see data_extraction/catalog.py)."""
    return load("price")["energy_price"]


# one colour per at_time, when several times are overlaid in a panel
_TIME_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange", "tab:brown"]


def _grid_shape(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _bin_edges(x: pd.Series, bins: int, method: str) -> np.ndarray:
    """Bin boundaries for the median line.

    "width" splits the x range into equal-width bins - right when x is
    roughly evenly spread (temperature, demand).

    "quantile" puts an equal number of readings in each bin instead. Needed
    for anything heavy-tailed: energy price runs -$1000 to $1100, but half
    the readings sit between $65 and $111, so equal-width bins put almost
    everything in one bin and leave the rest empty."""
    if method == "quantile":
        # duplicate edges where one value repeats across quantiles (price
        # pins at $738 for ~1% of intervals) would make pd.cut raise
        return np.unique(x.quantile(np.linspace(0, 1, bins + 1)).to_numpy())
    if method == "width":
        return np.linspace(x.min(), x.max(), bins + 1)
    raise ValueError(f"unknown bin method {method!r}; expected 'width' or 'quantile'")


def _binned_stat(
    x: pd.Series, y: pd.Series, bins: int, min_count: int, method: str = "width", stat: str = "median"
) -> tuple[np.ndarray, np.ndarray]:
    """Summary of y per x bin. The raw scatter is a 300k-point blob at any
    sensible alpha, so this is the line that actually carries the
    relationship. Bins holding fewer than `min_count` readings are dropped -
    the sparsest few bins otherwise swing the line around on a handful of
    readings.

    `stat` is "median" for a level like SOC, but must be "mean" for power:
    batteries are idle more than half the time, so the median power is
    exactly 0 in most bins (66 of 144 battery-hours) and hides the whole
    charge/discharge response.

    Each point is placed at the median x of its bin rather than the bin
    midpoint, so a wide sparse bin plots where its readings actually are
    instead of at the centre of an interval that may hold nothing."""
    if stat not in ("median", "mean"):
        raise ValueError(f"unknown stat {stat!r}; expected 'median' or 'mean'")

    edges = _bin_edges(x, bins, method)
    binned = pd.cut(x, edges)

    grouped_y = y.groupby(binned, observed=False)
    grouped_x = x.groupby(binned, observed=False)

    values = grouped_y.agg(stat).to_numpy(copy=True)
    values[grouped_y.count().to_numpy() < min_count] = np.nan
    return grouped_x.median().to_numpy(), values


def plot_soc_vs(
    soc_df: pd.DataFrame,
    x_series: pd.Series,
    x_label: str,
    title: str,
    filename: str,
    column_suffix: str = "_soc_pct",
    ylabel: str = "SOC (%)",
    at_time: str | list[str] | None = None,
    bins: int | None = None,
    min_bin_count: int | None = None,
    bin_method: str = "width",
    stat: str = "median",
    xscale: str | None = None,
    xscale_kwargs: dict | None = None,
    zero_line: bool = False,
    max_points: int | None = 20_000,
):
    """Scatter a per-battery quantity against an exogenous variable, one
    panel per battery. Despite the name it isn't SOC-only - pass the power
    frame with `column_suffix="_power_pct"` to plot power instead.

    soc_df   : wide frame indexed by dispatch interval, with a
               `<code><column_suffix>` column per battery
    x_series : Series indexed by timestamp (temperature, price, demand, ...);
               joined on the index, so it only has to overlap in time
    at_time  : "HH:MM", or a list of them, to restrict to that interval of
               each day. Pooling all 288 intervals mixes batteries at every
               point of their daily cycle, which is what flattens these
               plots; fixing the time holds the cycle constant so what's
               left is the response to x. With a list, each time gets its
               own colour and its own summary line in every panel, so the
               times can be compared directly.
    bins     : default 30, or 12 when `at_time` is set - one interval a day
               is ~1,000 readings, not ~300,000, so 30 bins would leave too
               few readings per bin to summarise
    min_bin_count: default 50, or 15 when `at_time` is set, for the same
               reason - at the pooled default an at_time line would be
               blanked entirely
    bin_method: "width" or "quantile" for the summary line - see _bin_edges
    stat     : "median" or "mean" for that line - see _binned_stat
    xscale   : optional matplotlib x scale, e.g. "symlog" for a heavy-tailed
               variable that also goes negative (price), with any extra
               arguments in `xscale_kwargs` (e.g. {"linthresh": 200})
    zero_line: draw y=0, for signed quantities like power
    max_points: subsample per panel for legibility; None keeps everything
    """
    d = soc_df.join(x_series.rename("_x"), how="inner")
    if d.empty:
        raise ValueError("no overlapping intervals - check index alignment and timezones")

    times = [at_time] if isinstance(at_time, str) else list(at_time) if at_time else []
    bins = bins if bins is not None else (12 if times else 30)
    min_bin_count = min_bin_count if min_bin_count is not None else (15 if times else 50)

    interval_of_day = d.index.strftime("%H:%M")
    # one (label, colour, rows) series to draw per panel, per time
    if times:
        subsets = []
        for j, t in enumerate(times):
            sub = d[interval_of_day == t]
            if sub.empty:
                raise ValueError(f"no intervals left at {t} - times must match the 5-min grid, e.g. '17:30'")
            subsets.append((t, _TIME_COLORS[j % len(_TIME_COLORS)], sub))
    else:
        subsets = [(None, None, d)]

    rows, cols = _grid_shape(len(battery_codes))
    fig, axes = plt.subplots(
        rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False, sharex=True, sharey=True
    )

    for i, code in enumerate(battery_codes):
        ax = axes[i // cols][i % cols]
        col = f"{code}{column_suffix}"

        if col not in d:
            ax.set_visible(False)
            continue

        if zero_line:
            ax.axhline(0, color="black", lw=0.8, zorder=1)

        panel_r = float("nan")
        for time_label, time_color, subset in subsets:
            g = subset[["_x", col]].dropna()
            if g.empty:
                continue

            panel_r = g["_x"].corr(g[col]) if len(g) > 2 else float("nan")
            # colour encodes the battery when everything is pooled (the
            # panel title already says which), but the time once we're
            # comparing several - that's the comparison being made
            point_color = time_color if time_label else UNIT_COLORS[code]

            plot_g = g.sample(max_points, random_state=0) if max_points and len(g) > max_points else g
            ax.scatter(
                plot_g["_x"], plot_g[col],
                s=4, alpha=0.3, color=point_color, edgecolors="none",
            )

            bin_x, values = _binned_stat(g["_x"], g[col], bins, min_bin_count, bin_method, stat)
            line_color = time_color if time_label else "black"
            line_label = (
                f"{time_label}  (r={panel_r:.2f}, n={len(g):,})"
                if time_label
                else f"binned {stat} ({bins} bins)"
            )
            ax.plot(bin_x, values, color=line_color, lw=1.8, label=line_label)

        if xscale is not None:
            ax.set_xscale(xscale, **(xscale_kwargs or {}))

        if times:
            # r and n differ per time, so the key has to live in the panel
            # rather than in one shared figure legend
            ax.set_title(code, fontsize=10)
            ax.legend(fontsize="x-small", loc="best")
        else:
            ax.set_title(f"{code}  (r={panel_r:.2f}, n={len(d):,})", fontsize=10)

    for i in range(len(battery_codes), rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    # sharex/sharey hides the interior tick labels, so only the outer edge
    # of the grid gets an axis label - an xlabel on a top-row panel with no
    # tick numbers under it just reads as clutter.
    for ax in axes[-1]:
        ax.set_xlabel(x_label)
    for row in axes:
        row[0].set_ylabel(ylabel)

    if not times:
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.975), fontsize="small"
        )

    subtitle = f" at {', '.join(times)}" if times else ""
    fig.suptitle(f"{title}{subtitle}", y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    save_figure(fig, local_plots_dir / filename)


# 16:00 is before the evening peak, 17:30 its start, 19:00 mid-peak.
PEAK_TIMES = ["16:00", "17:30", "19:00"]

# Price needs both non-default knobs: quantile bins so the summary line
# isn't decided by the $65-111 sliver half the readings sit in, and a symlog
# x-axis so the -$1000 and $1100 tails don't squash that sliver into a
# couple of pixels. linthresh=200 keeps the normal trading range linear and
# compresses only the spike/negative tails.
PRICE_AXIS_KWARGS = {
    "bin_method": "quantile",
    "xscale": "symlog",
    "xscale_kwargs": {"linthresh": 200},
}

# Power is signed and idle-heavy, so it needs the mean rather than the
# median (see _binned_stat) and a zero line to read the sign against.
POWER_KWARGS = {
    "column_suffix": "_power_pct",
    "ylabel": "Power (% of rated, + discharge / - charge)",
    "stat": "mean",
    "zero_line": True,
}


def _plot_pooled_and_by_time(slug: str, **kwargs) -> None:
    """Every comparison gets both framings, from one set of arguments.

    Pooled uses all 288 intervals a day, which mixes batteries at every
    point of their daily cycle and flattens most of these relationships -
    SOC vs demand is r=-0.02 pooled and -0.43 at 16:00. The by-time version
    holds the cycle constant. Keeping the pair in one call means the two can
    never drift apart in binning or axis treatment."""
    plot_soc_vs(filename=f"{slug}.png", **kwargs)
    plot_soc_vs(filename=f"{slug}_by_time.png", at_time=PEAK_TIMES, **kwargs)


def main():
    logging.basicConfig(level=logging.INFO)
    soc = load_clean("soc")
    power = load_clean("power")

    temperature = load_temperature()
    demand = clean_demand(load("demand")[DEMAND_COLUMN])
    price = load_energy_price()

    _plot_pooled_and_by_time(
        "soc_vs_temperature",
        soc_df=soc,
        x_series=temperature,
        x_label="Temperature (°C)",
        title="Battery SOC vs Perth temperature",
    )
    _plot_pooled_and_by_time(
        "soc_vs_demand",
        soc_df=soc,
        x_series=demand,
        x_label="System demand (MW)",
        title="Battery SOC vs demand",
    )
    _plot_pooled_and_by_time(
        "soc_vs_price",
        soc_df=soc,
        x_series=price,
        x_label="Energy price ($/MWh)",
        title="Battery SOC vs energy price",
        **PRICE_AXIS_KWARGS,
    )
    # power is the variable price actually drives: SOC is the stock, power
    # is the flow, and a battery answers a price by charging or discharging.
    # Correlation with price is ~3x the SOC-level one (mean |r| 0.14 vs 0.05).
    _plot_pooled_and_by_time(
        "power_vs_price",
        soc_df=power,
        x_series=price,
        x_label="Energy price ($/MWh)",
        title="Battery power vs energy price",
        **POWER_KWARGS,
        **PRICE_AXIS_KWARGS,
    )


if __name__ == "__main__":
    main()
