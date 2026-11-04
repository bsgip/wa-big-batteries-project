"""Per-battery SOC (or power) against an outside variable such as temperature, one scatter panel per battery."""

import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_extraction.catalog import load
from tools.io import load_clean
from tools.constants import battery_codes
from tools.paths import local_plots_dir
from tools.plot_style import UNIT_COLORS, save_figure
from visualisation.plot_variables_correlation import binned_stat


def load_temperature() -> pd.Series:
    """5-min Perth temperature, in Perth time."""
    return load_clean("5min_temperature")["temperature"].tz_convert("Australia/Perth")


def load_energy_price() -> pd.Series:
    """5-min energy price ($/MWh), from the extracted dispatchSolution data."""
    return load("price")["energy_price"]


# one colour per at_time, when several times are overlaid in a panel
_TIME_COLORS = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange", "tab:brown"]


def _grid_shape(n: int) -> tuple[int, int]:
    """(rows, cols) of the most square grid that fits n panels."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols



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
    """Scatter each battery's column against an outside variable, one panel per battery.

    Works for any per-battery quantity, e.g. power with column_suffix="_power_pct".

    Args:
        soc_df: Wide frame with a <code><column_suffix> column per battery.
        x_series: Variable to plot against (temperature, price, ...), joined on the index.
        x_label: X-axis label.
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.
        column_suffix: Column suffix to plot.
        ylabel: Y-axis label.
        at_time: "HH:MM" or a list of them, to use only those times of day (one colour and
            summary line per time); None uses every interval.
        bins: Bins for the summary line; defaults to 30, or 12 with at_time.
        min_bin_count: Bins with fewer readings are dropped; defaults to 50, or 15 with at_time.
        bin_method: "width" or "quantile"; see bin_edges.
        stat: "median" or "mean"; see binned_stat.
        xscale: Matplotlib x scale, e.g. "symlog"; extra arguments in xscale_kwargs.
        xscale_kwargs: Arguments for xscale, e.g. {"linthresh": 200}.
        zero_line: Draw y = 0.
        max_points: Points drawn per panel; None draws all.
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

            bin_x, values = binned_stat(g["_x"], g[col], bins, min_bin_count, bin_method, stat)
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
# median (see binned_stat) and a zero line to read the sign against.
POWER_KWARGS = {
    "column_suffix": "_power_pct",
    "ylabel": "Power (% of rated, + discharge / - charge)",
    "stat": "mean",
    "zero_line": True,
}


def _plot_pooled_and_by_time(slug: str, **kwargs) -> None:
    """Call plot_soc_vs twice: pooled over every interval, and at fixed times of day."""
    plot_soc_vs(filename=f"{slug}.png", **kwargs)
    plot_soc_vs(filename=f"{slug}_by_time.png", at_time=PEAK_TIMES, **kwargs)


def main():
    logging.basicConfig(level=logging.INFO)
    soc = load_clean("soc")
    power = load_clean("power")

    temperature = load_temperature()
    demand = load_clean("demand")["demand"]
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
