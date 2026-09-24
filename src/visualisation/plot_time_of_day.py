"""Time-of-day plots: what a typical day looks like for a series, as a fan
chart (median + percentile bands) or a box plot per hour.

Both work on a list of `(panel_title, series)` pairs, one panel per entry,
so they're not tied to per-battery data - a single system-wide series like
demand is just a one-panel list. The plot_soc_* wrappers build that list
from the `<code><suffix>` columns of a wide battery frame.
"""

import math

import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.patches import Patch

from tools.constants import PEAK_ESROI_END, PEAK_ESROI_START, battery_codes
from tools.paths import local_plots_dir
from tools.plot_style import DISCHARGE_COLOR, CHARGE_COLOR, save_figure

# (panel title, values indexed by timestamp)
Panel = tuple[str, pd.Series]


def _grid_shape(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _time_str_to_hour(t: str) -> float:
    hour, minute = t.split(":")
    return int(hour) + int(minute) / 60


# x-axis here is hour-of-day (float 0-24), not a real timestamp like in
# plot_soc.py's per-day plots, so the Peak ESROI window is expressed the
# same way rather than as a pd.Timestamp.
_PEAK_START_HOUR = _time_str_to_hour(PEAK_ESROI_START)
_PEAK_END_HOUR = _time_str_to_hour(PEAK_ESROI_END)

_BOX_XTICKS = range(0, 24, 4)

# A median of exactly 0 does NOT mean the battery is idle: for a signed
# quantity it means charge and discharge readings balance around zero, so
# the middle one lands on 0. Across the 66 such battery-hours only ~22% of
# readings are actually zero and ~47% are active (|power| > 1% of rated) -
# KWINANA_ESR1 at 08:00 is 73% active and still lands here. It gets its own
# colour rather than being rounded into charging or discharging, because
# the median genuinely can't say which way the hour went.
_NO_NET_BOX_COLOR = "lightgrey"


def _signed_box_color(median: float) -> str:
    if pd.isna(median) or median == 0:
        return _NO_NET_BOX_COLOR
    return DISCHARGE_COLOR if median > 0 else CHARGE_COLOR

# Fliers here are the 10% of readings outside the 5-95 whiskers - tens of
# thousands of them per panel, stacked on a single x position per hour.
# Small, near-transparent dots let that pile read as a density gradient
# (where the tail thins out) rather than one opaque blob.
_DENSE_FLIERPROPS = {
    "marker": ".",
    "markersize": 3,
    "markerfacecolor": "black",
    "markeredgecolor": "none",
    "alpha": 0.06,
}


def _lookback_slice(data: pd.Series | pd.DataFrame, lookback_days: int | None):
    if lookback_days is None:
        return data
    cutoff = data.index.max() - pd.Timedelta(days=lookback_days)
    return data[data.index > cutoff]


def battery_panels(df: pd.DataFrame, column_suffix: str) -> list[Panel]:
    """One panel per battery, reading the `<code><column_suffix>` column of a
    wide frame. Batteries with no such column still get a (blank) panel, so
    the grid layout stays the same whichever batteries are present."""
    panels = []
    for code in battery_codes:
        col = f"{code}{column_suffix}"
        series = df[col] if col in df else pd.Series(dtype="float64", index=df.index)
        panels.append((code, series))
    return panels


def _hour_of_day_frame(series: pd.Series, by_hour_bin: bool) -> pd.DataFrame:
    """Long (hour, value) frame with NaNs dropped. `by_hour_bin` rounds to
    the whole hour (box plots, 24 groups); otherwise the full 5-min-of-day
    resolution is kept (fan charts)."""
    hour = series.index.hour if by_hour_bin else series.index.hour + series.index.minute / 60
    return pd.DataFrame({"hour": hour, "value": series.to_numpy()}).dropna()


def _finish_grid(fig, axes, n_panels: int, title: str, filename: str, extra_handles=None):
    for i in range(n_panels, axes.size):
        axes.flat[i].set_visible(False)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    for handle in extra_handles or []:
        handles.append(handle)
        labels.append(handle.get_label())
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.975), fontsize="small")

    fig.suptitle(title, y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    save_figure(fig, local_plots_dir / filename)


def plot_timeofday_fan_grid(
    panels: list[Panel],
    ylabel: str,
    title: str,
    filename: str,
    lookback_days: int | None = None,
):
    """Grid of one time-of-day fan chart per panel: median line with 25-75%,
    5-95%, and min-max (0-100%) shaded bands, binned to 5-min-of-day
    resolution across the last `lookback_days` of data (None uses
    everything)."""
    rows, cols = _grid_shape(len(panels))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    quantiles = [0, 0.05, 0.25, 0.5, 0.75, 0.95, 1]

    for ax, (label, series) in zip(axes.flat, panels):
        ax.axvspan(
            _PEAK_START_HOUR,
            _PEAK_END_HOUR,
            color="gold",
            alpha=0.2,
            zorder=0,
            label=f"Peak ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})",
        )

        long_df = _hour_of_day_frame(_lookback_slice(series, lookback_days), by_hour_bin=False)
        if not long_df.empty:
            q = long_df.groupby("hour")["value"].quantile(quantiles).unstack()
            ax.fill_between(q.index, q[0], q[1], color="tab:blue", alpha=0.15, label="min-max")
            ax.fill_between(q.index, q[0.05], q[0.95], color="tab:blue", alpha=0.20, label="5th-95th percentile")
            ax.fill_between(q.index, q[0.25], q[0.75], color="tab:blue", alpha=0.35, label="25th-75th percentile")
            ax.plot(q.index, q[0.5], color="tab:blue", linewidth=1.9, label="median")

        ax.set_title(label)
        ax.set_xlabel("time of day (hour)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))

    _finish_grid(fig, axes, len(panels), title, filename)


def plot_timeofday_box_grid(
    panels: list[Panel],
    ylabel: str,
    title: str,
    filename: str,
    lookback_days: int | None = None,
    whis: float | tuple[float, float] = (5, 95),
    showfliers: bool = True,
    flierprops: dict | None = None,
    signed_colors: bool = False,
):
    """Grid of one time-of-day box plot per panel, one box per hour, over the
    last `lookback_days` of data (None uses everything).

    `whis` defaults to the 5th/95th percentiles rather than matplotlib's
    1.5*IQR. SOC's IQR is wide enough (25-37 points) that the 1.5*IQR fence
    falls below 0% for most batteries, which lets a single reading out of
    ~10,000 in an hourly box drag the whisker to zero - the whisker ends up
    reporting the rarest value in the box rather than its range. Percentile
    whiskers also match the fan chart's 5-95 band, so the two plots can be
    read against each other.

    Everything outside the whiskers is drawn as a flier, so no reading is
    left out of the plot. With percentile whiskers that's 10% of every box
    (~1,200 points per hour), which matplotlib's default 6pt opaque circles
    would render as a solid black bar - _DENSE_FLIERPROPS shrinks them to
    near-transparent dots so the tail reads as a density gradient instead.
    Pass `flierprops` to override.

    `signed_colors` fills each box by the sign of its median instead of one
    flat blue - for signed quantities like power, where the sign is the
    whole point (discharge/injection vs charge/withdrawal). It also draws a
    zero line, since that's the axis the colour splits on."""
    rows, cols = _grid_shape(len(panels))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for ax, (label, series) in zip(axes.flat, panels):
        ax.axvspan(
            _PEAK_START_HOUR,
            _PEAK_END_HOUR,
            color="gold",
            alpha=0.2,
            zorder=0,
            label=f"Peak ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})",
        )

        if signed_colors:
            ax.axhline(0, color="black", lw=0.8, zorder=1)

        long_df = _hour_of_day_frame(_lookback_slice(series, lookback_days), by_hour_bin=True)
        if not long_df.empty:
            groups = [long_df.loc[long_df["hour"] == h, "value"] for h in range(24)]
            drawn = ax.boxplot(
                groups,
                positions=range(24),
                widths=0.6,
                whis=whis,
                showfliers=showfliers,
                flierprops=flierprops if flierprops is not None else _DENSE_FLIERPROPS,
                patch_artist=True,
                boxprops={"facecolor": "tab:blue", "alpha": 0.5},
                medianprops={"color": "black"},
            )

            if signed_colors:
                for patch, group in zip(drawn["boxes"], groups, strict=True):
                    patch.set_facecolor(_signed_box_color(group.median()))

        ax.set_title(label)
        ax.set_xlabel("time of day (hour)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-1, 24)
        # boxplot() leaves a fixed tick formatter behind, which labels ticks
        # by their position in the tick list rather than by value - set_xticks
        # alone would label hours 0,4,8,...,20 as "0,1,2,3,4,5". Passing
        # labels explicitly replaces that formatter.
        ax.set_xticks(_BOX_XTICKS, labels=[str(h) for h in _BOX_XTICKS])

    legend_keys = None
    if signed_colors:
        legend_keys = [
            Patch(facecolor=_DISCHARGE_BOX_COLOR, alpha=0.5, label="discharging (injection)"),
            Patch(facecolor=_CHARGE_BOX_COLOR, alpha=0.5, label="charging (withdrawal)"),
            Patch(facecolor=_NO_NET_BOX_COLOR, alpha=0.5, label="no net direction (median 0)"),
        ]

    _finish_grid(fig, axes, len(panels), title, filename, extra_handles=legend_keys)


def plot_soc_timeofday_fan_grid(
    df: pd.DataFrame,
    column_suffix: str,
    ylabel: str,
    title: str,
    filename: str,
    lookback_days: int | None = None,
):
    """One fan chart per battery, from the `<code><column_suffix>` columns."""
    plot_timeofday_fan_grid(battery_panels(df, column_suffix), ylabel, title, filename, lookback_days)


def plot_soc_timeofday_box_grid(
    df: pd.DataFrame,
    column_suffix: str,
    ylabel: str,
    title: str,
    filename: str,
    lookback_days: int | None = None,
    **box_kwargs,
):
    """One box plot per battery, from the `<code><column_suffix>` columns."""
    plot_timeofday_box_grid(
        battery_panels(df, column_suffix), ylabel, title, filename, lookback_days, **box_kwargs
    )
