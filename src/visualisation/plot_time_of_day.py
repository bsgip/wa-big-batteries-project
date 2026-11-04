"""Time-of-day plots (fan charts and hourly box plots), one panel per (title, series) pair."""

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
    """(rows, cols) of the most square grid that fits n panels."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _time_str_to_hour(t: str) -> float:
    """"HH:MM" as hours after midnight."""
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
    """Discharge colour for a positive median, charge colour for negative, grey for 0 or NaN."""
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
    """The last lookback_days of data (all of it if None)."""
    if lookback_days is None:
        return data
    cutoff = data.index.max() - pd.Timedelta(days=lookback_days)
    return data[data.index > cutoff]


def battery_panels(df: pd.DataFrame, column_suffix: str) -> list[Panel]:
    """Build one (code, series) panel per battery from a wide frame.

    Args:
        df: Wide frame with a <code><column_suffix> column per battery.
        column_suffix: Column suffix, e.g. "_soc_pct".

    Returns:
        One panel per battery in battery_codes; the series is empty where the column is missing.
    """
    panels = []
    for code in battery_codes:
        col = f"{code}{column_suffix}"
        series = df[col] if col in df else pd.Series(dtype="float64", index=df.index)
        panels.append((code, series))
    return panels


def _hour_of_day_frame(series: pd.Series, by_hour_bin: bool) -> pd.DataFrame:
    """Long (hour, value) frame without NaNs; by_hour_bin rounds to the whole hour."""
    hour = series.index.hour if by_hour_bin else series.index.hour + series.index.minute / 60
    return pd.DataFrame({"hour": hour, "value": series.to_numpy()}).dropna()


def _finish_grid(fig, axes, n_panels: int, title: str, filename: str, extra_handles=None):
    """Hide unused panels, add the shared legend and title, and save to filename."""
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



def draw_timeofday_box(
    ax,
    series: pd.Series,
    lookback_days: int | None = None,
    whis: float | tuple[float, float] = (5, 95),
    showfliers: bool = True,
    flierprops: dict | None = None,
    signed_colors: bool = False,
):
    """Draw one box per hour of day for series on ax, with the Peak ESROI shaded.

    Args:
        ax: Axes to draw on.
        series: 5-min values.
        lookback_days: Only use the last N days; None uses all.
        whis: Whisker percentiles.
        showfliers: Draw the points outside the whiskers.
        flierprops: Style of those points; None for the default.
        signed_colors: Colour each box by the sign of its median (for power).
    """
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

    ax.set_xlabel("Hour of day")
    ax.set_xlim(-1, 24)
    # boxplot() leaves a fixed tick formatter behind, which labels ticks
    # by their position in the tick list rather than by value - set_xticks
    # alone would label hours 0,4,8,...,20 as "0,1,2,3,4,5". Passing
    # labels explicitly replaces that formatter.
    ax.set_xticks(_BOX_XTICKS, labels=[str(h) for h in _BOX_XTICKS])


def _signed_legend_keys(signed_colors: bool) -> list:
    """Legend entries for the box colours, if signed_colors is on."""
    if not signed_colors:
        return []
    return [
        Patch(facecolor=DISCHARGE_COLOR, alpha=0.5, label="discharging (injection)"),
        Patch(facecolor=CHARGE_COLOR, alpha=0.5, label="charging (withdrawal)"),
        Patch(facecolor=_NO_NET_BOX_COLOR, alpha=0.5, label="no net direction (median 0)"),
    ]


def plot_timeofday_box(
    series: pd.Series,
    ylabel: str,
    title: str,
    filename: str,
    legend_loc: str = "upper left",
    **box_kwargs,
):
    """Plot a time-of-day box plot for one series.

    Args:
        series: 5-min values.
        ylabel: Y-axis label.
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.
        legend_loc: Legend position.
        **box_kwargs: Passed to draw_timeofday_box.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    draw_timeofday_box(ax, series, **box_kwargs)

    ax.set_ylabel(ylabel)
    ax.set_title(title)

    handles = ax.get_legend_handles_labels()[0] + _signed_legend_keys(box_kwargs.get("signed_colors", False))
    ax.legend(handles=handles, loc=legend_loc, fontsize="small")

    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)


def plot_timeofday_box_grid(
    panels: list[Panel],
    ylabel: str,
    title: str,
    filename: str,
    **box_kwargs,
):
    """Plot a grid of time-of-day box plots, one panel per series.

    Args:
        panels: [(panel title, series), ...], e.g. [("System demand", demand)].
        ylabel: Y-axis label.
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.
        **box_kwargs: Passed to draw_timeofday_box.
    """
    rows, cols = _grid_shape(len(panels))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for ax, (label, series) in zip(axes.flat, panels):
        draw_timeofday_box(ax, series, **box_kwargs)
        ax.set_title(label)
        ax.set_ylabel(ylabel)

    legend_keys = _signed_legend_keys(box_kwargs.get("signed_colors", False))
    _finish_grid(fig, axes, len(panels), title, filename, extra_handles=legend_keys)


def plot_soc_timeofday_box_grid(
    df: pd.DataFrame,
    column_suffix: str,
    ylabel: str,
    title: str,
    filename: str,
    lookback_days: int | None = None,
    **box_kwargs,
):
    """Plot a time-of-day box plot per battery.

    Args:
        df: Wide frame with a <code><column_suffix> column per battery.
        column_suffix: Column suffix, e.g. "_soc_pct".
        ylabel: Y-axis label.
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.
        lookback_days: Only use the last N days; None uses all.
        **box_kwargs: Passed to draw_timeofday_box.
    """
    plot_timeofday_box_grid(
        battery_panels(df, column_suffix), ylabel, title, filename,
        lookback_days=lookback_days, **box_kwargs,
    )
