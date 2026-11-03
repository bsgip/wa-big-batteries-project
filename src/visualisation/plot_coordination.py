import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from tools.constants import PEAK_ESROI_END, PEAK_ESROI_START
from tools.paths import local_plots_dir
from tools.plot_style import save_figure

CANCELLING_YLABEL = "Flow cancelling out\n(% of all battery flow)"


def _hours(time_str):
    hour, minute = time_str.split(":")
    return int(hour) + int(minute) / 60


def _stacked_bars(ax, x, table, colors, width):
    """One stacked bar per row of table, one segment per column."""
    bottom = pd.Series(0.0, index=table.index)
    for color, column in zip(colors, table.columns, strict=True):
        ax.bar(x, table[column], bottom=bottom, width=width, color=color, alpha=0.85,
               edgecolor="white", lw=1, label=column)
        bottom += table[column]


def plot_cancelling_monthly(monthly_pct, colors, title, filename):
    """Stacked monthly bars: share of battery flow cancelling out, one segment per column
    (e.g. within each owner, across owners).

    monthly_pct: from fleet_coordination.share_cancelling_out, indexed by month start
    colors: one colour per column, in column order

    Returns (fig, ax).
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    _stacked_bars(ax, monthly_pct.index, monthly_pct, colors, width=20)

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_ylabel(CANCELLING_YLABEL)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(monthly_pct.columns), fontsize=9,
              frameon=False)
    ax.set_title(title, pad=28)  # room for the legend row

    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)

    return fig, ax


def plot_cancelling_by_hour(hourly_pct, colors, title, filename):
    """Stacked bars, one per hour of the day: share of battery flow cancelling out, one segment
    per column (e.g. within each owner, across owners). ESROI shaded behind.

    hourly_pct: from fleet_coordination.share_cancelling_out, indexed by hour (0-23)
    colors: one colour per column, in column order

    Returns (fig, ax).
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    _stacked_bars(ax, hourly_pct.index + 0.5, hourly_pct, colors, width=0.85)  # bar h spans h:00-h+1:00

    ax.axvspan(_hours(PEAK_ESROI_START), _hours(PEAK_ESROI_END), color="gold", alpha=0.25, lw=0, zorder=0)
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(CANCELLING_YLABEL)
    ax.grid(axis="y", alpha=0.3)

    handles, _ = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="gold", alpha=0.25, label=f"ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), fontsize=9,
              frameon=False)
    ax.set_title(title, pad=28)  # room for the legend row

    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)

    return fig, ax
