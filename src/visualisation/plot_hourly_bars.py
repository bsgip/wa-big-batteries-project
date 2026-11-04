import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tools.constants import PEAK_ESROI_END, PEAK_ESROI_START
from tools.paths import local_plots_dir
from tools.plot_style import save_figure


def _hours(time_str):
    """"HH:MM" as hours after midnight."""
    hour, minute = time_str.split(":")
    return int(hour) + int(minute) / 60


def plot_hourly_stacked_bars(panels, colors, ylabel, title, filename, legend_columns=4, largest_at_bottom=False):
    """Plot one panel per period, each a stacked bar per hour of day, with the ESROI marked.

    Args:
        panels: {panel title: one row per hour (0-23), one column per segment}, e.g. from
            price_setting.hourly_price_setter_shares.
        colors: {column: colour}; also the stacking and legend order. Columns not in any panel
            are skipped.
        ylabel: Y-axis label.
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.
        legend_columns: Legend entries per row.
        largest_at_bottom: Stack by average size over all panels instead, largest at the bottom.

    Returns:
        (fig, axes).
    """
    columns = [c for c in colors if any(c in table.columns for table in panels.values())]
    if largest_at_bottom:
        size = pd.concat(list(panels.values())).reindex(columns=columns).fillna(0).mean()
        columns = list(size.sort_values(ascending=False).index)
    fig, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5), sharey=True, squeeze=False)
    axes = axes[0]

    esroi = (_hours(PEAK_ESROI_START), _hours(PEAK_ESROI_END))
    for ax, (panel_title, table) in zip(axes, panels.items(), strict=True):
        table = table.reindex(columns=columns, fill_value=0)
        bottom = pd.Series(0.0, index=table.index)
        for column in columns:
            ax.bar(table.index + 0.5, table[column], bottom=bottom, width=0.85, color=colors[column],
                   edgecolor="white", lw=0.5)  # bar h spans h:00-h+1:00
            bottom += table[column]
        for x in esroi:
            ax.axvline(x, color="black", ls="--", lw=1.2)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        ax.set_xlabel("Hour of day")
        ax.set_title(panel_title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(ylabel)

    handles = [Patch(color=colors[c], label=c) for c in columns]
    handles.append(Line2D([], [], color="black", ls="--", lw=1.2, label=f"ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=legend_columns, fontsize=9,
               frameon=False)
    legend_rows = math.ceil(len(handles) / legend_columns)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95 - 0.06 * legend_rows))

    save_figure(fig, local_plots_dir / filename)

    return fig, axes
