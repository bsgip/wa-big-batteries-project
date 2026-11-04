import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from tools.paths import local_plots_dir
from tools.plot_style import save_figure


def plot_monthly_stacked_bars(monthly_pct, colors, ylabel, title, filename, legend_columns=4):
    """Plot stacked monthly bars, one segment per column.

    Args:
        monthly_pct: One row per month start, e.g. from price_setting.monthly_price_setter_shares
            or generation_mix.monthly_generation_share.
        colors: {column: colour}.
        ylabel: Y-axis label.
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.
        legend_columns: Legend entries per row.

    Returns:
        (fig, ax).
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    bottom = pd.Series(0.0, index=monthly_pct.index)
    for column in monthly_pct.columns:
        ax.bar(monthly_pct.index, monthly_pct[column], bottom=bottom, width=20, color=colors[column],
               alpha=0.9, edgecolor="white", lw=0.8, label=column)
        bottom += monthly_pct[column]

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=legend_columns, fontsize=8.5, frameon=False)
    legend_rows = math.ceil(len(monthly_pct.columns) / legend_columns)
    ax.set_title(title, pad=14 + 14 * legend_rows)  # room for the legend rows

    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)

    return fig, ax
