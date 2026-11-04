import math

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from tools.paths import local_plots_dir
from tools.plot_style import save_figure


def plot_offered_vs_rcoq_grid(offered, rcoq, below, capacity_mw, title, filename, ncols=2):
    """Plot, per battery, the injection offered against RCOQ, with intervals below RCOQ in red.

    Only intervals with an obligation (RCOQ > 0) and an offer are drawn.

    Args:
        offered: Offered injection (MW), from obligations.offered_injection_mw.
        rcoq: RCOQ (MW), same index and columns.
        below: From obligations.below_rcoq, same index and columns.
        capacity_mw: {code: rated MW}; sets each panel's y-range.
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.
        ncols: Panels per row.

    Returns:
        (fig, axes, summary), where summary has per battery the intervals with an obligation and
        an offer, how many were below RCOQ, and on how many days.
    """
    codes = list(rcoq.columns)
    nrows = math.ceil(len(codes) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.8 * nrows + 0.8), sharex=True, squeeze=False)
    axes = axes.ravel()
    offered = offered.reindex(index=rcoq.index, columns=codes)

    rows = []
    for ax, code in zip(axes, codes):
        counted = (rcoq[code] > 0) & offered[code].notna()
        short = below[code] & counted
        met = counted & ~short

        ax.scatter(offered.index[met], offered[code][met], s=2, color='C0', alpha=0.3, lw=0,
                   label='Offered, at or above RCOQ')
        ax.scatter(offered.index[short], offered[code][short], s=6, color='tab:red', lw=0,
                   label='Offered, below RCOQ')
        ax.step(rcoq.index, rcoq[code].where(rcoq[code] > 0), where='post', color='black', lw=1.2,
                label='RCOQ')

        n_short, n = int(short.sum()), int(counted.sum())
        n_days = short[short].index.normalize().nunique()
        rows.append({'code': code, 'intervals': n, 'intervals_below': n_short, 'days_below': n_days})
        note = f"{n_short} of {n} intervals below, on {n_days} day{'' if n_days == 1 else 's'}" if n else "no obligation in period"

        ax.set_title(f"{code}  ({note})", fontsize=9.5)
        ax.set_ylim(-0.05 * capacity_mw[code], capacity_mw[code] * 1.1)
        ax.set_ylabel("MW", fontsize=9)
        ax.grid(alpha=0.3)

    for ax in axes[len(codes):]:  # spare panels when codes don't fill the grid
        ax.set_visible(False)

    for ax in axes[:len(codes)]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.tick_params(axis='x', labelbottom=True, labelsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.965),
               ncol=len(labels), fontsize=9, frameon=False, markerscale=3)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    save_figure(fig, local_plots_dir / filename)

    return fig, axes, pd.DataFrame(rows).set_index('code')
