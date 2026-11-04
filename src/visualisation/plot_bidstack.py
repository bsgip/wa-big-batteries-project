"""Battery bid stack plots: MW$ by tranche, rolling churn, tranche x time heatmap.
Data helpers are in analysis/bidstack.py."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analysis.bidstack import rolling_change, tranche_grid, trim_to_window
from tools.plot_style import DIVERGING_CMAP

# Tranche is an ordered position in the bid stack, not an unordered category,
# so it gets a sequential ramp rather than a categorical cycle. Keyed by
# tranche number over the full 1..10 range so tranche 3 is the same colour in
# the four-tranche panels as in the ten-tranche ones.
MAX_TRANCHE = 10
TRANCHE_COLORS = {t: plt.get_cmap("viridis")((t - 1) / (MAX_TRANCHE - 1)) for t in range(1, MAX_TRANCHE + 1)}

# Heatmap colour for a cell with no value. Distinct from the near-white
# midpoint of the diverging ramp, so an empty cell never reads as a
# zero-valued bid.
_NO_BID_COLOR = "0.78"


def plot_mw_dollars_by_tranche(
    df: pd.DataFrame,
    freq: str = "D",
    agg: str = "mean",
    start: str | None = None,
    end: str | None = None,
    linthresh: float = 1e3,
):
    """Plot MW$ over time, one panel per unit and one line per tranche, on a symlog y-axis.

    Args:
        df: Bid stack from analysis.bidstack.load_bidstack.
        freq: Resampling frequency, e.g. "D"; None for every dispatch interval.
        agg: How to aggregate within each period, e.g. "mean".
        start: First timestamp to plot; None for no lower bound.
        end: Last timestamp to plot; None for no upper bound.
        linthresh: Range around 0 where the y-axis is linear.

    Returns:
        The figure (not saved).
    """
    df = trim_to_window(df, start, end)

    codes = sorted(df["code"].unique())
    n = len(codes)
    cols = min(2, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 3.2 * rows), squeeze=False, sharex=True, sharey=True)

    index = pd.Grouper(key="dispatch_interval", freq=freq) if freq else "dispatch_interval"

    for ax, code in zip(axes.flat, codes):
        wide = (
            df[df["code"] == code]
            .pivot_table(index=index, columns="tranche", values="mw_dollars", aggfunc=agg)
            .sort_index()
        )
        for tranche in wide.columns:
            ax.plot(wide.index, wide[tranche], lw=0.9, color=TRANCHE_COLORS[tranche], label=str(tranche))

        ax.axhline(0, color="grey", lw=0.8, zorder=0)
        # linscale shrinks the linear band around zero so it doesn't eat half
        # the panel; the explicit ticks stop matplotlib stacking +/-10^2 and 0
        # on top of each other where the log and linear regions meet.
        ax.set_yscale("symlog", linthresh=linthresh, linscale=0.4)
        decades = [10**p for p in range(int(np.log10(linthresh)), 8, 2)]
        ax.set_yticks([-d for d in reversed(decades)] + [0] + decades)
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.set_title(f"{code}  ({len(wide.columns)} tranches)", fontsize=9)
        ax.set_xlabel("")
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    # Built from the colour map, not from one panel's handles: the panels hold
    # different tranche counts (four vs ten), so a legend lifted off any single
    # panel would silently omit the tranches the others have.
    present = sorted(df["tranche"].unique())
    handles = [Line2D([], [], color=TRANCHE_COLORS[t], lw=2, label=str(t)) for t in present]
    fig.legend(handles=handles, title="Tranche", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

    period = f"{agg} per {freq}" if freq else "per dispatch interval"
    fig.supylabel("Quantity x submitted price (MW$, symlog)")
    fig.suptitle(f"Battery bid stacks as MW$ by tranche ({period})")
    fig.tight_layout()
    return fig


def plot_rolling_change(
    df: pd.DataFrame,
    window: str = "1D",
    signed: bool = False,
    freq: str = "5min",
    points_per_window: int = 4,
    linthresh: float = 1.0,
):
    """Plot the rolling change in MW$, one panel per unit and one line per tranche.

    Args:
        df: Bid stack from analysis.bidstack.load_bidstack.
        window: Rolling window, e.g. "1D".
        signed: Plot the signed change instead of the absolute change.
        freq: Grid spacing of the data.
        points_per_window: How many points to draw per window length.
        linthresh: Range around 0 where the y-axis is linear.

    Returns:
        The figure (not saved).
    """
    change = rolling_change(df, window=window, signed=signed, freq=freq)

    periods = max(1, int(pd.Timedelta(window) / pd.Timedelta(freq)))
    change = change.iloc[:: max(1, periods // points_per_window)]

    codes = sorted(change.columns.get_level_values("code").unique())
    n = len(codes)
    cols = min(2, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 3.2 * rows), squeeze=False, sharex=True, sharey=True)

    for ax, code in zip(axes.flat, codes):
        panel = change[code]
        for tranche in panel.columns:
            ax.plot(panel.index, panel[tranche], lw=0.9, color=TRANCHE_COLORS[tranche], label=str(tranche))

        # Symlog even for the non-negative absolute version, which a plain log
        # axis would misrepresent: ~28% of the rolling means are exactly zero -
        # a tranche whose bid didn't move at all across the window, which is a
        # real and common state, not a gap - and log sends those to -inf, so
        # they'd drop off the bottom rather than reading as "frozen".
        ax.axhline(0, color="grey", lw=0.8, zorder=0)
        ax.set_yscale("symlog", linthresh=linthresh, linscale=0.4)

        ax.set_title(f"{code}  ({len(panel.columns)} tranches)", fontsize=9)
        ax.set_xlabel("")
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=8)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    # Pinned after the loop, not inside it: set_ylim during plotting switches
    # autoscale off, which froze the top at whatever the first panel had drawn
    # and clipped the largest churn spikes off the shared axis.
    if not signed:
        axes.flat[0].set_ylim(0, np.nanmax(change.to_numpy()) * 1.5)

    present = sorted(change.columns.get_level_values("tranche").unique())
    handles = [Line2D([], [], color=TRANCHE_COLORS[t], lw=2, label=str(t)) for t in present]
    fig.legend(handles=handles, title="Tranche", loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)

    delta = "change" if signed else "|change|"
    fig.supylabel(f"{delta} in MW$ per 5-min interval ({window} rolling mean)")
    fig.suptitle(f"Battery bid stack churn: {delta} between dispatch intervals, {window} rolling mean")
    fig.tight_layout()
    return fig


def plot_tranche_heatmap(
    df: pd.DataFrame,
    freq: str | None = "D",
    agg: str = "mean",
    start: str | None = None,
    end: str | None = None,
    values: str = "level",
    percentile: float = 99.0,
    linthresh: float = 1e3,
):
    """Plot a tranche x time heatmap of MW$, one panel per unit, on one shared colour scale.

    Blue = charging, red = discharging; grey = no bid.

    Args:
        df: Bid stack from analysis.bidstack.load_bidstack.
        freq: Resampling frequency, e.g. "D"; None for every dispatch interval.
        agg: How to aggregate within each period, e.g. "mean".
        start: First timestamp to plot; None for no lower bound.
        end: Last timestamp to plot; None for no upper bound.
        values: "level" for MW$, or "change" for the change from the previous period.
        percentile: Percentile of |MW$| where the colour scale saturates.
        linthresh: Range around 0 where the colour scale is linear.

    Returns:
        The figure (not saved).
    """
    grids = tranche_grid(df, freq=freq, agg=agg, start=start, end=end, values=values)
    codes = list(grids)

    # One symmetric norm shared by every panel, so a colour means the same MW$
    # in each - the point of the figure is comparing units, which a per-panel
    # scale would quietly break.
    stacked = np.concatenate([g.to_numpy().ravel() for g in grids.values()])
    limit = np.nanpercentile(np.abs(stacked), percentile)
    norm = SymLogNorm(linthresh=linthresh, vmin=-limit, vmax=limit, base=10)
    # Missing has to be clearly separable from zero, not a shade off it: half
    # the cells are exactly zero, which sits near-white at the middle of a
    # diverging ramp, and "no bid submitted / unit not in service" is a
    # different statement from "bid submitted at zero value".
    cmap = plt.get_cmap(DIVERGING_CMAP).with_extremes(bad=_NO_BID_COLOR)

    # Panel height tracks tranche count so a cell is about the same shape in a
    # four-tranche unit as in a ten-tranche one.
    heights = [len(grids[c].columns) for c in codes]
    fig, axes = plt.subplots(
        len(codes), 1,
        figsize=(16, 0.26 * sum(heights) + 0.6 * len(codes)),
        squeeze=False,
        sharex=True,
        gridspec_kw={"height_ratios": heights},
        layout="constrained",  # tight_layout can't place a colorbar spanning several axes
    )

    for ax, code in zip(axes.flat, codes):
        grid = grids[code]
        tranches = list(grid.columns)

        # pcolormesh wants cell edges, not centres: passing the index directly
        # drops the final column and shifts every cell half a step left.
        step = grid.index.to_series().diff().median()
        x_edges = grid.index.append(pd.DatetimeIndex([grid.index[-1] + step]))
        y_edges = np.arange(0.5, len(tranches) + 1.5)

        mesh = ax.pcolormesh(
            x_edges, y_edges, np.ma.masked_invalid(grid.to_numpy().T), cmap=cmap, norm=norm, shading="flat"
        )
        ax.set_yticks(range(1, len(tranches) + 1), [str(t) for t in tranches], fontsize=7)
        ax.set_ylabel(code, fontsize=8, rotation=0, ha="right", va="center")
        ax.set_facecolor(_NO_BID_COLOR)
        ax.tick_params(labelsize=8)

    if values == "change":
        what = "interval-to-interval change in MW$"
        sign = "blue = moved down, red = moved up"
    else:
        what = "MW$ (quantity x submitted price)"
        sign = "blue = charging, red = discharging"

    period = f"{agg} per {freq}" if freq else "per dispatch interval"
    bar = fig.colorbar(mesh, ax=axes.ravel().tolist(), pad=0.01, shrink=0.55, extend="both")
    bar.set_label(f"{what} - {sign}", fontsize=8)
    # In change mode grey also covers cells whose previous interval is missing,
    # so there is nothing to difference against - not only absent bids.
    missing = "no bid / no prior interval" if values == "change" else "no bid submitted"
    fig.legend(
        handles=[Patch(facecolor=_NO_BID_COLOR, label=missing)],
        loc="upper right", bbox_to_anchor=(1.0, 0.99), bbox_transform=fig.transFigure,
        frameon=False, fontsize=8,
    )
    fig.suptitle(f"Battery bid stacks by tranche: {what}, {period}")
    fig.supylabel("Tranche", fontsize=9)
    return fig
