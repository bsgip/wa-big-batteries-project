"""Battery bid stacks as MW$ (quantity x submitted price), one panel per unit
and one line per tranche.

The raw frame is ~4.9M rows: six units x up to ten tranches x three years of
5-minute dispatch intervals. Drawn raw that's a solid block of ink and a
minute of rendering, so everything here resamples before it plots.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tools.constants import battery_codes
from tools.paths import extracted_data_dir, local_data_dir, local_plots_dir
from tools.plot_style import DIVERGING_CMAP, save_figure

BIDSTACK_CSV = local_data_dir / "battery_bidstack.csv"

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


def extract_battery_bidstack():
    filename = extracted_data_dir / "bidstack.parquet"
    df = pd.read_parquet(filename)
    df = df.loc[df["code"].isin(battery_codes)]
    cols = ["dispatch_interval", "code", "tranche", "quantity", "submitted_price"]
    df = df[cols]

    try:
        df.to_csv(BIDSTACK_CSV, index=False)
        print("succesfully saved to csv")
    except Exception as e:
        print(e)


def load_bidstack() -> pd.DataFrame:
    df = pd.read_csv(BIDSTACK_CSV, index_col=0, parse_dates=["dispatch_interval"])
    df["mw_dollars"] = df["quantity"] * df["submitted_price"]
    return df


def _window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Trim to [start, end]. Timestamps are localised to the frame's own tz -
    the data is Perth time, and comparing a naive bound against a tz-aware
    column raises rather than silently doing the wrong thing."""
    if start is None and end is None:
        return df
    tz = df["dispatch_interval"].dt.tz
    bounds = slice(
        pd.Timestamp(start, tz=tz) if start else None,
        pd.Timestamp(end, tz=tz) if end else None,
    )
    return df.set_index("dispatch_interval").sort_index().loc[bounds].reset_index()


def plot_mw_dollars_by_tranche(
    df: pd.DataFrame,
    freq: str = "D",
    agg: str = "mean",
    start: str | None = None,
    end: str | None = None,
    linthresh: float = 1e3,
):
    """One panel per unit, one line per tranche, y = quantity x submitted price.

    `freq`/`agg` set the resampling - "D"/"mean" is the typical bid value per
    dispatch interval on a given day. Pass freq=None with a tight
    `start`/`end` to see the raw 5-minute stack for a short window.

    The y-axis is symlog: submitted prices run to the +/-$99,999 caps, so the
    cap tranches are three orders of magnitude above the mid-stack ones and a
    linear axis flattens everything else onto zero. Symlog keeps the sign
    (negative = charging bids) and stays linear through zero.
    """
    df = _window(df, start, end)

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


def rolling_change(
    df: pd.DataFrame,
    window: str = "1D",
    signed: bool = False,
    freq: str = "5min",
    min_frac: float = 0.5,
) -> pd.DataFrame:
    """Interval-to-interval change in MW$ per unit/tranche, smoothed over `window`.

    Returns a wide frame indexed by dispatch_interval with (code, tranche)
    columns, so the caller can plot or aggregate it however it likes.

    Two things this has to get right:

    Gaps. The record has ~29k breaks in the 5-minute grid, the longest 156
    days - units enter service at different dates and the feed drops out.
    Diffing the rows as they come would read straight across a 156-day hole
    and call it one interval's change. Everything is reindexed onto a
    continuous `freq` grid first, so a gap becomes NaN, the diff across it
    becomes NaN, and the line breaks instead of spiking.

    Signed vs absolute. A rolling MEAN of the signed diff telescopes - over a
    window of w intervals it collapses to (x[t] - x[t-w]) / w, which is just
    a slope, and says nothing about how much the bid moved in between. The
    question "how much does this tranche churn" is answered by the absolute
    change, so that is the default. `signed=True` is there if you actually
    want the drift.
    """
    wide = df.pivot_table(index="dispatch_interval", columns=["code", "tranche"], values="mw_dollars", aggfunc="mean")
    print(wide)
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq=freq, tz=wide.index.tz)
    wide = wide.reindex(grid)

    change = wide.diff()
    if not signed:
        change = change.abs()

    periods = max(1, int(pd.Timedelta(window) / pd.Timedelta(freq)))
    return change.rolling(window, min_periods=max(1, int(periods * min_frac))).mean()


def plot_rolling_change(
    df: pd.DataFrame,
    window: str = "1D",
    signed: bool = False,
    freq: str = "5min",
    points_per_window: int = 4,
    linthresh: float = 1.0,
):
    """One panel per unit, one line per tranche, y = rolling change in MW$.

    The rolling mean has already smoothed the series, so it is point-sampled
    for drawing rather than resampled again - no second round of averaging,
    and every drawn point is a value the rolling window actually produced.
    The stride comes from the window rather than a fixed point budget: a
    `window`-long rolling mean can't move faster than its window, so
    `points_per_window` samples across it is enough to draw it faithfully,
    and a fixed budget would alias a short window on a long record.
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


def tranche_grid(
    df: pd.DataFrame,
    freq: str | None = "D",
    agg: str = "mean",
    start: str | None = None,
    end: str | None = None,
    values: str = "level",
) -> dict[str, pd.DataFrame]:
    """One tranche x time grid per unit: index = interval, columns = tranche.

    `values="level"` is MW$ itself; `values="change"` is the interval-to-interval
    difference, computed on a continuous grid so the ~29k breaks in the record
    don't get differenced across (see rolling_change for why that matters).
    """
    df = _window(df, start, end)
    index = pd.Grouper(key="dispatch_interval", freq=freq) if freq else "dispatch_interval"

    grids = {}
    for code in sorted(df["code"].unique()):
        wide = (
            df[df["code"] == code]
            .pivot_table(index=index, columns="tranche", values="mw_dollars", aggfunc=agg)
            .sort_index()
        )
        if values == "change":
            step = pd.Timedelta(freq) if freq else pd.Timedelta("5min")
            wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq=step, tz=wide.index.tz))
            wide = wide.diff()
        grids[code] = wide
    return grids


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
    """Tranche x time heatmap, one panel per unit, colour = MW$.

    Reads down a column to see the whole bid stack at one interval, and across
    a row to see one tranche move over time - so a tranche switching on, going
    quiet, or flipping sign shows up as a colour break without having to
    untangle ten overlapping lines.

    Colour is diverging on zero and follows the repo's sign convention (blue =
    charging, red = discharging), on a symlog norm: half the grid is exactly
    zero and the extremes run 40x the 99th percentile, so a linear ramp would
    put nearly every cell in the neutral middle. `percentile` clips the ends
    so the caps don't set the scale for everything else.
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


if __name__ == "__main__":
    bidstack = load_bidstack()
    rolling_change(bidstack)

    # fig = plot_mw_dollars_by_tranche(bidstack, freq="D")
    # save_figure(fig, local_plots_dir / "bidstack" / "mw_dollars_by_tranche_daily.png")

    # fig = plot_rolling_change(bidstack, window="1D")
    # save_figure(fig, local_plots_dir / "bidstack" / "mw_dollars_rolling_change_daily.png")

    # fig = plot_tranche_heatmap(bidstack, freq="D")
    # save_figure(fig, local_plots_dir / "bidstack" / "tranche_heatmap_daily.png")

    # # Raw 5-minute resolution over a few days: this is where interval-to-interval
    # # movement in the stack is actually visible rather than averaged away.
    # fig = plot_tranche_heatmap(bidstack, freq=None, start="2026-01-10", end="2026-01-14")
    # save_figure(fig, local_plots_dir / "bidstack" / "tranche_heatmap_window.png")

    # # Same window, but each cell is the move since the previous interval, so a
    # # repriced tranche lights up and an untouched one stays blank.
    # fig = plot_tranche_heatmap(bidstack, freq=None, start="2026-01-10", end="2026-01-14", values="change")
    # save_figure(fig, local_plots_dir / "bidstack" / "tranche_heatmap_window_change.png")
