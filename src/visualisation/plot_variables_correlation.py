"""Scatter any two columns of a frame against each other, with a binned
summary line - the same treatment exploration/variables_correlations.py gives
each battery panel, on a single pair of columns.

The binning helpers live here rather than there so both share one copy;
exploration imports visualisation, not the other way round.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import pandas as pd

from tools.paths import local_plots_dir
from tools.plot_style import SEASON_COLORS, save_figure
from tools.seasons import SEASON_GROUPS, SEASONS, season_group_of, season_of
from visualisation.plot_fleet_capacity import semester_labels


def bin_edges(x: pd.Series, bins: int, method: str) -> np.ndarray:
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


def binned_stat(
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

    edges = bin_edges(x, bins, method)
    binned = pd.cut(x, edges)

    grouped_y = y.groupby(binned, observed=False)
    grouped_x = x.groupby(binned, observed=False)

    values = grouped_y.agg(stat).to_numpy(copy=True)
    values[grouped_y.count().to_numpy() < min_count] = np.nan
    return grouped_x.median().to_numpy(), values


def _point_groups(points: pd.DataFrame, color_by: str | None):
    """(group label per point, group order, colour per group) for color_by, or (None, [], []) for no colouring."""
    if color_by == "season_group":
        groups = season_group_of(points.index)
        levels = [s for s in SEASON_GROUPS if s in set(groups)]
        return groups, levels, [SEASON_COLORS[s] for s in levels]
    if color_by == "season":
        groups = season_of(points.index)
        levels = [s for s in SEASONS if s in set(groups)]
        return groups, levels, [SEASON_COLORS[s] for s in levels]
    if color_by == "semester":
        groups = semester_labels(points.index)
        levels = sorted(groups.unique())
        # viridis stopped short of its palest yellow, which vanishes on white
        # at this alpha; dark-to-light still reads as earliest-to-latest
        return groups, levels, plt.get_cmap("viridis")(np.linspace(0.05, 0.85, len(levels)))
    if color_by is None:
        return None, [], []
    raise ValueError(f"color_by must be 'season_group', 'season', 'semester' or None, got {color_by!r}")


def _draw_correlation(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    sse_df: pd.DataFrame | None = None,
    line_fit: bool = False,
    bins: int = 30,
    min_bin_count: int = 50,
    bin_method: str = "width",
    stat: str = "median",
    max_points: int | None = 20_000,
    color_by: str | None = "season_group",
) -> float:
    """Scatter x_col against y_col on ax (SSE days in red if sse_df is given), with r and n in the legend; returns r."""
    data = df[[x_col, y_col]].dropna()
    if data.empty:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return float("nan")

    r = data[x_col].corr(data[y_col])
    points = data.sample(max_points, random_state=0) if max_points and len(data) > max_points else data

    # subsample first, then split, so the total stays under max_points and
    # each group keeps its real share of the points
    groups, levels, colors = _point_groups(points, color_by)
    if groups is None:
        ax.scatter(points[x_col], points[y_col], s=10, alpha=0.5, color="tab:blue", edgecolors="none")
    else:
        for level, color in zip(levels, colors):
            in_group = groups == level
            ax.scatter(
                points.loc[in_group, x_col], points.loc[in_group, y_col],
                s=10, alpha=0.5, color=color, edgecolors="none", label=level,
            )

    # plot SSE datapoints
    if isinstance(sse_df, pd.DataFrame):
        ax.scatter(sse_df[x_col], sse_df[y_col], s=12, alpha=0.6, color="tab:red", edgecolors="none")

    if line_fit:
        bin_x, values = binned_stat(data[x_col], data[y_col], bins, min_bin_count, bin_method, stat)
        ax.plot(bin_x, values, color="black", lw=1.8, label=f"binned {stat} ({bins} bins)")

    # r and n ride in the legend on a handle that draws nothing, so they sit
    # beside the line's key rather than in the title
    stats_key = Line2D([], [], linestyle="none", label=f"r={r:.2f}, n={len(data):,}")
    handles = [*ax.get_legend_handles_labels()[0], stats_key]
    ax.legend(handles=handles, fontsize="small", handlelength=1.5, markerscale=2)
    return r


def plot_variables_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    overlay_sse: bool = False,
    sse_df: pd.DataFrame | None = None,
    title: str | None = None,
    filename: str | None = None,
    filename_suffix: str = "",
    line_fit: bool = False,
    bins: int = 30,
    min_bin_count: int = 50,
    bin_method: str = "width",
    stat: str = "median",
    max_points: int | None = 20_000,
    color_by: str | None = "season_group",
) -> float:
    """Scatter x_col against y_col with Pearson r, save the figure and return r.

    color_by: "season_group" (Summer / Shoulder / Winter), "season" (4 seasons), "semester" or None.
    overlay_sse + sse_df: also plot SSE days in red.
    See binned_stat for bins / bin_method / stat (only used with line_fit).
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    r = _draw_correlation(
        ax, df, x_col, y_col, sse_df if overlay_sse else None,
        line_fit, bins, min_bin_count, bin_method, stat, max_points, color_by,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title)

    save_figure(fig, local_plots_dir / (filename or f"corr_{x_col}_vs_{y_col}{filename_suffix}.png"))
    return r


def plot_variables_correlation_by_season(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    sse_df: pd.DataFrame | None = None,
    title: str | None = None,
    filename: str | None = None,
    filename_suffix: str = "",
    **draw_kwargs,
) -> dict[str, float]:
    """2x2 grid: all days, then one panel per season group (Summer / Shoulder / Winter). Returns r per panel.

    sse_df: also plot these SSE days in red, split the same way.
    draw_kwargs: passed to the scatter (line_fit, bins, color_by, ...).
    """
    panels = {"All days": df}
    sse_panels = {"All days": sse_df}
    for group in SEASON_GROUPS:
        panels[group] = df[season_group_of(df.index) == group]
        if sse_df is not None:
            sse_panels[group] = sse_df[season_group_of(sse_df.index) == group]
        else:
            sse_panels[group] = None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    r_by_panel = {}
    for ax, (name, panel_df) in zip(axes.flat, panels.items()):
        r_by_panel[name] = _draw_correlation(ax, panel_df, x_col, y_col, sse_panels[name], **draw_kwargs)
        ax.set_title(name)
    for ax in axes[1]:
        ax.set_xlabel(x_label)
    for ax in axes[:, 0]:
        ax.set_ylabel(y_label)
    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    save_figure(fig, local_plots_dir / (filename or f"corr_{x_col}_vs_{y_col}{filename_suffix}_season_grid.png"))
    return r_by_panel
