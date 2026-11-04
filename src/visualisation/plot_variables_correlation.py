"""Scatter two columns of a frame against each other, with a binned summary line and Pearson r."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import pandas as pd

from tools.paths import local_plots_dir
from tools.plot_style import SEASON_COLORS, save_figure
from tools.seasons import SEASON_GROUPS, SEASONS, season_group_of, season_of
from visualisation.plot_fleet_capacity import semester_labels


def bin_edges(x: pd.Series, bins: int, method: str) -> np.ndarray:
    """Bin edges for the summary line.

    Args:
        x: Values to bin.
        bins: Number of bins.
        method: "width" for equal-width bins, "quantile" for equal counts (for heavy-tailed x
            such as price).

    Returns:
        Bin edges.
    """
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
    """Summary of y in each x bin, placed at the bin's median x.

    Args:
        x: X values.
        y: Y values, same index.
        bins: Number of bins.
        min_count: Bins with fewer readings are dropped.
        method: "width" or "quantile"; see bin_edges.
        stat: "median", or "mean" for power (whose median is 0 most of the time).

    Returns:
        (x positions, y values).
    """
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
    """(group label per point, group order, colour per group) for color_by, or (None, [], [])."""
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
    """Scatter x_col against y_col on ax (SSE days in red if sse_df), with r and n in the legend. Returns r."""
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
    """Scatter x_col against y_col with Pearson r, and save the figure.

    Args:
        df: Frame holding both columns.
        x_col: X column.
        y_col: Y column.
        x_label: X-axis label.
        y_label: Y-axis label.
        overlay_sse: Also plot sse_df's days in red.
        sse_df: SSE days, same columns as df.
        title: Figure title.
        filename: Path under local_plots_dir; defaults to "corr_<x_col>_vs_<y_col><suffix>.png".
        filename_suffix: Added to the default filename.
        line_fit: Draw the binned summary line.
        bins: Bins for the summary line.
        min_bin_count: Bins with fewer readings are dropped.
        bin_method: "width" or "quantile"; see bin_edges.
        stat: "median" or "mean"; see binned_stat.
        max_points: Points drawn; None draws all.
        color_by: "season_group", "season", "semester" or None.

    Returns:
        Pearson r.
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
    """Scatter x_col against y_col in a 2x2 grid: all days, then one panel per season group.

    Args:
        df: Frame holding both columns.
        x_col: X column.
        y_col: Y column.
        x_label: X-axis label.
        y_label: Y-axis label.
        sse_df: SSE days to plot in red, split the same way.
        title: Figure title.
        filename: Path under local_plots_dir; defaults to one built from the column names.
        filename_suffix: Added to the default filename.
        **draw_kwargs: Passed to the scatter (line_fit, bins, color_by, ...).

    Returns:
        {panel label: Pearson r}.
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


def plot_predictor_correlations(df, flags, predictors, labels):
    """Plot each predictor against fleet SOC: normal days, SSE days, both overlaid, and by season.

    Args:
        df: Daily feature table, from soc_features.build_feature_table.
        flags: Frame with a boolean is_sse column, from sse_events.sse_day_flags.
        predictors: Feature columns to plot.
        labels: {column: axis label}, e.g. soc_features.FEATURE_LABELS.
    """
    normal = df.loc[~flags["is_sse"]]
    sse = df.loc[flags["is_sse"]]

    y_col = "fleet_soc_pct"
    y_label = "Fleet SOC (%) at 17:30"

    for col in predictors:
        plot_variables_correlation(
            normal, col, y_col, labels[col], y_label,
            title="Variables Correlation - Normal Days",
        )
        plot_variables_correlation(
            sse, col, y_col, labels[col], y_label,
            title="Variables Correlation - SSE Days",
            filename_suffix="_sse",
        )
        plot_variables_correlation(
            normal, col, y_col, labels[col], y_label,
            title="Variables Correlation\nNormal days: blue, SSE days: red",
            filename_suffix="_sse_overlay",
            overlay_sse=True,
            sse_df=sse,
        )
        plot_variables_correlation(
            normal, col, y_col, labels[col], y_label,
            title="Variables Correlation\nNormal days: blue, SSE days: red",
            filename_suffix="_sse_overlay_by_season",
            overlay_sse=True,
            sse_df=sse,
            color_by="season_group",
        )
        plot_variables_correlation_by_season(
            normal, col, y_col, labels[col], y_label,
            sse_df=sse,
            title=f"{labels[col]} vs fleet SOC by season\nNormal days coloured by season, SSE days in red",
        )
