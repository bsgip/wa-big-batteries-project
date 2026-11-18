"""Plots for analysis/price_spread_soc.py: r^2 heatmap and per-battery spread vs SOC scatters."""

import math

import matplotlib.pyplot as plt
import pandas as pd

from tools.paths import local_plots_dir
from tools.plot_style import save_figure
from visualisation.plot_variables_correlation import _draw_correlation


def spread_label(col: str) -> str:
    """Readable name for a spread column: "Actual", or "Forecast, 4h ahead"."""
    if col == "actual_spread":
        return "Actual"
    return f"Forecast, {col.removeprefix('forecast_spread_')} ahead"


def plot_r2_heatmap(r2: pd.DataFrame, filename: str, at: str = "17:30"):
    """Heatmap of r^2, batteries by spread measure, each cell labelled with its value.

    Args:
        r2: From spread_soc_r2.
        filename: Path under local_plots_dir.
        at: Time of day the SOC was taken at, for the title.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(r2.to_numpy(), cmap="Blues", vmin=0, vmax=1, aspect="auto")

    for i in range(r2.shape[0]):
        for j in range(r2.shape[1]):
            value = r2.iat[i, j]
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize="small",
                    color="white" if value > 0.5 else "black")

    ax.set_xticks(range(r2.shape[1]), [spread_label(c) for c in r2.columns], rotation=30, ha="right")
    ax.set_yticks(range(r2.shape[0]), r2.index)
    fig.colorbar(im, ax=ax, label="r²")
    ax.set_title(f"r² of daily price spread vs battery SOC at {at}")

    save_figure(fig, local_plots_dir / filename)


def plot_spread_soc_grid(
    daily: pd.DataFrame,
    code: str,
    filename: str,
    panels: dict[str, str] | None = None,
    x_label: str = "Daily price spread ($/MWh)",
    title: str | None = None,
    at: str = "17:30",
):
    """One battery's SOC against each spread measure, one panel per measure, with r and n.

    Args:
        daily: From daily_spread_soc, optionally with more spread columns joined on.
        code: Battery code.
        filename: Path under local_plots_dir.
        panels: {spread column: panel title}; defaults to every spread column, titled by spread_label.
        x_label: X-axis label.
        title: Figure title; defaults to "<code>: daily price spread vs SOC at <at>".
        at: Time of day the SOC was taken at, for the labels.
    """
    soc_col = f"{code}_soc_pct"
    if panels is None:
        panels = {c: spread_label(c) for c in daily.columns if not c.endswith("_soc_pct")}

    plot_scatter_grid(
        daily,
        {panel_title: (col, soc_col) for col, panel_title in panels.items()},
        x_label=x_label,
        y_label=f"SOC (%) at {at}",
        title=title or f"{code}: daily price spread vs SOC at {at}",
        filename=filename,
    )


def plot_scatter_grid(
    df: pd.DataFrame,
    panels: dict[str, tuple[str, str]],
    x_label: str,
    y_label: str,
    title: str,
    filename: str,
    n_cols: int = 3,
):
    """Scatter pairs of columns, one panel per pair, with r and n in each legend.

    Args:
        df: Frame holding every column.
        panels: {panel title: (x column, y column)}.
        x_label: X-axis label, on the bottom row.
        y_label: Y-axis label, on the left column.
        title: Figure title.
        filename: Path under local_plots_dir.
        n_cols: Panels per row.
    """
    n_rows = math.ceil(len(panels) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), sharex=True, sharey=True, squeeze=False
    )
    for ax, (panel_title, (x_col, y_col)) in zip(axes.flat, panels.items()):
        _draw_correlation(ax, df, x_col, y_col, color_by=None)
        ax.set_title(panel_title)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(x_label)
    for ax in axes[:, 0]:
        ax.set_ylabel(y_label)
    fig.suptitle(title)

    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)
