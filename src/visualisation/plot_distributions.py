import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.constants import battery_codes
from tools.paths import local_plots_dir
from tools.plot_style import CHARGE_COLOR, DISCHARGE_COLOR, SOC_BINS, save_figure
from tools.seasons import SEASONS, season_of


def _grid_shape(n: int) -> tuple[int, int]:
    """(rows, cols) of the most square grid that fits n panels."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _plot_one_distribution(ax: plt.Axes, values: pd.Series, bicolor: bool, bins=75):
    """Draw a histogram of values on ax, with a fitted normal curve.

    Args:
        ax: Axes to draw on.
        values: Values to plot; NaNs are dropped.
        bicolor: Colour bars below 0 as charging and above 0 as discharging.
        bins: Number of bins, or fixed bin edges so plots can be compared.
    """
    values = values.dropna()
    if values.empty:
        return

    _, bins, patches = ax.hist(values, bins=bins, density=True, color="tab:gray", alpha=0.8)

    if bicolor:
        for patch, left_edge, right_edge in zip(patches, bins[:-1], bins[1:]):
            bin_center = (left_edge + right_edge) / 2
            patch.set_facecolor(DISCHARGE_COLOR if bin_center >= 0 else CHARGE_COLOR)
        
        ax.axvline(0, color="grey", linewidth=0.8, zorder=0)

    mu, sigma = values.mean(), values.std()
    if sigma > 0:
        x = np.linspace(values.min(), values.max(), 200)
        pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        ax.plot(x, pdf, color="black", linewidth=1.5, label=f"fit: μ={mu:.1f}, σ={sigma:.1f}")
        ax.legend(fontsize="small", loc="upper right")


def plot_distribution_grid(
    df: pd.DataFrame, column_suffix: str, xlabel: str, title: str, filename: str, bicolor: bool = False
):
    """Plot one histogram with a fitted normal curve per battery.

    Args:
        df: Wide frame with a <code><column_suffix> column per battery.
        column_suffix: Column suffix, e.g. "_soc_pct"; "" for the bare <code> columns.
        xlabel: X-axis label.
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.
        bicolor: Colour bars by sign (for power).
    """
    rows, cols = _grid_shape(len(battery_codes))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for i, code in enumerate(battery_codes):
        ax = axes[i // cols][i % cols]
        col = f"{code}{column_suffix}"
        if col in df:
            _plot_one_distribution(ax, df[col], bicolor)
        ax.set_title(code)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")

    for i in range(len(battery_codes), rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)


def plot_fleet_soc_distribution(soc: pd.Series, title: str, filename: str):
    """Plot a histogram of fleet SOC with a fitted normal curve.

    Args:
        soc: Fleet SOC (%).
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_one_distribution(ax, soc, bicolor=False, bins=SOC_BINS)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Fleet SOC (%)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)


def plot_seasonal_fleet_soc_distribution(soc: pd.Series, filename: str):
    """Plot fleet SOC histograms, one panel per season.

    Args:
        soc: Fleet SOC (%).
        filename: Path under local_plots_dir to save the figure to.
    """
    period = f"{soc.index.min():%b %Y} - {soc.index.max():%b %Y}"
    seasons = season_of(soc.index)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), squeeze=False)

    for i, s in enumerate(SEASONS):
        ax = axes[i // 2][i % 2]
        mask = seasons == s

        _plot_one_distribution(ax, soc.loc[mask], bicolor=False, bins=SOC_BINS)
        ax.set_xlim(0, 100)

        ax.set_title(s)
        ax.set_xlabel("SOC (%)")
        ax.set_ylabel("Density")

    fig.suptitle(f"Fleet SOC by season ({period})")
    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)


def plot_owner_soc_distributions(owner_soc: pd.DataFrame, title: str, filename: str):
    """Plot one SOC histogram per owner, side by side on the same 0-100% axis.

    Args:
        owner_soc: One SOC % column per owner, e.g. from owner_fleets.owner_fleet_soc_pct.
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.
    """
    owners = list(owner_soc.columns)
    fig, axes = plt.subplots(1, len(owners), figsize=(6 * len(owners), 5), sharey=True, squeeze=False)

    for ax, owner in zip(axes[0], owners):
        _plot_one_distribution(ax, owner_soc[owner], bicolor=False, bins=SOC_BINS)
        ax.set_xlim(0, 100)
        ax.set_title(owner)
        ax.set_xlabel("Fleet SOC (%)")
    axes[0][0].set_ylabel("Density")

    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)
