import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.constants import battery_codes
from tools.paths import repo_plots_dir
from tools.plot_style import CHARGE_COLOR, DISCHARGE_COLOR, save_figure


def _grid_shape(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _plot_one_distribution(ax: plt.Axes, values: pd.Series, bicolor: bool):
    values = values.dropna()
    if values.empty:
        return

    _, bins, patches = ax.hist(values, bins=60, density=True, color="tab:gray", alpha=0.8)

    if bicolor:
        for patch, left_edge, right_edge in zip(patches, bins[:-1], bins[1:]):
            bin_center = (left_edge + right_edge) / 2
            patch.set_facecolor(DISCHARGE_COLOR if bin_center >= 0 else CHARGE_COLOR)

    mu, sigma = values.mean(), values.std()
    if sigma > 0:
        x = np.linspace(values.min(), values.max(), 200)
        pdf = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        ax.plot(x, pdf, color="black", linewidth=1.5, label=f"fit: μ={mu:.1f}, σ={sigma:.1f}")
        ax.legend(fontsize="x-small", loc="upper right")

    ax.axvline(0, color="grey", linewidth=0.8, zorder=0)


def plot_distribution_grid(
    df: pd.DataFrame, column_suffix: str, xlabel: str, title: str, filename: str, bicolor: bool = False
):
    """Grid of one histogram + fitted normal curve per battery, built from
    the `<code><column_suffix>` columns of `df`."""
    rows, cols = _grid_shape(len(battery_codes))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for i, code in enumerate(battery_codes):
        ax = axes[i // cols][i % cols]
        col = f"{code}{column_suffix}"
        if col in df:
            _plot_one_distribution(ax, df[col], bicolor)
        ax.set_title(code)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("density")

    for i in range(len(battery_codes), rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout()
    save_figure(fig, repo_plots_dir / filename)
