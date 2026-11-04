import matplotlib.pyplot as plt
import pandas as pd

from tools.paths import local_plots_dir
from tools.plot_style import SEASON_COLORS, save_figure
from tools.seasons import SEASONS, season_of


def plot_by_season_scatter(x, y, xlabel, ylabel, title, filename):
    """Plot daily x against y, one panel per season (2x2), with Pearson r in each panel title.

    Args:
        x: Daily values; only days with both x and y are drawn.
        y: Daily values.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.

    Returns:
        (fig, axes, data), where data has x, y and season per day.
    """
    data = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    data["season"] = season_of(data.index)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, season in zip(axes.ravel(), SEASONS, strict=True):
        d = data[data["season"] == season]
        ax.scatter(d["x"], d["y"], s=14, color=SEASON_COLORS[season], alpha=0.7, lw=0)
        ax.set_title(f"{season} ({len(d)} days, r = {d['x'].corr(d['y']):.2f})", fontsize=10)
        ax.grid(alpha=0.3)

    for ax in axes[1]:
        ax.set_xlabel(xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()

    save_figure(fig, local_plots_dir / filename)

    return fig, axes, data
