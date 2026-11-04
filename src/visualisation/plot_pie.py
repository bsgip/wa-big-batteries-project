import matplotlib.pyplot as plt
import pandas as pd

from tools.paths import local_plots_dir
from tools.plot_style import save_figure


def plot_share_pie(counts: pd.Series, legend_title: str, title: str, filename: str):
    """Plot a pie of each item's share of counts, with the counts in the legend.

    Args:
        counts: Count per item, indexed by item name.
        legend_title: Legend title.
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    wedges, _, _ = ax.pie(
        counts.values,
        autopct=lambda p: f"{p:.0f}%" if p >= 4 else "",  # hide % on tiny slices
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
        pctdistance=0.75,
    )

    ax.legend(
        wedges,
        [f"{c} ({n})" for c, n in counts.items()],
        title=legend_title,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()

    save_figure(fig, local_plots_dir / filename)
