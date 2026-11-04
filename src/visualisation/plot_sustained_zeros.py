import math

import matplotlib.pyplot as plt
import pandas as pd

from tools.constants import battery_codes
from tools.paths import local_plots_dir
from tools.plot_style import save_figure


def plot_sustained_zero_timeline(raw_df: pd.DataFrame, masked_df: pd.DataFrame, filename: str = "consecutive_zeros_timeline.png"):
    """Plot raw SOC per battery, with the readings mask_sustained_zero_runs masked in red.

    Args:
        raw_df: SOC (MWh) before masking.
        masked_df: The same frame after mask_sustained_zero_runs.
        filename: Path under local_plots_dir to save the figure to.
    """
    cols = math.ceil(math.sqrt(len(battery_codes)))
    rows = math.ceil(len(battery_codes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows), squeeze=False)

    for i, code in enumerate(battery_codes):
        ax = axes[i // cols][i % cols]

        if code not in raw_df:
            ax.set_visible(False)
            continue

        s = raw_df[code]
        ax.plot(s.index, s, color="tab:blue", linewidth=0.5, label="SOC (MWh)")

        sustained_zero = (s == 0) & masked_df[code].isna()
        ax.scatter(
            s.index[sustained_zero], s[sustained_zero],
            color="red", s=8, zorder=3, label="Consecutive zeros",
        )

        ax.set_title(code)
        ax.set_ylabel("SOC (MWh)")
        ax.legend(fontsize="x-small")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle("Consecutive zeros over time")
    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)
