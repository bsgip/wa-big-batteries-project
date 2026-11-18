import math

import matplotlib.pyplot as plt
import pandas as pd

from tools.constants import battery_codes
from tools.paths import local_plots_dir
from tools.plot_style import save_figure


def plot_masked_zero_timeline(raw_df: pd.DataFrame, masked_df: pd.DataFrame,
                              filename: str = "data_description/soc_zeros_timeline.png"):
    """Plot raw SOC per battery, marking zeros cleaning masked (red) and kept as really empty (black).

    Args:
        raw_df: SOC (MWh) before masking zeros.
        masked_df: The same frame after soc.mask_unknown_zeros.
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

        is_zero = s == 0
        masked = is_zero & masked_df[code].isna()
        kept = is_zero & masked_df[code].notna()
        ax.scatter(s.index[masked], s[masked], color="red", s=8, zorder=3,
                   label="Zero masked (outage / impossible drop)")
        ax.scatter(s.index[kept], s[kept], color="black", s=8, zorder=4, label="Zero kept (really empty)")

        ax.set_title(code)
        ax.set_ylabel("SOC (MWh)")
        ax.legend(fontsize="x-small")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle("Zero SOC readings: masked vs kept")
    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)
