"""Per-battery comparison plots (data/plots/batteries_comparison/), over the analysis period.

    python -m visualisation.build_plots
"""

import logging

from tools.io import load_clean
from tools.plot_style import DIVERGING_CMAP
from visualisation.plot_distributions import plot_distribution_grid
from visualisation.plot_heatmap import plot_heatmap_grid
from visualisation.plot_time_of_day import plot_soc_timeofday_box_grid

FOLDER = "batteries_comparison"


def main():
    soc = load_clean("soc")  # analysis period only (from START_DATE)
    power = load_clean("power")
    period = f"{soc.index.min():%b %Y} - {soc.index.max():%b %Y}"

    plot_distribution_grid(
        soc, "_soc_pct", "SOC (%)",
        f"Battery SOC distribution ({period})",
        f"{FOLDER}/timeframe_soc_distribution_grid.png",
    )
    plot_distribution_grid(
        soc, "", "SOC (MWh)",
        f"Battery SOC distribution (MWh) ({period})",
        f"{FOLDER}/timeframe_soc_distribution_grid_mwh.png",
    )
    plot_soc_timeofday_box_grid(
        soc, "_soc_pct", "SOC (%)",
        f"Battery SOC by time of day ({period})",
        f"{FOLDER}/soc_timeofday_box_grid.png",
    )
    plot_soc_timeofday_box_grid(
        power, "_power_pct", "Power (% of rated, + discharge / - charge)",
        f"Battery power by time of day ({period})",
        f"{FOLDER}/power_timeofday_box_grid.png",
        signed_colors=True,
    )
    plot_heatmap_grid(
        power, "_power_pct",
        cmap=DIVERGING_CMAP,
        center=0,
        colorbar_label="Power (% of rated, + discharge / - charge)",
        title=f"Battery power heatmap ({period})",
        filename=f"{FOLDER}/timeframe_power_heatmap_grid.png",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
