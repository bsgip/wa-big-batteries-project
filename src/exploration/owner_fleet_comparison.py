"""Neoen vs Synergy fleet SOC at ESROI start, side by side.

    python -m exploration.owner_fleet_comparison
"""

import matplotlib.pyplot as plt

from analysis.owner_fleets import owner_fleet_soc_pct
from tools.constants import PEAK_ESROI_START, commissioning_dates
from tools.io import load_clean
from tools.paths import local_plots_dir
from tools.plot_style import SOC_BINS, save_figure
from visualisation.plot_distributions import _plot_one_distribution

FOLDER = "owner_fleet_comparison"


def plot_owner_soc_distributions(owner_soc, title, filename):
    """One SOC histogram per owner (column of owner_soc), side by side on the same 0-100% axis."""
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


def main():
    soc = load_clean("soc")  # analysis period only (from START_DATE)
    period = f"{soc.index.min():%b %Y} - {soc.index.max():%b %Y}"
    commissioning = commissioning_dates(load_clean("soc", start_date=None))  # full history
    owner_soc = owner_fleet_soc_pct(soc, commissioning)

    plot_owner_soc_distributions(
        owner_soc.at_time(PEAK_ESROI_START),
        f"Fleet SOC at ESROI start ({PEAK_ESROI_START}) by owner ({period})",
        f"{FOLDER}/owner_soc_distribution_esroi_start.png",
    )


if __name__ == "__main__":
    main()
