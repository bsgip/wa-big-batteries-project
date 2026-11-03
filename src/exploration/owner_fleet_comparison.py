"""Neoen vs Synergy fleet SOC at ESROI start, side by side.

    python -m exploration.owner_fleet_comparison
"""

import matplotlib.pyplot as plt

from analysis.energy import discharge_cycles_by_window
from analysis.owner_fleets import OWNER_BATTERIES, owner_fleet_soc_pct
from tools.constants import PEAK_ESROI_START, battery_capacity_MWh, commissioning_dates
from tools.io import load_clean
from tools.paths import local_plots_dir
from tools.plot_style import SOC_BINS, save_figure
from visualisation.plot_cycles import plot_cycles_by_window
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

    # How many cycles a day each battery does, and when: one deep evening cycle, or morning + evening.
    # Each battery counted from its commissioning date, or the analysis start if that's later.
    power = load_clean("power")
    owners = {code: owner for owner, codes in OWNER_BATTERIES.items() for code in codes}
    first_days = {code: max(commissioning[code], power.index.min()) for code in owners}
    cycles = discharge_cycles_by_window(power, battery_capacity_MWh, first_days)
    plot_cycles_by_window(
        cycles,
        owners,
        f"Battery cycles per day, by part of the day ({period})",
        f"{FOLDER}/cycles_per_day_by_window.png",
    )
    print(cycles.assign(total=cycles.sum(axis=1)).round(2))


if __name__ == "__main__":
    main()
