"""Neoen vs Synergy fleet SOC at ESROI start, side by side.

    python -m exploration.owner_fleet_comparison
"""

from analysis.energy import discharge_cycles_by_window
from analysis.owner_fleets import OWNER_BATTERIES, owner_fleet_soc_pct
from tools.constants import PEAK_ESROI_START, battery_capacity_MWh, commissioning_dates
from tools.io import load_clean
from visualisation.plot_cycles import plot_cycles_by_window
from visualisation.plot_distributions import plot_owner_soc_distributions

FOLDER = "owner_fleet_comparison"


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
