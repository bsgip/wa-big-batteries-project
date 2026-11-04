"""How does wind affect charging? Fleet net charge against net wind, by season: daytime and overnight.

    python -m exploration.wind_charging
"""

import pandas as pd

from analysis.energy import fleet_net_energy_pct, window_energy_mwh
from tools.constants import DAYTIME_END, DAYTIME_START, commissioning_dates
from tools.io import load_clean
from visualisation.plot_pv_charging import plot_by_season_scatter

# overnight window, same as the "Overnight" slice in the cycles plot
NIGHT_START = "22:00"
NIGHT_END = "05:00"
# a 22:00-05:00 night crosses midnight; shifting every timestamp 2h later puts the whole night
# (22:00 -> 05:00) in one calendar day, 00:00-07:00, labelled by the morning it ends
NIGHT_SHIFT = pd.Timedelta(hours=2)


def main():
    soc = load_clean("soc")
    power = load_clean("power")
    commissioning = commissioning_dates(soc, power)

    # net wind = all wind farms (AEMO's unconstrained forecast, i.e. what they could produce),
    # taken over the same window the charge is measured in - the wind on the grid while the batteries charge
    wind_mw = load_clean("wind").sum(axis=1, min_count=1)

    # daytime - same y as the solar plot: net charge as % of fleet capacity, discharge-positive flipped
    wind_day_gwh = window_energy_mwh(wind_mw, DAYTIME_START, DAYTIME_END) / 1000
    charge_day_pct = -fleet_net_energy_pct(power, commissioning, DAYTIME_START, DAYTIME_END)

    _, _, day = plot_by_season_scatter(
        wind_day_gwh,
        charge_day_pct,
        xlabel=f"Net wind generation, {DAYTIME_START}-{DAYTIME_END} (GWh)",
        ylabel=f"Daytime net charge, {DAYTIME_START}-{DAYTIME_END}\n(% of fleet capacity)",
        title=f"Daytime fleet net charge against net wind generation ({DAYTIME_START}-{DAYTIME_END}), by season",
        filename="wind/daytime_net_charge_vs_wind_by_season.png",
    )
    print("daytime\n", day.groupby("season").apply(lambda d: d["x"].corr(d["y"])).round(2).rename("correlation"))

    # overnight - the same, over each whole night (see NIGHT_SHIFT)
    shifted_start = (pd.Timestamp(NIGHT_START) + NIGHT_SHIFT).strftime("%H:%M")
    shifted_end = (pd.Timestamp(NIGHT_END) + NIGHT_SHIFT).strftime("%H:%M")
    wind_night_gwh = window_energy_mwh(wind_mw.shift(freq=NIGHT_SHIFT), shifted_start, shifted_end) / 1000
    power_night = power.shift(freq=NIGHT_SHIFT)
    charge_night_pct = -fleet_net_energy_pct(power_night, commissioning, shifted_start, shifted_end)

    _, _, night = plot_by_season_scatter(
        wind_night_gwh,
        charge_night_pct,
        xlabel=f"Net wind generation, {NIGHT_START}-{NIGHT_END} (GWh)",
        ylabel=f"Overnight net charge, {NIGHT_START}-{NIGHT_END}\n(% of fleet capacity)",
        title=f"Overnight fleet net charge against net wind generation ({NIGHT_START}-{NIGHT_END}), by season",
        filename="wind/overnight_net_charge_vs_wind_by_season.png",
    )
    print("overnight\n", night.groupby("season").apply(lambda d: d["x"].corr(d["y"])).round(2).rename("correlation"))


if __name__ == "__main__":
    main()
