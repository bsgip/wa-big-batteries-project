"""Do high-demand days also have poor charging? Daytime fleet net charge on high-demand days vs other days,
by season.

High-demand day: ESROI demand (MWh) in the top quarter of its season. Within each season, not overall:
over the whole year the top quarter would be almost all summer and winter days, and the comparison
would really be between seasons.

    python -m exploration.high_demand_charging
"""

from analysis.energy import fleet_net_energy_pct, window_energy_mwh
from exploration.solar_charging import DAYTIME_END, DAYTIME_START
from tools.constants import PEAK_ESROI_END, PEAK_ESROI_START, commissioning_dates
from tools.io import load_clean
from tools.plot_style import DISCHARGE_COLOR
from tools.seasons import season_of
from visualisation.plot_group_boxes import plot_two_groups_by_season

HIGH_DEMAND_QUANTILE = 0.75  # top quarter


def main():
    soc = load_clean("soc")
    power = load_clean("power")
    demand = load_clean("demand")["demand"]

    # same-day charging, i.e. the charging before that evening's peak
    net_charge_pct = -fleet_net_energy_pct(power, commissioning_dates(soc, power), DAYTIME_START, DAYTIME_END)

    esroi_demand = window_energy_mwh(demand, PEAK_ESROI_START, PEAK_ESROI_END).dropna()
    threshold = esroi_demand.groupby(season_of(esroi_demand.index)).transform(lambda s: s.quantile(HIGH_DEMAND_QUANTILE))
    high_demand = esroi_demand >= threshold

    _, _, summary = plot_two_groups_by_season(
        net_charge_pct,
        high_demand,
        group_labels=("High-demand days (top 25% ESROI demand)", "Other days"),
        colors=(DISCHARGE_COLOR, "C0"),
        ylabel=f"Daytime net charge, {DAYTIME_START}-{DAYTIME_END}\n(% of fleet capacity)",
        title="Daytime fleet net charge on high-demand days vs other days, by season",
        filename="demand/daytime_net_charge_high_demand_vs_other_by_season.png",
    )
    print(summary.round(1))
    print("high-demand threshold (ESROI demand MWh) by season:")
    print(esroi_demand.groupby(season_of(esroi_demand.index)).quantile(HIGH_DEMAND_QUANTILE).round(0))


if __name__ == "__main__":
    main()
