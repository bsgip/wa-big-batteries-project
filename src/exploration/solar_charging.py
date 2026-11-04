"""How does solar affect daytime charging? Daytime fleet net charge against daily PV, by season.

    python -m exploration.solar_charging
"""

from analysis.energy import daily_energy_mwh, fleet_net_energy_pct
from tools.constants import DAYTIME_END, DAYTIME_START, commissioning_dates
from tools.io import load_clean
from visualisation.plot_pv_charging import plot_by_season_scatter


def main():
    soc = load_clean("soc")
    power = load_clean("power")

    # PV = utility-scale farms (AEMO's unconstrained forecast, i.e. what they could produce) + rooftop (DPV).
    # Batteries don't charge from rooftop PV directly, but it lowers midday demand and prices.
    pv_mw = load_clean("pv").sum(axis=1, min_count=1) + load_clean("dpv")["dpv"]
    pv_gwh = daily_energy_mwh(pv_mw) / 1000

    # net charge as % of fleet capacity, not MWh, so the fleet growing doesn't read as more charging;
    # fleet_net_energy_pct is discharge-positive, so flip the sign
    net_charge_pct = -fleet_net_energy_pct(power, commissioning_dates(soc, power), DAYTIME_START, DAYTIME_END)

    _, _, data = plot_by_season_scatter(
        pv_gwh,
        net_charge_pct,
        xlabel="Daily PV generation (GWh)\nutility-scale + rooftop",
        ylabel=f"Daytime net charge, {DAYTIME_START}-{DAYTIME_END}\n(% of fleet capacity)",
        title="Daytime fleet net charge against daily PV generation, by season",
        filename="solar/daytime_net_charge_vs_daily_pv_by_season.png",
    )
    print(data.groupby("season").apply(lambda d: d["x"].corr(d["y"])).round(2).rename("correlation"))


if __name__ == "__main__":
    main()
