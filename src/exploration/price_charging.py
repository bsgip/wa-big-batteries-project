"""Do cheap daytime prices go with more charging? Daytime fleet net charge against the daytime
energy price, by season.

    python -m exploration.price_charging
"""

from analysis.energy import fleet_net_energy_pct
from analysis.price_measures import daily_price_measures
from data_extraction.catalog import load
from exploration.solar_charging import DAYTIME_END, DAYTIME_START
from tools.constants import commissioning_dates
from tools.io import load_clean
from visualisation.plot_pv_charging import plot_by_season_scatter

# a day's price counts if it has at least this many of the window's 72 five-minute prices
MIN_PRICE_INTERVALS = 60


def main():
    soc = load_clean("soc")
    power = load_clean("power")

    # median, not mean: one price spike (prices run from -1000 to 1100 $/MWh) shouldn't decide a day
    price = daily_price_measures(load("price")["energy_price"], DAYTIME_START, DAYTIME_END, min_intervals=MIN_PRICE_INTERVALS)

    # same y as the solar and wind plots: net charge as % of fleet capacity, discharge-positive flipped
    net_charge_pct = -fleet_net_energy_pct(power, commissioning_dates(soc, power), DAYTIME_START, DAYTIME_END)

    _, _, data = plot_by_season_scatter(
        price["median"],
        net_charge_pct,
        xlabel=f"Median energy price, {DAYTIME_START}-{DAYTIME_END} ($/MWh)",
        ylabel=f"Daytime net charge, {DAYTIME_START}-{DAYTIME_END}\n(% of fleet capacity)",
        title=f"Daytime fleet net charge against daytime energy price ({DAYTIME_START}-{DAYTIME_END}), by season",
        filename="price/daytime_net_charge_vs_price_by_season.png",
    )
    print(data.groupby("season").apply(lambda d: d["x"].corr(d["y"])).round(2).rename("correlation"))


if __name__ == "__main__":
    main()
