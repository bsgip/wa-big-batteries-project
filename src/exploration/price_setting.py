"""Who sets the energy price? Each battery's share of the month's energy price setters, and which
of the batteries' offers (energy, FCESS or both) set it.

    python -m exploration.price_setting
"""

from analysis.price_setting import (
    ENERGY_AND_FCESS,
    ENERGY_ONLY,
    FCESS_ONLY,
    NON_OFFER,
    OTHER_FACILITIES,
    monthly_battery_energy_offer_types,
    monthly_price_setter_shares,
)
from tools.constants import battery_codes
from tools.plot_style import UNIT_COLORS
from visualisation.plot_monthly_bars import plot_monthly_stacked_bars

# the whole market history, to show the fleet growing; Sep 2023 is only the last 3 days, so start at Oct
START = "2023-10-01"


def main():
    shares = monthly_price_setter_shares(battery_codes, service="energy")
    shares = shares.loc[shares.index >= START]  # tz-aware index vs date string: compared in its own tz

    plot_monthly_stacked_bars(
        shares,
        colors={**UNIT_COLORS, OTHER_FACILITIES: "0.6", NON_OFFER: "0.85"},
        ylabel="Share of energy price setters\n(% of all price setters in the month)",
        title="Who sets the energy price, by month",
        filename="price/battery_energy_price_setting_monthly.png",
    )
    print(shares.assign(batteries=shares[battery_codes].sum(axis=1)).round(1).to_string())

    # when a battery sets the energy price, which of its offers do - fleet-wide
    offer_types = monthly_battery_energy_offer_types(battery_codes)
    offer_types = offer_types.loc[offer_types.index >= START]
    plot_monthly_stacked_bars(
        offer_types,
        colors={ENERGY_ONLY: "C0", ENERGY_AND_FCESS: "tab:purple", FCESS_ONLY: "tab:red"},
        ylabel="Battery price setters by offer type\n(% of battery price setters in the month)",
        title="Which battery offers set the energy price, by month",
        filename="price/battery_energy_price_setting_offer_type_monthly.png",
    )
    print(offer_types.round(1).to_string())


if __name__ == "__main__":
    main()
