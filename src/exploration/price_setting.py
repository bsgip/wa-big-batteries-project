"""Who sets the energy price? Battery shares by month, which battery offers set it, and technologies by hour.

    python -m exploration.price_setting
"""

from analysis.generation_mix import facility_groups
from analysis.price_setting import (
    ENERGY_AND_FCESS,
    ENERGY_ONLY,
    FCESS_ONLY,
    NON_OFFER,
    OTHER_FACILITIES,
    hourly_price_setter_shares,
    monthly_battery_energy_offer_types,
    monthly_price_setter_shares,
)
from tools.constants import battery_codes
from tools.plot_style import TECHNOLOGY_COLORS, UNIT_COLORS
from visualisation.plot_hourly_bars import plot_hourly_stacked_bars
from visualisation.plot_monthly_bars import plot_monthly_stacked_bars

# the whole market history, to show the fleet growing; Sep 2023 is only the last 3 days, so start at Oct
START = "2023-10-01"
# by time of day: one panel per year of the analysis period - the mix changed too much to pool them
# (batteries set about half the price setters in the first year, over 80% in the second)
HOURLY_PERIODS = {
    "Dec 2024 - Nov 2025": ("2024-12-01", "2025-12-01"),
    "Dec 2025 - Oct 2026": ("2025-12-01", "2026-12-01"),
}


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

    # which technologies set the energy price, by hour of day - batteries counted as one technology
    groups = facility_groups().replace({"Batteries (discharge)": "Batteries"})  # they also set it with charging bids
    hourly = {label: hourly_price_setter_shares(groups, start, end) for label, (start, end) in HOURLY_PERIODS.items()}
    plot_hourly_stacked_bars(
        hourly,
        colors={
            **{k: v for k, v in TECHNOLOGY_COLORS.items() if k != "Batteries (discharge)"},
            OTHER_FACILITIES: "0.45",
            NON_OFFER: "0.9",
        },
        ylabel="Share of energy price setters\n(% of all price setters in the hour)",
        title="Price setters, by technology and hour of day",
        filename="price/price_setter_technology_by_hour.png",
        largest_at_bottom=True,
    )
    for label, table in hourly.items():
        print(f"\n{label}\n{table.round(1).to_string()}")


if __name__ == "__main__":
    main()
