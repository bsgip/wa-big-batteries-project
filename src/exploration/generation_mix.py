"""What is the trend in the generation mix over the years? Monthly share of generation (energy sent
out) and of capacity (MW in service) by technology.

    python -m exploration.generation_mix
"""

import pandas as pd

from analysis.generation_mix import (
    facility_capacity_mw,
    facility_groups,
    monthly_capacity_mw,
    monthly_generation_mwh,
    monthly_generation_share,
)
from data_extraction.catalog import load
from visualisation.plot_monthly_bars import plot_monthly_stacked_bars

# the whole market history; Sep 2023 is only the last few days, so start at Oct
START = "2023-10-01"
# year-on-year: the same months each year (Dec-Sep, the full months every year has)
SAME_MONTHS = [("2023-12-01", "2024-09-30"), ("2024-12-01", "2025-09-30"), ("2025-12-01", "2026-09-30")]

COLORS = {
    "Coal": "#4D4D4D",
    "Gas": "#E69F00",
    "Other (liquid, waste, bioenergy)": "0.75",
    "Wind": "#56B4E9",
    "Grid-scale solar": "#F0E442",
    "Batteries (discharge)": "tab:red",
}


def main():
    power = load("power")  # every facility, not just the batteries
    power = power.loc[power.index >= START]  # tz-aware index vs date string: compared in its own tz
    groups = facility_groups()

    monthly_mwh = monthly_generation_mwh(power, groups)
    shares = monthly_generation_share(monthly_mwh)
    plot_monthly_stacked_bars(
        shares,
        colors=COLORS,
        ylabel="Share of generation\n(% of energy sent out in the month)",
        title="Generation mix by month",
        filename="generation_mix/generation_mix_monthly.png",
        legend_columns=3,
    )
    print(shares.round(1).to_string())

    # same months each year, so seasonality doesn't read as change
    tz = monthly_mwh.index.tz
    yearly = pd.DataFrame({
        f"{start[:7]} - {end[:7]}": monthly_mwh.loc[pd.Timestamp(start, tz=tz):pd.Timestamp(end, tz=tz)].sum()
        for start, end in SAME_MONTHS
    }).T
    print("\nShare of generation (%), same months each year")
    print(monthly_generation_share(yearly).round(1).to_string())
    print("\nGeneration (GWh), same months each year")
    print((yearly / 1000).round(0).to_string())

    # the same, by MW in service rather than energy sent out
    capacity = facility_capacity_mw(power, groups)
    monthly_mw = monthly_capacity_mw(power, groups, capacity)
    plot_monthly_stacked_bars(
        monthly_generation_share(monthly_mw),
        colors=COLORS,
        ylabel="Share of capacity\n(% of MW in service in the month)",
        title="Capacity mix by month",
        filename="generation_mix/capacity_mix_monthly.png",
        legend_columns=3,
    )
    print("\nCapacity in service (MW), monthly average")
    print(monthly_mw.assign(total=monthly_mw.sum(axis=1)).round(0).to_string())


if __name__ == "__main__":
    main()
