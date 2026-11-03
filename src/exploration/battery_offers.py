"""At what prices do the batteries offer to discharge, and are their ESROI offers explainable as
opportunity cost (Offer Construction Guidelines)?

    python -m exploration.battery_offers
"""

from analysis.offers import (
    cheapest_offer_price,
    discharge_offers,
    esroi_offer_gap,
    esroi_opportunity_benchmark,
    weighted_offer_price,
)
from analysis.owner_fleets import OWNER_BATTERIES
from tools.constants import PEAK_ESROI_END, PEAK_ESROI_START, battery_codes
from visualisation.plot_battery_boxes import plot_boxes_by_battery, plot_window_boxes_by_battery

FOLDER = "offers"
OWNER_COLORS = {"Neoen": "C0", "Synergy": "C1"}
ESROI = f"ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})"
OUTSIDE_ESROI = "Outside ESROI"


def main():
    offers = discharge_offers(battery_codes)  # every offer each battery has made (whole history)

    # 1. how high: the average price of the MW each battery offers to discharge, every interval -
    #    in the ESROI apart from the rest of the day, when batteries can price high to hold their charge
    weighted = weighted_offer_price(offers)
    in_esroi = weighted.index.isin(weighted.between_time(PEAK_ESROI_START, PEAK_ESROI_END, inclusive="left").index)
    by_window = {ESROI: weighted.loc[in_esroi], OUTSIDE_ESROI: weighted.loc[~in_esroi]}
    _, _, medians = plot_window_boxes_by_battery(
        by_window,
        window_colors={ESROI: "#D55E00", OUTSIDE_ESROI: "0.6"},
        owners=OWNER_BATTERIES,
        ylabel="Discharge offer price ($/MWh)\nMW-weighted average per interval",
        title="Discharge offer prices per battery, in and outside the ESROI",
        filename=f"{FOLDER}/discharge_offer_price_by_battery.png",
        ylim=(-200, 1000),
    )
    for window, values in by_window.items():
        print(f"\n{window}\n{values.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(0).T}")
    print("owner medians\n", medians.round(0))

    # 2. Option A: in the ESROI, cheapest discharge offer vs the best price forecast for later in the ESROI
    gap = esroi_offer_gap(cheapest_offer_price(offers), esroi_opportunity_benchmark())
    _, _, gap_medians = plot_boxes_by_battery(
        gap,
        owners=OWNER_BATTERIES,
        owner_colors=OWNER_COLORS,
        ylabel="Cheapest discharge offer − highest price\nforecast for the rest of the ESROI ($/MWh)",
        title="ESROI offers vs the best price forecast later in the ESROI (above 0: offer above it)",
        filename=f"{FOLDER}/esroi_offer_vs_forecast_by_battery.png",
        value_format="{:+,.0f} $/MWh",
        ylim=(-300, 600),
        zero_line=True,
    )
    print(gap.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(1).T)
    # % of each battery's ESROI intervals with its offer above the benchmark, by more than a few margins
    for margin in (0, 10, 100):
        share = gap.gt(margin).where(gap.notna()).mean() * 100
        print(f"% of ESROI intervals with the offer more than ${margin}/MWh above:", share.round(1).to_dict())
    print("owner median gaps", gap_medians.round(1).to_dict())


if __name__ == "__main__":
    main()
