"""Is low SOC at the start of the ESROI linked to high prices beforehand - through discharging
earlier, or charging less? Days grouped by the highest price before the ESROI.

    python -m exploration.pre_esroi_charging
"""

from analysis.pre_esroi import PRICE_BAND_LABELS, daily_max_price, fleet_charged_discharged_pct, price_band
from data_extraction.catalog import load
from tools.constants import PEAK_ESROI_START, START_DATE, commissioning_dates
from tools.io import load_clean
from tools.plot_style import CHARGE_COLOR, DISCHARGE_COLOR
from visualisation.plot_group_boxes import plot_panels_by_group

# the hours before the ESROI: from midday, when prices are low and batteries mostly charge
WINDOW_START = "12:00"


def main():
    soc = load_clean("soc")
    power = load_clean("power")
    price = load("price")["energy_price"].loc[START_DATE:]  # clean price is out of date; raw has every interval

    soc_start = soc["fleet_soc_pct"].at_time(PEAK_ESROI_START)
    soc_start.index = soc_start.index.normalize()
    energy = fleet_charged_discharged_pct(power, commissioning_dates(soc, power), WINDOW_START, PEAK_ESROI_START)
    band = price_band(daily_max_price(price, WINDOW_START, PEAK_ESROI_START))

    window = f"{WINDOW_START}-{PEAK_ESROI_START}"
    _, _, summary = plot_panels_by_group(
        panels=[
            (soc_start, f"Fleet SOC at ESROI start, {PEAK_ESROI_START} (%)", "C0"),
            (energy["charged"], f"Charged, {window}\n(% of fleet capacity)", CHARGE_COLOR),
            (energy["discharged"], f"Discharged, {window}\n(% of fleet capacity)", DISCHARGE_COLOR),
        ],
        groups=band.loc[soc_start.dropna().index.intersection(band.dropna().index)],
        order=PRICE_BAND_LABELS,
        title=f"Comparison of SOC at ESROI start and midday charging with the highest price at {window}",
        filename="price/pre_esroi_soc_charge_discharge_by_price_band.png",
    )
    print(summary.round(1).to_string())


if __name__ == "__main__":
    main()
