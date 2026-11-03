"""Why do Synergy's batteries discharge in the morning? FCESS enablement through the day next to
discharging, and morning discharge against the morning energy price.

    python -m exploration.morning_discharge
"""

from analysis.energy import DAY_WINDOWS
from analysis.fcess import enabled_mw, mean_by_window
from analysis.owner_fleets import OWNER_BATTERIES
from analysis.pre_esroi import fleet_charged_discharged_pct, price_band
from data_extraction.catalog import load
from tools.constants import START_DATE, commissioning_dates
from tools.io import load_clean
from visualisation.plot_fcess import plot_fcess_and_power_by_window
from visualisation.plot_group_boxes import plot_panels_by_group

FOLDER = "fcess"
# A: from when all six batteries were running, so the averages aren't diluted by batteries not yet built
ALL_RUNNING = "2025-12-01"
MORNING = DAY_WINDOWS["Morning (05:00-09:00)"]
# D: morning average energy price bands ($/MWh) - roughly 30% / 50% / 20% of days
MORNING_PRICE_EDGES = [80, 120]
MORNING_PRICE_LABELS = ["Under $80", "$80–120", "Over $120"]
OWNER_COLORS = {"Neoen": "C0", "Synergy": "C1"}  # as in battery_offers


def main():
    soc = load_clean("soc")
    power = load_clean("power")

    # A. FCESS enabled and power by part of the day, per owner
    by_owner = {}
    for owner, codes in OWNER_BATTERIES.items():
        enabled = enabled_mw(codes).loc[ALL_RUNNING:]
        owner_power = power.loc[ALL_RUNNING:, codes]
        flows = owner_power.clip(lower=0).sum(axis=1).to_frame("discharging")
        flows["charging"] = -owner_power.clip(upper=0).sum(axis=1)
        by_owner[owner] = (mean_by_window(enabled, DAY_WINDOWS), mean_by_window(flows, DAY_WINDOWS))
        print(f"\n{owner}\n{by_owner[owner][0].join(by_owner[owner][1]).round(0)}")
    period = f"{power.loc[ALL_RUNNING:].index.min():%b %Y} - {power.index.max():%b %Y}"
    plot_fcess_and_power_by_window(
        by_owner,
        title=f"FCESS enabled and battery power by part of the day, per owner ({period})",
        filename=f"{FOLDER}/fcess_enabled_and_power_by_window.png",
    )

    # D. morning discharge against the morning energy price
    price = load("price")["energy_price"].loc[START_DATE:]  # clean price is out of date; raw has every interval
    morning_price = price.between_time(*MORNING, inclusive="left").resample("D").mean()
    band = price_band(morning_price, MORNING_PRICE_EDGES, MORNING_PRICE_LABELS)
    commissioning = commissioning_dates(soc, power)
    window = "-".join(MORNING)
    discharged = {owner: fleet_charged_discharged_pct(power[codes], commissioning, *MORNING)["discharged"]
                  for owner, codes in OWNER_BATTERIES.items()}
    _, _, summary = plot_panels_by_group(
        panels=[(discharged[owner], f"{owner}: discharged, {window}\n(% of {owner}'s capacity)",
                 OWNER_COLORS[owner]) for owner in OWNER_BATTERIES],
        groups=band.dropna(),
        order=MORNING_PRICE_LABELS,
        title=f"Morning discharge by average energy price {window}",
        filename=f"{FOLDER}/morning_discharge_by_morning_price.png",
    )
    print(summary.round(1).to_string())


if __name__ == "__main__":
    main()
