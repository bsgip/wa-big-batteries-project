from analysis.energy import fleet_net_energy_pct
from tools.io import load_clean
from tools.constants import commissioning_dates
from tools.registration import registered_capacity_mw

START = "09:00"
END = "15:00"
WINDOW_HOURS = 6


def main():
    registered_mw = registered_capacity_mw()

    for name in ["pv", "wind"]:
        mw = load_clean(name)
        codes = list(mw.columns)
        commissioning = commissioning_dates(mw, candidates=codes)

        # most a farm could produce in the window: registered MW x window hours
        capacity_mwh = {code: registered_mw[code] * WINDOW_HOURS for code in codes}

        pct = fleet_net_energy_pct(mw, commissioning, START, END, codes, capacity_mwh)
        print(f"\n{name}: energy {START}-{END} as % of capacity")
        print(pct.describe().round(1))


if __name__ == "__main__":
    main()
