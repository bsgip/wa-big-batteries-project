"""Do some batteries charge while others discharge? Share of flow cancelling out, by hour and by month.

    python -m exploration.fleet_coordination
"""

import pandas as pd

from analysis.fleet_coordination import opposite_flow_by_owner, share_cancelling_out
from analysis.owner_fleets import OWNER_BATTERIES
from tools.constants import battery_codes
from tools.io import load_clean
from visualisation.plot_coordination import plot_cancelling_by_hour, plot_cancelling_monthly

FOLDER = "fleet_coordination"
COLORS = ["C0", "C1", "tab:purple"]  # within Neoen, within Synergy, across owners


def main():
    power = load_clean("power")[battery_codes]  # analysis period only (from START_DATE)
    period = f"{power.index.min():%b %Y} - {power.index.max():%b %Y}"
    flow = opposite_flow_by_owner(power, OWNER_BATTERIES)

    # 1. by hour of day, over the whole period
    hourly = share_cancelling_out(flow, power, by=power.index.hour)
    plot_cancelling_by_hour(
        hourly,
        colors=COLORS,
        title=f"Battery flow cancelling out (charging while others discharge), by hour of day ({period})",
        filename=f"{FOLDER}/cancelling_by_hour.png",
    )
    print(hourly.assign(total=hourly.sum(axis=1)).round(1))

    # 2. by month (a share, not MWh, so the fleet growing doesn't read as a trend)
    monthly = share_cancelling_out(flow, power, by=pd.Grouper(freq="MS"))
    plot_cancelling_monthly(
        monthly,
        colors=COLORS,
        title="Battery flow cancelling out (charging while others discharge), by month",
        filename=f"{FOLDER}/cancelling_monthly.png",
    )
    print(monthly.assign(total=monthly.sum(axis=1)).round(1))


if __name__ == "__main__":
    main()
