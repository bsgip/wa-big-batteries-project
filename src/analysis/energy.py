from collections.abc import Mapping

import pandas as pd

from tools.constants import battery_capacity_MWh, battery_codes, codes_in, commissioned_capacity

INTERVAL_HOURS = 5 / 60

def window_energy_mwh(mw: pd.DataFrame | pd.Series, start: str, end: str) -> pd.DataFrame | pd.Series:
    """Daily energy (MWh) of a 5-min MW series between start and end."""
    mwh = (
        (mw.between_time(start, end, inclusive="left") * INTERVAL_HOURS)
        .resample("D").sum(min_count=1)
    )

    return mwh


def fleet_net_energy_mwh(power: pd.DataFrame, start: str, end: str, codes: list[str] | None = None) -> pd.Series:
    """Daily fleet net energy (MWh) between start and end, summed over the codes columns.

    codes: units to sum (default: the batteries in power)
    """
    if codes is None:
        codes = codes_in(power, battery_codes)
    daily_energy = window_energy_mwh(power[codes], start, end)
    return daily_energy.sum(axis=1, min_count=1).rename("fleet_net_energy_mwh")


def fleet_required_energy_mwh(rcoq: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Daily energy (MWh) the fleet must be able to discharge between start and end to meet its RCOQ.

    rcoq: load_clean("rcoq"), one MW column per facility
    """
    return fleet_net_energy_mwh(rcoq, start, end).rename("fleet_required_energy_mwh")


def fleet_net_energy_pct(
    power: pd.DataFrame,
    commissioning: dict[str, pd.Timestamp],
    start: str,
    end: str,
    codes: list[str] | None = None,
    capacity: Mapping[str, float] = battery_capacity_MWh,
) -> pd.Series:
    """Daily fleet net energy between start and end, as % of commissioned fleet capacity (MWh).

    commissioning: from constants.commissioning_dates(...)
    codes: units to sum (default: the batteries in power)
    capacity: {code: MWh} (default: battery_capacity_MWh)
    """
    if codes is None:
        codes = codes_in(power, battery_codes)
    net_energy = fleet_net_energy_mwh(power, start, end, codes)

    dates = {code: commissioning[code] for code in codes if code in commissioning}
    commissioned = commissioned_capacity(net_energy.index, dates, capacity)

    return (net_energy / commissioned * 100).rename("fleet_net_energy_pct")


def fleet_soc_drop_pct(soc: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Daily fleet SOC (%) at start minus at end. Positive = SOC used."""
    soc_start = soc["fleet_soc_pct"].at_time(start)
    soc_end = soc["fleet_soc_pct"].at_time(end)

    # index by day so the two times line up
    soc_start.index = soc_start.index.normalize()
    soc_end.index = soc_end.index.normalize()

    return (soc_start - soc_end).rename("fleet_soc_drop_pct")
