"""How is the generation mix changing? Monthly energy sent out, and MW in service, by each technology.

Technology per facility comes from data/registration/facility_technology.csv. Generation is a
facility's positive sent-out power (batteries: discharging only; their charging is load), from the
extracted all-facility power data. Rooftop solar isn't a registered facility, so it isn't included.
"""

from collections.abc import Mapping

import pandas as pd

from analysis.energy import INTERVAL_HOURS
from tools.constants import battery_capacity_MW
from tools.paths import registration_dir
from tools.registration import registered_capacity_mw

technology_path = registration_dir / "facility_technology.csv"

# technology (as in the csv) -> group shown; loads and unknown facilities are left out
TECHNOLOGY_GROUPS = {
    "coal": "Coal",
    "gas": "Gas",
    "liquid": "Other (liquid, waste, bioenergy)",
    "waste_to_energy": "Other (liquid, waste, bioenergy)",
    "bioenergy": "Other (liquid, waste, bioenergy)",
    "wind": "Wind",
    "solar": "Grid-scale solar",
    "battery": "Batteries (discharge)",
}


def facility_groups(path=technology_path) -> pd.Series:
    """{facility code: technology group} for the facilities in TECHNOLOGY_GROUPS."""
    technology = pd.read_csv(path).set_index("code")["technology"]
    return technology.map(TECHNOLOGY_GROUPS).dropna()


def monthly_generation_mwh(power: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Energy sent out (MWh) per month by each technology group.

    power: sent-out power (MW) per facility, any facilities - those not in groups are ignored
    groups: {facility code: group}, from facility_groups

    Returns one row per month, one column per group in TECHNOLOGY_GROUPS order.
    """
    codes = [c for c in power.columns if c in groups.index]
    energy = power[codes].clip(lower=0) * INTERVAL_HOURS
    by_group = energy.T.groupby(groups[codes]).sum().T  # facilities -> groups, per interval
    order = list(dict.fromkeys(TECHNOLOGY_GROUPS.values()))
    return by_group.resample("MS").sum().reindex(columns=order, fill_value=0)


def facility_capacity_mw(power: pd.DataFrame, groups: pd.Series) -> pd.Series:
    """{facility code: MW} for the facilities in groups that are in power.

    Registered system size (data/registration/facilities.csv), except:
      - batteries: their rated MW (constants.battery_capacity_MW) - the registered size counts
        charging and discharging, about double the rated MW
      - facilities with no registered size (e.g. MUJA_G6, retired before the list was taken): the
        highest output seen in power
    """
    codes = [c for c in power.columns if c in groups.index]
    registered = pd.Series(registered_capacity_mw()).reindex(codes)
    capacity = registered.fillna(power[codes].max())
    return capacity.where(~capacity.index.isin(list(battery_capacity_MW)), pd.Series(battery_capacity_MW))


def monthly_capacity_mw(power: pd.DataFrame, groups: pd.Series, capacity: Mapping[str, float]) -> pd.DataFrame:
    """Average MW in service per month by each technology group.

    A facility is in service from its first to its last power reading (non-missing, zero included,
    so a peaker that rarely runs still counts); a facility commissioned mid-month counts for the
    part of the month it was in service.

    power: sent-out power (MW) per facility - used only for when each facility has readings
    groups: {facility code: group}, from facility_groups
    capacity: {facility code: MW}, from facility_capacity_mw

    Returns one row per month, one column per group in TECHNOLOGY_GROUPS order.
    """
    codes = [c for c in power.columns if c in groups.index and c in capacity]
    has_reading = power[codes].notna()
    in_service = has_reading.cummax() & has_reading[::-1].cummax()[::-1]  # first to last reading

    mw = in_service * pd.Series(capacity)[codes]
    by_group = mw.T.groupby(groups[codes]).sum().T
    order = list(dict.fromkeys(TECHNOLOGY_GROUPS.values()))
    return by_group.resample("MS").mean().reindex(columns=order, fill_value=0)


def monthly_generation_share(monthly: pd.DataFrame) -> pd.DataFrame:
    """Each group's share (%) of the month's total - generation (MWh) or capacity (MW)."""
    return monthly.div(monthly.sum(axis=1), axis=0) * 100
