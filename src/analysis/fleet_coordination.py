"""Do some batteries charge while others discharge? How much energy flows in opposite directions at the
same time (opposite flows), within each owner and across owners.
"""

from collections.abc import Mapping

import pandas as pd

from analysis.energy import INTERVAL_HOURS


def opposite_flow_mw(power: pd.DataFrame) -> pd.Series:
    """Per interval, MW flowing in opposite directions: the smaller of total discharging and total
    charging across the batteries given - what one part of the group discharges while another charges."""
    discharging = power.clip(lower=0).sum(axis=1)
    charging = -power.clip(upper=0).sum(axis=1)
    return pd.concat([discharging, charging], axis=1).min(axis=1)


def opposite_flow_by_owner(power: pd.DataFrame, owners: Mapping[str, list[str]]) -> pd.DataFrame:
    """Per interval, opposite flow (MW) split into each owner's own ("within <owner>") and the rest
    ("across owners"), which only exists between batteries of different owners.

    owners: {owner: [battery codes]} - together they should cover power's columns
    """
    out = {f"Within {owner}": opposite_flow_mw(power[codes]) for owner, codes in owners.items()}
    out["Across owners"] = opposite_flow_mw(power) - sum(out.values())
    return pd.DataFrame(out)


def share_cancelling_out(flow_mw: pd.DataFrame, power: pd.DataFrame, by: pd.Index | pd.Grouper) -> pd.DataFrame:
    """Each column of flow_mw as % of all battery flow (charging + discharging energy) in each group:
    the share of the batteries' flow that cancels out. 0% = all batteries going the same way,
    100% = charging exactly cancelling discharging.

    Opposite flow of X MW is X going out and X coming in, so 2X of the flow cancels - hence the x2.

    flow_mw: per interval opposite flow (MW), e.g. from opposite_flow_by_owner
    power: battery power (MW), one column per battery
    by: what to group intervals by, e.g. power.index.hour, or pd.Grouper(freq="MS") for months
    """
    flow = (flow_mw * INTERVAL_HOURS).groupby(by).sum()
    all_flow = (power.abs().sum(axis=1) * INTERVAL_HOURS).groupby(by).sum()
    return flow.div(all_flow, axis=0) * 2 * 100
