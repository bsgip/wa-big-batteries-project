"""Do some batteries charge while others discharge? How much energy flows in opposite directions at the
same time (opposite flows), within each owner and across owners.
"""

from collections.abc import Mapping

import pandas as pd

from analysis.energy import INTERVAL_HOURS


def opposite_flow_mw(power: pd.DataFrame) -> pd.Series:
    """MW flowing in opposite directions within a group of batteries.

    Args:
        power: Battery power (MW), one column per battery in the group.

    Returns:
        Per interval, the smaller of total discharging and total charging (MW).
    """
    discharging = power.clip(lower=0).sum(axis=1)
    charging = -power.clip(upper=0).sum(axis=1)
    return pd.concat([discharging, charging], axis=1).min(axis=1)


def opposite_flow_by_owner(power: pd.DataFrame, owners: Mapping[str, list[str]]) -> pd.DataFrame:
    """Split opposite flow into flow within each owner's batteries and flow across owners.

    Args:
        power: Battery power (MW), one column per battery.
        owners: {owner: [battery codes]}; together they should cover power's columns.

    Returns:
        One row per interval, columns "within <owner>" for each owner and "across owners" (MW).
    """
    out = {f"Within {owner}": opposite_flow_mw(power[codes]) for owner, codes in owners.items()}
    out["Across owners"] = opposite_flow_mw(power) - sum(out.values())
    return pd.DataFrame(out)


def share_cancelling_out(flow_mw: pd.DataFrame, power: pd.DataFrame, by: pd.Index | pd.Grouper) -> pd.DataFrame:
    """Opposite flow as a share of all battery flow, per group of intervals.

    0% means every battery is going the same way; 100% means charging fully cancels discharging.

    Args:
        flow_mw: Opposite flow (MW) per interval, e.g. from opposite_flow_by_owner.
        power: Battery power (MW), one column per battery.
        by: How to group intervals, e.g. power.index.hour or pd.Grouper(freq="MS").

    Returns:
        One row per group, one % column per column of flow_mw.
    """
    flow = (flow_mw * INTERVAL_HOURS).groupby(by).sum()
    all_flow = (power.abs().sum(axis=1) * INTERVAL_HOURS).groupby(by).sum()
    return flow.div(all_flow, axis=0) * 2 * 100
