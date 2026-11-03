"""Stress events under different definitions, each as one start time per event.

Every definition comes out the same shape - a Series of event start times in
local (Australia/Perth) wall-clock time, tz-naive - so they can be counted and
compared side by side.
"""

import pandas as pd

from analysis.sse_events import filter_advisories, raw_advisories_path, stated_times
from data_extraction.catalog import load
from tools.constants import system_stress_events

SPARE_CAPACITY_THRESHOLD_MW = 750
# low-spare-capacity intervals this close together are one event, so a brief
# recovery partway through doesn't split it in two
EVENT_MERGE_GAP = pd.Timedelta(minutes=60)


def review_event_starts() -> pd.Series:
    """Start of each event in the review's list (constants.system_stress_events)."""
    starts = [pd.Timestamp(f"{e['date']} {e['event_start_time']}") for e in system_stress_events]
    return pd.Series(sorted(starts), name="review")


def advisory_starts(keywords: list[str] | None = None) -> pd.Series:
    """Start of the period each AEMO market advisory mentioning keywords warns about.

    keywords: default sse_events.ADVISORY_KEYWORDS (LRC, LOR)

    The stated period start (the 17:00 in "17:00 - 21:00") on the advisory's Start
    Interval date - Start Interval itself is when it was issued. Advisories that
    state no period keep Start Interval. Withdrawn advisories are kept: each was
    still a warning AEMO issued.
    """
    advisories = filter_advisories(pd.read_csv(raw_advisories_path, encoding="cp1252"), keywords)

    starts = []
    for _, advisory in advisories.iterrows():
        start = pd.Timestamp(advisory["Start Interval"])
        times = stated_times(advisory["Details"])
        if times is not None:
            start = start.normalize() + pd.Timedelta(times[0] + ":00")
        starts.append(start)
    return pd.Series(sorted(starts), name="advisories")


def in_service_capacity_mw() -> pd.Series:
    """In-service energy injection capacity per dispatch interval (MW)."""
    return load("in_service_quantities")["energyInjectionCapacity"].sort_index().rename("in_service_capacity_mw")


def spare_capacity_mw() -> pd.Series:
    """In-service energy injection capacity minus the energy injection dispatched, per interval (MW).

    Capacity offered as AVAILABLE (needs notice to come online) isn't counted:
    with it added, spare capacity never falls below ~880 MW.
    """
    dispatched = load("dispatch_total")["energyInjectionCapacity"]
    return (in_service_capacity_mw() - dispatched).sort_index().rename("spare_capacity_mw")


def low_spare_capacity_starts(
    spare: pd.Series,
    threshold_mw: float = SPARE_CAPACITY_THRESHOLD_MW,
    merge_gap: pd.Timedelta = EVENT_MERGE_GAP,
) -> pd.Series:
    """Start of each run of intervals with spare capacity below threshold_mw.

    Runs no more than merge_gap apart count as one event.
    spare: from spare_capacity_mw
    """
    low = spare.index[spare < threshold_mw].to_series()
    gap = low.diff()
    starts = low[gap.isna() | (gap > merge_gap)]
    return pd.Series(starts.dt.tz_localize(None).to_numpy(), name="spare_capacity")
