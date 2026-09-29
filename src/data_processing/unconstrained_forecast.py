"""Turn the long extracted unconstrained forecast into one wide frame per
generation technology.

Two steps, in this order:

    net_by_facility()   long -> wide: sums the injection and withdrawal rows
                        into one signed MW value per (interval, facility).
                        Signed at extraction time (injection positive,
                        withdrawal negative), so the sum IS the net - nothing
                        here has to know which direction is which.

    classify_columns()  splits those columns by technology. There is no
                        technology field anywhere in caseInputData - the only
                        thing the JSON gives us is the facility code - so the
                        split goes on the naming convention AEMO uses for WEM
                        facility codes:

                            *PV*   solar farm       AMBRISOLAR_PV1, NORTHAM_SF_PV1
                            *WF*   wind farm        YANDIN_WF1, ALINTA_WWF, EDWFMAN_WF1
                            rest   other renewable  BIOGAS01, RED_HILL, PRDSO_WALPOLE_HG1

The "rest" bucket is landfill gas, biogas and the Walpole hydro unit -
everything in unconstrainedForecast is renewable, which is why it's called
other_renewables rather than "other". It's a residual, so a facility whose
code stops following the convention lands there rather than being dropped, and
the counts are logged on every run so a mis-bucketed new facility shows up as
a surprise in the log instead of quietly skewing a technology total.

PV is tested before wind because the substrings aren't mutually exclusive in
principle (a hybrid could carry both), and a code carrying both is a solar
facility with wind in its name, not the other way round.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

PV_MARKER = "PV"
WIND_MARKER = "WF"


def net_by_facility(df: pd.DataFrame) -> pd.DataFrame:
    """Long extracted forecast -> wide: dispatch_interval index, one column
    per facility, values the net signed MW forecast (injection + withdrawal).

    min_count=1 keeps a facility that has no rows at all for an interval as
    NaN instead of 0: a farm that isn't commissioned yet is missing, not
    forecasting zero output, and the commissioning-date helpers in
    tools.constants read that distinction off the first non-NaN value."""
    net = df.groupby(["dispatch_interval", "code"])["value_mw"].sum(min_count=1)
    return net.unstack("code")


def technology_of(code: str) -> str:
    """Which bucket one facility code belongs in: "pv", "wind" or
    "other_renewables"."""
    upper = code.upper()
    if PV_MARKER in upper:
        return "pv"
    if WIND_MARKER in upper:
        return "wind"
    return "other_renewables"


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group a wide frame's facility columns by technology, each list in the
    column order the frame already has. Buckets with no facilities still
    appear, holding an empty list, so callers can index them unconditionally."""
    groups: dict[str, list[str]] = {"pv": [], "wind": [], "other_renewables": []}
    for code in df.columns:
        groups[technology_of(code)].append(code)
    return groups


def process(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Long extracted forecast -> {"pv", "wind", "other_renewables"}, each a
    wide frame on a shared dispatch_interval index holding only that
    technology's facility columns, in net signed MW.

    Nothing is aggregated across facilities: the per-facility columns are kept
    as they are, so a fleet total is a `.sum(axis=1)` away and a single farm is
    still addressable. NaNs stay NaN for the same reason - summing them away
    here would hide a not-yet-commissioned farm inside a fleet total."""
    wide = net_by_facility(df)
    groups = classify_columns(wide)

    logger.info(
        "unconstrained_forecast: "
        + ", ".join(f"{name} {len(codes)} facilities" for name, codes in groups.items())
    )
    for name, codes in groups.items():
        if not codes:
            logger.warning(f"unconstrained_forecast: no {name} facilities found")
    logger.info(f"unconstrained_forecast: other_renewables = {groups['other_renewables']}")

    withdrawing = df.loc[df["value_mw"] < 0, "code"].unique()
    if len(withdrawing):
        logger.info(f"unconstrained_forecast: facilities forecasting withdrawal: {sorted(withdrawing)}")

    return {name: wide[codes] for name, codes in groups.items()}
