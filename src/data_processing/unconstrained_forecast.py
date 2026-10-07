"""Turn the long extracted unconstrained forecast into one wide frame per
generation technology.

Three steps, in this order:

    mask_sentinel()     drops the 1000 MW placeholder the go-live corpus
                        publishes in place of a real withdrawal forecast.
                        Before netting, or it lands in the net.

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

FORECAST_SENTINEL = 1000.0


def mask_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """unconstrainedForecast publishes a fixed 1000 MW placeholder instead of
    a real withdrawal forecast over the WEMDE go-live period - every interval
    from the start of the corpus (2023-09-26 08:00) through 2023-09-28 08:55,
    for every facility except ALINTA_WWF, then never again. Treat it as
    missing rather than a real forecast.

    1000 MW is not a number any facility in this field could produce: the
    largest injection anywhere in the corpus is ~222 MW, and the biggest
    registered WEM wind farm is ~214 MW. Nor is it a plausible withdrawal -
    the only facility that ever genuinely forecasts one is
    PRDSO_WALPOLE_HG1, at most 0.99 MW, and it isn't in the affected set.

    Matched on the absolute value so the rule holds whichever sign convention
    the extracted frame carries: extraction signs withdrawal negative, so the
    sentinel normally arrives as -1000, but a frame extracted before that
    convention landed carries +1000 and must clean the same way.

    Only the withdrawal row is dropped, not the facility's whole interval -
    injection over that window is ordinary data (0-203 MW, mean 18), and
    netting skips the NaN, so those rows come out as injection alone. That is
    the intended reading: withdrawal is unknown there, but every facility
    involved forecasts a flat zero withdrawal for the following three years.
    """
    sentinel = df["value_mw"].abs() == FORECAST_SENTINEL
    if not sentinel.any():
        return df

    affected = df.loc[sentinel]
    logger.info(
        f"unconstrained_forecast: masking {sentinel.sum()} sentinel ({FORECAST_SENTINEL:g} MW) rows as missing - "
        f"{affected['code'].nunique()} facilities, {affected['direction'].unique().tolist()}, "
        f"{affected['dispatch_interval'].min()} -> {affected['dispatch_interval'].max()}"
    )

    df = df.copy()
    df.loc[sentinel, "value_mw"] = float("nan")
    return df


def net_by_facility(df: pd.DataFrame) -> pd.DataFrame:
    """Long extracted forecast -> wide: dispatch_interval index, one column
    per facility, values the net signed MW forecast (injection + withdrawal).

    min_count=1 keeps a facility with no usable rows for an interval as NaN
    instead of 0: a farm that isn't commissioned yet is missing, not
    forecasting zero output, and the commissioning-date helpers in
    tools.constants read that distinction off the first non-NaN value. It also
    means a masked sentinel (see mask_sentinel) nets to the surviving
    direction rather than dragging the whole interval to NaN."""
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
    technology's facility columns, in net signed MW, with the go-live
    sentinel masked out.

    Nothing is aggregated across facilities: the per-facility columns are kept
    as they are, so a fleet total is a `.sum(axis=1)` away and a single farm is
    still addressable. NaNs stay NaN for the same reason - summing them away
    here would hide a not-yet-commissioned farm inside a fleet total."""
    df = mask_sentinel(df)
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

    # On the direction column, not on the sign: a handful of facilities
    # forecast a very slightly negative *injection* (-0.002 MW), which is not
    # the same thing as forecasting a withdrawal.
    withdrawal_rows = df.loc[df["direction"] == "withdrawal", ["code", "value_mw"]]
    real = withdrawal_rows["value_mw"].notna() & withdrawal_rows["value_mw"].ne(0)
    withdrawing = withdrawal_rows.loc[real, "code"].unique()
    if len(withdrawing):
        logger.info(f"unconstrained_forecast: facilities forecasting a withdrawal: {sorted(withdrawing)}")

    return {name: wide[codes] for name, codes in groups.items()}
