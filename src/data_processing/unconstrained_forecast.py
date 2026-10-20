"""Long extracted unconstrained forecast -> one wide frame (MW per facility) per technology.

Steps: mask_sentinel -> net_by_facility -> clip_to_capacity -> classify_columns.
Technology comes from the facility code: *PV* = pv, *WF* = wind, anything else = other_renewables.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

PV_MARKER = "PV"
WIND_MARKER = "WF"

FORECAST_SENTINEL = 1000.0


def mask_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """Set the 1000 MW go-live placeholder rows (2023-09-26 to 2023-09-28) to NaN."""
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
    """Long -> wide: one column per facility, net MW per interval (injection +, withdrawal -).

    NaN (not 0) where a facility has no data, so not-yet-commissioned farms stay missing.
    """
    # withdrawal made negative whatever sign the extracted file stored it with
    df = df.copy()
    is_withdrawal = df["direction"] == "withdrawal"
    df.loc[is_withdrawal, "value_mw"] = -df.loc[is_withdrawal, "value_mw"].abs()

    net = df.groupby(["dispatch_interval", "code"])["value_mw"].sum(min_count=1)
    return net.unstack("code")


def technology_of(code: str) -> str:
    """ "pv", "wind" or "other_renewables" for one facility code."""
    upper = code.upper()
    if PV_MARKER in upper:
        return "pv"
    if WIND_MARKER in upper:
        return "wind"
    return "other_renewables"


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """{technology: [facility codes]} for the columns of df."""
    groups: dict[str, list[str]] = {"pv": [], "wind": [], "other_renewables": []}
    for code in df.columns:
        groups[technology_of(code)].append(code)
    return groups


def clip_to_capacity(wide: pd.DataFrame, capacity: dict[str, float]) -> pd.DataFrame:
    """Cap each facility's forecast at its registered capacity (MW); log how many intervals were capped."""
    wide = wide.copy()
    for code in wide.columns:
        if code not in capacity:
            logger.warning(f"unconstrained_forecast: {code} has no registered capacity, not clipped")
            continue
        rated = capacity[code]
        n_clipped = wide[code].gt(rated).sum()
        if n_clipped:
            logger.info(f"unconstrained_forecast: {code}: clipping {n_clipped} intervals above {rated:g} MW")
        wide[code] = wide[code].clip(upper=rated)
    return wide


def process(df: pd.DataFrame, capacity: dict[str, float]) -> dict[str, pd.DataFrame]:
    """Extracted forecast -> {"pv", "wind", "other_renewables"}: MW per facility, sentinel masked, capped at capacity."""
    df = mask_sentinel(df)
    wide = net_by_facility(df)
    wide = clip_to_capacity(wide, capacity)
    groups = classify_columns(wide)

    logger.info(
        "unconstrained_forecast: "
        + ", ".join(f"{name} {len(codes)} facilities" for name, codes in groups.items())
    )
    for name, codes in groups.items():
        if not codes:
            logger.warning(f"unconstrained_forecast: no {name} facilities found")
    logger.info(f"unconstrained_forecast: other_renewables = {groups['other_renewables']}")

    # by the direction column, not the sign: some facilities forecast a tiny negative injection
    withdrawal_rows = df.loc[df["direction"] == "withdrawal", ["code", "value_mw"]]
    real = withdrawal_rows["value_mw"].notna() & withdrawal_rows["value_mw"].ne(0)
    withdrawing = withdrawal_rows.loc[real, "code"].unique()
    if len(withdrawing):
        logger.info(f"unconstrained_forecast: facilities forecasting a withdrawal: {sorted(withdrawing)}")

    return {name: wide[codes] for name, codes in groups.items()}
