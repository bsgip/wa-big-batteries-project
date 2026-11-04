"""Parse fields from every caseInputData file in one parallel pass.
To add a field: write _extract_<name>(item) and _finish_<name>(rows), and add both to _FIELDS.
"""

import functools
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import ijson
import pandas as pd

from tools.constants import CASE_INPUT_DATASET, esr_codes
from tools.paths import raw_dataset_dir
from data_extraction.download import log_and_record_parse_failures

logger = logging.getLogger(__name__)


# --- per-field extractors: one caseData item in, that field's rows out ---


def _extract_charge_level(item: dict) -> list[dict]:
    """Battery SOC from the chargeLevel SCADA tag.

    Args:
        item: One caseData item from the JSON.

    Returns:
        One row per battery (MWh).
    """
    dispatch_interval = item["dispatchInterval"]
    rows = []
    for scada in item["scada"]:
        tag = scada["tag"]
        if "." not in tag:
            continue
        code, field = tag.rsplit(".", 1)
        if code in esr_codes and field == "chargeLevel":
            rows.append({"dispatch_interval": dispatch_interval, "code": code, "value": scada["value"]})
    return rows


def _extract_demand(item: dict) -> list[dict]:
    """System-wide demand and DPV SCADA tags.

    Args:
        item: One caseData item from the JSON.

    Returns:
        One row per tag.
    """
    dispatch_interval = item["dispatchInterval"]
    rows = []
    for scada in item["scada"]:
        tag = scada["tag"]
        if "demand" in tag.lower() or "dpv" in tag.lower():
            rows.append({"dispatch_interval": dispatch_interval, "tag": tag, "value": scada["value"]})
    return rows


def _extract_power(item: dict) -> list[dict]:
    """Measured output (MW) per facility, from the <code>.energy.sentOut SCADA tag.

    Covers every facility with SCADA (+ sending out, - drawing in); same values as initialMw.

    Args:
        item: One caseData item from the JSON.

    Returns:
        One row per facility.
    """
    dispatch_interval = item["dispatchInterval"]
    rows = []
    for scada in item["scada"]:
        tag = scada["tag"]
        if tag.endswith(".energy.sentOut"):
            code = tag.removesuffix(".energy.sentOut")
            rows.append({"dispatch_interval": dispatch_interval, "code": code, "value": scada["value"]})
    return rows


def _extract_dispatch_condition(item: dict) -> list[dict]:
    """System-wide dispatch condition SCADA tags (e.g. loadRelief).

    Args:
        item: One caseData item from the JSON.

    Returns:
        One row per tag.
    """
    dispatch_interval = item["dispatchInterval"]
    rows = []
    for scada in item["scada"]:
        tag = scada["tag"]
        if "dispatchCondition" in tag:
            rows.append({"dispatch_interval": dispatch_interval, "tag": tag, "value": scada["value"]})
    return rows


def _extract_bidstack(item: dict) -> list[dict]:
    """Battery energy offer tranches (markets.energy.facilities[].tranches[]).

    Args:
        item: One caseData item from the JSON.

    Returns:
        One row per battery tranche.
    """
    dispatch_interval = item["dispatchInterval"]
    rows = []
    for facility in item["markets"]["energy"]["facilities"]:
        code = facility.get("facilityCode")
        if code not in esr_codes:
            continue
        submission_id = facility.get("submissionId")
        for tranche in facility["tranches"]:
            rows.append(
                {
                    "dispatch_interval": dispatch_interval,
                    "code": code,
                    "tranche": tranche["tranche"],
                    "quantity": tranche["quantity"],
                    "submitted_price": tranche["submittedPrice"],
                    "lfa_price": tranche["lfaPrice"],
                    "capacity_type": tranche["capacityType"],
                    "fuel_type": tranche["fuelType"],
                    "notice_time": tranche["noticeTime"],
                    "submission_id": submission_id,
                }
            )
    return rows


FCESS_MARKET_SERVICES = ["regulationRaise", "regulationLower", "contingencyRaise", "contingencyLower", "rocof"]

# Enablement limits per facility (JSON key -> column), as offered and as used by
# WEMDE. Per facility, so repeated on every tranche row.
_FCESS_FACILITY_FIELDS = {
    "maximumCapacity": "maximum_capacity",
    "enablementMinimum": "enablement_minimum",
    "enablementMinimumValueUsed": "enablement_minimum_used",
    "lowBreakpoint": "low_breakpoint",
    "lowBreakpointValueUsed": "low_breakpoint_used",
    "highBreakpoint": "high_breakpoint",
    "highBreakpointValueUsed": "high_breakpoint_used",
    "enablementMaximum": "enablement_maximum",
    "enablementMaximumValueUsed": "enablement_maximum_used",
}


def _extract_fcess_bidstack(item: dict) -> list[dict]:
    """Battery offer tranches in the five FCESS markets, with enablement limits.

    Args:
        item: One caseData item from the JSON.

    Returns:
        One row per battery tranche per market.
    """
    dispatch_interval = item["dispatchInterval"]
    rows = []
    for market_service in FCESS_MARKET_SERVICES:
        for facility in item["markets"][market_service]["facilities"]:
            code = facility.get("facilityCode")
            if code not in esr_codes:
                continue
            submission_id = facility.get("submissionId")
            limits = {column: facility.get(key) for key, column in _FCESS_FACILITY_FIELDS.items()}
            for tranche in facility.get("tranches") or []:
                rows.append(
                    {
                        "dispatch_interval": dispatch_interval,
                        "market_service": market_service,
                        "code": code,
                        "tranche": tranche["tranche"],
                        "quantity": tranche["quantity"],
                        "submitted_price": tranche["submittedPrice"],
                        "lfa_price": tranche["lfaPrice"],
                        "capacity_type": tranche["capacityType"],
                        "fuel_type": tranche["fuelType"],
                        "notice_time": tranche["noticeTime"],
                        "submission_id": submission_id,
                        **limits,
                    }
                )
    return rows


_FORECAST_DIRECTIONS = {
    # direction -> (JSON key, sign). AEMO publishes both as positive
    # magnitudes; the sign here is what puts them on one axis - injection to
    # the grid positive, withdrawal from it negative - so that netting a
    # facility is a plain sum rather than a subtraction somebody has to
    # remember to get the right way round.
    "injection": ("unconstrainedInjectionForecastMw", 1.0),
    "withdrawal": ("unconstrainedWithdrawalForecastMw", -1.0),
}


def _extract_unconstrained_forecast(item: dict) -> list[dict]:
    """Unconstrained output forecast (MW) per renewable facility.

    Args:
        item: One caseData item from the JSON.

    Returns:
        Two signed rows per facility, one per direction (hybrids can forecast a charge).
    """
    dispatch_interval = item["dispatchInterval"]
    rows = []
    for facility in item["unconstrainedForecast"]:
        code = facility["facilityCode"]
        for direction, (key, sign) in _FORECAST_DIRECTIONS.items():
            magnitude = facility[key]
            rows.append(
                {
                    "dispatch_interval": dispatch_interval,
                    "code": code,
                    "direction": direction,
                    # sign * 0.0 is -0.0 - equal to zero, but it reads as a
                    # typo, and almost every withdrawal row is zero.
                    "value_mw": sign * magnitude if magnitude else 0.0,
                }
            )
    return rows


# --- per-field finishers: that field's raw row list in, final DataFrame out ---


def _finish_wide(index_col: str, columns_col: str, values_col: str = "value"):
    """Finisher that pivots rows to one column per columns_col value, on a tz-aware dispatch_interval index."""

    def finish(rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows).pivot(index=index_col, columns=columns_col, values=values_col).astype(float)
        df.index = pd.to_datetime(df.index).tz_convert("Australia/Perth")
        return df

    return finish


_FORECAST_COLUMNS = ["dispatch_interval", "code", "direction", "value_mw"]


def _finish_unconstrained_forecast(rows: list[dict]) -> pd.DataFrame:
    """Long frame, two rows per (dispatch_interval, facility); netting is done in data_processing."""
    df = pd.DataFrame(rows, columns=_FORECAST_COLUMNS)
    df["dispatch_interval"] = pd.to_datetime(df["dispatch_interval"]).dt.tz_convert("Australia/Perth")
    df["value_mw"] = df["value_mw"].astype(float)
    return df.sort_values(["dispatch_interval", "code", "direction"]).reset_index(drop=True)


_BIDSTACK_COLUMNS = [
    "dispatch_interval",
    "code",
    "tranche",
    "quantity",
    "submitted_price",
    "lfa_price",
    "capacity_type",
    "fuel_type",
    "notice_time",
    "submission_id",
]


def _finish_bidstack(rows: list[dict]) -> pd.DataFrame:
    """Long frame, one row per (dispatch_interval, code, tranche)."""
    df = pd.DataFrame(rows, columns=_BIDSTACK_COLUMNS)
    df["dispatch_interval"] = pd.to_datetime(df["dispatch_interval"]).dt.tz_convert("Australia/Perth")
    return df.sort_values(["dispatch_interval", "code", "tranche"]).reset_index(drop=True)


_FCESS_BIDSTACK_COLUMNS = (
    ["dispatch_interval", "market_service"] + _BIDSTACK_COLUMNS[1:] + list(_FCESS_FACILITY_FIELDS.values())
)


def _finish_fcess_bidstack(rows: list[dict]) -> pd.DataFrame:
    """Long frame, one row per (dispatch_interval, market_service, code, tranche)."""
    df = pd.DataFrame(rows, columns=_FCESS_BIDSTACK_COLUMNS)
    df["dispatch_interval"] = pd.to_datetime(df["dispatch_interval"]).dt.tz_convert("Australia/Perth")
    return df.sort_values(["dispatch_interval", "market_service", "code", "tranche"]).reset_index(drop=True)


# name -> (extractor, finisher). This is the whole registry - add a field by
# adding one entry here.
_FIELDS = {
    "charge_level": (_extract_charge_level, _finish_wide("dispatch_interval", "code")),
    "demand": (_extract_demand, _finish_wide("dispatch_interval", "tag")),
    "bidstack": (_extract_bidstack, _finish_bidstack),
    "fcess_bidstack": (_extract_fcess_bidstack, _finish_fcess_bidstack),
    "dispatch_condition": (_extract_dispatch_condition, _finish_wide("dispatch_interval", "tag")),
    "unconstrained_forecast": (_extract_unconstrained_forecast, _finish_unconstrained_forecast),
    "power": (_extract_power, _finish_wide("dispatch_interval", "code")),
}

# For catalog.py to validate its entries against at import time.
FIELD_NAMES = list(_FIELDS)


def _get_case_input_rows(path: Path, field_names: list[str]) -> tuple[dict[str, list[dict]], str | None]:
    """Run the requested extractors on every interval of one file.

    Args:
        path: caseInputData JSON file.
        field_names: Names in _FIELDS to run.

    Returns:
        ({field: rows}, error message or None). Rows parsed before a failure are kept.
    """
    rows_by_field: dict[str, list[dict]] = {name: [] for name in field_names}

    try:
        with open(path, "rb") as f:
            case_data = ijson.items(f, "data.caseData.item", use_float=True)

            for item in case_data:
                for name in field_names:
                    extract, _finish = _FIELDS[name]
                    rows_by_field[name].extend(extract(item))
    except Exception as e:
        return rows_by_field, f"{type(e).__name__}: {e}"

    return rows_by_field, None


def build_case_input_data(max_workers: int = 12, fields: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Extract fields from every caseInputData file in one parallel pass.

    Args:
        max_workers: Parallel processes.
        fields: Names in _FIELDS to extract; defaults to all.

    Returns:
        {field: DataFrame}.
    """
    field_names = fields if fields is not None else list(_FIELDS)
    unknown = [name for name in field_names if name not in _FIELDS]
    if unknown:
        raise ValueError(f"unknown field(s) {unknown}; expected one of {list(_FIELDS)}")

    paths = sorted(raw_dataset_dir(CASE_INPUT_DATASET).rglob("*.json"))
    logger.info(f"Processing {len(paths)} files across {max_workers} workers for fields: {field_names}")

    worker = functools.partial(_get_case_input_rows, field_names=field_names)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(worker, paths, chunksize=1))

    log_and_record_parse_failures(paths, [error for _, error in results], CASE_INPUT_DATASET)

    dfs = {}
    for name in field_names:
        _extract, finish = _FIELDS[name]
        rows = [row for rows_by_field, _ in results for row in rows_by_field[name]]
        dfs[name] = finish(rows)
    return dfs
