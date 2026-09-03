"""Parse per-dispatch-interval fields out of every caseInputData file
downloaded by download.py, in a single parallel pass over the corpus.

Each file is ~450MB (~70GB total across the ~1,056 files), so re-walking
the corpus once per field would be expensive - instead, every field has a
small extractor registered in _FIELDS, and build_case_input_data() walks
each file exactly once, running every registered extractor against each
dispatch interval as it goes.

To add a new field: write a small `_extract_<name>(item) -> list[dict]`
function (one caseData item in, that field's rows out) and a matching
`_finish_<name>(rows) -> DataFrame` function (raw row list in, final
DataFrame out - a pivot, a sort, whatever that field needs), then add both
to _FIELDS. Nothing else needs to change.
"""

import functools
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import ijson
import pandas as pd

from tools.constants import CASE_INPUT_DATASET, battery_codes
from tools.paths import raw_dataset_dir
from wem_data.download import log_and_record_parse_failures

logger = logging.getLogger(__name__)


# --- per-field extractors: one caseData item in, that field's rows out ---


def _extract_charge_level(item: dict) -> list[dict]:
    """Battery SOC (MWh), off the chargeLevel SCADA tag."""
    dispatch_interval = item["dispatchInterval"]
    rows = []
    for scada in item["scada"]:
        tag = scada["tag"]
        if "." not in tag:
            continue
        code, field = tag.rsplit(".", 1)
        if code in battery_codes and field == "chargeLevel":
            rows.append({"dispatch_interval": dispatch_interval, "code": code, "value": scada["value"]})
    return rows


def _extract_demand(item: dict) -> list[dict]:
    """System-wide demand/DPV SCADA tags (not per-battery)."""
    dispatch_interval = item["dispatchInterval"]
    rows = []
    for scada in item["scada"]:
        tag = scada["tag"]
        if "demand" in tag.lower() or "dpv" in tag.lower():
            rows.append({"dispatch_interval": dispatch_interval, "tag": tag, "value": scada["value"]})
    return rows


def _extract_bidstack(item: dict) -> list[dict]:
    """Battery offer tranches (bid stack), off
    markets.energy.facilities[].tranches[]."""
    dispatch_interval = item["dispatchInterval"]
    rows = []
    for facility in item["markets"]["energy"]["facilities"]:
        code = facility.get("facilityCode")
        if code not in battery_codes:
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


# --- per-field finishers: that field's raw row list in, final DataFrame out ---


def _finish_wide(index_col: str, columns_col: str, values_col: str = "value"):
    """Most SCADA-tag fields (chargeLevel, demand) share the same shape:
    pivot to one column per code/tag, tz-aware dispatch_interval index."""

    def finish(rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows).pivot(index=index_col, columns=columns_col, values=values_col).astype(float)
        df.index = pd.to_datetime(df.index).tz_convert("Australia/Perth")
        return df

    return finish


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
    """Long, not wide - each battery submits multiple tranches per
    interval, so dispatch_interval is a plain column, not the index, with
    one row per (dispatch_interval, code, tranche)."""
    df = pd.DataFrame(rows, columns=_BIDSTACK_COLUMNS)
    df["dispatch_interval"] = pd.to_datetime(df["dispatch_interval"]).dt.tz_convert("Australia/Perth")
    return df.sort_values(["dispatch_interval", "code", "tranche"]).reset_index(drop=True)


# name -> (extractor, finisher). This is the whole registry - add a field by
# adding one entry here.
_FIELDS = {
    "charge_level": (_extract_charge_level, _finish_wide("dispatch_interval", "code")),
    "demand": (_extract_demand, _finish_wide("dispatch_interval", "tag")),
    "bidstack": (_extract_bidstack, _finish_bidstack),
}


def _get_case_input_rows(path: Path, field_names: list[str]) -> tuple[dict[str, list[dict]], str | None]:
    """Walk one file, running the requested extractors (a subset of
    _FIELDS, or all of it) against each dispatch interval. One bad file
    must never take down the whole run - failures here are always caught
    and reported, never raised. Keeps whatever rows were parsed before a
    failure rather than discarding the whole (huge, ~450MB, ~1 day of
    data) file."""
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
    """Extract the requested fields (default: every field registered in
    _FIELDS) from every caseInputData file in one pass, returning
    {field_name: DataFrame}. Pass `fields` to skip running extractors you
    don't need this time (e.g. you already have soc/demand cached and only
    want bidstack) - the walk still only happens once either way, this
    just controls what work happens during it.

    The corpus is only ~1,056 files, but each is ~450MB (~70GB total) - too
    slow single-threaded, so it's parsed in parallel across `max_workers`
    processes."""
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
