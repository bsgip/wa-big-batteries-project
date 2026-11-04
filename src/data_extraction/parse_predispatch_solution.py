"""Parse forecast fields from every predispatchData file: each file is one run forecasting 96 half-hours.
Long rows with primary_dispatch_interval (the run) and dispatch_interval (the half-hour forecast),
yielded a month at a time.
"""

import functools
import logging
import os
import re
from collections.abc import Collection, Iterator
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import ijson
import pandas as pd

from tools.constants import PREDISPATCH_SOLUTION_DATASET
from tools.paths import raw_dataset_dir
from data_extraction.download import log_and_record_parse_failures

logger = logging.getLogger(__name__)

# Files are ReferencePre-DispatchSolution_<YYYYMMDDHHMM>.json. Unlike
# dispatchData there are no correction/annotation variants in this corpus (as
# of 2026-10, all 53,423 files match the plain pattern), but matching on the
# leading timestamp rather than the whole name means a variant appearing later
# still parses rather than failing.
_FILE_PREFIX = "ReferencePre-DispatchSolution_"
_LEADING_TIMESTAMP = re.compile(_FILE_PREFIX + r"(\d{12})")

_TIMESTAMP = "datetime64[us, Australia/Perth]"


# --- per-field extractors: one solutionData item in, that field's rows out ---


def _extract_price(item: dict) -> list[dict]:
    """Forecast price in energy and the five FCESS markets.

    Args:
        item: One solutionData item from the JSON.

    Returns:
        One row.
    """
    prices = item["prices"]
    return [
        {
            "dispatch_interval": item["dispatchInterval"],
            "energy_price": prices["energy"],
            "regulation_raise": prices["regulationRaise"],
            "regulation_lower": prices["regulationLower"],
            "contingency_raise": prices["contingencyRaise"],
            "contingency_lower": prices["contingencyLower"],
            "rocof": prices["rocof"],
        }
    ]


def _extract_price_setting(item: dict) -> list[dict]:
    """What set the forecast price in each market service.

    Args:
        item: One solutionData item from the JSON.

    Returns:
        One row per contribution per market service, or one empty row for a service with
        none (to keep isMarketServiceCapped).
    """
    rows = []
    for service in item["priceSetting"]:
        base = {
            "dispatch_interval": item["dispatchInterval"],
            "market_service": service["marketService"],
            "is_market_service_capped": service["isMarketServiceCapped"],
        }
        if not service["contributions"]:
            rows.append(base)
        for contribution in service["contributions"]:
            rows.append(
                {
                    **base,
                    "contributing_variable": contribution["contributingVariable"],
                    "by_how_much": contribution["byHowMuch"],
                    "objective_coefficient": contribution["objectiveCoefficient"],
                }
            )
    return rows


def _extract_dict(key: str):
    """Extractor for a flat {name: number} dict on the item (e.g. marketShortfalls), as one row."""

    def extract(item: dict) -> list[dict]:
        return [{"dispatch_interval": item["dispatchInterval"], **item[key]}]

    return extract


def _extract_projected_level(item: dict) -> list[dict]:
    """projectedChargeLevel / projectedDischargeLevel per facility (runs from 2025-10-30 09:30 on).

    Args:
        item: One solutionData item from the JSON.

    Returns:
        One row per facility with either value set.
    """
    return [
        {
            "dispatch_interval": item["dispatchInterval"],
            "code": facility["facilityCode"],
            "projected_charge_level": facility["projectedChargeLevel"],
            "projected_discharge_level": facility["projectedDischargeLevel"],
        }
        for facility in item["facilityScheduleDetails"]
        if facility.get("projectedChargeLevel") is not None
        or facility.get("projectedDischargeLevel") is not None
    ]


def _extract_energy_schedule(item: dict) -> list[dict]:
    """Scheduled energy per facility.

    Args:
        item: One solutionData item from the JSON.

    Returns:
        One row per facility (MW; batteries negative when charging).
    """
    return [
        {
            "dispatch_interval": item["dispatchInterval"],
            "code": facility["facilityCode"],
            "quantity": facility["quantity"],
        }
        for service in item["schedule"]
        if service["marketService"] == "energy"
        for facility in service["facilitySchedule"]
    ]


# --- the registry ---

_KEY_COLUMNS = {"primary_dispatch_interval": _TIMESTAMP, "dispatch_interval": _TIMESTAMP}


def _float_columns(names: list[str]) -> dict[str, str]:
    """{name: "float64"} for each name."""
    return {name: "float64" for name in names}


# name -> (extractor, {column: dtype} after the two timestamp columns, extra
# sort keys after the two timestamps). Columns are listed explicitly rather
# than inferred so every monthly file has an identical schema - including
# months where a field has no rows at all (projected_level before Oct 2025) -
# because the monthly files are merged into one. Add a field by adding
# one entry here.
_FIELDS = {
    "price": (
        _extract_price,
        _float_columns(
            [
                "energy_price",
                "regulation_raise",
                "regulation_lower",
                "contingency_raise",
                "contingency_lower",
                "rocof",
            ]
        ),
        [],
    ),
    "price_setting": (
        _extract_price_setting,
        {
            "market_service": "str",
            "is_market_service_capped": "bool",
            "contributing_variable": "str",
            "by_how_much": "float64",
            "objective_coefficient": "float64",
        },
        ["market_service"],
    ),
    "market_shortfalls": (
        _extract_dict("marketShortfalls"),
        _float_columns(
            [
                "energyDeficit",
                "regulationRaiseDeficit",
                "regulationLowerDeficit",
                "contingencyRaiseDeficit",
                "contingencyLowerDeficit",
                "rocofDeficit",
            ]
        ),
        [],
    ),
    "market_service_requirements": (
        _extract_dict("marketServiceRequirements"),
        _float_columns(
            ["energy", "regulationRaise", "regulationLower", "contingencyRaise", "contingencyLower", "rocof"]
        ),
        [],
    ),
    "dispatch_total": (
        _extract_dict("dispatchTotal"),
        _float_columns(
            [
                "energyInjectionCapacity",
                "energyWithdrawalCapacity",
                "contingencyRaise",
                "contingencyLower",
                "regulationRaise",
                "regulationLower",
                "rocof",
            ]
        ),
        [],
    ),
    "projected_level": (
        _extract_projected_level,
        {"code": "str", **_float_columns(["projected_charge_level", "projected_discharge_level"])},
        ["code"],
    ),
    "energy_schedule": (_extract_energy_schedule, {"code": "str", "quantity": "float64"}, ["code"]),
}

# For catalog.py to validate its entries against at import time.
FIELD_NAMES = list(_FIELDS)


def _finish(name: str, rows: list[dict]) -> pd.DataFrame:
    """Turn one field's raw rows into its final long DataFrame.

    Keys not in the field's columns are logged, since they would be dropped.

    Args:
        name: Field name in _FIELDS.
        rows: Raw rows.

    Returns:
        Long frame on a RangeIndex with primary_dispatch_interval and dispatch_interval columns.
    """
    _extract, value_dtypes, extra_sort = _FIELDS[name]
    dtypes = {**_KEY_COLUMNS, **value_dtypes}

    unknown = {key for row in rows for key in row} - dtypes.keys()
    if unknown:
        logger.warning(f"{name}: dropping unexpected key(s) {sorted(unknown)} - add them to _FIELDS")

    df = pd.DataFrame(rows, columns=list(dtypes))
    for col in _KEY_COLUMNS:
        # utc=True parses the +08:00 offsets, and also copes with an empty column.
        df[col] = pd.to_datetime(df[col], utc=True).dt.tz_convert("Australia/Perth")
    df = df.astype(dtypes)

    return df.sort_values(list(_KEY_COLUMNS) + extra_sort, kind="stable").reset_index(drop=True)


def _get_predispatch_solution_rows(
    path: Path, field_names: list[str]
) -> tuple[dict[str, list[dict]], str | None]:
    """Run the requested extractors on every item of one forecast run.

    Args:
        path: predispatchData JSON file.
        field_names: Names in _FIELDS to run.

    Returns:
        ({field: rows}, error message or None). A file that fails gives no rows.
    """
    rows_by_field: dict[str, list[dict]] = {name: [] for name in field_names}

    match = _LEADING_TIMESTAMP.match(path.stem)
    if match is None:
        return rows_by_field, "filename doesn't match the expected pattern"

    dt = datetime.strptime(match.group(1), "%Y%m%d%H%M")
    primary_interval = dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")

    try:
        with open(path, "rb") as f:
            for item in ijson.items(f, "data.solutionData.item", use_float=True):
                for name in field_names:
                    extract = _FIELDS[name][0]
                    for row in extract(item):
                        rows_by_field[name].append({"primary_dispatch_interval": primary_interval, **row})
    except Exception as e:
        return {name: [] for name in field_names}, f"{type(e).__name__}: {e}"

    if not any(rows_by_field.values()):
        return rows_by_field, "no solutionData items in file"

    return rows_by_field, None


def _month_of(path: Path) -> str:
    """"YYYY-MM" from a predispatch file name."""
    stamp = path.name[len(_FILE_PREFIX) :]
    return f"{stamp[:4]}-{stamp[4:6]}"


def iter_predispatch_solution_months(
    fields: list[str] | None = None,
    dates: list[str] | None = None,
    max_workers: int = 12,
    skip_months: Collection[str] = (),
) -> Iterator[tuple[str, dict[str, pd.DataFrame]]]:
    """Extract fields from every predispatchData file, one month of runs at a time.

    Args:
        fields: Names in _FIELDS to extract; defaults to all.
        dates: Only runs made on these days, e.g. ["2025-01-21"] (run sequentially).
        max_workers: Parallel processes.
        skip_months: Months ("YYYY-MM") to leave out, e.g. ones already written.

    Yields:
        (month, {field: DataFrame}).
    """
    field_names = fields if fields is not None else list(_FIELDS)
    unknown = [name for name in field_names if name not in _FIELDS]
    if unknown:
        raise ValueError(f"unknown field(s) {unknown}; expected one of {list(_FIELDS)}")

    dataset_dir = raw_dataset_dir(PREDISPATCH_SOLUTION_DATASET)
    worker = functools.partial(_get_predispatch_solution_rows, field_names=field_names)

    if dates is None:
        paths = sorted(dataset_dir.rglob("*.json"))
    else:
        prefixes = tuple(f"{_FILE_PREFIX}{date.replace('-', '')}" for date in dates)
        with os.scandir(dataset_dir) as it:
            paths = sorted(Path(entry.path) for entry in it if entry.name.startswith(prefixes))

    paths_by_month: dict[str, list[Path]] = {}
    for path in paths:
        paths_by_month.setdefault(_month_of(path), []).append(path)

    skipped = sorted(set(skip_months) & paths_by_month.keys())
    if skipped:
        logger.info(f"skipping {len(skipped)} month(s) already done: {skipped[0]} .. {skipped[-1]}")
    months = [month for month in sorted(paths_by_month) if month not in skip_months]
    logger.info(
        f"Processing {sum(len(paths_by_month[m]) for m in months)} files over {len(months)} month(s) "
        f"for fields: {field_names}"
    )

    executor = ProcessPoolExecutor(max_workers=max_workers) if dates is None else None
    try:
        for month in months:
            month_paths = paths_by_month[month]
            if executor is not None:
                results = list(executor.map(worker, month_paths, chunksize=4))
            else:
                results = [worker(path) for path in month_paths]

            log_and_record_parse_failures(
                month_paths, [error for _, error in results], PREDISPATCH_SOLUTION_DATASET
            )

            dfs = {}
            for name in field_names:
                rows = [row for rows_by_field, _ in results for row in rows_by_field[name]]
                dfs[name] = _finish(name, rows)
            del results

            logger.info(f"{month}: {len(month_paths)} files done")
            yield month, dfs
    finally:
        if executor is not None:
            executor.shutdown()
