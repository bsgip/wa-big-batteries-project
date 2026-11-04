"""Parse fields from every dispatchSolution/dispatchData file in one parallel pass, a month at a time.
Each file is used only for its own interval. To add a field: write an extractor and finisher, add them to _FIELDS.
"""

import functools
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import ijson
import pandas as pd

from tools.constants import DISPATCH_SOLUTION_DATASET
from tools.paths import raw_dataset_dir
from data_extraction.download import log_and_record_parse_failures

logger = logging.getLogger(__name__)

# Most dispatchData files are ReferenceDispatchSolution_<YYYYMMDDHHMM>.json,
# but AEMO also republishes correction/annotation variants for a small
# fraction of intervals with extra suffixes after the timestamp - e.g.
# ..._AmmendingRule_2023-145.json, ..._AffectedDispatchInterval.json,
# ..._DispatchEngineFailedToRun.json, ..._MarketAnalystOverride.json. The
# leading 12-digit timestamp right after the prefix is always present, so
# pull it from there rather than from the last "_"-separated token (which
# breaks on these variants).
_FILE_PREFIX = "ReferenceDispatchSolution_"
_LEADING_TIMESTAMP = re.compile(_FILE_PREFIX + r"(\d{12})")

_TIMESTAMP = "datetime64[us, Australia/Perth]"


# --- per-field extractors: one solutionData item in, that field's rows out ---


def _extract_price(item: dict) -> list[dict]:
    """Clearing price in energy ($/MWh) and the five FCESS markets.

    Args:
        item: One solutionData item from the JSON.

    Returns:
        One row.
    """
    return [
        {
            "dispatch_interval": item["dispatchInterval"],
            "energy_price": item["prices"]["energy"],
            "regulation_raise": item["prices"]["regulationRaise"],
            "regulation_lower": item["prices"]["regulationLower"],
            "contingency_raise": item["prices"]["contingencyRaise"],
            "contingency_lower": item["prices"]["contingencyLower"],
            "rocof": item["prices"]["rocof"],
        }
    ]


def _extract_price_setting(item: dict) -> list[dict]:
    """What set the price in each market service.

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
    """projectedChargeLevel / projectedDischargeLevel per facility (files from late Oct 2025 on).

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
        if facility.get("projectedChargeLevel") is not None or facility.get("projectedDischargeLevel") is not None
    ]


def _extract_energy_schedule(item: dict) -> list[dict]:
    """Dispatched energy per facility: a dispatch target, not measured output.

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


def _extract_fcess_schedule(item: dict) -> list[dict]:
    """Enabled FCESS quantity per facility per FCESS market.

    Args:
        item: One solutionData item from the JSON.

    Returns:
        One row per facility per market (MW, including 0).
    """
    return [
        {
            "dispatch_interval": item["dispatchInterval"],
            "market_service": service["marketService"],
            "code": facility["facilityCode"],
            "quantity": facility["quantity"],
            "ess_pre_processing_condition_flag": facility["essPreProcessingConditionFlag"],
        }
        for service in item["schedule"]
        if service["marketService"] != "energy"
        for facility in service["facilitySchedule"]
    ]


# --- per-field finishers: that field's raw row list in, final DataFrame out ---


def _rows_to_df(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    """Turn raw rows into a DataFrame, keeping the correction file's rows where an interval has one.

    Keys not in columns are logged, since they would be dropped.

    Args:
        rows: Raw rows, each tagged with is_correction.
        columns: Columns to keep.

    Returns:
        One frame with the winning file's rows per interval.
    """
    unknown = {key for row in rows for key in row} - set(columns) - {"is_correction"}
    if unknown:
        logger.warning(f"dropping unexpected key(s) {sorted(unknown)} next to {columns} - add them to _FIELDS")

    df = pd.DataFrame(rows, columns=columns + ["is_correction"])
    winning_file = df["is_correction"] == df.groupby("dispatch_interval")["is_correction"].transform("max")
    df = df[winning_file].drop(columns="is_correction")
    # utc=True parses the +08:00 offsets, and also copes with an empty column.
    df["dispatch_interval"] = pd.to_datetime(df["dispatch_interval"], utc=True).dt.tz_convert("Australia/Perth")
    return df


def _finish_wide(columns: list[str], pivot: tuple[str, str] | None = None):
    """Finisher for numeric fields: one row per interval, or pivot=(key, value) for one column per key."""

    def finish(rows: list[dict]) -> pd.DataFrame:
        df = _rows_to_df(rows, columns)

        if pivot is not None:
            columns_col, values_col = pivot
            # pivot raises on a repeated (interval, code), which would kill the
            # whole run - none have been seen, but warn and keep one rather than crash.
            duplicated = df.duplicated(subset=["dispatch_interval", columns_col], keep="last")
            if duplicated.any():
                logger.warning(f"dropping {duplicated.sum()} repeated (dispatch_interval, {columns_col}) rows")
                df = df[~duplicated]
            df = df.pivot(index="dispatch_interval", columns=columns_col, values=values_col)
        else:
            df = df.set_index("dispatch_interval")

        return df.sort_index().astype(float)

    return finish


def _finish_long(value_dtypes: dict[str, str], extra_sort: list[str]):
    """Finisher for long fields: indexed by dispatch_interval (repeated), with fixed columns and dtypes."""

    def finish(rows: list[dict]) -> pd.DataFrame:
        df = _rows_to_df(rows, ["dispatch_interval", *value_dtypes])
        df = df.astype({"dispatch_interval": _TIMESTAMP, **value_dtypes})
        return df.sort_values(["dispatch_interval", *extra_sort], kind="stable").set_index("dispatch_interval")

    return finish


def _dict_field(key: str, names: list[str]):
    """(extractor, finisher) pair for a flat dict field with the given column names."""
    return (_extract_dict(key), _finish_wide(["dispatch_interval", *names]))


# name -> (extractor, finisher). This is the whole registry - add a field by
# adding one entry here.
_FIELDS = {
    "price": (_extract_price, _finish_wide(
        [
            "dispatch_interval", 
            "energy_price",
            "regulation_raise",
            "regulation_lower",
            "contingency_raise",
            "contingency_lower",
            "rocof"
            ])
              ),
    "price_setting": (
        _extract_price_setting,
        _finish_long(
            {
                "market_service": "str",
                "is_market_service_capped": "bool",
                "contributing_variable": "str",
                "by_how_much": "float64",
                "objective_coefficient": "float64",
            },
            ["market_service"],
        ),
    ),
    "market_shortfalls": _dict_field(
        "marketShortfalls",
        [
            "energyDeficit",
            "regulationRaiseDeficit",
            "regulationLowerDeficit",
            "contingencyRaiseDeficit",
            "contingencyLowerDeficit",
            "rocofDeficit",
        ],
    ),
    "market_service_requirements": _dict_field(
        "marketServiceRequirements",
        ["energy", "regulationRaise", "regulationLower", "contingencyRaise", "contingencyLower", "rocof"],
    ),
    "dispatch_total": _dict_field(
        "dispatchTotal",
        [
            "energyInjectionCapacity",
            "energyWithdrawalCapacity",
            "contingencyRaise",
            "contingencyLower",
            "regulationRaise",
            "regulationLower",
            "rocof",
        ],
    ),
    "available_quantities": _dict_field(
        "availableQuantities",
        [
            "energyInjectionCapacity",
            "energyWithdrawalCapacity",
            "contingencyRaise",
            "contingencyLower",
            "regulationRaise",
            "regulationLower",
            "rocof",
        ],
    ),
    "in_service_quantities": _dict_field(
        "inServiceQuantities",
        [
            "energyInjectionCapacity",
            "energyWithdrawalCapacity",
            "contingencyRaise",
            "contingencyLower",
            "regulationRaise",
            "regulationLower",
            "rocof",
        ],
    ),
    "projected_level": (
        _extract_projected_level,
        _finish_long(
            {"code": "str", "projected_charge_level": "float64", "projected_discharge_level": "float64"}, ["code"]
        ),
    ),
    "energy_schedule": (
        _extract_energy_schedule,
        _finish_wide(["dispatch_interval", "code", "quantity"], pivot=("code", "quantity")),
    ),
    "fcess_schedule": (
        _extract_fcess_schedule,
        _finish_long(
            {
                "market_service": "str",
                "code": "str",
                "quantity": "float64",
                "ess_pre_processing_condition_flag": "bool",
            },
            ["market_service", "code"],
        ),
    ),
}

# For catalog.py to validate its entries against at import time.
FIELD_NAMES = list(_FIELDS)


def _get_dispatch_solution_rows(path: Path, field_names: list[str]) -> tuple[dict[str, list[dict]], str | None]:
    """Run the requested extractors on the one interval a file is for (its filename timestamp).

    Args:
        path: dispatchSolution JSON file.
        field_names: Names in _FIELDS to run.

    Returns:
        ({field: rows}, error message or None).
    """
    rows_by_field: dict[str, list[dict]] = {name: [] for name in field_names}

    match = _LEADING_TIMESTAMP.match(path.stem)
    if match is None:
        return rows_by_field, "filename doesn't match the expected pattern"

    is_correction = match.end() < len(path.stem)
    dt = datetime.strptime(match.group(1), "%Y%m%d%H%M")
    expected_interval = dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")

    try:
        with open(path, "rb") as f:
            solution_data = ijson.items(f, "data.solutionData.item", use_float=True)

            for item in solution_data:
                if item["dispatchInterval"] != expected_interval:
                    continue

                for name in field_names:
                    extract, _finish = _FIELDS[name]
                    for row in extract(item):
                        rows_by_field[name].append({**row, "is_correction": is_correction})

                return rows_by_field, None
    except Exception as e:
        # A file that fails partway contributes nothing, rather than some
        # fields but not others.
        return {name: [] for name in field_names}, f"{type(e).__name__}: {e}"

    return rows_by_field, "expected dispatch interval not found in file"


def _month_of(path: Path) -> str:
    """"YYYY-MM" from a dispatchSolution file name."""
    stamp = path.name[len(_FILE_PREFIX) :]
    return f"{stamp[:4]}-{stamp[4:6]}"


def build_dispatch_solution_data(
    fields: list[str] | None = None, dates: list[str] | None = None, max_workers: int = 12
) -> dict[str, pd.DataFrame]:
    """Extract fields from every dispatchSolution file, a month at a time.

    Args:
        fields: Names in _FIELDS to extract; defaults to all.
        dates: Only these days, e.g. ["2025-01-21"] (run sequentially); None for every file.
        max_workers: Parallel processes.

    Returns:
        {field: DataFrame}.
    """
    field_names = fields if fields is not None else list(_FIELDS)
    unknown = [name for name in field_names if name not in _FIELDS]
    if unknown:
        raise ValueError(f"unknown field(s) {unknown}; expected one of {list(_FIELDS)}")

    dataset_dir = raw_dataset_dir(DISPATCH_SOLUTION_DATASET)
    worker = functools.partial(_get_dispatch_solution_rows, field_names=field_names)

    if dates is None:
        paths = sorted(dataset_dir.rglob("*.json"))
    else:
        prefixes = tuple(f"{_FILE_PREFIX}{date.replace('-', '')}" for date in dates)
        with os.scandir(dataset_dir) as it:
            paths = sorted(Path(entry.path) for entry in it if entry.name.startswith(prefixes))

    paths_by_month: dict[str, list[Path]] = {}
    for path in paths:
        paths_by_month.setdefault(_month_of(path), []).append(path)
    logger.info(f"Processing {len(paths)} files over {len(paths_by_month)} month(s) for fields: {field_names}")

    parts: dict[str, list[pd.DataFrame]] = {name: [] for name in field_names}
    executor = ProcessPoolExecutor(max_workers=max_workers) if dates is None else None
    try:
        for month in sorted(paths_by_month):
            month_paths = paths_by_month[month]
            if executor is not None:
                results = list(executor.map(worker, month_paths, chunksize=8))
            else:
                results = [worker(path) for path in month_paths]

            log_and_record_parse_failures(month_paths, [error for _, error in results], DISPATCH_SOLUTION_DATASET)

            for name in field_names:
                _extract, finish = _FIELDS[name]
                rows = [row for rows_by_field, _ in results for row in rows_by_field[name]]
                parts[name].append(finish(rows))
            del results

            logger.info(f"{month}: {len(month_paths)} files done")
    finally:
        if executor is not None:
            executor.shutdown()

    dfs = {}
    for name, months in parts.items():
        # Empty months (projected_level before Oct 2025) are left out so they
        # can't affect the concatenated dtypes.
        non_empty = [df for df in months if not df.empty] or months[:1]
        df = pd.concat(non_empty)
        # Pivoted fields gain facility columns over time, which concat appends
        # at the end - re-sort them so they stay alphabetical like each month.
        if any(not part.columns.equals(non_empty[0].columns) for part in non_empty):
            df = df.sort_index(axis=1)
        dfs[name] = df
    return dfs
