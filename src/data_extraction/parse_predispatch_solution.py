"""Parse forecast fields out of every dispatchSolution/predispatchData file
downloaded by download.py, in a single parallel pass over the corpus.

Same shape as parse_dispatch_solution.py, with one key difference in what a
file means. A dispatchData file is authoritative for ONE interval, so only
that interval is kept. A predispatchData file is one forecast run: AEMO runs
pre-dispatch every 30 minutes (the file's primaryDispatchInterval, also in
its filename) and each run forecasts 96 half-hour intervals - 48 hours ahead,
starting at the primary interval itself. Every one of those 96 is the point
here, so nothing is thrown away.

That gives every row two timestamps:

    primary_dispatch_interval  when the forecast run was made
    dispatch_interval          the interval being forecast

and any given dispatch_interval is forecast by ~96 different runs. Every
output is long format, one row per (run, forecast interval, ...) - see
_finish for why.

Output is produced ONE MONTH AT A TIME (see iter_predispatch_solution_months),
not as one DataFrame at the end like the other parsers. The corpus is ~53k
files of ~100MB (~650GB); the energy schedule alone is ~500M rows (~25GB as a
DataFrame) and price setting up to ~190M more, which doesn't fit in memory in
one go. catalog.py writes each month to its own temporary parquet and merges
them into one file per dataset at the end.

Every file has to be parsed end to end (all 96 items are wanted, so there's
no early exit like dispatch has): ~1.8s CPU per file with every field below,
so roughly 2.2h on 12 workers.

To add a new field: write a small `_extract_<name>(item) -> list[dict]`
function (one solutionData item in, that field's rows out), add it to _FIELDS
with its column dtypes, then add a CATALOG entry in data_extraction/catalog.py.
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
    """Forecast clearing price in every market service for the item's
    interval: energy ($/MWh) and the five FCESS markets. Same columns as
    price.parquet from parse_dispatch_solution.py."""
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
    """What set the price in each market service: one row per contribution.
    A market service with no contributions still gets one row (contribution
    columns empty), so its isMarketServiceCapped flag isn't lost."""
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
    """A flat {name: number} dict on the item (marketShortfalls, dispatchTotal,
    ...) as one row, its keys kept verbatim as column names."""

    def extract(item: dict) -> list[dict]:
        return [{"dispatch_interval": item["dispatchInterval"], **item[key]}]

    return extract


def _extract_projected_level(item: dict) -> list[dict]:
    """projectedChargeLevel / projectedDischargeLevel per facility, only where
    at least one isn't None. These keys only exist in files from the 2025-10-30 09:30 run on,
    and in practice are only ever non-None for batteries."""
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
    """Scheduled energy quantity (MW) per facility from schedule[marketService
    == "energy"] - every facility in the solve: thermal, wind/solar, batteries
    (negative = charging) and demand side programmes alike."""
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
    """Raw rows for one field in, final DataFrame out.

    Long format, both timestamps as plain columns on a RangeIndex - the same
    way bidstack.parquet is stored. Long rather than wide (one column per
    forecast step, or per run) because the two natural questions slice it in
    opposite directions - "what did the run at 14:00 forecast?" filters on
    primary_dispatch_interval, "how did the forecast for 18:00 evolve?"
    filters on dispatch_interval - and long format answers both with a plain
    filter. It also copes with a run that has fewer than 96 items.

    How far ahead each forecast is (dispatch_interval -
    primary_dispatch_interval) isn't stored, because it's derived, not
    extracted - work it out in data_processing.

    Keys in the JSON that aren't in _FIELDS' column list would be silently
    dropped, so they're logged - a new key appearing means _FIELDS needs
    updating."""
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
    """Walk one file (one forecast run), running the requested extractors
    against every solutionData item in it. Every row is tagged with the run's
    primary_dispatch_interval, taken from the filename.

    A file that fails partway through contributes no rows at all, rather than
    a truncated run that would look like a short forecast horizon.

    One bad file must never take down the whole run, so failures here are
    always caught and reported, never raised."""
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
    """"2025-01" from ReferencePre-DispatchSolution_202501210030.json."""
    stamp = path.name[len(_FILE_PREFIX) :]
    return f"{stamp[:4]}-{stamp[4:6]}"


def iter_predispatch_solution_months(
    fields: list[str] | None = None,
    dates: list[str] | None = None,
    max_workers: int = 12,
    skip_months: Collection[str] = (),
) -> Iterator[tuple[str, dict[str, pd.DataFrame]]]:
    """Extract the requested fields (default: every field in _FIELDS) from
    every predispatchData file, or just the runs made on `dates` (e.g.
    ["2025-01-21"]) if given, yielding (month, {field_name: DataFrame}) one
    month at a time - months by the forecast run's date, e.g. "2025-01".

    `skip_months` are left out entirely - for resuming a run that died
    partway, without redoing the months it already wrote.

    The full corpus is parsed in parallel across `max_workers` processes. A
    `dates` subset is ~48 files per day and runs sequentially (~1.8s a file)
    - useful for exploring a new field before committing to a full walk."""
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
