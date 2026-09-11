"""Parse per-dispatch-interval fields out of every
dispatchSolution/dispatchData file downloaded by download.py, in a single
parallel pass over the corpus.

The corpus is ~300k files of ~20MB, so re-walking it once per field would be
prohibitive (~45min per walk on 12 workers) - instead, every field has a small
extractor registered in _FIELDS, and build_dispatch_solution_data() walks each
file exactly once, running every requested extractor against the one dispatch
interval that file is authoritative for.

To add a new field: write a small `_extract_<name>(item) -> list[dict]`
function (one solutionData item in, that field's rows out) and a matching
finisher (raw row list in, final DataFrame out), then add both to _FIELDS.
Nothing else needs to change here - see data_extraction/catalog.py to make it
loadable by name.
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

from tools.constants import DISPATCH_SOLUTION_DATASET, battery_codes
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
_LEADING_TIMESTAMP = re.compile(r"ReferenceDispatchSolution_(\d{12})")


# --- per-field extractors: one solutionData item in, that field's rows out ---


def _extract_price(item: dict) -> list[dict]:
    """Energy market clearing price ($/MWh) for the interval."""
    return [{"dispatch_interval": item["dispatchInterval"], "energy_price": item["prices"]["energy"]}]


def _extract_power(item: dict) -> list[dict]:
    """Per-battery power (MW) off initialMw - the facility's SCADA-measured
    output at the start of the interval, positive discharging / negative
    charging. Note this is measured telemetry, not the solver's target (that
    would be schedule[marketService=="energy"])."""
    return [
        {
            "dispatch_interval": item["dispatchInterval"],
            "code": facility["facilityCode"],
            "power_mw": facility["initialMw"],
        }
        for facility in item["facilityScheduleDetails"]
        if facility["facilityCode"] in battery_codes
    ]


# --- per-field finishers: that field's raw row list in, final DataFrame out ---


def _finish_dedup(columns: list[str], subset: list[str], pivot: tuple[str, str] | None = None):
    """Every finisher here has to resolve the same duplicate problem: ~0.6% of
    intervals have both a regular file and a correction/annotation variant (see
    _LEADING_TIMESTAMP), and the correction should win.

    Sorting on is_correction puts regular (False) before correction (True), so
    drop_duplicates(keep="last") prefers the correction where one exists and
    falls back to the regular row where it doesn't."""

    def finish(rows: list[dict]) -> pd.DataFrame:
        df = (
            pd.DataFrame(rows, columns=columns + ["is_correction"])
            .sort_values("is_correction")
            .drop_duplicates(subset=subset, keep="last")
            .drop(columns="is_correction")
        )

        if pivot is not None:
            columns_col, values_col = pivot
            df = df.pivot(index="dispatch_interval", columns=columns_col, values=values_col)
        else:
            df = df.set_index("dispatch_interval")

        df.index = pd.to_datetime(df.index).tz_convert("Australia/Perth")

        # sort_values("is_correction") above uses pandas' default unstable sort,
        # which scrambles the relative order of same-valued rows (~99.4% of them
        # share is_correction=False) - restore chronological order explicitly
        # rather than relying on downstream operations (like pivot's implicit
        # sort) to fix it back up.
        return df.sort_index().astype(float)

    return finish


# name -> (extractor, finisher). This is the whole registry - add a field by
# adding one entry here.
_FIELDS = {
    "price": (_extract_price, _finish_dedup(["dispatch_interval", "energy_price"], ["dispatch_interval"])),
    "power": (
        _extract_power,
        _finish_dedup(
            ["dispatch_interval", "code", "power_mw"],
            ["dispatch_interval", "code"],
            pivot=("code", "power_mw"),
        ),
    ),
}

# For catalog.py to validate its entries against at import time.
FIELD_NAMES = list(_FIELDS)


def _get_dispatch_solution_rows(
    path: Path, field_names: list[str]
) -> tuple[dict[str, list[dict]], str | None]:
    """Walk one file, running the requested extractors against the single
    dispatch interval that file is authoritative for.

    Each file reports the realised solution for its own filename timestamp plus
    a couple of hours of forward-looking forecast intervals (which get
    superseded by later files), so only the entry matching the file's own
    timestamp is kept - and the walk stops as soon as it's found rather than
    parsing the rest of a ~20MB file.

    Every row is tagged with is_correction (derived from the filename, not the
    item) so the finishers can prefer corrections - see _finish_dedup.

    One bad file must never take down the whole ~45min run, so failures here
    are always caught and reported, never raised."""
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
        return rows_by_field, f"{type(e).__name__}: {e}"

    return rows_by_field, "expected dispatch interval not found in file"


def build_dispatch_solution_data(
    fields: list[str] | None = None, dates: list[str] | None = None, max_workers: int = 12
) -> dict[str, pd.DataFrame]:
    """Extract the requested fields (default: every field registered in
    _FIELDS) from every dispatchSolution file, or just the files for `dates`
    (e.g. ["2025-01-21"]) if given, returning {field_name: DataFrame}.

    The full corpus is ~300k files (~20MB each) - too slow to walk
    single-threaded (~3.75h measured), so it's parsed in parallel across
    `max_workers` processes (~45min measured on 12 workers). A `dates` subset
    is a handful of files and runs sequentially - useful for exploring a new
    field on a couple of days before committing to a full walk."""
    field_names = fields if fields is not None else list(_FIELDS)
    unknown = [name for name in field_names if name not in _FIELDS]
    if unknown:
        raise ValueError(f"unknown field(s) {unknown}; expected one of {list(_FIELDS)}")

    dataset_dir = raw_dataset_dir(DISPATCH_SOLUTION_DATASET)
    worker = functools.partial(_get_dispatch_solution_rows, field_names=field_names)

    if dates is None:
        paths = sorted(dataset_dir.rglob("*.json"))
        logger.info(f"Processing {len(paths)} files across {max_workers} workers for fields: {field_names}")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(worker, paths, chunksize=8))
    else:
        prefixes = tuple(f"ReferenceDispatchSolution_{date.replace('-', '')}" for date in dates)
        with os.scandir(dataset_dir) as it:
            paths = sorted(Path(entry.path) for entry in it if entry.name.startswith(prefixes))
        logger.info(f"Processing {len(paths)} files for fields: {field_names}")
        results = [worker(path) for path in paths]

    log_and_record_parse_failures(paths, [error for _, error in results], DISPATCH_SOLUTION_DATASET)

    dfs = {}
    for name in field_names:
        _extract, finish = _FIELDS[name]
        rows = [row for rows_by_field, _ in results for row in rows_by_field[name]]
        dfs[name] = finish(rows)
    return dfs
