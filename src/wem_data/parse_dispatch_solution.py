"""Parse energy price and battery power (initialMw) out of every
dispatchSolution/dispatchData file downloaded by download.py. Both fields
live in the same JSON files, so both are extracted in a single pass over the
(very large - ~300k files) corpus rather than walking it twice.
"""

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
from wem_data.download import log_and_record_parse_failures

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


def _get_price_and_power_rows(path: Path) -> tuple[dict | None, list[dict], str | None]:
    """Each dispatchData file reports the realised solution for its own
    filename timestamp plus a couple of hours of forward-looking forecast
    intervals (which get superseded by later files). Keep only the entry
    matching the file's own timestamp so each dispatch interval appears once.

    A small fraction (~0.6%) of intervals also have a correction/annotation
    variant filename for the same timestamp as a regular file (see above) -
    `is_correction` marks those so build_price_and_power_df can prefer the
    correction over the regular file where both exist for the same interval.

    The third return value is an error message if this file couldn't be
    parsed (e.g. truncated/corrupted download), else None - one bad file
    must never take down the whole ~45min run, so failures here are always
    caught and reported, never raised."""
    match = _LEADING_TIMESTAMP.match(path.stem)
    if match is None:
        return None, [], "filename doesn't match the expected pattern"

    is_correction = match.end() < len(path.stem)
    dt = datetime.strptime(match.group(1), "%Y%m%d%H%M")
    expected_interval = dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")

    try:
        with open(path, "rb") as f:
            solution_data = ijson.items(f, "data.solutionData.item", use_float=True)

            for item in solution_data:
                if item["dispatchInterval"] != expected_interval:
                    continue

                price_row = {
                    "dispatch_interval": item["dispatchInterval"],
                    "energy_price": item["prices"]["energy"],
                    "is_correction": is_correction,
                }
                power_rows = [
                    {
                        "dispatch_interval": item["dispatchInterval"],
                        "code": facility["facilityCode"],
                        "power_mw": facility["initialMw"],
                        "is_correction": is_correction,
                    }
                    for facility in item["facilityScheduleDetails"]
                    if facility["facilityCode"] in battery_codes
                ]
                return price_row, power_rows, None
    except Exception as e:
        return None, [], f"{type(e).__name__}: {e}"

    return None, [], "expected dispatch interval not found in file"


def _rows_to_dfs(price_rows: list[dict], power_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    # sort regular (False) before correction (True) rows so drop_duplicates
    # keep="last" prefers the correction when both exist for one interval,
    # and falls back to the regular row when no correction exists
    price_df = (
        pd.DataFrame(price_rows, columns=["dispatch_interval", "energy_price", "is_correction"])
        .sort_values("is_correction")
        .drop_duplicates(subset="dispatch_interval", keep="last")
        .set_index("dispatch_interval")
        .drop(columns="is_correction")
    )

    power_long = (
        pd.DataFrame(power_rows, columns=["dispatch_interval", "code", "power_mw", "is_correction"])
        .sort_values("is_correction")
        .drop_duplicates(subset=["dispatch_interval", "code"], keep="last")
        .drop(columns="is_correction")
    )
    power_df = power_long.pivot(index="dispatch_interval", columns="code", values="power_mw")

    for df in (price_df, power_df):
        df.index = pd.to_datetime(df.index).tz_convert("Australia/Perth")

    # sort_values("is_correction") above uses pandas' default unstable sort,
    # which scrambles the relative order of same-valued rows (~99.4% of them
    # share is_correction=False) - restore chronological order explicitly
    # rather than relying on downstream operations (like pivot's implicit
    # sort) to fix it back up.
    return price_df.sort_index(), power_df.sort_index()


def build_price_and_power_df(
    dates: list[str] | None = None, max_workers: int = 12
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract energy price and per-battery power (MW) from every
    dispatchSolution file, or just the files for `dates` (e.g.
    ["2025-01-21"]) if given.

    The full corpus is ~300k files (~20MB each) - too slow to walk
    single-threaded (~3.75h measured), so it's parsed in parallel across
    `max_workers` processes (~45min measured on 12 workers). A `dates`
    subset is a handful of files and runs sequentially."""
    dataset_dir = raw_dataset_dir(DISPATCH_SOLUTION_DATASET)

    if dates is None:
        paths = sorted(dataset_dir.rglob("*.json"))
        logger.info(f"Processing {len(paths)} files across {max_workers} workers")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_get_price_and_power_rows, paths, chunksize=8))
    else:
        prefixes = tuple(f"ReferenceDispatchSolution_{date.replace('-', '')}" for date in dates)
        with os.scandir(dataset_dir) as it:
            paths = sorted(Path(entry.path) for entry in it if entry.name.startswith(prefixes))
        logger.info(f"Processing {len(paths)} files")
        results = [_get_price_and_power_rows(path) for path in paths]

    price_rows = [price_row for price_row, _, _ in results if price_row is not None]
    power_rows = [row for _, power_rows, _ in results for row in power_rows]
    log_and_record_parse_failures(paths, [error for _, _, error in results], DISPATCH_SOLUTION_DATASET)

    price_df, power_df = _rows_to_dfs(price_rows, power_rows)
    return price_df.astype(float), power_df.astype(float)


def build_energy_price_series_for_dates(dates: list[str]) -> pd.Series:
    price_df, _ = build_price_and_power_df(dates=dates)
    return price_df["energy_price"].sort_index().fillna(0)
