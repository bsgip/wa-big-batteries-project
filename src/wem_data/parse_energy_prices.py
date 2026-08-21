import logging
import os
from datetime import datetime
from pathlib import Path

import ijson
import pandas as pd

from tools.constants import DISPATCH_SOLUTION_DATASET
from tools.paths import raw_dataset_dir

logger = logging.getLogger(__name__)


def _get_energy_price_rows(path: Path) -> list[dict]:
    """Each dispatchData file reports the realised price for its own
    filename timestamp plus a couple of hours of forward-looking forecast
    intervals (which get superseded by later files). Keep only the entry
    matching the file's own timestamp so each dispatch interval appears once."""
    dt = datetime.strptime(path.stem.rsplit("_", 1)[-1], "%Y%m%d%H%M")
    expected_interval = dt.strftime("%Y-%m-%dT%H:%M:%S+08:00")

    rows = []
    with open(path, "rb") as f:
        solution_data = ijson.items(f, "data.solutionData.item", use_float=True)

        for item in solution_data:
            if item["dispatchInterval"] != expected_interval:
                continue
            rows.append({"dispatch_interval": item["dispatchInterval"], "energy_price": item["prices"]["energy"]})
            break

    return rows


def build_energy_price_df() -> pd.DataFrame:
    rows = []
    for path in sorted(raw_dataset_dir(DISPATCH_SOLUTION_DATASET).rglob("*.json")):
        logger.info(f"Processing file: {path}")
        rows.extend(_get_energy_price_rows(path))

    return pd.DataFrame(rows).set_index("dispatch_interval")


def build_energy_price_df_for_dates(dates: list[str]) -> pd.DataFrame:
    """Like build_energy_price_df, but only reads files whose timestamped
    filename falls on one of `dates` (e.g. "2025-01-21"), instead of walking
    the whole dispatchData folder - much faster when only a handful of dates
    are needed. The folder holds hundreds of thousands of files, so this
    scans it once with os.scandir rather than re-globbing per date."""
    prefixes = tuple(f"ReferenceDispatchSolution_{date.replace('-', '')}" for date in dates)

    with os.scandir(raw_dataset_dir(DISPATCH_SOLUTION_DATASET)) as it:
        matched_paths = sorted(Path(entry.path) for entry in it if entry.name.startswith(prefixes))

    rows = []
    for path in matched_paths:
        logger.info(f"Processing file: {path}")
        rows.extend(_get_energy_price_rows(path))

    return pd.DataFrame(rows, columns=["dispatch_interval", "energy_price"]).set_index("dispatch_interval")


def build_energy_price_series_for_dates(dates: list[str]) -> pd.Series:
    df = build_energy_price_df_for_dates(dates)
    df.index = pd.to_datetime(df.index)
    return df["energy_price"].sort_index().fillna(0)
