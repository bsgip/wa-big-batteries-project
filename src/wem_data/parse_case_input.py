"""Parse battery chargeLevel (SOC) and system demand out of every
caseInputData file downloaded by download.py. Both fields live in the same
JSON files (and each file is ~450MB), so both are extracted in a single
parallel pass over the corpus rather than walking it twice.
"""

import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import ijson
import pandas as pd

from tools.constants import CASE_INPUT_DATASET, battery_codes
from tools.paths import raw_dataset_dir
from wem_data.download import log_and_record_parse_failures

logger = logging.getLogger(__name__)


def _get_charge_level_and_demand_rows(path: Path) -> tuple[list[dict], list[dict], str | None]:
    """One bad file must never take down the whole run - failures here are
    always caught and reported, never raised. Keeps whatever rows were
    parsed before a failure rather than discarding the whole (huge, ~450MB,
    ~1 day of data) file."""
    charge_level_rows = []
    demand_rows = []

    try:
        with open(path, "rb") as f:
            case_data = ijson.items(f, "data.caseData.item", use_float=True)

            for item in case_data:
                dispatch_interval = item["dispatchInterval"]

                for scada in item["scada"]:
                    tag = scada["tag"]

                    if "." in tag:
                        code, field = tag.rsplit(".", 1)
                        if code in battery_codes and field == "chargeLevel":
                            charge_level_rows.append(
                                {"dispatch_interval": dispatch_interval, "code": code, "value": scada["value"]}
                            )

                    if "demand" in tag.lower() or "dpv" in tag.lower():
                        demand_rows.append(
                            {"dispatch_interval": dispatch_interval, "tag": tag, "value": scada["value"]}
                        )
    except Exception as e:
        return charge_level_rows, demand_rows, f"{type(e).__name__}: {e}"

    return charge_level_rows, demand_rows, None


def build_charge_level_and_demand_df(max_workers: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract battery SOC (chargeLevel, MWh) and system demand from every
    caseInputData file in one pass. The corpus is only ~1,056 files, but
    each is ~450MB (~70GB total) - too slow single-threaded, so it's parsed
    in parallel across `max_workers` processes."""
    paths = sorted(raw_dataset_dir(CASE_INPUT_DATASET).rglob("*.json"))
    logger.info(f"Processing {len(paths)} files across {max_workers} workers")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_get_charge_level_and_demand_rows, paths, chunksize=1))

    charge_level_rows = [row for rows, _, _ in results for row in rows]
    demand_rows = [row for _, rows, _ in results for row in rows]
    log_and_record_parse_failures(paths, [error for _, _, error in results], CASE_INPUT_DATASET)

    charge_level_df = (
        pd.DataFrame(charge_level_rows).pivot(index="dispatch_interval", columns="code", values="value").astype(float)
    )
    demand_df = (
        pd.DataFrame(demand_rows).pivot(index="dispatch_interval", columns="tag", values="value").astype(float)
    )

    for df in (charge_level_df, demand_df):
        df.index = pd.to_datetime(df.index).tz_convert("Australia/Perth")

    return charge_level_df, demand_df
