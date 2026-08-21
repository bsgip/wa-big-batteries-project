"""Parse and concatenate battery chargeLevel scada tags out of every
caseInputData file downloaded by download.py.
"""

from pathlib import Path

import logging
import ijson
import pandas as pd

from tools.constants import CASE_INPUT_DATASET, esr_codes
from tools.paths import raw_dataset_dir

# facilities in esr_codes that aren't actually batteries
DROP = {"TESLA_PICTON_G1", "PRDSO_WALPOLE_HG1", "SBSOLAR1_CUNDERDIN_PV1"}
battery_codes = [c for c in esr_codes if c not in DROP]


logger = logging.getLogger(__name__)


def _get_charge_level_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, "rb") as f:
        case_data = ijson.items(f, "data.caseData.item", use_float=True)

        for item in case_data:
            dispatch_interval = item["dispatchInterval"]

            for scada in item["scada"]:
                code, field = scada["tag"].rsplit(".", 1)

                if code not in battery_codes:
                    continue
                if field != "chargeLevel":
                    continue

                rows.append(
                    {
                        "dispatch_interval": dispatch_interval,
                        "code": code,
                        "value": scada["value"],
                    }
                )
    return rows


def build_charge_level_df() -> pd.DataFrame:
    rows = []
    for path in sorted(raw_dataset_dir(CASE_INPUT_DATASET).rglob("*.json")):
        logger.info(f"Processing file: {path}")
        rows.extend(_get_charge_level_rows(path))

    return pd.DataFrame(rows).pivot(index="dispatch_interval", columns="code", values="value")


def _get_demand_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, "rb") as f:
        case_data = ijson.items(f, "data.caseData.item")

        for data in case_data:
            scada = data["scada"]

            for item in scada:
                if "demand" in item["tag"].lower() or "dpv" in item["tag"].lower():
                    rows.append(
                        {"dispatch_interval": data["dispatchInterval"], "tag": item["tag"], "value": item["value"]}
                    )
    return rows


def build_demand_df() -> pd.DataFrame:
    rows = []
    for path in sorted(raw_dataset_dir(CASE_INPUT_DATASET).rglob("*.json")):
        logger.info(f"Processing file: {path}")
        rows.extend(_get_demand_rows(path))

    return pd.DataFrame(rows).pivot(index="dispatch_interval", columns="tag", values="value")
