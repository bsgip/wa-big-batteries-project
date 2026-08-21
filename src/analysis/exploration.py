import logging
from itertools import islice
from pathlib import Path
from pprint import pprint

import ijson
import matplotlib.pyplot as plt
import pandas as pd

from tools.constants import DISPATCH_SOLUTION_DATASET, esr_codes
from tools.paths import case_input_dir, dispatch_solution_data_dir, predispatch_data_dir, raw_dataset_dir
from tools.utils import search_keyword

DROP = {"TESLA_PICTON_G1", "PRDSO_WALPOLE_HG1", "SBSOLAR1_CUNDERDIN_PV1"}

battery_codes = [c for c in esr_codes if c not in DROP]

case_input_path = case_input_dir / "ReferenceDispatchCase_202608120800.json"
dispatch_data_path = dispatch_solution_data_dir / "ReferenceDispatchSolution_202608140800.json"
predispatch_data_path = predispatch_data_dir / "ReferencePre-DispatchSolution_202608171100.json"


def explore_dispatch():
    print(f"Dispatch Solution")
    with open(dispatch_data_path, "rb") as f:
        data = ijson.items(f, "data.solutionData.item")
        print(next(data).keys())


def explore_case_input() -> pd.DataFrame:
    rows = []
    with open(case_input_path, "rb") as f:
        case_data = ijson.items(f, "data.caseData.item")

        for data in case_data:
            scada = data["scada"]

            for item in scada:
                if "demand" in item["tag"].lower() or "dpv" in item["tag"].lower():
                    rows.append(
                        {"dispatch_interval": data["dispatchInterval"], "tag": item["tag"], "value": item["value"]}
                    )

    return pd.DataFrame(rows).pivot(index="dispatch_interval", columns="tag", values="value")


df = explore_case_input()
print(df.head(20))

# search_keyword(dispatch_data_path, "scada")


def get_battery_df():
    rows = []
    with open(filepath, "rb") as f:
        case_data = ijson.items(f, "data.caseData.item", use_float=True)

        for item in case_data:
            dispatch_interval = item["dispatchInterval"]

            for scada in item["scada"]:
                code, field = scada["tag"].rsplit(".", 1)

                if code not in battery_codes:
                    continue

                if field != "chargeLevel":
                    continue

                rows.append({"dispatch_interval": dispatch_interval, "code": code, "value": scada["value"]})

    df = pd.DataFrame(rows).pivot(index="dispatch_interval", columns="code", values="value")

    df.to_csv("src/charge_level.csv")
    print(df.head())


# get_battery_df()


def get_df():
    df = (
        pd.read_csv("src/charge_level.csv", parse_dates=["dispatch_interval"])
        .assign(dispatch_interval=lambda d: d["dispatch_interval"].dt.tz_convert("Australia/Perth"))
        .set_index("dispatch_interval")
    )

    return df


def plot():
    df = get_df()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df)
    fig.savefig("charge_level.png", dpi=200)
