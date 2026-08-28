import logging
from itertools import islice
from pathlib import Path
from pprint import pprint

import ijson
import matplotlib.pyplot as plt
import pandas as pd

from tools.constants import DISPATCH_SOLUTION_DATASET, battery_codes
from tools.paths import case_input_dir, dispatch_solution_data_dir, predispatch_data_dir, raw_dataset_dir
from tools.utils import search_keyword

case_input_path = case_input_dir / "ReferenceDispatchCase_202608120800.json"
dispatch_data_path = dispatch_solution_data_dir / "ReferenceDispatchSolution_202608140800.json"
predispatch_data_path = predispatch_data_dir / "ReferencePre-DispatchSolution_202608171100.json"


def explore_dispatch():
    print(f"Dispatch Solution")
    with open(dispatch_data_path, "rb") as f:
        solution_data = ijson.items(f, "data.solutionData.item", use_float=True)
        data = next(solution_data)
        # pprint(list(data.keys()))

        fields = [
            # "schedule",
            "facilityScheduleDetails",
            # "availableQuantities",
            # "marketShortfalls",
            # "prices",
            # "priceSetting",
        ]

        for field in fields:
            for d in data[f"{field}"]:
                if any(code in d["facilityCode"] for code in battery_codes):
                    print()
                    print(f"--- {field} ---")
                    print()
                    pprint(d)


# explore_dispatch()


def explore_case_input() -> pd.DataFrame:
    rows = []
    with open(case_input_path, "rb") as f:
        case_data = ijson.items(f, "data.caseData.item")
        tag_set = set()

        for data in case_data:
            scada = data["scada"]

            for item in scada:
                tag = item["tag"]

                if "tranche" in tag.lower():
                    print(tag)

        pprint(tag_set)
        #     for item in scada:
        #         if "demand" in item["tag"].lower() or "dpv" in item["tag"].lower():
        #             rows.append(
        #                 {"dispatch_interval": data["dispatchInterval"], "tag": item["tag"], "value": item["value"]}
        #             )

    # return pd.DataFrame(rows).pivot(index="dispatch_interval", columns="tag", values="value")


explore_case_input()

# df = explore_case_input()
# print(df.head(20))

# search_keyword(dispatch_data_path, "scada")


def get_df(filepath: str):
    df = (
        pd.read_csv(filepath, parse_dates=["dispatch_interval"])
        .assign(dispatch_interval=lambda d: d["dispatch_interval"].dt.tz_convert("Australia/Perth"))
        .set_index("dispatch_interval")
    )

    return df


def plot(filepath: str, saved_filename: str):
    df = get_df(filepath)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df)
    fig.savefig(saved_filename, dpi=200)
