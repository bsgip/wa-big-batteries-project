import logging
from itertools import islice
from pathlib import Path
from pprint import pprint

import ijson
import matplotlib.pyplot as plt
import pandas as pd

from tools.constants import DISPATCH_SOLUTION_DATASET, battery_codes
from tools.utils import search_keyword



def main():
    pd.set_option("display.max_columns", None)


    # Temp dirs
    root_dir = Path.cwd()
    data_dir = root_dir / "data" / "sample_raw_data"


    case_input_path = data_dir / "ReferenceDispatchCase_202608120800.json"
    dispatch_data_path = data_dir / "ReferenceDispatchSolution_202608140800.json"
    predispatch_data_path = data_dir / "ReferencePre-DispatchSolution_202608171100.json"
    
    # explore_case_input(case_input_path)
    # explore_forecast(case_input_path)
    explore_scada(case_input_path)



def explore_dispatch(dispatch_data_path):
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




def get_bid_stack_rows(case_input_path):
    with open(case_input_path, "rb") as f:
        rows = []
        case_data = ijson.items(f, "data.caseData.item", use_float=True)

        for item in case_data:
            dispatch_interval = item["dispatchInterval"]

            for market_data in item["markets"]["energy"]["facilities"]:
                if market_data["facilityCode"] in battery_codes:
                    for tranche in market_data["tranches"]:
                        rows.append({
                            "dispatch_interval": dispatch_interval,
                            "code": market_data["facilityCode"],
                            "quantity": tranche["quantity"],
                            "submitted_price": tranche["submittedPrice"],
                            "tranche": tranche["tranche"],
                            "submission_id": market_data["submissionId"],
                            "capacity_type": tranche["capacityType"],
                            "fuel_type": tranche["fuelType"],
                            "lfa_price": tranche["lfaPrice"],
                            "notice_time": tranche["noticeTime"],
                        })
        
        return rows



def save_df(rows, data_dir):
    df = pd.DataFrame(rows).set_index("dispatch_interval")
    try:
        filename = data_dir / "processed_data" / "bidstack.csv"
        df.to_csv(filename)
        print(f"csv saved to {filename}")
    except Exception as e:
        print(e)




def explore_case_input(case_input_path):
    rows = []
    with open(case_input_path, "rb") as f:
        case_data = ijson.items(f, "data.caseData.item", use_float=True)
        tag_set = set()
        
        data = next(case_data)
        
        pprint(list(data.keys()))
        
        for field in data:
            item = data[field]

            print(f"\n--- Fieldname: {field} ---")
            print(f"Type: {type(item)}")
            print(f"Example:")
            
            if isinstance(item, str):
                pprint(item)
            
            if isinstance(item, list):
                pprint(item[0])
                pprint(item[-1])
                
            if isinstance(item, dict):
                dict_keys = list(item.keys())
                print(f"Dict keys:")
                pprint(dict_keys)
                
                for key in dict_keys:
                    dict_content = data[field][key]
                    print(f"\nKey: {key}")
                    print(f"Type: {type(dict_content)}")
                    
                    if isinstance(dict_content, str):
                        print(dict_content)
                    
                    if isinstance(dict_content, list):
                        try:
                            print("Example:")
                            pprint(item[0])
                            pprint(item[-1])
                        except KeyError:
                            if len(item) < 1:
                                print(f"List is empty")
                            else:
                                print(f"Unknown error")
                            continue
                    
                    if isinstance(dict_content, dict):
                        dict_keys = list(dict_content.keys())
                        print(f"Dict keys:")
                        pprint(dict_keys)


def explore_forecast(case_input_path):
    with open(case_input_path, "rb") as f:
        forecast = ijson.items(f, "data.caseData.item.unconstrainedForecast", use_float=True)

        lst = []
        codes = set()
        for item in islice(forecast, 22, 24):
            for tag in item:
                print(tag)
                codes.add(tag["facilityCode"])
                if tag["facilityCode"] in battery_codes:
                    lst.append(tag)
                    
        pprint(lst)
        pprint(codes)



def explore_rcm(case_input_path):
    with open(case_input_path, "rb") as f:
        rcm = ijson.items(f, "data.caseData.item.rcmData", use_float=True)
        facilities = set()
        for item in rcm:
            facilities.add(item["facilityCode"])
            


def explore_scada(case_input_path):
    with open(case_input_path, "rb") as f:
        rows = []
        case_data = ijson.items(f, "data.caseData.item", use_float=True)

        for item in case_data:
            for scada in item["scada"]:
                if "dispatchCondition" in scada["tag"]:
                    rows.append({
                        "dispatch_interval": item["dispatchInterval"],
                        "tag": scada["tag"],
                        "value": scada["value"]
                    })
    
    df = pd.DataFrame(rows)
    print(df.head())



def get_df(filepath: str):
    df = (
        pd.read_csv(filepath, parse_dates=["dispatch_interval"])
        .assign(dispatch_interval=lambda d: d["dispatch_interval"].dt.tz_convert("Australia/Perth"))
        .set_index("dispatch_interval")
    )

    return df



if __name__ == "__main__":
    main()