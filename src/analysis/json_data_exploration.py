from pathlib import Path
from itertools import islice
from pprint import pprint
import ijson
import pandas as pd
import matplotlib.pyplot as plt

from utils.dirs import data_dir
from scripts.defaults import esr_codes


DROP = {
    "TESLA_PICTON_G1", 
    "PRDSO_WALPOLE_HG1", 
    "SBSOLAR1_CUNDERDIN_PV1"
}

battery_codes = [c for c in esr_codes if c not in DROP]


filepath = data_dir / "case_input.json"

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
                
                rows.append({
                    "dispatch_interval": dispatch_interval,
                    "code": code,
                    "value": scada["value"]
                })

    df = (
        pd.DataFrame(rows)
        .pivot(
            index="dispatch_interval", 
            columns="code", 
            values="value"
        )
    )

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




import requests
import json

url = "https://data.wa.aemo.com.au/public/market-data/wemde/caseInputData/current/ReferenceDispatchCase_202608110800.json"

resp = requests.get(url)

d = json.loads(resp.text)

json_str = json.dumps(d, indent=4)
with open("caseinput_json.json", "w") as f:
    f.write(json_str)
