import glob
import pandas as pd
from scripts.defaults import esr_codes
from utils.dirs import data_dir

# --- rated power lookup -------------------------------------------------
def get_rated_power_df(filename):
    facilities = pd.read_csv(data_dir / filename, usecols=["Facility Code", "System Size (MW)"])
    battery_max_rated_power = (
        facilities[facilities["Facility Code"].isin(esr_codes)]
        .set_index("Facility Code")["System Size (MW)"]
        # .to_dict()
    )

    missing_rating = set(esr_codes) - set(battery_max_rated_power)
    if missing_rating:
        print(f"warning: no rated power in facilities.csv for {sorted(missing_rating)}")
        
    return battery_max_rated_power

battery_max_rated_power = get_rated_power_df("facilities.csv")
print(battery_max_rated_power)

# # --- load SCADA ---------------------------------------------------------
# files = sorted(glob.glob(f"{data_dir}/SCADA_*.csv"))
# if not files:
#     raise FileNotFoundError(f"no SCADA_*.csv files in {data_dir}")


# def load_scada(path):
#     """Read one SCADA file, keeping only battery rows."""
#     chunk = pd.read_csv(path, usecols=["dispatchInterval", "code", "quantity"])
#     return chunk[chunk["code"].isin(battery_codes)]


# df = pd.concat((load_scada(f) for f in files), ignore_index=True)

# # --- clean & convert ----------------------------------------------------
# df["dispatchInterval"] = pd.to_datetime(df["dispatchInterval"], utc=True, errors="coerce")

# bad_timestamps = df["dispatchInterval"].isna().sum()
# if bad_timestamps:
#     print(f"warning: dropping {bad_timestamps:,} rows with unparseable timestamps")
#     df = df.dropna(subset=["dispatchInterval"])

# df["quantity"] = df["quantity"] * 12  # MWh per 5-min interval -> average MW
# df["quantity"] = df["quantity"] / df["code"].map(battery_max_rated_power) * 100  # MW -> % of rated power
# # df["quantity"] = -df["quantity"]  # load convention: positive = charging, negative = discharging

# # parse as UTC, then convert to WA time before splitting into date / time-of-day
# df["local"] = df["dispatchInterval"].dt.tz_convert("Australia/Perth")
# df["time_of_day"] = df["local"].dt.strftime("%H:%M")
# df["date_of_year"] = df["local"].dt.date


# print(df.head())

