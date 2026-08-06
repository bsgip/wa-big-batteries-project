import glob
import math
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scripts.defaults import battery_codes
from utils.dirs import data_dir

# --- rated power lookup -------------------------------------------------
facilities = pd.read_csv(data_dir / "facilities.csv", usecols=["Facility Code", "System Size (MW)"])
battery_max_rated_power = (
    facilities[facilities["Facility Code"].isin(battery_codes)]
    .set_index("Facility Code")["System Size (MW)"]
    .to_dict()
)

missing_rating = set(battery_codes) - set(battery_max_rated_power)
if missing_rating:
    print(f"warning: no rated power in facilities.csv for {sorted(missing_rating)}")

# --- load SCADA ---------------------------------------------------------
files = sorted(glob.glob(f"{data_dir}/SCADA_*.csv"))
if not files:
    raise FileNotFoundError(f"no SCADA_*.csv files in {data_dir}")


def load_scada(path):
    """Read one SCADA file, keeping only battery rows."""
    chunk = pd.read_csv(path, usecols=["dispatchInterval", "code", "quantity"])
    return chunk[chunk["code"].isin(battery_codes)]


df = pd.concat((load_scada(f) for f in files), ignore_index=True)

# --- clean & convert ----------------------------------------------------
df["dispatchInterval"] = pd.to_datetime(df["dispatchInterval"], utc=True, errors="coerce")

bad_timestamps = df["dispatchInterval"].isna().sum()
if bad_timestamps:
    print(f"warning: dropping {bad_timestamps:,} rows with unparseable timestamps")
    df = df.dropna(subset=["dispatchInterval"])

df["quantity"] = df["quantity"] * 12  # MWh per 5-min interval -> average MW
df["quantity"] = df["quantity"] / df["code"].map(battery_max_rated_power) * 100  # MW -> % of rated power
# df["quantity"] = -df["quantity"]  # load convention: positive = charging, negative = discharging

# parse as UTC, then convert to WA time before splitting into date / time-of-day
df["local"] = df["dispatchInterval"].dt.tz_convert("Australia/Perth")
df["time_of_day"] = df["local"].dt.strftime("%H:%M")
df["date_of_year"] = df["local"].dt.date

# --- build one 2-D grid per battery -------------------------------------
# rows = time of day (sorted 00:00 -> 23:55), columns = date
grids = {}
for code in battery_codes:
    unit_df = df[df["code"] == code]
    if unit_df.empty:
        print(f"{code}: no data found in {data_dir}, skipping")
        continue

    grids[code] = unit_df.pivot_table(
        index="time_of_day",
        columns="date_of_year",
        values="quantity",
    ).sort_index()

if not grids:
    raise ValueError("no battery data found — check battery_codes against the SCADA files")

# --- individual figures -------------------------------------------------
for code, grid in grids.items():
    if code == "PRDSO_WALPOLE_HG1":
        fig = go.Figure(data=go.Heatmap(
            z=grid.values,
            y=grid.index,
            x=grid.columns,
            colorscale="rdbu_r", zmid=0,
            colorbar={"title": "% of rated power<br>(- charging / + discharging)"},
            hovertemplate="%{y} %{x}:<br>%{z:.1f}%<extra></extra>",
        ))
        fig.update_layout(
            title=code,
            yaxis_title="Time of day (AWST)",
            xaxis_title="Date",
        )
        fig.update_yaxes(nticks=6)
        fig.update_xaxes(nticks=6)
        fig.show()

# --- combined view ------------------------------------------------------
# shared colour scale so colours are directly comparable across batteries
zmax = max(grid.abs().max().max() for grid in grids.values())

n = len(grids)
cols = min(4, n)
rows = math.ceil(n / cols)

combined_fig = make_subplots(rows=rows, cols=cols, subplot_titles=list(grids.keys()))
for i, (code, grid) in enumerate(grids.items()):
    row, col = i // cols + 1, i % cols + 1
    combined_fig.add_trace(
        go.Heatmap(
            z=grid.values,
            y=grid.index,
            x=grid.columns,
            colorscale="rdbu_r", zmid=0, zmin=-zmax, zmax=zmax,
            showscale=(i == 0),
            colorbar={"title": "% of rated power<br>(- charging / + discharging)"},
            hovertemplate="%{y} %{x}:<br>%{z:.1f}%<extra></extra>",
        ),
        row=row, col=col,
    )

combined_fig.update_yaxes(nticks=6)
combined_fig.update_xaxes(nticks=6)
combined_fig.update_layout(title="Battery Output — SCADA Data", height=350 * rows)
combined_fig.show()
