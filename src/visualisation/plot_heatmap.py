import math

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tools.constants import esr_codes

from tools.paths import data_dir


def plot_heatmap(df):
    # --- build one 2-D grid per battery -------------------------------------
    # rows = time of day (sorted 00:00 -> 23:55), columns = date
    grids = {}
    for code in esr_codes:
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
            fig = go.Figure(
                data=go.Heatmap(
                    z=grid.values,
                    y=grid.index,
                    x=grid.columns,
                    colorscale="rdbu_r",
                    zmid=0,
                    colorbar={"title": "% of rated power<br>(- charging / + discharging)"},
                    hovertemplate="%{y} %{x}:<br>%{z:.1f}%<extra></extra>",
                )
            )
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
                colorscale="rdbu_r",
                zmid=0,
                zmin=-zmax,
                zmax=zmax,
                showscale=(i == 0),
                colorbar={"title": "% of rated power<br>(- charging / + discharging)"},
                hovertemplate="%{y} %{x}:<br>%{z:.1f}%<extra></extra>",
            ),
            row=row,
            col=col,
        )

    combined_fig.update_yaxes(nticks=6)
    combined_fig.update_xaxes(nticks=6)
    combined_fig.update_layout(title="Battery Output — SCADA Data", height=350 * rows)
    combined_fig.show()
