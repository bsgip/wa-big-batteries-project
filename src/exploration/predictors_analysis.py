import pandas as pd

from analysis.soc_predictors import (
    DRIVERS,
    SOC_BASELINES,
    add_demand_features,
    add_generation_features,
    add_soc_baselines,
    forecast_error_scores,
)
from analysis.sse_events import sse_day_flags
from data_extraction.catalog import load
from data_processing.demand import DEMAND_COLUMN, clean_demand
from data_processing.store import load_clean
from tools.paths import clean_data_dir
from visualisation.plot_variables_correlation import plot_variables_correlation


pd.set_option("display.max_columns", None)

delta_peak_trough_path = clean_data_dir / "demand_delta_peak_trough.csv"

# Columns to plot against fleet SOC. Labels come from SOC_BASELINES / DRIVERS.
# To plot every driver: PLOT_VARS = list(DRIVERS)
PLOT_VARS = [
    # "soc_yesterday",
    # "soc_last_week",
    # "soc_30d_median",
    # "soc_7d_median",
    "soc_season_median",
    # "peak_demand",
    # "delta_peak_trough",

    "net_pv_midday",
    "net_wind_midday",
    "net_pv_wind_midday",

    "net_pv_24h",
    "net_wind_24h",
    "net_pv_wind_24h",
]

LABELS = SOC_BASELINES | DRIVERS


def load_predictors():
    """Fleet SOC at 17:30 with every predictor column."""
    soc = load_clean("soc")
    df = pd.DataFrame(soc["fleet_soc_pct"].at_time("17:30"))
    df = add_soc_baselines(df)

    demand = clean_demand(load("demand")[DEMAND_COLUMN])
    delta = pd.read_csv(delta_peak_trough_path, index_col="date", parse_dates=True)["delta"]
    df = add_demand_features(df, demand, delta)

    pv_mw = load_clean("pv").sum(axis=1)
    wind_mw = load_clean("wind").sum(axis=1)
    df = add_generation_features(df, pv_mw, wind_mw)

    # add new data sources here: load the data, then call its add_* helper
    return df


def print_error_summary(df, flags):
    """Print SOC baseline error scores for all, normal and SSE days."""
    subsets = {
        "All": df,
        "Normal": df.loc[~flags["is_sse"]],
        "SSE": df.loc[flags["is_sse"]],
    }
    for day_type, d in subsets.items():
        print("\n======================================\n"
              f"Forecast Error Metrics - {day_type} Days"
              "\n======================================\n")
        print(forecast_error_scores(d, list(SOC_BASELINES)))


def plot_correlations(df, flags):
    """Plot each variable against fleet SOC: normal days, SSE days, and both overlaid."""
    normal = df.loc[~flags["is_sse"]]
    sse = df.loc[flags["is_sse"]]

    y_col = "fleet_soc_pct"
    y_label = "Fleet SOC (%) at 17:30"

    for col in PLOT_VARS:
        # plot_variables_correlation(
        #     normal, col, y_col, LABELS[col], y_label,
        #     title="Variables Correlation - Normal Days",
        # )
        # plot_variables_correlation(
        #     sse, col, y_col, LABELS[col], y_label,
        #     title="Variables Correlation - SSE Days",
        #     filename_suffix="_sse",
        # )
        # plot_variables_correlation(
        #     normal, col, y_col, LABELS[col], y_label,
        #     title="Variables Correlation\nNormal days: blue, SSE days: red",
        #     filename_suffix="_sse_overlay",
        #     overlay_sse=True,
        #     sse_df=sse,
        # )
        plot_variables_correlation(
            normal, col, y_col, LABELS[col], y_label,
            title="Variables Correlation\nNormal days: blue, SSE days: red",
            filename_suffix="_sse_overlay_by_season",
            overlay_sse=True,
            sse_df=sse,
            color_by="season",
        )


def main():
    df = load_predictors()
    flags = sse_day_flags(df.index)

    print_error_summary(df, flags)
    plot_correlations(df, flags)


if __name__ == "__main__":
    main()
