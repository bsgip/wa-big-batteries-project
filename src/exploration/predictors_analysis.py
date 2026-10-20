import pandas as pd

from analysis.soc_features import FEATURE_LABELS, build_feature_table
from analysis.sse_events import sse_day_flags
from visualisation.plot_variables_correlation import plot_variables_correlation, plot_variables_correlation_by_season


pd.set_option("display.max_columns", None)

# Feature columns analysed as predictors of fleet SOC. Labels come from FEATURE_LABELS.
# Any column in FEATURE_LABELS can go here; it can also be in BASELINES.
PREDICTORS = [
    # "soc_yesterday",
    # "soc_last_week",
    # "soc_30d_median",
    # "soc_7d_median",
    "soc_season_median",
    # "peak_demand",
    # "demand_range",

    "net_pv_midday",
    "net_wind_midday",
    "net_pv_wind_midday",

    "net_pv_24h",
    "net_wind_24h",
    "net_pv_wind_24h",

    "dpv_midday",
    "dpv_24h",
]


def plot_correlations(df, flags, predictors=PREDICTORS):
    """Plot each predictor against fleet SOC: normal days, SSE days, and both overlaid."""
    normal = df.loc[~flags["is_sse"]]
    sse = df.loc[flags["is_sse"]]

    y_col = "fleet_soc_pct"
    y_label = "Fleet SOC (%) at 17:30"

    for col in predictors:
        plot_variables_correlation(
            normal, col, y_col, FEATURE_LABELS[col], y_label,
            title="Variables Correlation - Normal Days",
        )
        plot_variables_correlation(
            sse, col, y_col, FEATURE_LABELS[col], y_label,
            title="Variables Correlation - SSE Days",
            filename_suffix="_sse",
        )
        plot_variables_correlation(
            normal, col, y_col, FEATURE_LABELS[col], y_label,
            title="Variables Correlation\nNormal days: blue, SSE days: red",
            filename_suffix="_sse_overlay",
            overlay_sse=True,
            sse_df=sse,
        )
        plot_variables_correlation(
            normal, col, y_col, FEATURE_LABELS[col], y_label,
            title="Variables Correlation\nNormal days: blue, SSE days: red",
            filename_suffix="_sse_overlay_by_season",
            overlay_sse=True,
            sse_df=sse,
            color_by="season_group",
        )
        plot_variables_correlation_by_season(
            normal, col, y_col, FEATURE_LABELS[col], y_label,
            sse_df=sse,
            title=f"{FEATURE_LABELS[col]} vs fleet SOC by season\nNormal days coloured by season, SSE days in red",
        )


def main():
    df = build_feature_table()
    flags = sse_day_flags(df.index)
    plot_correlations(df, flags)


if __name__ == "__main__":
    main()
