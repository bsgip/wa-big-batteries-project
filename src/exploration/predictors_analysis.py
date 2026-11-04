import pandas as pd

from analysis.soc_features import FEATURE_LABELS, build_feature_table
from analysis.sse_events import sse_day_flags
from visualisation.plot_variables_correlation import plot_predictor_correlations


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


def main():
    df = build_feature_table()
    flags = sse_day_flags(df.index)
    plot_predictor_correlations(df, flags, PREDICTORS, FEATURE_LABELS)


if __name__ == "__main__":
    main()
