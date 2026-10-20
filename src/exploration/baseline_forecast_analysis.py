import pandas as pd

from analysis.baseline_forecast import BASELINES, forecast_error_scores
from analysis.soc_features import build_feature_table
from analysis.sse_events import sse_day_flags


pd.set_option("display.max_columns", None)


def print_error_summary(df, flags):
    """Print baseline forecast error scores for all, normal and SSE days."""
    subsets = {
        "All": df,
        "Normal": df.loc[~flags["is_sse"]],
        "SSE": df.loc[flags["is_sse"]],
    }
    for day_type, d in subsets.items():
        print("\n======================================\n"
              f"Forecast Error Metrics - {day_type} Days"
              "\n======================================\n")
        print(forecast_error_scores(d, BASELINES))


def main():
    df = build_feature_table()
    flags = sse_day_flags(df.index)
    print_error_summary(df, flags)


if __name__ == "__main__":
    main()
