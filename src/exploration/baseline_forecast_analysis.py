import pandas as pd

from analysis.baseline_forecast import forecast_error_scores_by_day_type
from analysis.soc_features import build_feature_table
from analysis.sse_events import sse_day_flags


pd.set_option("display.max_columns", None)


def main():
    df = build_feature_table()
    flags = sse_day_flags(df.index)

    for day_type, scores in forecast_error_scores_by_day_type(df, flags).items():
        print("\n======================================\n"
              f"Forecast Error Metrics - {day_type} Days"
              "\n======================================\n")
        print(scores)


if __name__ == "__main__":
    main()
