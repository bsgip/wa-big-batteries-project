import pandas as pd


# Feature columns that are themselves a fleet SOC forecast (in %), scored against the actual SOC.
# Any column in soc_features.FEATURE_LABELS can go here; it can also be in PREDICTORS.
BASELINES = [
    "soc_static_50",
    "soc_overall_median",
    "soc_yesterday",
    "soc_last_week",
    "soc_30d_median",
    "soc_7d_median",
    "soc_season_median",
]


def forecast_error_scores(df, preds, target="fleet_soc_pct"):
    """Score each forecast column against the target.

    Args:
        df: Daily frame holding the forecast columns and the target.
        preds: Forecast column names (fleet SOC forecasts, in %).
        target: Column with the actual value.

    Returns:
        One row per forecast, sorted by MAE, with columns bias, median_err, MAE, RMSE, n
        and rmse_mae_ratio.
    """
    err = df[preds].sub(df[target], axis=0).dropna()

    scores = pd.DataFrame({
        "bias": err.mean(),
        "median_err": err.median(),
        "MAE": err.abs().mean(),
        "RMSE": (err ** 2).mean() ** 0.5,
    }).sort_values("MAE")

    scores["n"] = len(err)
    scores["rmse_mae_ratio"] = scores["RMSE"] / scores["MAE"]
    return scores


def forecast_error_scores_by_day_type(df, flags, preds=BASELINES):
    """Score each forecast separately for all days, normal days and SSE days.

    Args:
        df: Daily feature table, e.g. from soc_features.build_feature_table.
        flags: Frame with a boolean is_sse column on the same index, from sse_events.sse_day_flags.
        preds: Forecast column names.

    Returns:
        {"All" | "Normal" | "SSE": forecast_error_scores table}.
    """
    subsets = {
        "All": df,
        "Normal": df.loc[~flags["is_sse"]],
        "SSE": df.loc[flags["is_sse"]],
    }
    return {day_type: forecast_error_scores(d, preds) for day_type, d in subsets.items()}
