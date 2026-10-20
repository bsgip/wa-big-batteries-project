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
    """Bias, median error, MAE and RMSE of each forecast column against the target."""
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
