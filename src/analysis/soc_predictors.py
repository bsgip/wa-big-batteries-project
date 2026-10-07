import pandas as pd

from tools.seasons import season_of, season_year_of


# {column: axis label}. Add a new column here once, then compute it in an add_* helper.
SOC_BASELINES = {
    "soc_static_50":      "Static 50% SOC (%)",
    "soc_overall_median": "Overall median SOC (%)",
    "soc_yesterday":      "SOC Yesterday (%)",
    "soc_last_week":      "SOC Last Week (%)",
    "soc_30d_median":     "30D Median SOC (%)",
    "soc_7d_median":      "7D Median SOC (%)",
    "soc_season_median":  "Seasonal Median SOC (%)",
}

DRIVERS = {
    "peak_demand":        "Peak Demand (MW)",
    "delta_peak_trough":  "Demand delta peak-trough (MWh)",
    "net_pv_midday":      "Net PV generation (MWh) between 09:00-15:00",
    "net_wind_midday":    "Net wind generation (MWh) between 09:00-15:00",
    "net_pv_wind_midday": "Net PV and wind generation (MWh) between 09:00-15:00",
    "net_pv_24h":         "Net PV generation (MWh) 24h prior to 17:30",
    "net_wind_24h":       "Net wind generation (MWh) 24h prior to 17:30",
    "net_pv_wind_24h":    "Net PV and wind generation (MWh) 24h prior to 17:30",
}


def add_soc_baselines(df):
    """Add simple SOC forecasts built from past fleet SOC."""
    df["soc_yesterday"] = df["fleet_soc_pct"].shift(1)
    df["soc_last_week"] = df["fleet_soc_pct"].shift(7)
    df["soc_30d_median"] = df["fleet_soc_pct"].rolling("30D", closed="left", min_periods=28).median()
    df["soc_7d_median"] = df["fleet_soc_pct"].rolling("7D", closed="left", min_periods=7).median()
    df["soc_overall_median"] = df["fleet_soc_pct"].median()
    df["soc_static_50"] = 50

    # Seasonal Median
    df["season"] = season_of(df.index)
    df["season_year"] = season_year_of(df.index)

    df["soc_season_median"] = (
    df.groupby(["season", "season_year"], observed=True)["fleet_soc_pct"]
      .transform(lambda s: s.expanding(min_periods=7).median().shift(1))
    )

    return df


def add_demand_features(df, demand, delta):
    """Add that day's peak demand and demand peak-trough delta."""
    peak_demand = demand.resample("D").max()
    df["peak_demand"] = df.index.normalize().map(peak_demand)
    df["delta_peak_trough"] = df.index.normalize().map(delta)
    return df


def add_generation_features(df, pv_mw, wind_mw):
    """Add PV/wind energy (MWh) for 09:00-15:00 that day and for the 24h before each row."""
    pv_wind_mw = pv_mw + wind_mw
    sources = {"pv": pv_mw, "wind": wind_mw, "pv_wind": pv_wind_mw}

    # 5-min MW values, so MWh = sum / 12
    for name, mw in sources.items():
        midday = mw.between_time("09:00", "15:00").resample("D").sum() / 12
        df[f"net_{name}_midday"] = df.index.normalize().map(midday)

        last_24h = mw.rolling("24h", closed="right").sum() / 12
        df[f"net_{name}_24h"] = df.index.map(last_24h)

    return df


def forecast_error_scores(df, preds, target="fleet_soc_pct"):
    """Bias, median error, MAE and RMSE of each predictor against the target."""
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
