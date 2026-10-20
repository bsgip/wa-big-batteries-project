import pandas as pd

from tools.io import load_clean
from tools.seasons import season_of, season_year_of


# Every feature column and its axis label: {column: label}.
# To add a feature: compute it in an add_* helper and give it a label here.
# Which analysis uses which column is decided in that analysis (BASELINES, PREDICTORS).
FEATURE_LABELS = {
    # past fleet SOC
    "soc_static_50": "Static 50% SOC (%)",
    "soc_overall_median": "Overall median SOC (%)",
    "soc_yesterday": "SOC Yesterday (%)",
    "soc_last_week": "SOC Last Week (%)",
    "soc_30d_median": "30D Median SOC (%)",
    "soc_7d_median": "7D Median SOC (%)",
    "soc_season_median": "Seasonal Median SOC (%)",

    # demand
    "peak_demand": "Peak Demand (MW)",
    "demand_range": "Daily demand range (MW)",

    # renewable generation
    "net_pv_midday": "Net PV generation (MWh) between 09:00-15:00",
    "net_wind_midday": "Net wind generation (MWh) between 09:00-15:00",
    "net_pv_wind_midday": "Net PV and wind generation (MWh) between 09:00-15:00",
    "net_pv_24h": "Net PV generation (MWh) 24h prior to 17:30",
    "net_wind_24h": "Net wind generation (MWh) 24h prior to 17:30",
    "net_pv_wind_24h": "Net PV and wind generation (MWh) 24h prior to 17:30",
    "dpv_midday": "Distributed PV (MWh) between 09:00-15:00",
    "dpv_24h": "Distributed PV (MWh) 24h prior to 17:30",

    # price measures
    "price_range": "Daily price range",
    "price_std_dev": "Daily price standard deviation",
}


def add_soc_history_features(df):
    """Add columns built from past fleet SOC: lags, rolling and seasonal medians, constants."""
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


def add_demand_features(df, demand, demand_range):
    """Add that day's peak demand and demand range (peak - trough)."""
    peak_demand = demand.resample("D").max()
    df["peak_demand"] = df.index.normalize().map(peak_demand)
    df["demand_range"] = df.index.normalize().map(demand_range)
    return df


def add_energy_window_features(df, name, mw):
    """Add <name>_midday (MWh, 09:00-15:00 that day) and <name>_24h (MWh, 24h before each row)."""
    # 5-min MW values, so MWh = sum / 12; min_count=1 keeps days with no data as NaN, not 0
    midday = mw.between_time("09:00", "15:00").resample("D").sum(min_count=1) / 12
    df[f"{name}_midday"] = df.index.normalize().map(midday)

    last_24h = mw.rolling("24h", closed="right").sum() / 12
    df[f"{name}_24h"] = df.index.map(last_24h)
    return df


def add_generation_features(df, pv_mw, wind_mw):
    """Add net PV, wind and PV+wind energy (MWh) for 09:00-15:00 and the 24h before each row."""
    df = add_energy_window_features(df, "net_pv", pv_mw)
    df = add_energy_window_features(df, "net_wind", wind_mw)
    df = add_energy_window_features(df, "net_pv_wind", pv_mw + wind_mw)
    return df


def build_feature_table(at="17:30"):
    """Fleet SOC at time `at` each day, with every feature column."""
    soc = load_clean("soc")
    df = pd.DataFrame(soc["fleet_soc_pct"].at_time(at))
    df = add_soc_history_features(df)

    demand = load_clean("demand")["demand"]
    demand_range = load_clean("demand_range")["range"]
    df = add_demand_features(df, demand, demand_range)

    pv_mw = load_clean("pv").sum(axis=1)
    wind_mw = load_clean("wind").sum(axis=1)
    df = add_generation_features(df, pv_mw, wind_mw)

    dpv_mw = load_clean("dpv")["dpv"]
    df = add_energy_window_features(df, "dpv", dpv_mw)

    # add new data sources here: load the data, then call its add_* helper
    return df
