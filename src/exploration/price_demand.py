import pandas as pd
import matplotlib.pyplot as plt
from tools.io import load_clean
from visualisation.plot_variables_correlation import plot_variables_correlation, plot_variables_correlation_by_season


pd.set_option("display.max_columns", None)

demand = load_clean("demand")
price = load_clean("price")
price_measures = load_clean("price_measures")
demand_range = load_clean("demand_range")
price_measures.rename(columns={"range": "price_range"}, inplace=True)
demand_range.rename(columns={"range": "demand_range"}, inplace=True)

df = demand.join([price, price_measures, demand_range]).at_time("17:30")
print(df.head())



plot_variables_correlation(
    df=df,
    x_col="demand",
    y_col="energy_price",
    x_label="Demand (MW)",
    y_label="Price ($/MWh)",
    title="Correlations between demand and price at ESROI start (17:30)",
)


plot_variables_correlation_by_season(
    df=df,
    x_col="demand",
    y_col="energy_price",
    x_label="Demand (MW)",
    y_label="Price ($/MWh)",
    title="Correlations between demand and price at ESROI start (17:30) by season",
)