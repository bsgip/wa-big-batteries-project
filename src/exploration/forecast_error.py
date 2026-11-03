import pandas as pd

from data_extraction.catalog import load
from tools.io import load_clean
from visualisation.plot_trend_band import trend_band_event_plot, trend_band_plot

pd.set_option("display.max_columns", None)

price_forecast = load(
    "predispatch_price", columns=["primary_dispatch_interval", "dispatch_interval", "energy_price"]
).rename(columns={"energy_price": "price_forecast"})
price = load("price")[["energy_price"]]


price_forecast["lead_time"] = (
    price_forecast["dispatch_interval"] - price_forecast["primary_dispatch_interval"]
).dt.total_seconds() / 3600


price_30 = price.resample("30min").mean()


merged = price_30.merge(price_forecast, how="inner", on="dispatch_interval")


merged = merged.dropna(subset=["energy_price"])


merged = merged.loc[merged["dispatch_interval"] >= "2023-10-01"]

forecast_4h = merged.loc[merged["lead_time"] == 4].copy()


forecast_4h["forecast_error"] = forecast_4h["price_forecast"] - forecast_4h["energy_price"]

forecast_4h = forecast_4h.set_index("dispatch_interval")
# print(f"\nForecast error: {forecast_4h['forecast_error'].describe()}")

# print(forecast_4h)

trend_band_plot(
    df=forecast_4h,
    col="forecast_error",
    show_outliers=False,
    ylabel="Forecast − actual price ($/MWh)",
    title="4h-ahead price forecast error",
    filename="price_forecast_error_4h_month_trend.png",
)


rcoq = load_clean("rcoq")
rcoq_esroi = rcoq.between_time("17:30", "21:30")
# print(rcoq_esroi.loc[rcoq_esroi.index > "2026"])

d = load("predispatch_dispatch_total")
print(d.columns)
