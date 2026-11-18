import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from tools.constants import battery_codes
from tools.io import load_clean
from tools.paths import local_plots_dir
from tools.plot_style import save_figure

pd.set_option("display.max_columns", None)

FACILITY = "KWINANA_ESR2"
WINDOW = ["next_1h", "1_4h", "4_8h", "8_12h", "12-48h"]

# Load data
fc = load_clean("predispatch_price")
price = load_clean("price")
soc = load_clean("soc")
power = load_clean("power")
fcess = load_clean("fcess_schedule")
fcess = fcess.loc[fcess["code"].isin(battery_codes)]


# tau is the interval for when a forecast is used for
fc["tau"] = fc.index + pd.Timedelta("30min")
fc["lead_min"] = (fc["dispatch_interval"] - fc.index).dt.total_seconds() / 60  # lead time in minutes
fc["forecast_window"] = pd.cut(
    x=fc["lead_min"],
    bins=[30, 90, 270, 510, 750, 2910],
    right=False,
    labels=WINDOW,
)

fc = fc.groupby(["tau", "forecast_window"], observed=True)["energy_price"].mean().unstack()

power_ti = power[FACILITY].resample("30min").mean().rename("power_mw")
soc_ti = soc[FACILITY].resample("30min").first().rename("soc_mwh")


df = fc.join([soc_ti, power_ti], how="inner")
df = df.loc["2025-05-01":]


for w in WINDOW:
    if w == "next_1h":
        continue

    df[f"far_minus_near_{w}"] = df[w] - df["next_1h"]

# Drop trading intervals missing any model input or the power used to label the action
df = df.dropna(subset=["far_minus_near_1_4h", "far_minus_near_4_8h", "soc_mwh", "power_mw"])

df["action"] = np.select(
    condlist=[df["power_mw"] > 5, df["power_mw"] < -5], choicelist=["discharge", "charge"], default="hold"
)


def action_chance(df: pd.DataFrame, col: str):
    dis_chance = df[col].value_counts()["discharge"] / len(df[col]) * 100
    charge_chance = df[col].value_counts()["charge"] / len(df[col]) * 100
    hold_chance = df[col].value_counts()["hold"] / len(df[col]) * 100

    print(f"Discharge: {dis_chance:.2f}%\nCharge: {charge_chance:.2f}%\nHold: {hold_chance:.2f}%")


ACTIONS = ["charge", "discharge", "hold"]


def action_percent(actions: pd.Series) -> pd.Series:
    """Percent of trading intervals with each action (0% if an action never occurs)."""
    return actions.value_counts(normalize=True).reindex(ACTIONS, fill_value=0) * 100


df["hour"] = df.index.hour
df["far_above_near_1_4h"] = (df["far_minus_near_1_4h"] > 0).astype(int)
df["far_above_near_4_8h"] = (df["far_minus_near_4_8h"] > 0).astype(int)

print(df.columns)


# Define inputs
inputs = ["far_above_near_1_4h", "far_above_near_4_8h", "far_minus_near_1_4h", "far_minus_near_4_8h", "soc_mwh", "hour"]
X = pd.get_dummies(
    df[inputs],
    columns=["hour"],
    drop_first=True,
    dtype=float,
)
X = sm.add_constant(X)
# print(X)

# Output
CLASS_LABELS = {0: "hold", 1: "discharge", 2: "charge"}
y = df["action"].map({label: code for code, label in CLASS_LABELS.items()})
# print(y)


# Setting up training and test sets
train = df.index < "2026-05-01"
test = ~train

X_train = X.loc[train]
X_test = X.loc[test]
y_train = y.loc[train]
y_test = y.loc[test]

# print(X_train.shape, X_test.shape)
# print(X_train)


def run_model():
    max_iter = 1000
    # Fit on training dataset
    model = sm.MNLogit(y_train, X_train).fit(maxiter=max_iter)

    # Predict on test and score
    probs = model.predict(X_test)  # columns 0, 1, 2 = P(hold), P(discharge), P(charge)
    pred = probs.values.argmax(axis=1)  # pick the biggest probability
    acc_full = (pred == y_test.values).mean()

    # Baselines
    clock_cols = ["const", "soc_mwh"] + [c for c in X.columns if c.startswith("hour_")]
    m_clock = sm.MNLogit(y_train, X_train[clock_cols]).fit(maxiter=max_iter)
    acc_clock = (m_clock.predict(X_test[clock_cols]).values.argmax(axis=1) == y_test.values).mean()

    prev = y.shift(1)
    acc_lag = (prev[test] == y[test]).mean()

    print(f"Accuracy  full model: {acc_full:.3f}  clock + SOC only: {acc_clock:.3f}  previous action: {acc_lag:.3f}")

    # Percent chance of each action over the test period
    chance = pd.DataFrame(
        {
            "actual": action_percent(y_test.map(CLASS_LABELS)),
            "model_avg_probability": probs.rename(columns=CLASS_LABELS).mean() * 100,
            "model_predicted_action": action_percent(pd.Series(pred, index=y_test.index).map(CLASS_LABELS)),
        }
    )
    print(chance.round(1))


run_model()
