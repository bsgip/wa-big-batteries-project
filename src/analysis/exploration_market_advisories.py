from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_columns", None)

data_path = Path.cwd() / "data"
filename = data_path / "market-advisories-filtered.csv"

print(filename)
df = pd.read_csv(filename, parse_dates=["End Interval", "Start Interval"])
print(df["Interval Duration"])

df["duration_td"] = pd.to_timedelta(df["Interval Duration"])
df["duration_days"] = df["duration_td"].dt.total_seconds() / 86400


bins = [0, 1, 2, 3, 4, 5, float("inf")]
labels = ["<1 day", "1–2 days", "2–3 days", "3–4 days", "4–5 days", ">5 days"]

df["duration_bin"] = pd.cut(df["duration_days"], bins=bins, labels=labels, right=False)

counts = pd.crosstab(df["duration_bin"], df["Withdrawn"])

ax = counts.plot(kind="bar", figsize=(8, 5), edgecolor="black")
ax.set_xlabel("Interval duration")
ax.set_ylabel("Count")
ax.set_title("Duration distribution by withdrawal status")
ax.legend(title="Withdrawn")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()
