from pathlib import Path
from tools.paths import local_plots_dir
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd

pd.set_option("display.max_columns", None)

data_path = Path.cwd() / "data"
filename = data_path / "market-advisories-filtered.csv"


df = pd.read_csv(filename, parse_dates=True)
mask = (df["Details"].str.contains("LRC")) & ~(df["Details"].str.contains("LOR"))
df["duration_td"] = pd.to_timedelta(df["Interval Duration"])
df["duration_days"] = df["duration_td"].dt.total_seconds() / 86400
print(df.loc[df["duration_days"] > 5])


def plot_sse_duration():
    df = pd.read_csv(filename, parse_dates=["End Interval", "Start Interval"])
    print(df["Interval Duration"])

    df["duration_td"] = pd.to_timedelta(df["Interval Duration"])
    df["duration_days"] = df["duration_td"].dt.total_seconds() / 86400
    print(df.loc[df["duration_days"] > 1])


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
    # plt.show()
    try:
        filename = local_plots_dir / "LRC_dates.png"
        plt.savefig(filename, dpi=200)
        print(f"Plot saved to {filename}")
    except Exception as e:
        print(e)
