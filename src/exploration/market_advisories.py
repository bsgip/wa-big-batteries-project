from pathlib import Path
from tools.paths import local_plots_dir, local_data_dir
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re

pd.set_option("display.max_columns", None)

sse_dir = local_data_dir / "processed_data" / "sse_data"
events_path = sse_dir / "market-advisories.csv"
events_filtered_path = sse_dir / "market-advisories-filtered.csv"



cols = ["Start Interval", "Details", "Interval Duration"]
df = pd.read_csv(events_filtered_path, parse_dates=True, usecols=cols)

print(df.head())

def pie_chart():

    keywords = ["temperature", "heatwave", "outage"]

    details = df["Details"].fillna("").str.lower()
    matches = pd.DataFrame({k: details.str.contains(k, regex=False) for k in keywords})

    n_hits = matches.sum(axis=1)
    df["category"] = np.select(
        [n_hits == 0, n_hits > 1],
        ["Other", "Multiple"],
        default=matches.idxmax(axis=1),     # the single matching keyword
    )

    counts = df["category"].value_counts()

    fig, ax = plt.subplots(figsize=(9, 6))
    wedges, _, _ = ax.pie(
        counts.values,
        autopct=lambda p: f"{p:.0f}%" if p >= 4 else "",
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
        pctdistance=0.75,
    )
    ax.legend(
        wedges,
        [f"{c.capitalize()} ({n})" for c, n in counts.items()],
        title="Details mentions",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )
    ax.set_title(f"Events by keyword (n={len(df)})")
    ax.set_aspect("equal")
    fig.tight_layout()
    plt.show()


pie_chart()


# mask = (df["Details"].str.contains("LRC")) & ~(df["Details"].str.contains("LOR"))
# df["duration_td"] = pd.to_timedelta(df["Interval Duration"])
# df["duration_days"] = df["duration_td"].dt.total_seconds() / 86400
# print(df.loc[df["duration_days"] > 5])


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
