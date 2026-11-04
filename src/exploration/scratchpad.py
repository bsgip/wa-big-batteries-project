import matplotlib.pyplot as plt
import pandas as pd

from data_extraction.catalog import load
from tools.io import load_clean
from tools.paths import local_plots_dir
from tools.plot_style import save_figure

price = load("price")["energy_price"]
demand = load_clean("demand")
date = "2026-10-04"
df = pd.concat([price, demand])
df = df.loc[date]

print(df.iloc[30:40])

fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(10, 6))
axes[0].plot(df.index, df["energy_price"], label="price")
axes[1].plot(df.index, df["demand"], label="demand")
axes[0].legend()
axes[1].legend()

fig.tight_layout()

save_figure(fig, local_plots_dir / f"{date}.png")
