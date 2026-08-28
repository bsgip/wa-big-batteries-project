import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from tools.constants import battery_codes
from tools.paths import repo_plots_dir
from tools.plot_style import save_figure


def _grid_shape(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def plot_heatmap_grid(
    df: pd.DataFrame,
    column_suffix: str,
    cmap: str,
    center: float | None,
    colorbar_label: str,
    title: str,
    filename: str,
    vmax: float | None = 100,
):
    """Grid of one time-of-day (y) x date (x) heatmap per battery, built
    from the `<code><column_suffix>` columns of `df`. `center=0` gives a
    diverging colour scale shared across all batteries (for signed power);
    `center=None` gives a sequential 0-`vmax` scale shared across all
    batteries (`vmax=100` by default, for SOC %; pass the data's own max -
    e.g. via the grids once built - for an absolute-unit version like MWh,
    so batteries of different rated capacity stay comparably scaled)."""
    time_of_day = df.index.strftime("%H:%M")
    date = df.index.normalize()  # real calendar date, not day-of-year - avoids year collisions

    grids = {}
    for code in battery_codes:
        col = f"{code}{column_suffix}"
        if col not in df:
            continue
        long_df = pd.DataFrame({"time_of_day": time_of_day, "date": date, "value": df[col].to_numpy()})
        grid = long_df.pivot_table(index="time_of_day", columns="date", values="value").sort_index()
        if not grid.empty:
            grids[code] = grid

    if not grids:
        raise ValueError("no battery data found - check battery_codes against the input dataframe")

    norm = None
    vmin = 0
    if center is not None:
        abs_max = max(grid.abs().max().max() for grid in grids.values())
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=center, vmax=abs_max)
        vmin, vmax = None, None

    rows, cols = _grid_shape(len(grids))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    im = None
    for i, (code, grid) in enumerate(grids.items()):
        ax = axes[i // cols][i % cols]
        im = ax.imshow(grid.values, aspect="auto", origin="lower", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)
        ax.set_title(code)

        yticks = np.linspace(0, len(grid.index) - 1, min(6, len(grid.index))).astype(int)
        ax.set_yticks(yticks)
        ax.set_yticklabels(grid.index[yticks])

        xticks = np.linspace(0, len(grid.columns) - 1, min(6, len(grid.columns))).astype(int)
        ax.set_xticks(xticks)
        ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in grid.columns[xticks]], rotation=45, ha="right")

    for i in range(len(grids), rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 0.92, 0.95))
    cbar_ax = fig.add_axes((0.94, 0.15, 0.015, 0.7))
    fig.colorbar(im, cax=cbar_ax, label=colorbar_label)
    save_figure(fig, repo_plots_dir / filename)
