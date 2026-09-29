"""Scatter any two columns of a frame against each other, with their
correlation. For the per-battery version, see
exploration/variables_correlations.py - this one is deliberately plain.
"""

import matplotlib.pyplot as plt
import pandas as pd

from tools.paths import local_plots_dir
from tools.plot_style import save_figure


def plot_variables_correlation(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    filename: str | None = None,
) -> float:
    """Scatter `x_col` against `y_col`, annotate Pearson r, save the figure
    and return r.

    Rows where either column is missing are dropped, so r and the point
    count both describe the overlap rather than the frame.
    """
    data = df[[x_col, y_col]].dropna()
    if data.empty:
        raise ValueError(f"no rows where both {x_col} and {y_col} are present")

    r = data[x_col].corr(data[y_col])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(data[x_col], data[y_col], s=4, alpha=0.3, color="tab:blue", edgecolors="none")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {x_col}\n(r={r:.2f}, n={len(data):,})")

    save_figure(fig, local_plots_dir / (filename or f"corr_{x_col}_vs_{y_col}.png"))
    return r
