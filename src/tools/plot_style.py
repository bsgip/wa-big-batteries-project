from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from tools.constants import esr_codes

# sign convention (AEMO): discharge/generation positive, charge negative.
# colour convention: red = discharge, green = charge - used wherever
# colour directly encodes a signed value (heatmaps, distribution bars).
DISCHARGE_COLOR = "tab:red"
CHARGE_COLOR = "tab:green"
# negative (charge) -> green, zero -> white, positive (discharge) -> red
DIVERGING_CMAP = LinearSegmentedColormap.from_list("charge_discharge", [CHARGE_COLOR, "white", DISCHARGE_COLOR])
SEQUENTIAL_CMAP = "viridis"  # unsigned 0-100% (SOC)

# same 0-100% bins on every SOC histogram, so they can be compared
SOC_BINS = np.linspace(0, 100, 81)  # 1.25% wide

# one stable colour per battery, reused across every plot type so a given
# battery is always the same colour regardless of which figure it's in
_TAB10 = plt.get_cmap("tab10").colors
BATTERY_COLORS = {code: _TAB10[i % len(_TAB10)] for i, code in enumerate(esr_codes)}

# separate fixed palette for visualisation/plotting.py's per-unit lines,
# keyed to unit name so filtering never repaints a series. Deliberately not
# merged into BATTERY_COLORS above, which the older
# plot_soc.py/plot_heatmap.py/plot_distributions.py functions already use
# and aren't being touched by this addition.
#
# Hues spread ~60 degrees apart around the wheel (blue / green / vermillion
# / magenta / gold / purple) rather than clustered in the orange-red-brown
# range, which is where the original palette here was hard to tell apart at
# thin line widths.
UNIT_COLORS = {
    "ALINTA_WGP_ESR1": "#6B8E23",
    "COLLIE_BESS2": "#0072B2",
    "COLLIE_ESR1": "#009E73",
    "COLLIE_ESR4": "#D55E00",
    "COLLIE_ESR5": "#CC3399",
    "KWINANA_ESR1": "#D4A017",
    "KWINANA_ESR2": "#6A3D9A",
}


SEASON_COLORS = {
    "Summer": "#E69F00",  # orange
    "Autumn": "#8C510A",  # brown
    "Winter": "#56B4E9",  # light sky blue
    "Spring": "#009E73",  # green
    "Shoulder": "#333333",  # dark grey (autumn + spring)
}


def save_figure(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"plot successfully saved to {path}")
