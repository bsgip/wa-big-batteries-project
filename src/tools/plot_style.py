from pathlib import Path

import matplotlib.pyplot as plt

from tools.constants import battery_codes

# sign convention (AEMO): discharge/generation positive, charge negative.
# colour convention: warm/red = discharge, cool/blue = charge - used wherever
# colour directly encodes a signed value (heatmaps, distribution bars).
DISCHARGE_COLOR = "tab:red"
CHARGE_COLOR = "tab:blue"
DIVERGING_CMAP = "RdBu_r"  # negative (charge) -> blue, positive (discharge) -> red
SEQUENTIAL_CMAP = "viridis"  # unsigned 0-100% (SOC)

# one stable colour per battery, reused across every plot type so a given
# battery is always the same colour regardless of which figure it's in
_TAB10 = plt.get_cmap("tab10").colors
BATTERY_COLORS = {code: _TAB10[i % len(_TAB10)] for i, code in enumerate(battery_codes)}

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
    "COLLIE_BESS2": "#0072B2",
    "COLLIE_ESR1": "#009E73",
    "COLLIE_ESR4": "#D55E00",
    "COLLIE_ESR5": "#CC3399",
    "KWINANA_ESR1": "#D4A017",
    "KWINANA_ESR2": "#6A3D9A",
}


def save_figure(fig: plt.Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"plot successfully saved to {path}")
