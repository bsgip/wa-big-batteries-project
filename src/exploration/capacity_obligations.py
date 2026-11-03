"""Do batteries offer their capacity obligation (RCOQ) at every ESROI dispatch interval?

    python -m exploration.capacity_obligations
"""

from analysis.obligations import below_rcoq, offered_injection_mw
from tools.constants import PEAK_ESROI_END, PEAK_ESROI_START, START_DATE, battery_capacity_MW, battery_codes
from tools.io import load_clean
from visualisation.plot_obligations import plot_offered_vs_rcoq_grid


def main():
    # ESROI intervals only, analysis period only (load_clean starts at START_DATE)
    rcoq = load_clean("rcoq")[battery_codes].between_time(PEAK_ESROI_START, PEAK_ESROI_END, inclusive="left")
    offered = offered_injection_mw(battery_codes).loc[START_DATE:]
    below = below_rcoq(offered, rcoq)

    _, _, summary = plot_offered_vs_rcoq_grid(
        offered,
        rcoq,
        below,
        battery_capacity_MW,
        title=f"Injection offered vs RCOQ at each ESROI interval ({PEAK_ESROI_START}-{PEAK_ESROI_END}), per battery",
        filename="batteries_comparison/offered_vs_rcoq_esroi_per_battery.png",
    )
    print(summary)


if __name__ == "__main__":
    main()
