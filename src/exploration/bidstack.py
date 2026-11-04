"""Battery bid stacks as MW$ by tranche - uncomment the plot you want.

    python -m exploration.bidstack
"""

from analysis.bidstack import load_bidstack, rolling_change
from tools.paths import local_plots_dir
from tools.plot_style import save_figure
from visualisation.plot_bidstack import plot_mw_dollars_by_tranche, plot_rolling_change, plot_tranche_heatmap


def main():
    bidstack = load_bidstack()
    rolling_change(bidstack)

    # fig = plot_mw_dollars_by_tranche(bidstack, freq="D")
    # save_figure(fig, local_plots_dir / "bidstack" / "mw_dollars_by_tranche_daily.png")

    # fig = plot_rolling_change(bidstack, window="1D")
    # save_figure(fig, local_plots_dir / "bidstack" / "mw_dollars_rolling_change_daily.png")

    # fig = plot_tranche_heatmap(bidstack, freq="D")
    # save_figure(fig, local_plots_dir / "bidstack" / "tranche_heatmap_daily.png")

    # # Raw 5-minute resolution over a few days: this is where interval-to-interval
    # # movement in the stack is actually visible rather than averaged away.
    # fig = plot_tranche_heatmap(bidstack, freq=None, start="2026-01-10", end="2026-01-14")
    # save_figure(fig, local_plots_dir / "bidstack" / "tranche_heatmap_window.png")

    # # Same window, but each cell is the move since the previous interval, so a
    # # repriced tranche lights up and an untouched one stays blank.
    # fig = plot_tranche_heatmap(bidstack, freq=None, start="2026-01-10", end="2026-01-14", values="change")
    # save_figure(fig, local_plots_dir / "bidstack" / "tranche_heatmap_window_change.png")


if __name__ == "__main__":
    main()
