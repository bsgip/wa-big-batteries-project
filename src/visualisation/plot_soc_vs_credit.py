import math

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tools.paths import local_plots_dir
from tools.plot_style import UNIT_COLORS, save_figure


def plot_soc_vs_credit_grid(soc_start, credit_soc, capacity, title, filename, ncols=2):
    """One panel per battery: daily SOC at the window start against the charge its RCOQ needs.

    soc_start: daily SOC (MWh) at the window start, one column per battery
    credit_soc: daily credit-equivalent SOC (MWh), same columns - from energy.credit_equivalent_soc_mwh
    capacity: {code: rated MWh}, sets each panel's y-range and dotted ceiling line
    ncols: panels per row

    The credit line is only drawn where RCOQ > 0: a zero obligation is "nothing
    required", not a requirement of 0 MWh worth comparing against.

    Returns (fig, axes, summary) - summary has, per battery, the days with an
    obligation and SOC data, and how many of those started below the credit line.
    """
    codes = list(soc_start.columns)
    nrows = math.ceil(len(codes) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 2.8 * nrows + 0.8),
                             sharex=True, squeeze=False)
    axes = axes.ravel()

    # RCOQ data runs past the end of the SOC data - keep to the SOC period
    credit_soc = credit_soc.reindex(soc_start.index)

    rows = []
    for ax, code in zip(axes, codes):
        soc = soc_start[code]
        credit = credit_soc[code].where(credit_soc[code] > 0)

        ax.scatter(soc.index, soc, s=4, color='C0', alpha=0.6, lw=0,
                   label='SOC at window start')
        ax.step(credit.index, credit, where='post', color='black', lw=1.4,
                label='Credit-equivalent SOC (RCOQ × window)')
        ax.axhline(capacity[code], color='0.5', ls=':', lw=1,
                   label='Rated capacity')

        # days the battery had an obligation and an SOC reading
        both = soc.notna() & credit.notna()
        n_below = int((soc[both] < credit[both]).sum())
        n_days = int(both.sum())
        rows.append({'code': code, 'obligation_days': n_days, 'days_below': n_below})
        note = f"{n_below} of {n_days} obligation days below" if n_days else "no obligation in period"

        ax.set_title(f"{code}  ({note})", fontsize=9.5)
        ax.set_ylim(0, capacity[code] * 1.08)
        ax.set_ylabel("MWh", fontsize=9)
        ax.grid(alpha=0.3)

    for ax in axes[len(codes):]:  # spare panels when codes don't fill the grid
        ax.set_visible(False)

    for ax in axes[:len(codes)]:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
        ax.tick_params(axis='x', labelbottom=True, labelsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.965),
               ncol=len(labels), fontsize=9, frameon=False, markerscale=3)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    save_figure(fig, local_plots_dir / filename)

    return fig, axes, pd.DataFrame(rows).set_index('code')


def plot_below_credit_summary(soc_start, credit, title, filename, whis=(10, 90)):
    """Per battery, against one fixed credit-equivalent SOC: how often it starts the
    window below it (left), and how far above/below it it usually is (right).

    soc_start: daily SOC (MWh) at the window start, one column per battery
    credit: one credit-equivalent SOC (MWh) per battery - from energy.typical_credit_equivalent_mwh
    whis: whisker percentiles for the deviation boxes (default 10th-90th, as in the trend plots)

    Every day with an SOC reading counts, from the battery's first reading on,
    whether or not its RCOQ had started yet. Deviation is (SOC - credit) / credit
    x 100, so batteries of different size share one scale and 0 = exactly at credit.

    Returns (fig, axes, summary) - summary has, per battery, the credit, days,
    days below, % below and the median deviation.
    """
    codes = [c for c in soc_start.columns if pd.notna(credit.get(c))]
    soc = soc_start[codes]
    credit = credit[codes]

    days = soc.notna().sum()
    below = (soc < credit).sum()  # NaN SOC compares False, so missing days never count as below
    pct_below = below / days * 100
    deviation = (soc - credit) / credit * 100

    # first battery at the top
    y = range(len(codes))[::-1]
    fig, (ax_pct, ax_dev) = plt.subplots(
        1, 2, figsize=(11, 0.6 * len(codes) + 2.2), sharey=True,
        gridspec_kw={'width_ratios': [1, 1.6]},
    )
    # margins by hand: tight_layout can't place the figure-level legend
    fig.subplots_adjust(left=0.2, right=0.98, top=0.82, bottom=0.13, wspace=0.06)

    # left: share of days below the credit; day counts go in the row labels so
    # the bar labels stay short enough to fit inside the panel
    ax_pct.barh(y, pct_below, color='C0', alpha=0.7, height=0.6)
    for yi, code in zip(y, codes):
        ax_pct.text(pct_below[code] + 1.5, yi, f"{pct_below[code]:.0f}%", va='center', fontsize=8.5)
    ax_pct.set_xlim(0, 115)  # room for the labels past the longest bar
    ax_pct.set_xticks(range(0, 101, 25))
    ax_pct.set_xlabel("Days below credit equivalent (%)")
    ax_pct.set_yticks(
        list(y),
        labels=[f"{c}\n{credit[c]:,.0f} MWh credit, {below[c]} of {days[c]} days" for c in codes],
        fontsize=8.5,
    )
    ax_pct.grid(axis='x', alpha=0.3)

    # right: how far from the credit, same box style as period_box_plot
    ax_dev.boxplot(
        [deviation[c].dropna().to_numpy() for c in codes],
        positions=list(y),
        widths=0.6,
        whis=whis,
        vert=False,
        manage_ticks=False,
        patch_artist=True,
        boxprops={'facecolor': 'C0', 'alpha': 0.5},
        medianprops={'color': 'black', 'lw': 2},
        whiskerprops={'color': '0.4', 'lw': 1},
        capprops={'color': '0.4', 'lw': 1},
        flierprops={'marker': 'o', 'markersize': 3, 'markerfacecolor': 'C0', 'markeredgecolor': 'none',
                    'alpha': 0.75},
    )
    ax_dev.axvline(0, color='black', ls='--', lw=1)
    ax_dev.set_xlabel("SOC at window start vs credit equivalent (% of credit, below 0 = short)")
    ax_dev.grid(axis='x', alpha=0.3)

    lo, hi = whis
    handles = [
        Patch(facecolor='C0', alpha=0.5, edgecolor='black', label='p25–p75 (middle half)'),
        Line2D([], [], color='black', lw=2, label='Median'),
        Line2D([], [], color='0.4', lw=1, label=f'p{lo}–p{hi} (whiskers)'),
        Line2D([], [], ls='none', marker='o', markersize=3, color='C0', alpha=0.75, label=f'Days beyond p{lo}–p{hi}'),
        Line2D([], [], color='black', ls='--', lw=1, label='At credit equivalent'),
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.59, 0.83), ncol=len(handles),
               fontsize=8.5, frameon=False)
    fig.suptitle(title, y=0.97)

    save_figure(fig, local_plots_dir / filename)

    summary = pd.DataFrame({
        'credit_mwh': credit,
        'days': days,
        'days_below': below,
        'pct_below': pct_below,
        'median_deviation_pct': deviation.median(),
    })
    return fig, (ax_pct, ax_dev), summary


def plot_share_below_by_month(soc_start, share, capacity, ylabel, title, filename):
    """Monthly % of days below a threshold, one line per battery (top), over the
    capacity those %s are counted from (bottom), on one time axis.

    soc_start: the daily values the share was computed from, one column per battery
        (SOC in MWh or %) - only used to see which days each battery has a reading
    share: from energy.monthly_share_below - one % column per battery; its pooled "all" column isn't drawn
    capacity: {code: rated MWh}
    ylabel: what the top panel's % is of, e.g. "Days with SOC below 50% (%)"

    The bottom panel counts a battery's MWh only on days it has an SOC reading,
    so it is the fleet the top panel's %s are measured over - not the installed
    fleet, which can run months ahead of the SOC data.

    Returns (fig, (ax_share, ax_cap), counted) - counted is the monthly capacity (MWh).
    """
    codes = [c for c in share.columns if c != 'all']

    counted = (soc_start[codes].notna() * pd.Series(capacity)[codes]).sum(axis=1).resample('MS').max()

    fig, (ax_share, ax_cap) = plt.subplots(
        2, 1, figsize=(11, 6.8), sharex=True,
        gridspec_kw={'height_ratios': [3, 1.2], 'hspace': 0.08},
    )

    # one line per battery
    for code in codes:
        ax_share.plot(share.index, share[code], color=UNIT_COLORS.get(code, '0.5'), lw=1.6,
                      marker='o', ms=3.5, alpha=0.9, label=code)
    ax_share.set_ylim(-3, 103)
    ax_share.set_ylabel(ylabel)
    ax_share.grid(alpha=0.3)
    ax_share.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=8.5, frameon=False)
    ax_share.set_title(title, pad=40)  # room for the two legend rows

    ax_cap.step(counted.index, counted, where='post', color='0.3', lw=1.5)
    ax_cap.fill_between(counted.index, counted, step='post', color='0.3', alpha=0.15)
    ax_cap.set_ylim(0, counted.max() * 1.15)
    ax_cap.set_ylabel("Fleet counted\n(MWh)")
    ax_cap.grid(alpha=0.3)

    ax_cap.set_xlim(share.index.min(), share.index.max() + pd.offsets.MonthEnd(0))
    ax_cap.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    ax_cap.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax_cap.xaxis.set_minor_locator(mdates.MonthLocator())

    save_figure(fig, local_plots_dir / filename)

    return fig, (ax_share, ax_cap), counted
