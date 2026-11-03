import matplotlib.pyplot as plt
import pandas as pd

from tools.paths import local_plots_dir
from tools.plot_style import save_figure
from tools.seasons import SEASON_BY_MONTH, SEASON_GROUPS, season_group_of


def first_start_per_day(starts: pd.Series) -> pd.DatetimeIndex:
    """One timestamp per stress day: the start of that day's first event."""
    idx = pd.DatetimeIndex(starts)
    return pd.DatetimeIndex(pd.Series(idx, index=idx.normalize()).groupby(level=0).min())


def stress_days_by_season_hour(starts: pd.Series) -> pd.DataFrame:
    """Count of stress days per season group (rows: Summer / Shoulder / Winter, most days first)
    and start hour of the day's first event (columns 0-23)."""
    days = first_start_per_day(starts)
    groups = pd.MultiIndex.from_arrays([season_group_of(days), days.hour])
    counts = pd.Series(1, index=groups).groupby(level=[0, 1]).sum()
    counts = counts.unstack(fill_value=0).reindex(index=SEASON_GROUPS, columns=range(24), fill_value=0)

    # sort is stable, so groups with the same total keep SEASON_GROUPS order
    order = counts.sum(axis=1).sort_values(ascending=False, kind="stable").index
    return counts.loc[order]


def plot_stress_days_by_season_hour(definitions, title, filename):
    """One heatmap per definition: stress days counted by season group (rows, autumn and
    spring together as shoulder, most days at the top) and the start hour of the day's
    first event (columns), the count written in every non-empty cell.

    definitions: {panel label: Series of event start times}

    Each panel has its own colour scale - the definitions give very different
    numbers of stress days, so a shared one would leave the smaller ones blank.

    Returns (fig, axes, tables) - tables is {panel label: season group x hour counts}.
    """
    n = len(definitions)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.1 * n + 1), sharex=True, squeeze=False)
    axes = axes[:, 0]

    tables = {}
    for ax, (label, starts) in zip(axes, definitions.items()):
        counts = stress_days_by_season_hour(starts)
        tables[label] = counts
        peak = max(counts.to_numpy().max(), 1)

        ax.imshow(counts, cmap='Blues', vmin=0, vmax=peak, aspect='auto')
        rows = list(counts.index)
        for (row, hour), value in counts.stack().items():
            if value:
                ax.text(hour, rows.index(row), value, ha='center', va='center', fontsize=8.5,
                        color='white' if value > 0.6 * peak else 'black')

        # white lines between cells, so neighbouring counts read apart
        ax.set_xticks([h - 0.5 for h in range(25)], minor=True)
        ax.set_yticks([r - 0.5 for r in range(len(rows) + 1)], minor=True)
        ax.grid(which='minor', color='white', lw=1.5)
        ax.tick_params(which='minor', length=0)

        ax.set_yticks(range(len(rows)), labels=[f"{r} ({counts.loc[r].sum()})" for r in rows], fontsize=9)
        first, last = pd.Timestamp(starts.min()), pd.Timestamp(starts.max())
        ax.set_title(f"{label}: {counts.to_numpy().sum()} stress days, {first:%b %Y} – {last:%b %Y}",
                     fontsize=10, loc='left')

    axes[-1].set_xticks(range(24), labels=[f"{h:02d}" for h in range(24)], fontsize=9)
    axes[-1].set_xlabel("Start hour of the day's first event")
    fig.suptitle(title)
    fig.tight_layout()

    save_figure(fig, local_plots_dir / filename)

    return fig, axes, tables


# quarters starting in December, so each one is a southern season:
# Dec-Feb summer, Mar-May autumn, Jun-Aug winter, Sep-Nov spring
SEASON_FREQ = "QS-DEC"


def stress_days_by_season(starts: pd.Series, coverage: tuple) -> pd.DataFrame:
    """Stress days per season for one definition, over the seasons its source covers.

    starts: event start times; a stress day is a date with at least one event starting
    coverage: (first, last) time the definition's source covers

    Returns one row per covered season (indexed by season start), with
    days (number of stress days) and covered (share of the season's days the source covers).
    """
    first, last = (pd.Timestamp(t).normalize() for t in coverage)
    covered_days = pd.Series(1, index=pd.date_range(first, last, freq="D"))
    stress_days = pd.Series(1, index=pd.DatetimeIndex(starts).normalize().unique())

    covered = covered_days.resample(SEASON_FREQ).size()
    season_length = (covered.index + pd.DateOffset(months=3) - covered.index).days
    days = stress_days.resample(SEASON_FREQ).size().reindex(covered.index, fill_value=0)

    return pd.DataFrame({"days": days, "covered": covered / season_length})


def plot_stress_days_by_season(definitions, coverage, title, filename):
    """Number of stress days per season, one bar panel per definition on a shared season axis.

    definitions: {label: Series of event start times}
    coverage: {label: (first, last)} - the period each definition's source covers.
        Seasons it doesn't cover get "no data" instead of a bar, so they never read
        as seasons with no stress days.

    Returns (fig, axes, tables) - tables is {label: stress_days_by_season(...)}.
    """
    tables = {label: stress_days_by_season(starts, coverage[label]) for label, starts in definitions.items()}

    # every season from the earliest any definition covers to the latest
    seasons = pd.date_range(min(t.index.min() for t in tables.values()),
                            max(t.index.max() for t in tables.values()), freq=SEASON_FREQ)
    x = range(len(seasons))

    n = len(tables)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.3 * n + 1.2), sharex=True, squeeze=False)
    axes = axes[:, 0]

    for i, (ax, (label, table)) in enumerate(zip(axes, tables.items())):
        color = f"C{i}"
        table = table.reindex(seasons)  # NaN where the definition has no data
        top = max(table["days"].max(), 1)

        for xi, days in zip(x, table["days"]):
            if pd.isna(days):
                ax.text(xi, 0, "no data", ha="center", va="bottom", fontsize=7.5, color="0.55")
                continue
            ax.bar(xi, days, width=0.7, color=color, alpha=0.85)
            ax.text(xi, days + top * 0.03, f"{days:.0f}", ha="center", va="bottom", fontsize=8.5)

        ax.set_ylim(0, top * 1.25)  # room for the count labels
        ax.set_ylabel("Stress days")
        ax.grid(axis="y", alpha=0.3)
        ax.set_title(label, fontsize=10, loc="left")

    axes[-1].set_xticks(list(x), labels=[f"{SEASON_BY_MONTH[s.month]}\n{s:%b %Y}" for s in seasons], fontsize=8.5)

    fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()

    save_figure(fig, local_plots_dir / filename)

    return fig, axes, tables
