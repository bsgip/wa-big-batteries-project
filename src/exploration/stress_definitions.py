"""How many stress events each definition gives, and when (season, start hour),
and how the number of stress days per season has changed over time.

    python -m exploration.stress_definitions
"""

import pandas as pd

from analysis.sse_events import raw_advisories_path
from analysis.stress_events import (
    SPARE_CAPACITY_THRESHOLD_MW,
    advisory_starts,
    in_service_capacity_mw,
    low_spare_capacity_starts,
    review_event_starts,
    spare_capacity_mw,
)
from tools.constants import PEAK_ESROI_END, PEAK_ESROI_START
from tools.io import load_clean
from visualisation.plot_stress_events import (
    first_start_per_day,
    plot_stress_days_by_season,
    plot_stress_days_by_season_hour,
)
from visualisation.plot_stress_soc import plot_soc_moments_stress_vs_other
from visualisation.plot_trend_band import trend_band_plot

REVIEW = "CC2T review events"
ADVISORIES = "AEMO LRC/LOR market advisories"
SPARE = f"Spare capacity below {SPARE_CAPACITY_THRESHOLD_MW} MW"


def main():
    spare = spare_capacity_mw()
    review = review_event_starts()
    advisories_raw = pd.read_csv(raw_advisories_path, encoding="cp1252", usecols=["Start Interval"])

    definitions = {
        REVIEW: review,
        ADVISORIES: advisory_starts(),
        SPARE: low_spare_capacity_starts(spare),
    }

    # the period each source covers, so seasons with no data aren't read as seasons with no stress
    coverage = {
        REVIEW: (review.min(), review.max()),  # the review's own period isn't recorded - first to last event
        ADVISORIES: tuple(pd.to_datetime(advisories_raw["Start Interval"]).agg(["min", "max"])),  # the download
        SPARE: (spare.index.min().tz_localize(None), spare.index.max().tz_localize(None)),  # the dispatch data
    }

    _, _, tables = plot_stress_days_by_season_hour(
        definitions,
        title="Stress days by season and start hour, per definition",
        filename="stress_events/stress_days_by_season_hour.png",
    )
    for label, counts in tables.items():
        print(f"\n{label}\n{counts.loc[:, counts.sum() > 0]}")

    _, _, by_season = plot_stress_days_by_season(
        definitions,
        coverage,
        title="Stress days per season, per definition",
        filename="stress_events/stress_days_by_season.png",
    )
    for label, table in by_season.items():
        print(f"\n{label}\n{table.round(2)}")

    # what the spare capacity definition is built from, month by month
    trend_band_plot(
        df=in_service_capacity_mw(),
        ylabel="In-service capacity (MW)",
        title="In-service energy injection capacity: monthly median with spread",
        filename="stress_events/in_service_capacity_trend.png",
    )
    trend_band_plot(
        df=spare,
        ylabel="Spare capacity (MW)\nIn-service capacity − dispatched",
        title="Spare capacity: monthly median with spread",
        filename="stress_events/spare_capacity_trend.png",
    )

    # fleet SOC on stress days vs other days at three moments - before charging finishes, going into
    # the peak, coming out of it - over the analysis period (load_clean starts at START_DATE).
    # Spare capacity is left out: its last stress day (Oct 2024) is before the analysis period.
    soc_pct = load_clean("soc")["fleet_soc_pct"].dropna()
    soc_pct.index = soc_pct.index.tz_localize(None)  # stress days are tz-naive local dates
    stress_days = {label: first_start_per_day(definitions[label]).normalize() for label in (REVIEW, ADVISORIES)}
    _, _, moments = plot_soc_moments_stress_vs_other(
        soc_pct,
        stress_days,
        moments={"Midday\n12:00": "12:00", f"ESROI start\n{PEAK_ESROI_START}": PEAK_ESROI_START,
                 f"ESROI end\n{PEAK_ESROI_END}": PEAK_ESROI_END},
        title="Fleet SOC at midday, ESROI start and ESROI end: stress days vs other days",
        filename="stress_events/fleet_soc_moments_stress_vs_other.png",
    )
    print(moments.round(1))


if __name__ == "__main__":
    main()
