import re

import pandas as pd

from tools.paths import local_data_dir, registration_dir


sse_events_path = local_data_dir / "sse_events.csv"
raw_advisories_path = registration_dir / "market-advisories.csv"
# SSE events added by hand (not in the advisories, or not caught by the keywords); never overwritten
manual_events_path = local_data_dir / "sse_events_manual.csv"

# An advisory counts as an SSE event if its Details contain any of these (case-sensitive)
ADVISORY_KEYWORDS = ["LRC", "LOR"]
# Category for advisories not yet categorised by hand in sse_events.csv
DEFAULT_CATEGORY = "other"

# "period 2024-May-20, 16:00 – 19:00", "period on 01/02/2026, 17:00 - 21:00", "period 20-Feb-2024 15:00 – 20:30"
STATED_PERIOD = re.compile(r"period (?:on )?.{6,20}?,? (\d{1,2}:\d{2})\s*[–-]\s*(\d{1,2}:\d{2})")


def filter_advisories(advisories, keywords=None):
    """Keep advisories whose Details mention any keyword.

    Args:
        advisories: AEMO market advisories.
        keywords: Words to look for; defaults to ADVISORY_KEYWORDS.

    Returns:
        The matching rows.
    """
    keywords = ADVISORY_KEYWORDS if keywords is None else keywords
    pattern = "|".join(keywords)
    return advisories[advisories["Details"].str.contains(pattern, na=False)]


def stated_times(details):
    """Find the period stated in an advisory's text.

    Args:
        details: Advisory text.

    Returns:
        (start, end) as "HH:MM" strings, or None if no period is stated.
    """
    match = STATED_PERIOD.search(re.sub(r"\s+", " ", str(details)))
    if match is None:
        return None
    return match.group(1), match.group(2)


def corrected_period(advisory):
    """Period an advisory warns about.

    AEMO's Start/End Interval are the issue and withdrawal times, so the start is the time
    stated in the text and the end is the earlier of the stated end and the withdrawal.

    Args:
        advisory: One advisory row.

    Returns:
        (start, end) timestamps, or None if it was withdrawn before its period began.
    """
    start = pd.Timestamp(advisory["Start Interval"])
    end = pd.Timestamp(advisory["End Interval"]) if pd.notna(advisory["End Interval"]) else pd.NaT

    times = stated_times(advisory["Details"])
    if times is None:
        return start, end

    day = start.normalize()
    stated_start = day + pd.Timedelta(times[0] + ":00")
    stated_end = day + pd.Timedelta(times[1] + ":00")
    new_end = stated_end if pd.isna(end) else min(stated_end, end)

    if new_end <= stated_start:  # withdrawn (e.g. superseded by a correction) before the period began
        return None
    return stated_start, new_end


def build_sse_events(advisories, categories):
    """Build the sse_events table from advisories, with corrected periods.

    Args:
        advisories: Filtered advisories, from filter_advisories.
        categories: {advisory ID: category}; others get DEFAULT_CATEGORY.

    Returns:
        One row per event, with the sse_events.csv columns plus ID.
    """
    rows = []
    for _, advisory in advisories.iterrows():
        period = corrected_period(advisory)
        if period is None:
            continue
        start, end = period
        rows.append({
            "ID": advisory["ID"],
            "start_date": start.strftime("%Y-%m-%d"),
            "start_time": start.strftime("%H:%M:%S"),
            "end_date": end.strftime("%Y-%m-%d") if pd.notna(end) else None,
            "end_time": end.strftime("%H:%M:%S") if pd.notna(end) else None,
            "duration_td": end - start,
            "duration_days": (end - start) / pd.Timedelta(days=1),
            "category": categories.get(advisory["ID"], DEFAULT_CATEGORY),
        })
    return pd.DataFrame(rows)


def load_manual_events(path=manual_events_path):
    """Load hand-added SSE events.

    Args:
        path: CSV of manual events.

    Returns:
        One row per event: start/end date and time, category and source.
    """
    manual = pd.read_csv(path)
    start = pd.to_datetime(manual["start_date"] + " " + manual["start_time"])
    end = pd.to_datetime(manual["end_date"] + " " + manual["end_time"])
    manual["duration_td"] = end - start
    manual["duration_days"] = (end - start) / pd.Timedelta(days=1)
    return manual


def current_categories():
    """{advisory ID: category} from the current sse_events.csv."""
    current = pd.read_csv(sse_events_path, index_col=0)
    return dict(zip(current["ID"], current["category"]))


def main():
    raw = pd.read_csv(raw_advisories_path, encoding="cp1252")
    categories = current_categories()
    advisories = filter_advisories(raw)

    advisory_events = build_sse_events(advisories, categories)
    events = pd.concat([advisory_events, load_manual_events()], ignore_index=True)
    events["ID"] = events["ID"].astype("Int64")  # manual events have no ID; keep the rest whole numbers
    events.to_csv(sse_events_path)
    print(f"sse events saved to {sse_events_path} ({len(events)} events)")

    new_ids = [i for i in advisory_events["ID"] if i not in categories]
    if new_ids:
        print(f"{len(new_ids)} new events set to '{DEFAULT_CATEGORY}', categorise them by hand: {new_ids}")


def load_sse_events(path=sse_events_path):
    """Load SSE events as timestamps.

    Args:
        path: sse_events CSV.

    Returns:
        One row per event: start, end and category. A missing end is set to the start.
    """
    ev = pd.read_csv(path, index_col=0)

    start = pd.to_datetime(ev["start_date"] + " " + ev["start_time"].fillna("00:00:00"))
    end = pd.to_datetime(
        ev["end_date"].fillna(ev["start_date"]) + " " + ev["end_time"].fillna("23:59:59")
    )
    end = end.mask(ev["end_date"].isna(), start)

    return pd.DataFrame({"start": start, "end": end, "category": ev["category"]})


def sse_day_flags(index, long_event_days=5):
    """Flag the days that fall inside an SSE event.

    Args:
        index: Daily DatetimeIndex.
        long_event_days: Events lasting this many days or more are ignored.

    Returns:
        Frame on index with a boolean is_sse column.
    """
    sse_days = set()
    for _, e in load_sse_events().iterrows():
        days = pd.date_range(e["start"].normalize(), e["end"].normalize())
        if len(days) < long_event_days:
            sse_days.update(days.date)

    return pd.DataFrame({"is_sse": [day in sse_days for day in index.date]}, index=index)


if __name__ == "__main__":
    main()
