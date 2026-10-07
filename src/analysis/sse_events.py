import pandas as pd

from tools.paths import local_data_dir


sse_events_path = local_data_dir / "sse_events.csv"


def load_sse_events(path=sse_events_path):
    """SSE events as start/end timestamps plus category.

    Events with no recorded end are treated as ending when they started, i.e.
    drawn as a single line rather than a span.
    """
    ev = pd.read_csv(path, index_col=0)

    start = pd.to_datetime(ev["start_date"] + " " + ev["start_time"].fillna("00:00:00"))
    end = pd.to_datetime(
        ev["end_date"].fillna(ev["start_date"]) + " " + ev["end_time"].fillna("23:59:59")
    )
    end = end.mask(ev["end_date"].isna(), start)

    return pd.DataFrame({"start": start, "end": end, "category": ev["category"]})


def sse_day_flags(index, long_event_days=5):
    """Flag which days in index fall in a short (is_sse) or long (is_sse_long) SSE event."""
    short_days = set()
    long_days = set()

    for _, e in load_sse_events().iterrows():
        days = pd.date_range(e["start"].normalize(), e["end"].normalize())
        if len(days) < long_event_days:
            short_days.update(days.date)
        else:
            long_days.update(days.date)

    return pd.DataFrame({
        "is_sse": [day in short_days for day in index.date],
        "is_sse_long": [day in long_days for day in index.date],
    }, index=index)
