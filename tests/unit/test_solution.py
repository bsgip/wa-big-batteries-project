from datetime import UTC, datetime, timedelta, timezone

import pytest

from database.extract.solution import run_from_file_name

PERTH = timezone(timedelta(hours=8))


@pytest.mark.parametrize(
    ("file_name", "expected"),
    [
        (
            "ReferenceDispatchSolution_202608121800.json",
            {
                "schedule": "Dispatch",
                "scenario": "Reference",
                "issued_at": datetime(2026, 8, 12, 18, 0, tzinfo=PERTH),
                "file_variant": "regular",
            },
        ),
        (
            "ReferenceDispatchSolution_202310010835_AffectedDispatchInterval.json",
            {
                "schedule": "Dispatch",
                "scenario": "Reference",
                "issued_at": datetime(2023, 10, 1, 8, 35, tzinfo=PERTH),
                "file_variant": "AffectedDispatchInterval",
            },
        ),
        (
            "ReferenceDispatchSolution_202310040955_AffectedDispatchInterval_AmmendingRule_2023-145.json",
            {
                "schedule": "Dispatch",
                "scenario": "Reference",
                "issued_at": datetime(2023, 10, 4, 9, 55, tzinfo=PERTH),
                "file_variant": "AffectedDispatchInterval_AmmendingRule_2023-145",
            },
        ),
        (
            "ReferencePre-DispatchSolution_202309282000.json",
            {
                "schedule": "Pre-Dispatch",
                "scenario": "Reference",
                "issued_at": datetime(2023, 9, 28, 20, 0, tzinfo=PERTH),
                "file_variant": "regular",
            },
        ),
        (
            "ReferenceWeekAhead-DispatchSolution_202309290800.json",
            {
                "schedule": "Week-Ahead",
                "scenario": "Reference",
                "issued_at": datetime(2023, 9, 29, 8, 0, tzinfo=PERTH),
                "file_variant": "regular",
            },
        ),
    ],
)
def test_run_from_file_name(file_name, expected):
    assert run_from_file_name(file_name) == expected


def test_issued_at_is_the_right_moment():
    # 18:00 in Perth is 10:00 UTC: proves the +08:00 is attached, not just a bare 18:00.
    run = run_from_file_name("ReferenceDispatchSolution_202608121800.json")
    assert run["issued_at"] == datetime(2026, 8, 12, 10, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    "file_name",
    [
        "ReferenceDispatchCase_202608120800.json",  # a Case Input file, not a Solution
        "ReferenceDispatchSolution_20260812.json",  # 8 digits, not 12
        "ReferenceDispatchSolution_202608121800.csv",  # not .json
    ],
)
def test_bad_names_raise(file_name):
    with pytest.raises(ValueError):
        run_from_file_name(file_name)
