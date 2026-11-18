"""What each battery did vs what a rule-following battery would do, given the price forecast at the time.

    python -m exploration.ideal_response
"""

import pandas as pd

from analysis.ideal_response import (
    FAR_WINDOWS,
    compliance_by_time_of_day,
    compliance_summary,
    decision_table,
    fleet_outcome_shares,
    forecast_by_decision_step,
    outcome_shares,
    start_time,
)
from tools.constants import START_DATE, battery_codes
from tools.io import load_clean
from visualisation.plot_ideal_response import (
    plot_compliance_by_time_of_day,
    plot_compliance_dumbbell,
    plot_ideal_vs_actual,  # noqa: F401
    plot_outcome_bars,
)

CHECK_BATTERY = "KWINANA_ESR2"
CHECK_DAYS = ["2025-08-25", "2025-08-18"]
CHECK_WINDOW = ("15:00", "22:00")
RULES = {"ideal_rule_1": "Rule 1", "ideal_rule_1_2": "Rule 1+2"}


def day_view(table: pd.DataFrame, day: str, window: tuple[str, str]) -> pd.DataFrame:
    """One battery's decision_table on one day's window, one row per trading interval, far windows side by side."""
    start, end = (pd.Timestamp(f"{day} {t}", tz=table.index.tz) for t in window)
    rows = table.loc[(table.index >= start) & (table.index < end) & ~table["skipped"]]
    common = rows.loc[rows["far_window"] == "1-2h", ["N", "C", "P_bar", "actual"]].round(1)
    far = rows.pivot(columns="far_window", values="F").add_prefix("F_").round(1)
    ideal_1 = rows.pivot(columns="far_window", values="ideal_rule_1").add_prefix("rule_1_")
    ideal_1_2 = rows.pivot(columns="far_window", values="ideal_rule_1_2").add_prefix("rule_1_2_")
    return common.join([far, ideal_1, ideal_1_2])


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    # Load
    soc = load_clean("soc", start_date=None)
    power = load_clean("power", start_date=None)
    price = load_clean("price", start_date=None)["energy_price"]
    bidstack = load_clean("bidstack", start_date=None)
    forecast = forecast_by_decision_step(load_clean("predispatch_price", start_date=None).reset_index())

    # Start times
    starts = {code: start_time(code, soc, bidstack, START_DATE) for code in battery_codes}
    print(pd.Series(starts, name="start_time"))

    # Check 3: one battery on two days
    # check = decision_table(soc[CHECK_BATTERY], power[CHECK_BATTERY], price, forecast, starts[CHECK_BATTERY])
    # for day in CHECK_DAYS:
    #     print(f"\n{CHECK_BATTERY} {day} {CHECK_WINDOW[0]}-{CHECK_WINDOW[1]}")
    #     print(day_view(check, day, CHECK_WINDOW))

    # All batteries
    tables = pd.concat(
        {code: decision_table(soc[code], power[code], price, forecast, start) for code, start in starts.items()},
        names=["code"],
    )
    summary = compliance_summary(tables)
    print(summary.round(3))

    # Fleet outcomes (%), per rule and far window
    for rule, rule_label in RULES.items():
        print(f"\nFleet, {rule_label}\n{(fleet_outcome_shares(tables, rule) * 100).round(1)}")

    # Compliance rate per battery
    plot_compliance_dumbbell(
        summary,
        title="Trading intervals where the battery did what the price-forecast rule said",
        filename="ideal_response/compliance_by_battery.png",
    )

    # Ideal vs actual, as four outcomes per battery
    plot_outcome_bars(
        {label: outcome_shares(tables, rule) for rule, label in RULES.items()},
        title="How each battery's action compared with the price-forecast rule",
        filename="ideal_response/outcomes_by_battery.png",
    )

    # Ideal vs actual, full 3x3 heatmaps per rule and far window
    # for rule, rule_label in RULES.items():
    #     for far_window in FAR_WINDOWS:
    #         plot_ideal_vs_actual(
    #             tables, rule, far_window,
    #             title=f"{rule_label}, far window {far_window} ahead: "
    #                   "what the battery did when the rule said each action",
    #             filename=f"ideal_response/ideal_vs_actual_{rule.removeprefix('ideal_')}_{far_window}.png",
    #         )

    # Compliance through the day, per far window
    by_time = compliance_by_time_of_day(tables)
    for far_window in FAR_WINDOWS:
        plot_compliance_by_time_of_day(
            by_time, far_window,
            title=f"Trading intervals where the battery did what the rule said, by time of day "
                  f"(far window {far_window} ahead)",
            filename=f"ideal_response/compliance_by_time_of_day_{far_window}.png",
        )


if __name__ == "__main__":
    main()
