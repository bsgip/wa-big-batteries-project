"""Ideal response of each battery to the pre-dispatch price forecast, compared with what it did.

For each battery, from its start time:
    1. start time: low SOC, offering, and offering in every interval for the next few days
    2. weighted cost C of the stored energy, in 5-min steps
    3. near price N (0-1h ahead) and far price F (per far window: 1-2h, 1-4h, 1-6h ahead), from the run
       published 30 min before each trading interval
    4. ideal action (Rule 1, Rule 1+2) vs actual action, per far window
"""

import numpy as np
import pandas as pd

ETA = 0.85  # round-trip efficiency
CHARGE_EFFICIENCY = ETA**0.5  # share of the energy drawn that is stored: one side of the round trip
LOW_SOC_PCT = 10  # start time: SOC below this
OFFER_LOOKAHEAD_DAYS = 3  # start time: offering in every 5-min interval for this many days after
ACTION_MW = 5  # trading interval mean power beyond ±this counts as discharging / charging

STEP = pd.Timedelta("5min")
STEP_HOURS = 5 / 60
TRADING_INTERVAL = pd.Timedelta("30min")
PUBLISH_LEAD = pd.Timedelta("30min")  # decision step tau uses the run published at tau - 30 min
NEAR_WINDOW = pd.Timedelta("1h")  # N: [tau, tau + 1h)
# F: [tau + 1h, tau + end), labelled by how far ahead the window runs
FAR_WINDOWS = {"1-2h": pd.Timedelta("2h"), "1-4h": pd.Timedelta("4h"), "1-6h": pd.Timedelta("6h")}
TOP_K = 4  # F: mean of the top k forecast prices

CHARGE, HOLD, DISCHARGE = "charge", "hold", "discharge"
ACTIONS = [CHARGE, HOLD, DISCHARGE]

# the 3x3 ideal vs actual table, collapsed into four outcomes
AGREE = "Agree"
HELD = "Held instead of acting"  # rule said charge or discharge, battery held
ACTED = "Acted instead of holding"  # rule said hold, battery charged or discharged
OPPOSITE = "Opposite direction"  # rule said charge, battery discharged, or the reverse
OUTCOMES = [AGREE, HELD, ACTED, OPPOSITE]


def five_min_grid(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Every 5-min interval from start to end, both included."""
    return pd.date_range(start, end, freq=STEP)


# --- 1. Start time ---


def offered(bidstack: pd.DataFrame, code: str, index: pd.DatetimeIndex) -> pd.Series:
    """Whether code had a tranche of more than 0 MW (charge or discharge) in each interval of index."""
    rows = bidstack[(bidstack["code"] == code) & (bidstack["quantity"].abs() > 0)]
    return pd.Series(index.isin(rows.index), index=index)


def bidstack_gaps(bidstack: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    """Intervals of index with no bid stack rows for any facility (a missing file, not a missing offer)."""
    return pd.Series(~index.isin(bidstack.index), index=index)


def true_for_next(flag: pd.Series, days: int) -> pd.Series:
    """Whether flag is True in every interval of the next `days` days, not counting the interval itself.

    Args:
        flag: Booleans on a complete 5-min grid.
        days: Length of the look-ahead.

    Returns:
        Booleans like flag; False where the look-ahead runs past the end of the data.
    """
    n = days * pd.Timedelta("1D") // STEP
    # rolling(n) at row i covers rows i-n+1..i; shifting by -n moves that to row i-n, i.e. the n rows after it
    return flag.astype(int).rolling(n).min().shift(-n) == 1


def start_time(code: str, soc: pd.DataFrame, bidstack: pd.DataFrame, search_from: str) -> pd.Timestamp:
    """First interval where SOC < LOW_SOC_PCT, the battery offers, and it offers in every interval after.

    The look-ahead is OFFER_LOOKAHEAD_DAYS. Bid stack gaps don't count as not offering.

    Args:
        code: Battery, e.g. "KWINANA_ESR2".
        soc: Clean SOC, with a <code>_soc_pct column.
        bidstack: Clean bid stack, indexed by dispatch_interval.
        search_from: First date to look at, e.g. "2024-12-01".

    Returns:
        The start time.
    """
    grid = five_min_grid(pd.Timestamp(search_from, tz=soc.index.tz), soc.index.max())
    low_soc = soc[f"{code}_soc_pct"].reindex(grid) < LOW_SOC_PCT
    offering = offered(bidstack, code, grid)
    keeps_offering = true_for_next(offering | bidstack_gaps(bidstack, grid), OFFER_LOOKAHEAD_DAYS)

    candidates = low_soc & offering & keeps_offering
    if not candidates.any():
        raise ValueError(f"{code}: no start time from {search_from}")
    return candidates.idxmax()


# --- 2. Weighted cost of stored energy ---


def weighted_cost(soc: pd.Series, power: pd.Series, price: pd.Series) -> pd.Series:
    """Weighted average cost of the stored energy, C_t ($/MWh), in 5-min steps.

    C_0 = p_0. When charging (P_t < 0): C_t = (SOC_{t-1} C_{t-1} + E_t p_t) / (SOC_{t-1} + k E_t),
    with E_t = -P_t * 5/60 and k = CHARGE_EFFICIENCY. Otherwise C_t = C_{t-1}. A step missing
    SOC_{t-1}, P_t or p_t also carries C forward.

    Measured SOC_t isn't used: near empty it reads ~0 and the division blows up.

    Args:
        soc: SOC (MWh) on a complete 5-min grid, starting at the start time.
        power: Power (MW, + discharging), same index.
        price: Actual price ($/MWh), same index.

    Returns:
        C, same index.
    """
    known = soc.shift().notna() & power.notna() & price.notna()
    charging = (known & (power < 0)).to_numpy()
    s, p = soc.to_numpy(), price.to_numpy()
    energy = (-power * STEP_HOURS).to_numpy()  # E_t, MWh drawn in the step

    if np.isnan(p[0]):
        raise ValueError(f"no price at the start time {soc.index[0]}")
    cost = np.empty(len(s))
    cost[0] = p[0]
    for t in range(1, len(s)):
        if charging[t]:
            stored = s[t - 1] + CHARGE_EFFICIENCY * energy[t]
            cost[t] = (s[t - 1] * cost[t - 1] + energy[t] * p[t]) / stored
        else:
            cost[t] = cost[t - 1]
    return pd.Series(cost, index=soc.index)


# --- 3. Near and far forecast prices ---


def forecast_by_decision_step(predispatch_price: pd.DataFrame) -> pd.DataFrame:
    """Each run's energy price forecast, labelled by the decision step it serves.

    Args:
        predispatch_price: The extracted predispatch_price dataset.

    Returns:
        One row per (run, forecast trading interval): tau (run published + PUBLISH_LEAD), offset
        (trading interval start - tau) and price.
    """
    tau = predispatch_price["primary_dispatch_interval"] + PUBLISH_LEAD
    return pd.DataFrame({
        "tau": tau,
        "offset": predispatch_price["dispatch_interval"] - tau,
        "price": predispatch_price["energy_price"],
    })


def window(forecast: pd.DataFrame, start: pd.Timedelta, end: pd.Timedelta) -> pd.DataFrame:
    """Rows of forecast_by_decision_step whose trading interval starts in [tau + start, tau + end)."""
    return forecast[(forecast["offset"] >= start) & (forecast["offset"] < end)]


def trading_intervals_in(start: pd.Timedelta, end: pd.Timedelta) -> int:
    """Number of trading intervals in [start, end)."""
    return (end - start) // TRADING_INTERVAL


def near_price(forecast: pd.DataFrame) -> pd.Series:
    """N: mean forecast price over [tau, tau + 1h), per tau; NaN if the run lacks a trading interval of it."""
    prices = window(forecast, pd.Timedelta(0), NEAR_WINDOW).groupby("tau")["price"]
    complete = prices.size() == trading_intervals_in(pd.Timedelta(0), NEAR_WINDOW)
    return prices.mean().where(complete)


def far_price(forecast: pd.DataFrame, window_end: pd.Timedelta) -> pd.Series:
    """F: mean of the top k forecast prices over the far window [tau + 1h, tau + window_end), per tau.

    k = min(TOP_K, trading intervals in the window). NaN if the run lacks a trading interval of the window.

    Args:
        forecast: From forecast_by_decision_step.
        window_end: End of the far window, e.g. pd.Timedelta("4h") for 1-4h ahead.

    Returns:
        F, indexed by tau.
    """
    rows = window(forecast, NEAR_WINDOW, window_end)
    n = trading_intervals_in(NEAR_WINDOW, window_end)
    k = min(TOP_K, n)

    top_k = rows.sort_values("price", ascending=False).groupby("tau").head(k)
    complete = rows.groupby("tau").size() == n
    return top_k.groupby("tau")["price"].mean().where(complete)


# --- 4. Ideal and actual actions ---


def rule_1(near: pd.Series, far: pd.Series) -> pd.Series:
    """Discharge if N > F and N > 0; charge if N < ETA * F; otherwise hold."""
    discharge = (near > far) & (near > 0)
    charge = near < ETA * far
    return pd.Series(np.select([discharge, charge], [DISCHARGE, CHARGE], HOLD), index=near.index)


def rule_1_2(near: pd.Series, far: pd.Series, cost: pd.Series) -> pd.Series:
    """Rule 1, but discharge also needs N > C (the weighted cost of the stored energy)."""
    discharge = (near > far) & (near > 0) & (near > cost)
    charge = near < ETA * far
    return pd.Series(np.select([discharge, charge], [DISCHARGE, CHARGE], HOLD), index=near.index)


def mean_power(power: pd.Series) -> pd.Series:
    """P_bar: mean power per trading interval, labelled by its start; NaN unless all six 5-min values exist."""
    by_trading_interval = power.resample(TRADING_INTERVAL)
    return by_trading_interval.mean().where(by_trading_interval.count() == TRADING_INTERVAL // STEP)


def actual_action(mean_mw: pd.Series) -> pd.Series:
    """Discharge if P_bar > ACTION_MW, charge if P_bar < -ACTION_MW, otherwise hold."""
    discharge = mean_mw > ACTION_MW
    charge = mean_mw < -ACTION_MW
    return pd.Series(np.select([discharge, charge], [DISCHARGE, CHARGE], HOLD), index=mean_mw.index)


# --- Per battery ---


def decision_table(
    soc: pd.Series,
    power: pd.Series,
    price: pd.Series,
    forecast: pd.DataFrame,
    start: pd.Timestamp,
) -> pd.DataFrame:
    """Ideal and actual action of one battery in every trading interval from its start time, per far window.

    Trading intervals missing the forecast run, a forecast trading interval or a 5-min power value are marked
    skipped and get no actions.

    Args:
        soc: The battery's clean SOC (MWh).
        power: Its clean power (MW, + discharging).
        price: Actual energy price ($/MWh), 5-min.
        forecast: From forecast_by_decision_step.
        start: From start_time.

    Returns:
        One row per (far_window, tau), indexed by tau: far_window, N, F, C, P_bar, skipped, actual, ideal_rule_1
        and ideal_rule_1_2.
    """
    grid = five_min_grid(start, min(soc.index.max(), power.index.max()))
    soc, power, price = soc.reindex(grid), power.reindex(grid), price.reindex(grid)
    taus = pd.date_range(start.ceil(TRADING_INTERVAL), grid[-1], freq=TRADING_INTERVAL)

    common = pd.DataFrame({
        "N": near_price(forecast).reindex(taus),
        "C": weighted_cost(soc, power, price).reindex(taus),
        "P_bar": mean_power(power).reindex(taus),
    })
    per_far_window = [
        common.assign(far_window=label, F=far_price(forecast, end).reindex(taus))
        for label, end in FAR_WINDOWS.items()
    ]
    table = pd.concat(per_far_window)
    table["skipped"] = table[["N", "F", "C", "P_bar"]].isna().any(axis=1)

    used = ~table["skipped"]
    table["actual"] = actual_action(table["P_bar"]).where(used)
    table["ideal_rule_1"] = rule_1(table["N"], table["F"]).where(used)
    table["ideal_rule_1_2"] = rule_1_2(table["N"], table["F"], table["C"]).where(used)
    columns = ["far_window", "N", "F", "C", "P_bar", "skipped", "actual", "ideal_rule_1", "ideal_rule_1_2"]
    return table.rename_axis("tau")[columns]


# --- Summaries ---


def compliance_summary(tables: pd.DataFrame) -> pd.DataFrame:
    """Compliance rate per battery and far window.

    Args:
        tables: decision_table of each battery, concatenated with a "code" index level.

    Returns:
        One row per (code, far_window): trading_intervals used, skipped, rule_1 and rule_1_2 (share of used
        trading intervals where ideal == actual), and rule_2_changed (trading intervals where Rule 1+2 differs
        from Rule 1).
    """
    skipped = tables.groupby(["code", "far_window"])["skipped"].sum()
    tables = tables[~tables["skipped"]]
    df = tables.assign(
        rule_1=tables["ideal_rule_1"] == tables["actual"],
        rule_1_2=tables["ideal_rule_1_2"] == tables["actual"],
        rule_2_changed=tables["ideal_rule_1"] != tables["ideal_rule_1_2"],
    )
    by = df.groupby(["code", "far_window"])
    return pd.DataFrame({
        "trading_intervals": by.size(),
        "skipped": skipped,
        "rule_1":by["rule_1"].mean(),
        "rule_1_2": by["rule_1_2"].mean(),
        "rule_2_changed": by["rule_2_changed"].sum(),
    })


def compliance_by_time_of_day(tables: pd.DataFrame) -> pd.DataFrame:
    """Compliance rate per battery, far window and trading interval of the day.

    Args:
        tables: decision_table of each battery, concatenated with a "code" index level.

    Returns:
        One row per (code, far_window, time_of_day): trading_intervals used, rule_1 and rule_1_2 (share where
        ideal == actual). time_of_day is in hours, e.g. 18.5 for the trading interval starting 18:30.
    """
    df = tables[~tables["skipped"]]
    tau = df.index.get_level_values("tau")
    df = df.assign(
        time_of_day=tau.hour + tau.minute / 60,
        rule_1=df["ideal_rule_1"] == df["actual"],
        rule_1_2=df["ideal_rule_1_2"] == df["actual"],
    )
    by = df.groupby(["code", "far_window", "time_of_day"])
    return pd.DataFrame({
        "trading_intervals": by.size(),
        "rule_1": by["rule_1"].mean(),
        "rule_1_2": by["rule_1_2"].mean(),
    })


def outcome(ideal: pd.Series, actual: pd.Series) -> pd.Series:
    """Which of OUTCOMES each trading interval falls in, from its ideal and actual action."""
    agree = ideal == actual
    held = ~agree & (actual == HOLD)
    acted = ~agree & (ideal == HOLD)
    return pd.Series(np.select([agree, held, acted], [AGREE, HELD, ACTED], OPPOSITE), index=ideal.index)


def outcome_shares(tables: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Share of trading intervals in each outcome, per battery and far window.

    Args:
        tables: decision_table of each battery, concatenated with a "code" index level.
        rule: "ideal_rule_1" or "ideal_rule_1_2".

    Returns:
        One row per (code, far_window), one column per OUTCOMES; each row adds up to 1.
    """
    df = tables[~tables["skipped"]]
    outcomes = outcome(df[rule], df["actual"])
    code = df.index.get_level_values("code")
    shares = pd.crosstab([code, df["far_window"].to_numpy()], outcomes.to_numpy(), normalize="index")
    return shares.rename_axis(index=["code", "far_window"], columns=None).reindex(columns=OUTCOMES, fill_value=0)


def fleet_outcome_shares(tables: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Share of trading intervals in each outcome, all batteries pooled, per far window.

    Every trading interval of every battery counts once, so batteries with a longer record weigh more.

    Args:
        tables: decision_table of each battery, concatenated with a "code" index level.
        rule: "ideal_rule_1" or "ideal_rule_1_2".

    Returns:
        One row per far window, one column per OUTCOMES; each row adds up to 1.
    """
    df = tables[~tables["skipped"]]
    outcomes = outcome(df[rule], df["actual"])
    shares = pd.crosstab(df["far_window"].to_numpy(), outcomes.to_numpy(), normalize="index")
    return shares.rename_axis(index="far_window", columns=None).reindex(columns=OUTCOMES, fill_value=0)


def ideal_vs_actual(table: pd.DataFrame, rule: str) -> pd.DataFrame:
    """3x3 count of trading intervals: ideal action (rows) vs actual action (columns).

    Args:
        table: Rows of decision_table for one battery and one far window.
        rule: "ideal_rule_1" or "ideal_rule_1_2".

    Returns:
        Counts, rows and columns in ACTIONS order.
    """
    counts = pd.crosstab(table[rule], table["actual"])
    return counts.reindex(index=ACTIONS, columns=ACTIONS, fill_value=0).rename_axis(index="ideal", columns="actual")
