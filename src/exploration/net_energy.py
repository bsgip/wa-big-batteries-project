MIDDAY_START = "09:00"
MIDDAY_END = "15:00"
INTERVAL_HOURS = 5 / 60


def fleet_midday_net_energy_pct(
    power: pd.DataFrame,
    commissioning: dict[str, pd.Timestamp],
    start: str = MIDDAY_START,
    end: str = MIDDAY_END,
) -> pd.Series:
    """Daily net energy the fleet moves between `start` and `end`, as a
    percentage of the commissioned fleet's storage capacity.

    Sums the signed MW columns (discharge +, charge -) over the window and
    converts to MWh, so charge and discharge inside the window net off.
    Midday is normally a net charge, i.e. negative, so the result is returned
    as a magnitude - net energy withdrawn from the grid.

    The denominator is every battery commissioned by that date, not just the
    ones reporting that day, matching add_fleet_soc_columns - so a battery
    sitting out an outage drags the percentage down rather than quietly
    leaving the fleet. `commissioning` comes from constants.commissioning_dates;
    build it from the soc and power frames together so both metrics date the
    fleet identically.
    """
    codes = codes_in(power, battery_codes)
    window = power[codes].between_time(start, end, inclusive="left")

    # only the batteries this frame can report on: ALINTA_WGP_ESR1 is commissioned
    # and in the dict, but power.parquet carries no column for it, so counting its
    # capacity here would be dividing by energy the numerator can never see
    dates = {code: commissioning[code] for code in codes if code in commissioning}

    daily_energy = window.groupby(pd.Grouper(freq="D")).sum(min_count=1) * INTERVAL_HOURS
    net_energy = daily_energy.sum(axis=1, min_count=1)
    capacity = commissioned_capacity(daily_energy.index, dates, battery_capacity_MWh)

    pct = (net_energy / capacity * 100).abs()
    return pct.rename("fleet_midday_net_energy_pct")


commissioning = commissioning_dates(soc_clean, power_clean)
fleet_midday_net_energy = fleet_midday_net_energy_pct(power_clean, commissioning)

