"""Clean the extracted datasets: data/processed_data/extracted/ -> data/processed_data/clean/.
Comment out what you don't want. Fields with no cleaning step are copied as extracted, indexed by time.

    python -m data_processing.main
"""

import logging

from data_extraction.catalog import load
from data_processing import bidstack, power, soc, demand, dpv, rcoq, unconstrained_forecast
from tools.io import save_clean
from tools.registration import registered_capacity_mw

logger = logging.getLogger(__name__)


def main():
    clean_bidstack = bidstack.process(load("bidstack"))
    save_clean("bidstack", clean_bidstack)

    # the bid stack tells outages (not offering) from batteries that really are empty
    save_clean("soc", soc.process(load("soc"), clean_bidstack))
    save_clean("power", power.process(load("power")))

    # One extracted frame, three clean ones out (clean/demand, clean/underlying_demand, clean/dpv).
    dispatch_condition = load("dispatch_condition")
    save_clean("demand", demand.process(dispatch_condition))
    save_clean("underlying_demand", demand.process_underlying_demand(dispatch_condition))
    save_clean("dpv", dpv.process(dispatch_condition))

    save_clean("rcoq", rcoq.process(load("rcoq")))

    # One extracted frame in, three clean ones out
    # (clean/pv.parquet, clean/wind.parquet, clean/other_renewables.parquet).
    capacity = registered_capacity_mw()
    for name, frame in unconstrained_forecast.process(load("unconstrained_forecast"), capacity).items():
        save_clean(name, frame)

    # No cleaning step, already indexed by dispatch_interval
    for name in [
        "price",
        "price_setting",
        "market_shortfalls",
        "market_service_requirements",
        "dispatch_total",
        "available_quantities",
        "in_service_quantities",
        "projected_level",
        "energy_schedule",
        "fcess_schedule",
    ]:
        save_clean(name, load(name))

    # No cleaning step, indexed here like the rest of clean/
    save_clean("fcess_bidstack", load("fcess_bidstack").set_index("dispatch_interval"))

    # Predispatch: no cleaning step, indexed by forecast run (dispatch_interval stays a column)
    for name in [
        "predispatch_price",
        "predispatch_price_setting",
        "predispatch_market_shortfalls",
        "predispatch_market_service_requirements",
        "predispatch_dispatch_total",
        "predispatch_projected_level",
        "predispatch_energy_schedule",
    ]:
        save_clean(name, load(name).set_index("primary_dispatch_interval"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
