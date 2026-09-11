"""Stage 1 of the pipeline: extract/load the raw SOC, demand, bidstack, power
and price data (the actual JSON walk over caseInputData/dispatchSolution),
with on-disk caching so re-runs skip whatever's already extracted.

No cleaning, % conversion, or derived columns here - that split matters:
cleaning logic, rated capacities, and anything else derived from these raw
values can have bugs (like the KWINANA_ESR1/ESR2 capacity swap, or a wrong
sentinel value) that get fixed in tools/constants.py or tools/df_management.py
without ever needing to redo the expensive walk - only a bug in the raw
extraction itself (parsing, sign convention, row ordering) requires deleting
the cached parquet and re-extracting.
"""

import logging
from collections.abc import Callable

import pandas as pd

from tools.df_management import save_df_to_parquet
from tools.paths import repo_processed_data_dir
from data_extraction.parse_case_input import build_case_input_data
from data_extraction.parse_dispatch_solution import build_price_and_power_df

logger = logging.getLogger(__name__)


#  parse_case_input.build_case_input_data()'s field name -> saved filename.
#  "charge_level" saves as soc.parquet (the established name for that
#  processed output); everything else matches its field name. Adding a new
#  field there just needs one more entry here - nothing else in main()
#  changes.
_CASE_INPUT_FILENAMES = {
    "charge_level": "soc.parquet",
    "demand": "demand.parquet",
    "bidstack": "bidstack.parquet",
}


def _extract_case_input_data() -> dict[str, pd.DataFrame]:
    """Load whichever fields are already cached on disk, and only extract
    the ones that are missing - if e.g. soc.parquet and demand.parquet
    already exist and only bidstack.parquet doesn't, this walks the corpus
    once and runs just the bidstack extractor, not all three."""
    paths = {name: repo_processed_data_dir / filename for name, filename in _CASE_INPUT_FILENAMES.items()}

    cached = {name: pd.read_parquet(path) for name, path in paths.items() if path.exists()}
    missing = [name for name in _CASE_INPUT_FILENAMES if name not in cached]

    if not missing:
        logger.info(f"{', '.join(p.name for p in paths.values())} already exist, loading instead of re-extracting")
        return cached

    logger.info(f"extracting missing field(s) {missing} from caseInputData ({list(cached)} already cached)")
    fresh = build_case_input_data(max_workers=12, fields=missing)
    for name, df in fresh.items():
        save_df_to_parquet(df, paths[name])

    return {**cached, **fresh}


def _extract_power_and_price():
    power_path = repo_processed_data_dir / "power.parquet"
    price_path = repo_processed_data_dir / "price.parquet"

    if power_path.exists() and price_path.exists():
        logger.info(f"{power_path.name} and {price_path.name} already exist, loading instead of re-extracting")
        return pd.read_parquet(power_path), pd.read_parquet(price_path)

    price_df, power_df = build_price_and_power_df(dates=None, max_workers=12)
    save_df_to_parquet(power_df, power_path)
    save_df_to_parquet(price_df, price_path)
    return power_df, price_df


def extract_raw_data(
    step: Callable[..., object],
) -> tuple[pd.DataFrame | None, pd.Series | None, pd.DataFrame | None, pd.Series | None]:
    """Run stage 1: extract/load raw case input data (SOC, demand, bidstack)
    and raw power/price data. Returns (raw_soc_df, demand, raw_power_df,
    price)."""
    logger.info("Extracting/loading raw case input data (SOC, demand, bidstack) from caseInputData...")
    case_input_data = step("extract/load raw case input data", _extract_case_input_data) or {}
    raw_soc_df = case_input_data.get("charge_level")
    demand_df = case_input_data.get("demand")
    demand = demand_df["dispatchCondition.demand"] if demand_df is not None else None

    logger.info("Extracting/loading raw power and price from dispatchSolution (slow if not cached, ~45-60min)...")
    power_price = step("extract/load raw power and price", _extract_power_and_price)
    raw_power_df, price_df = power_price if power_price else (None, None)
    price = price_df["energy_price"] if price_df is not None else None

    return raw_soc_df, demand, raw_power_df, price
