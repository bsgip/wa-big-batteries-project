"""The named datasets this project extracts.
extract() walks the raw JSON and writes extracted/<name>.parquet (slow, one-time).
load() reads it, and never extracts.
"""

import logging
import shutil
from dataclasses import dataclass

import pandas as pd
import pyarrow.parquet as pq

from tools.io import save_df_to_parquet
from tools.paths import extracted_data_dir
from data_extraction import (
    parse_case_input,
    parse_dispatch_solution,
    parse_predispatch_solution,
    parse_rcoq,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dataset:
    """One extractable dataset: which parser produces it, under which field
    name there, and what it lands on disk as."""

    filename: str
    source: str
    field: str
    description: str = ""


CATALOG: dict[str, Dataset] = {
    "soc": Dataset(
        "soc.parquet", "case_input", "charge_level", "battery state of charge (MWh), per battery"
    ),
    "dispatch_condition": Dataset(
        "dispatch_condition.parquet",
        "case_input",
        "dispatch_condition",
        "system-wide dispatchCondition SCADA tags: demand, dpv, underlyingDemand, ... (MW)",
    ),
    "bidstack": Dataset(
        "bidstack.parquet", "case_input", "bidstack", "battery energy offer tranches, long format"
    ),
    "fcess_bidstack": Dataset(
        "fcess_bidstack.parquet",
        "case_input",
        "fcess_bidstack",
        "battery FCESS offer tranches (regulation/contingency raise/lower, rocof) + enablement limits, long format",
    ),
    "unconstrained_forecast": Dataset(
        "unconstrained_forecast.parquet",
        "case_input",
        "unconstrained_forecast",
        "unconstrained output forecast (signed MW) per intermittent facility, long format",
    ),
    "power": Dataset(
        "power.parquet",
        "case_input",
        "power",
        "measured output (MW) per facility, all facilities, off the <code>.energy.sentOut SCADA tag",
    ),
    "rcoq": Dataset(
        "rcoq.parquet",
        "rcoq_files",
        "rcoq",
        "reserve capacity obligation quantity (MW) per facility, 5-min, from the AEMO CSVs in data/rcoq/",
    ),
    "price": Dataset(
        "price.parquet",
        "dispatch_solution",
        "price",
        "clearing price per market service: energy ($/MWh) + the five FCESS markets, wide",
    ),
    # dispatchData: the realised (dispatched) value per 5-min interval - the
    # same fields as the predispatch_* datasets below, plus available/in-service
    # quantities.
    "price_setting": Dataset(
        "price_setting.parquet",
        "dispatch_solution",
        "price_setting",
        "what set each market service's price: one row per priceSetting contribution",
    ),
    "market_shortfalls": Dataset(
        "market_shortfalls.parquet",
        "dispatch_solution",
        "market_shortfalls",
        "marketShortfalls: energy/FCESS deficits (MW)",
    ),
    "market_service_requirements": Dataset(
        "market_service_requirements.parquet",
        "dispatch_solution",
        "market_service_requirements",
        "marketServiceRequirements: energy/FCESS requirements (MW)",
    ),
    "dispatch_total": Dataset(
        "dispatch_total.parquet",
        "dispatch_solution",
        "dispatch_total",
        "dispatchTotal: total injection/withdrawal and FCESS dispatched (MW)",
    ),
    "available_quantities": Dataset(
        "available_quantities.parquet",
        "dispatch_solution",
        "available_quantities",
        "availableQuantities: energy/FCESS capacity available (MW)",
    ),
    "in_service_quantities": Dataset(
        "in_service_quantities.parquet",
        "dispatch_solution",
        "in_service_quantities",
        "inServiceQuantities: energy/FCESS capacity in service (MW)",
    ),
    "projected_level": Dataset(
        "projected_level.parquet",
        "dispatch_solution",
        "projected_level",
        "projectedCharge/DischargeLevel per battery, long format (only exists from late Oct 2025)",
    ),
    "energy_schedule": Dataset(
        "energy_schedule.parquet",
        "dispatch_solution",
        "energy_schedule",
        "dispatched energy target (MW) per facility, wide, every facility incl. wind/solar/DSPs",
    ),
    "fcess_schedule": Dataset(
        "fcess_schedule.parquet",
        "dispatch_solution",
        "fcess_schedule",
        "enabled FCESS quantity (MW) per facility per FCESS market, every FCESS facility, long format",
    ),
    # predispatchData: every 30-min forecast run x its 96 forecast intervals,
    # long format.
    "predispatch_price": Dataset(
        "predispatch_price.parquet",
        "predispatch_solution",
        "price",
        "forecast clearing price per market service: energy ($/MWh) + the five FCESS markets",
    ),
    "predispatch_price_setting": Dataset(
        "predispatch_price_setting.parquet",
        "predispatch_solution",
        "price_setting",
        "what set each market service's price: one row per priceSetting contribution",
    ),
    "predispatch_market_shortfalls": Dataset(
        "predispatch_market_shortfalls.parquet",
        "predispatch_solution",
        "market_shortfalls",
        "marketShortfalls: energy/FCESS deficits (MW)",
    ),
    "predispatch_market_service_requirements": Dataset(
        "predispatch_market_service_requirements.parquet",
        "predispatch_solution",
        "market_service_requirements",
        "marketServiceRequirements: energy/FCESS requirements (MW)",
    ),
    "predispatch_dispatch_total": Dataset(
        "predispatch_dispatch_total.parquet",
        "predispatch_solution",
        "dispatch_total",
        "dispatchTotal: total injection/withdrawal and FCESS dispatched (MW)",
    ),
    "predispatch_projected_level": Dataset(
        "predispatch_projected_level.parquet",
        "predispatch_solution",
        "projected_level",
        "projectedCharge/DischargeLevel per battery (only exists from 2025-10-30)",
    ),
    "predispatch_energy_schedule": Dataset(
        "predispatch_energy_schedule.parquet",
        "predispatch_solution",
        "energy_schedule",
        "scheduled energy (MW) per facility, every facility incl. wind/solar/DSPs",
    ),
}


def _build_case_input(fields: list[str], dates: list[str] | None, max_workers: int):
    """Builder for caseInputData fields; raises ValueError if dates is given (no per-day option)."""
    if dates is not None:
        raise ValueError("caseInputData has no date fast path - omit dates for these fields")
    return parse_case_input.build_case_input_data(fields=fields, max_workers=max_workers)


def _build_dispatch_solution(fields: list[str], dates: list[str] | None, max_workers: int):
    """Builder for dispatchSolution/dispatchData fields."""
    return parse_dispatch_solution.build_dispatch_solution_data(
        fields=fields, dates=dates, max_workers=max_workers
    )


def _build_rcoq(fields: list[str], dates: list[str] | None, max_workers: int):
    """Builder for RCOQ (fields, dates and max_workers are ignored)."""
    return parse_rcoq.build_rcoq_data()


_BUILDERS = {
    "case_input": _build_case_input,
    "dispatch_solution": _build_dispatch_solution,
    "rcoq_files": _build_rcoq,
}

# Sources whose output is built and written one month at a time, by
# _extract_monthly, rather than returned whole by a _BUILDERS entry.
_MONTHLY_BUILDERS = {
    "predispatch_solution": parse_predispatch_solution.iter_predispatch_solution_months,
}

_FIELD_NAMES = {
    "case_input": parse_case_input.FIELD_NAMES,
    "dispatch_solution": parse_dispatch_solution.FIELD_NAMES,
    "predispatch_solution": parse_predispatch_solution.FIELD_NAMES,
    "rcoq_files": parse_rcoq.FIELD_NAMES,
}


def _validate_catalog() -> None:
    """Raise at import if a CATALOG entry names an unknown source or field."""
    for name, dataset in CATALOG.items():
        if dataset.source not in _FIELD_NAMES:
            raise ValueError(f"{name}: unknown source {dataset.source!r}; expected one of {list(_FIELD_NAMES)}")
        known = _FIELD_NAMES[dataset.source]
        if dataset.field not in known:
            raise ValueError(f"{name}: {dataset.source} has no field {dataset.field!r}; expected one of {known}")


_validate_catalog()


def _dataset(name: str) -> Dataset:
    """The CATALOG entry for name; raises ValueError listing the known names."""
    if name not in CATALOG:
        raise ValueError(f"unknown dataset {name!r}; expected one of {list(CATALOG)}")
    return CATALOG[name]


def extracted_path(name: str):
    """Path of extracted/<file> for dataset name."""
    return extracted_data_dir / _dataset(name).filename


def is_extracted(name: str) -> bool:
    """Whether dataset name has been extracted."""
    return extracted_path(name).exists()


def list_datasets() -> list[tuple[str, str, bool]]:
    """(name, description, already extracted?) for every registered dataset."""
    return [(name, ds.description, is_extracted(name)) for name, ds in CATALOG.items()]


def load(name: str, **read_parquet_kwargs) -> pd.DataFrame:
    """Read one extracted dataset. Never extracts.

    Args:
        name: Dataset name in CATALOG, e.g. "price".
        **read_parquet_kwargs: Passed to pd.read_parquet, e.g. columns=[...] or
            filters=[("code", "in", battery_codes)] to read part of a big dataset.

    Returns:
        The dataset.

    Raises:
        FileNotFoundError: If it hasn't been extracted.
    """
    path = extracted_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{name} not extracted yet ({path} doesn't exist) - run extract([{name!r}])")
    return pd.read_parquet(path, **read_parquet_kwargs)


def load_many(names: list[str]) -> dict[str, pd.DataFrame]:
    """{name: load(name)} for each name."""
    return {name: load(name) for name in names}


def extract(
    names: list[str], *, dates: list[str] | None = None, refresh: bool = False, max_workers: int = 12
) -> dict[str, pd.DataFrame]:
    """Walk the raw files and write extracted/<name>.parquet for each name.

    Names from the same source share one walk. Already-extracted names are skipped unless
    refresh is True.

    Args:
        names: Dataset names in CATALOG.
        dates: Only these days, e.g. ["2025-01-21"] (dispatch and predispatch sources);
            results are returned, not written.
        refresh: Re-extract names that already exist.
        max_workers: Parallel processes.

    Returns:
        {name: DataFrame} for a dates run; otherwise {}.
    """
    for name in names:
        _dataset(name)

    wanted = names if (refresh or dates is not None) else [n for n in names if not is_extracted(n)]
    for name in names:
        if name not in wanted:
            logger.info(f"{name}: already extracted, skipping (pass refresh=True to redo)")
    if not wanted:
        return {}

    by_source: dict[str, list[str]] = {}
    for name in wanted:
        by_source.setdefault(CATALOG[name].source, []).append(name)

    logger.info(f"extracting {wanted} via {len(by_source)} corpus walk(s): {list(by_source)}")

    extracted = {}
    for source, source_names in by_source.items():
        if source in _MONTHLY_BUILDERS:
            extracted.update(_extract_monthly(source, source_names, dates, refresh, max_workers))
            continue

        fields = [CATALOG[name].field for name in source_names]
        field_to_name = {CATALOG[name].field: name for name in source_names}

        dfs = _BUILDERS[source](fields, dates, max_workers)

        for field, df in dfs.items():
            if field not in field_to_name:
                continue
            name = field_to_name[field]
            extracted[name] = df
            if dates is None:
                save_df_to_parquet(df, extracted_path(name))

    if dates is not None:
        logger.info("dates= run: results returned but NOT saved (partial data)")

    return extracted


def _extract_monthly(
    source: str, names: list[str], dates: list[str] | None, refresh: bool, max_workers: int
) -> dict[str, pd.DataFrame]:
    """extract() for monthly sources: each month goes to extracted/<name>.partial/, then all
    months are merged into extracted/<name>.parquet.

    A run that died resumes from the months already written; refresh=True starts over.

    Returns:
        {name: DataFrame} for a dates run; otherwise {}.
    """
    fields = [CATALOG[name].field for name in names]
    field_to_name = {CATALOG[name].field: name for name in names}
    months = _MONTHLY_BUILDERS[source]

    if dates is not None:
        parts: dict[str, list[pd.DataFrame]] = {name: [] for name in names}
        for _month, dfs in months(fields, dates=dates, max_workers=max_workers):
            for field, df in dfs.items():
                parts[field_to_name[field]].append(df)
        return {name: pd.concat(dfs, ignore_index=True) for name, dfs in parts.items() if dfs}

    partial_dirs = {name: extracted_path(name).with_suffix(".partial") for name in names}
    if refresh:
        for partial_dir in partial_dirs.values():
            shutil.rmtree(partial_dir, ignore_errors=True)

    done_per_name = [
        {p.stem for p in partial_dir.glob("*.parquet")} if partial_dir.exists() else set()
        for partial_dir in partial_dirs.values()
    ]
    done = set.intersection(*done_per_name)

    for month, dfs in months(fields, max_workers=max_workers, skip_months=done):
        for field, df in dfs.items():
            partial_dir = partial_dirs[field_to_name[field]]
            partial_dir.mkdir(parents=True, exist_ok=True)
            # Write-then-rename, so a crash mid-write can't leave a truncated
            # month that a resume would then treat as done.
            tmp = partial_dir / f"{month}.parquet.tmp"
            df.to_parquet(tmp, index=False)
            tmp.rename(partial_dir / f"{month}.parquet")

    for name, partial_dir in partial_dirs.items():
        _merge_months(partial_dir, extracted_path(name))
        shutil.rmtree(partial_dir)
        logger.info(f"{name}: written to {extracted_path(name)}")

    return {}


def _merge_months(partial_dir, final_path) -> None:
    """Append partial_dir's monthly parquets, in month order, into one parquet at final_path."""
    month_files = sorted(partial_dir.glob("*.parquet"))
    tmp = final_path.with_suffix(".parquet.tmp")
    with pq.ParquetWriter(tmp, pq.read_schema(month_files[0])) as writer:
        for month_file in month_files:
            writer.write_table(pq.read_table(month_file))
    tmp.rename(final_path)
