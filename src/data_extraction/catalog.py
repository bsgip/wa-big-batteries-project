"""The named datasets this project extracts, and how to get at them.

Two operations, deliberately kept under separate names because they cost
wildly different amounts:

    extract(...)  walks the raw JSON corpus and writes extracted/<name>.parquet.
                  Expensive (~45min for dispatchSolution, ~70GB of reads for
                  caseInputData) and essentially a one-time job per field.

    load(name)    reads extracted/<name>.parquet. Cheap, and NEVER extracts -
                  if the file isn't there it says so instead of silently
                  starting a 45-minute walk.

extracted/ holds literally what came out of the JSON. Cleaning and derived
columns live in data_processing/, which reads these and writes clean/.

To add a new field: register an extractor in the relevant parse_*.py module,
then add one entry here.
"""

import logging
from dataclasses import dataclass

import pandas as pd

from tools.io import save_df_to_parquet
from tools.paths import extracted_data_dir
from data_extraction import parse_case_input, parse_dispatch_solution

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
    "demand": Dataset(
        "demand.parquet", "case_input", "demand", "system demand / DPV SCADA tags (MW)"
    ),
    "bidstack": Dataset(
        "bidstack.parquet", "case_input", "bidstack", "battery energy offer tranches, long format"
    ),
    "power": Dataset(
        "power.parquet", "dispatch_solution", "power", "battery power from initialMw (MW), per battery"
    ),
    "price": Dataset(
        "price.parquet", "dispatch_solution", "price", "energy market clearing price ($/MWh)"
    ),
}


def _build_case_input(fields: list[str], dates: list[str] | None, max_workers: int):
    if dates is not None:
        raise ValueError("caseInputData has no date fast path - omit dates for these fields")
    return parse_case_input.build_case_input_data(fields=fields, max_workers=max_workers)


def _build_dispatch_solution(fields: list[str], dates: list[str] | None, max_workers: int):
    return parse_dispatch_solution.build_dispatch_solution_data(
        fields=fields, dates=dates, max_workers=max_workers
    )


_BUILDERS = {
    "case_input": _build_case_input,
    "dispatch_solution": _build_dispatch_solution,
}

_FIELD_NAMES = {
    "case_input": parse_case_input.FIELD_NAMES,
    "dispatch_solution": parse_dispatch_solution.FIELD_NAMES,
}


def _validate_catalog() -> None:
    """Fail at import rather than at extraction time. A typo in a source or
    field name would otherwise only surface once someone had already kicked
    off a multi-hour walk."""
    for name, dataset in CATALOG.items():
        if dataset.source not in _BUILDERS:
            raise ValueError(f"{name}: unknown source {dataset.source!r}; expected one of {list(_BUILDERS)}")
        known = _FIELD_NAMES[dataset.source]
        if dataset.field not in known:
            raise ValueError(f"{name}: {dataset.source} has no field {dataset.field!r}; expected one of {known}")


_validate_catalog()


def _dataset(name: str) -> Dataset:
    if name not in CATALOG:
        raise ValueError(f"unknown dataset {name!r}; expected one of {list(CATALOG)}")
    return CATALOG[name]


def extracted_path(name: str):
    return extracted_data_dir / _dataset(name).filename


def is_extracted(name: str) -> bool:
    return extracted_path(name).exists()


def list_datasets() -> list[tuple[str, str, bool]]:
    """(name, description, already extracted?) for every registered dataset."""
    return [(name, ds.description, is_extracted(name)) for name, ds in CATALOG.items()]


def load(name: str) -> pd.DataFrame:
    """Read one extracted dataset. Never extracts - see this module's docstring."""
    path = extracted_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{name} not extracted yet ({path} doesn't exist) - run extract([{name!r}])")
    return pd.read_parquet(path)


def load_many(names: list[str]) -> dict[str, pd.DataFrame]:
    return {name: load(name) for name in names}


def extract(
    names: list[str], *, dates: list[str] | None = None, refresh: bool = False, max_workers: int = 12
) -> dict[str, pd.DataFrame]:
    """Walk the raw corpus for `names` and write extracted/<name>.parquet.

    Names are grouped by source, so asking for several fields from the same
    corpus walks it ONCE. Names already extracted are skipped unless
    `refresh` - a guard against accidentally redoing a one-time job, not a
    cache optimisation.

    `dates` (e.g. ["2025-01-21"]) restricts dispatchSolution to a handful of
    files instead of ~300k - useful for exploring a new field on a couple of
    days before committing to the full walk. Results from a `dates` run are
    NOT written to disk, since they'd be a partial dataset masquerading as a
    complete one; they're returned for inspection only.
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
