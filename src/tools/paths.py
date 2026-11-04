"""Every directory the project reads or writes.
local_*: the repo's data/ folder. vm_*: the raw WEMDE download, outside the repo (VM only).
"""

from pathlib import Path

# --- the two anchors ---

# This repo, found from this file rather than from wherever python started.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The raw corpus, outside the repo. Edit this line if it lives elsewhere.
VM_DATA_ROOT = Path.home() / "bigdata" / "wa-big-batteries-project"


# --- local: the repo's data/, on both machines ---

local_data_dir = PROJECT_ROOT / "data"
local_processed_data_dir = local_data_dir / "processed_data"
local_plots_dir = local_data_dir / "plots"
local_docs_dir = PROJECT_ROOT / "docs"
registration_dir = local_data_dir / "registration"  # AEMO facility registration lists
rcoq_dir = local_data_dir / "rcoq"  # AEMO Reserve Capacity Obligation Quantity files, one per month

# The two pipeline stages, kept in separate folders so it's always obvious
# which is which: extracted/ holds literally what came out of the raw JSON and
# is written once per field by data_extraction; clean/ holds the cleaned and
# derived versions, rewritten cheaply whenever data_processing changes.
extracted_data_dir = local_processed_data_dir / "extracted"
clean_data_dir = local_processed_data_dir / "clean"


# --- vm: the raw corpus, VM only ---

vm_raw_data_dir = VM_DATA_ROOT / "raw_data"
vm_case_input_dir = vm_raw_data_dir / "caseInputData"
vm_dispatch_solution_data_dir = vm_raw_data_dir / "dispatchSolution" / "dispatchData"
vm_predispatch_data_dir = vm_raw_data_dir / "dispatchSolution" / "predispatchData"


def raw_dataset_dir(dataset: str) -> Path:
    """Folder of downloaded .json files for one dataset, e.g. "caseInputData"."""
    return vm_raw_data_dir / dataset


def download_state_dir(dataset: str) -> Path:
    """Folder where download.py tracks failures and extracted zips for one dataset."""
    return VM_DATA_ROOT / "download_state" / dataset
