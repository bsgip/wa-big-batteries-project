"""Every directory the project reads or writes, in one place.

The work spans two machines, and the paths split cleanly along that line:

    local_*   The repo's own data/ folder - the extracted and cleaned
              parquets, plots, docs. Small enough to commit, so it travels
              with the checkout and is there on both machines.

    vm_*      The raw WEMDE corpus download.py pulls down (~70GB+ of JSON).
              Far too big for the repo, so it sits outside it, under your
              home directory. VM only - none of these exist on a laptop.

Both are anchored explicitly, and neither uses the current working
directory. Scripts here get run from the repo root, from src/ and from
notebooks; a cwd-derived root quietly resolves to a directory that doesn't
exist instead of failing, and a missing directory doesn't surface until
something much later chokes on an empty DataFrame.
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
    """Where download.py and the parsers keep files for one dataset, e.g.
    "caseInputData" or "dispatchSolution/dispatchData". Holds only the
    downloaded/extracted .json files - nothing else."""
    return vm_raw_data_dir / dataset


def download_state_dir(dataset: str) -> Path:
    """Where download.py tracks which zip archives it's already extracted for
    a dataset. Kept out of raw_dataset_dir so that directory stays pure data."""
    return VM_DATA_ROOT / "download_state" / dataset
