"""Download raw WEMDE files for a dataset directory under wemde/, e.g.
"caseInputData" or "dispatchSolution/dispatchData". Every file from both
current/ and previous/ lands flattened into raw_dataset_dir(dataset) as .json
- zips are extracted in memory and discarded, only their .json members are
kept.

Usage:
    python -m wem_data.download caseInputData "dispatchSolution/dispatchData"
    python -m wem_data.download caseInputData --retry-failed
    python -m wem_data.download --dataset caseInputData --url "<file-url>"

Failures (bad zips, network errors) are logged and skipped rather than
stopping the run, and the failing URL is recorded in
download_state_dir(dataset)/failed_downloads.txt for later retry via
--retry-failed or --url.
"""

import io
import logging
import zipfile
from pathlib import Path
from urllib.parse import urljoin

import requests
import zipfile_deflate64  # noqa: F401  — registers deflate64 support in zipfile
from bs4 import BeautifulSoup

from tools.constants import DATASET_FOLDERS, ROOT_URL
from tools.paths import download_state_dir, raw_dataset_dir

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        " AppleWebKit/537.36 (KHTML, like Gecko)"
        " Chrome/80.0.3987.87 Safari/537.36"
    )
}

session = requests.Session()
session.headers.update(HEADERS)


def list_directory(url: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (subdirs, files) as (name, url) pairs for an IIS listing page."""
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    subdirs, files = [], []
    for a in soup.find_all("a", href=True):
        href, text = str(a["href"]), a.get_text(strip=True)
        if text == "[To Parent Directory]" or href in ("../", "/"):
            continue
        full = urljoin(url, href)
        if not full.startswith(ROOT_URL):
            continue
        name = text or href.rstrip("/").split("/")[-1]
        (subdirs if href.endswith("/") else files).append((name, full))
    return subdirs, files


def _failed_file(state_dir: Path) -> Path:
    return state_dir / "failed_downloads.txt"


def _record_failure(state_dir: Path, url: str) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = _failed_file(state_dir)
    existing = path.read_text().splitlines() if path.exists() else []
    if url not in existing:
        with path.open("a") as f:
            f.write(url + "\n")


def _clear_failure(state_dir: Path, url: str) -> None:
    path = _failed_file(state_dir)
    if not path.exists():
        return
    remaining = [line for line in path.read_text().splitlines() if line.strip() != url]
    if remaining:
        path.write_text("\n".join(remaining) + "\n")
    else:
        path.unlink()


def _download_plain_file(url: str, dest_path: Path, state_dir: Path, force: bool) -> bool:
    if dest_path.exists() and not force:
        return True
    logger.info(f"downloading {url}")
    try:
        resp = session.get(url, timeout=60)
        resp.raise_for_status()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(resp.content)
    except (requests.RequestException, OSError) as e:
        logger.error(f"failed to download {url}: {e}")
        _record_failure(state_dir, url)
        return False
    logger.info(f"downloaded {dest_path.name}")
    _clear_failure(state_dir, url)
    return True


def _download_and_extract_zip(url: str, dest_dir: Path, state_dir: Path, force: bool) -> bool:
    # The extracted member's filename isn't known until the zip is opened, so
    # a marker file (rather than checking for the extracted file itself) is
    # what lets a re-run skip the download. Kept in state_dir, not dest_dir,
    # so it never sits next to the real .json output.
    marker = state_dir / f"{Path(url).name}.done"
    if marker.exists() and not force:
        return True

    logger.info(f"downloading {url}")
    try:
        resp = session.get(url, timeout=120)
        resp.raise_for_status()
        dest_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            for member in zf.namelist():
                if not member.lower().endswith(".json"):
                    continue
                (dest_dir / member).write_bytes(zf.read(member))
                logger.info(f"extracted {member}")
    except (requests.RequestException, zipfile.BadZipFile, OSError) as e:
        logger.error(f"failed to download/extract {url}: {e}")
        _record_failure(state_dir, url)
        return False

    state_dir.mkdir(parents=True, exist_ok=True)
    marker.touch()
    _clear_failure(state_dir, url)
    return True


def download_file(url: str, dataset: str, force: bool = True) -> bool:
    """Manually (re)download a single file URL belonging to `dataset`, e.g.
    to retry one entry from failed_downloads.txt. Returns True on success."""
    dest_dir = raw_dataset_dir(dataset)
    state_dir = download_state_dir(dataset)
    name = Path(url).name
    if name.lower().endswith(".zip"):
        return _download_and_extract_zip(url, dest_dir, state_dir, force)
    return _download_plain_file(url, dest_dir / name, state_dir, force)


def retry_failed_downloads(dataset: str) -> None:
    """Retry every URL previously logged as failed for `dataset`. URLs that
    succeed are cleared from failed_downloads.txt; URLs that fail again stay
    logged for another retry."""
    state_dir = download_state_dir(dataset)
    failed_file = _failed_file(state_dir)
    if not failed_file.exists():
        logger.info(f"no failed downloads recorded for {dataset}")
        return

    urls = [line.strip() for line in failed_file.read_text().splitlines() if line.strip()]
    logger.info(f"retrying {len(urls)} failed download(s) for {dataset}")
    for url in urls:
        download_file(url, dataset, force=True)


def download_dataset(dataset: str, folders: tuple[str, ...] = DATASET_FOLDERS, force: bool = False) -> None:
    """Download every file under wemde/<dataset>/{folders}/, flattened into
    raw_dataset_dir(dataset) - current/ and previous/ files land in the same
    directory since only the extracted .json content matters. `dataset` can
    be nested, e.g. "dispatchSolution/dispatchData"."""
    dest_dir = raw_dataset_dir(dataset)
    state_dir = download_state_dir(dataset)

    for folder in folders:
        url = f"{ROOT_URL}{dataset}/{folder}/"
        logger.info(f"listing {url}")
        _, files = list_directory(url)

        logger.info(f"{dataset}/{folder}: {len(files)} file(s)")
        for name, file_url in files:
            if name.lower().endswith(".zip"):
                _download_and_extract_zip(file_url, dest_dir, state_dir, force)
            else:
                _download_plain_file(file_url, dest_dir / name, state_dir, force)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help='e.g. caseInputData "dispatchSolution/dispatchData"')
    parser.add_argument("--force", action="store_true", help="re-download even if already present")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="only retry URLs previously logged as failed for the given dataset(s)",
    )
    parser.add_argument("--url", help="manually (re)download a single file URL; requires --dataset")
    parser.add_argument("--dataset", help="dataset the --url file belongs to")
    args = parser.parse_args()

    if args.url:
        if not args.dataset:
            parser.error("--url requires --dataset")
        download_file(args.url, args.dataset, force=True)
    elif args.retry_failed:
        if not args.datasets:
            parser.error("--retry-failed requires at least one dataset")
        for dataset in args.datasets:
            retry_failed_downloads(dataset)
    else:
        if not args.datasets:
            parser.error("at least one dataset is required")
        for dataset in args.datasets:
            download_dataset(dataset, force=args.force)
