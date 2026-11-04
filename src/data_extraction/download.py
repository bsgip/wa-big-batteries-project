"""Download raw WEMDE files for a dataset (e.g. caseInputData) into raw_dataset_dir(dataset), unzipped.
Failed URLs are logged to failed_downloads.txt for --retry-failed.

Usage:
    python -m data_extraction.download caseInputData "dispatchSolution/dispatchData"
    python -m data_extraction.download caseInputData --from-date 2025-01-01 --to-date 2025-01-31
    python -m data_extraction.download caseInputData --retry-failed
    python -m data_extraction.download --dataset caseInputData --url "<file-url>"
"""

import io
import logging
import re
import zipfile
from datetime import date
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
    """List an IIS directory page.

    Args:
        url: Directory URL.

    Returns:
        (subdirs, files), each a list of (name, url).
    """
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


_DATE_RE = re.compile(r"_(\d{8})\d*(?:_\d+)?\.\w+$")


def _extract_date(name: str) -> date | None:
    """The leading YYYYMMDD date in a file name, or None."""
    m = _DATE_RE.search(name)
    if not m:
        return None
    try:
        return date(int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8]))
    except ValueError:
        return None


def _in_date_range(name: str, start: date | None, end: date | None) -> bool:
    """Whether a file name's date is within [start, end]; names without a date always are."""
    if start is None and end is None:
        return True
    file_date = _extract_date(name)
    if file_date is None:
        return True
    if start and file_date < start:
        return False
    if end and file_date > end:
        return False
    return True


def _failed_file(state_dir: Path) -> Path:
    """Path of the failed-downloads log in state_dir."""
    return state_dir / "failed_downloads.txt"


def _record_failure(state_dir: Path, url: str) -> None:
    """Add url to the failed-downloads log."""
    state_dir.mkdir(parents=True, exist_ok=True)
    path = _failed_file(state_dir)
    existing = path.read_text().splitlines() if path.exists() else []
    if url not in existing:
        with path.open("a") as f:
            f.write(url + "\n")


def _clear_failure(state_dir: Path, url: str) -> None:
    """Remove url from the failed-downloads log."""
    path = _failed_file(state_dir)
    if not path.exists():
        return
    remaining = [line for line in path.read_text().splitlines() if line.strip() != url]
    if remaining:
        path.write_text("\n".join(remaining) + "\n")
    else:
        path.unlink()


def _download_plain_file(url: str, dest_path: Path, state_dir: Path, force: bool) -> bool:
    """Download url to dest_path (skipped if it exists, unless force). Returns True on success."""
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
    """Download a zip and extract its .json files into dest_dir (skipped if done, unless force). Returns True on success."""
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


def record_corrupted_file(dataset: str, filename: str) -> None:
    """Log a file that failed to parse as a failed download, so --retry-failed fetches it again.

    Assumes the file came from current/.

    Args:
        dataset: Dataset folder, e.g. "caseInputData".
        filename: Name of the local file.
    """
    url = f"{ROOT_URL}{dataset}/current/{filename}"
    _record_failure(download_state_dir(dataset), url)


def log_and_record_parse_failures(paths: list[Path], errors: list[str | None], dataset: str) -> None:
    """Log a summary of parse failures and record each with record_corrupted_file.

    Args:
        paths: Files that were parsed.
        errors: Error message per file, or None if it parsed.
        dataset: Dataset folder, e.g. "caseInputData".
    """
    failed = [(path, error) for path, error in zip(paths, errors, strict=True) if error is not None]

    if not failed:
        logger.info(f"{dataset}: all {len(paths)} files parsed cleanly")
        return

    logger.warning(f"{dataset}: {len(failed)} of {len(paths)} files failed to parse, recorded for --retry-failed")
    for path, error in failed:
        logger.debug(f"{path.name}: {error}")
        record_corrupted_file(dataset, path.name)


def download_file(url: str, dataset: str, force: bool = True) -> bool:
    """Download one file URL of a dataset, e.g. to retry an entry in failed_downloads.txt.

    Args:
        url: File URL.
        dataset: Dataset folder, e.g. "caseInputData".
        force: Download even if the file is already there.

    Returns:
        True on success.
    """
    dest_dir = raw_dataset_dir(dataset)
    state_dir = download_state_dir(dataset)
    name = Path(url).name
    if name.lower().endswith(".zip"):
        return _download_and_extract_zip(url, dest_dir, state_dir, force)
    return _download_plain_file(url, dest_dir / name, state_dir, force)


def retry_failed_downloads(dataset: str) -> None:
    """Retry every URL logged as failed for dataset; successes are removed from the log."""
    state_dir = download_state_dir(dataset)
    failed_file = _failed_file(state_dir)
    if not failed_file.exists():
        logger.info(f"no failed downloads recorded for {dataset}")
        return

    urls = [line.strip() for line in failed_file.read_text().splitlines() if line.strip()]
    logger.info(f"retrying {len(urls)} failed download(s) for {dataset}")
    for url in urls:
        download_file(url, dataset, force=True)


def download_dataset(
    dataset: str,
    folders: tuple[str, ...] = DATASET_FOLDERS,
    force: bool = False,
    start_date: date | None = None,
    end_date: date | None = None,
) -> None:
    """Download every file under wemde/<dataset>/<folder>/ into raw_dataset_dir(dataset).

    Args:
        dataset: Dataset folder, e.g. "caseInputData" or "dispatchSolution/dispatchData".
        folders: Sub-folders to download, e.g. ("current", "previous").
        force: Download files that are already there.
        start_date: Only files dated on or after this; None for no limit.
        end_date: Only files dated on or before this; None for no limit.
    """
    dest_dir = raw_dataset_dir(dataset)
    state_dir = download_state_dir(dataset)

    for folder in folders:
        url = f"{ROOT_URL}{dataset}/{folder}/"
        logger.info(f"listing {url}")
        _, files = list_directory(url)
        if start_date or end_date:
            files = [(name, file_url) for name, file_url in files if _in_date_range(name, start_date, end_date)]

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
    parser.add_argument("--from-date", help="only download files dated on/after this date (YYYY-MM-DD)")
    parser.add_argument("--to-date", help="only download files dated on/before this date (YYYY-MM-DD)")
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
        start_date = date.fromisoformat(args.from_date) if args.from_date else None
        end_date = date.fromisoformat(args.to_date) if args.to_date else None
        for dataset in args.datasets:
            download_dataset(dataset, force=args.force, start_date=start_date, end_date=end_date)
