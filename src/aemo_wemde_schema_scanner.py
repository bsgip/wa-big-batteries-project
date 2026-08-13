#!/usr/bin/env python3
"""
Builds a field library for AEMO WA WEMDE market data.

Approach:
  * JSON is parsed as a STREAM (ijson), so a 500 MB file uses ~constant memory
    and never needs to be written to disk. The parser's path prefixes ARE the
    schema, so we just collect them.
  * XML is streamed the same way via ElementTree.iterparse, recording element
    paths, attributes and text-bearing leaves.
  * Array elements are collapsed to $.data.x[].y and unioned across every item
    the parser actually reaches. Reading stops early once MAX_JSON_BYTES or
    PLATEAU_EVENTS is hit, so on very large files this is a SAMPLE of the
    document, not a guaranteed-complete schema. The read_status column records
    exactly why a read stopped:
        ok          - document read to the end, schema is complete
        plateau     - stopped after PLATEAU_EVENTS with no new path
        byte_cap    - stopped at MAX_JSON_BYTES
        parse_error - malformed (or truncated) document
    Anything other than "ok" means treat that file's paths as a lower bound.
  * One exemplar per distinct FILENAME PATTERN per folder, not per folder,
    because a folder can hold several different report types.
  * Output is a tidy CSV you can query/diff, written incrementally so a crash
    or Ctrl-C keeps everything discovered so far.

Install:
    pip install requests beautifulsoup4 ijson

Run:
    python aemo_wemde_schema_scanner.py
"""

import csv
import io
import os
import re
import time
import zipfile
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import ijson

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT_URL = "https://data.wa.aemo.com.au/public/market-data/wemde/"
OUT_CSV = "wemde_fields.csv"

MAX_DEPTH = 6
MAX_FOLDERS = 300
SLEEP = 0.4

# Skip on the first pass:
#   previous/ - huge dated archives whose schema normally matches current/
#   schema/   - confirmed empty on this server, so not worth a request each
SKIP_FOLDERS = {"previous", "schema"}

# Stop reading a JSON/XML stream after this many bytes. Set to None for no cap.
MAX_JSON_BYTES = 80 * 1024 * 1024

# Stop early once we've gone this many parse events without discovering a new
# path. Repetitive files plateau quickly. Set to None to disable.
#
# NOTE: on the big dispatch files this is what fires, not the byte cap - they
# yield ~175 paths from ~12 MB of a much larger document. Before trusting those
# counts as complete, re-run one file with PLATEAU_EVENTS = None and confirm the
# path count doesn't climb.
PLATEAU_EVENTS = 400_000

# Don't download zips bigger than this (they must land on disk to be read).
MAX_ZIP_BYTES = 250 * 1024 * 1024

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; WEMDE-SchemaScanner/1.0)"}
session = requests.Session()
session.headers.update(HEADERS)

SCALAR = {"string", "number", "integer", "double", "boolean", "null"}

FIELDNAMES = [
    "dataset", "folder_url", "source_file", "member_file",
    "kind", "path", "types", "occurrences", "position",
    "read_status", "example",
]

visited = set()
errors = []
writer = None       # set up in main()


# ---------------------------------------------------------------------------
# Incremental CSV output
# ---------------------------------------------------------------------------

class FieldWriter:
    """Writes field rows to CSV as they are discovered, not all at the end."""

    def __init__(self, path):
        self.fh = open(path, "w", newline="", encoding="utf-8")
        self.csv = csv.DictWriter(self.fh, fieldnames=FIELDNAMES)
        self.csv.writeheader()
        self.fh.flush()
        self.count = 0
        self.datasets = set()

    def write(self, row):
        self.csv.writerow(row)
        self.count += 1
        self.datasets.add(row["dataset"])

    def flush(self):
        self.fh.flush()

    def close(self):
        self.fh.close()


# ---------------------------------------------------------------------------
# Directory listing
# ---------------------------------------------------------------------------

def list_directory(url):
    """Return (subdir_entries, file_entries) for an IIS listing page."""
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    subdirs, files = [], []
    for a in soup.find_all("a", href=True):
        href, text = a["href"], a.get_text(strip=True)
        if text == "[To Parent Directory]" or href in ("../", "/"):
            continue
        full = urljoin(url, href)
        if not full.startswith(ROOT_URL):
            continue
        name = text or href.rstrip("/").split("/")[-1]
        (subdirs if href.endswith("/") else files).append((name, full))
    return subdirs, files


# Month names must be bounded by non-letters, otherwise "Mar" eats the start of
# "market-advisories.csv" and "Aug" eats "AugustReport".
MONTHS = (r"(?<![A-Za-z])"
          r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
          r"(?:uary|ruary|ch|il|e|y|ust|tember|ober|ember)?"
          r"(?![A-Za-z])")


def pattern_of(filename):
    """Collapse dates/sequence numbers so files of the same report type group.

    ReferenceDispatchSolution_202608060800.json -> ReferenceDispatchSolution_#.json
    AemoProcured_TW_28_Jun_2026.json            -> AemoProcured_TW_#_#_#.json
    """
    s = re.sub(MONTHS, "#", filename, flags=re.IGNORECASE)
    return re.sub(r"\d+", "#", s)


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

class ByteBudget(io.RawIOBase):
    """Wrap a file-like object and pretend EOF once `budget` bytes are read."""

    def __init__(self, raw, budget):
        self.raw, self.budget, self.used = raw, budget, 0
        self.truncated = False

    def readinto(self, buf):
        # RawIOBase derives read() from readinto(); the ijson C backend calls
        # readinto() directly, so this is the one that must exist.
        if self.budget is not None and self.used >= self.budget:
            self.truncated = True
            return 0
        want = len(buf)
        if self.budget is not None:
            want = min(want, self.budget - self.used)
        chunk = self.raw.read(want)
        self.used += len(chunk)
        buf[:len(chunk)] = chunk
        return len(chunk)

    def readable(self):
        return True


def read_status(byte_capped, plateaued, parse_error=False):
    """Single label describing why a stream stopped. Plateau wins: it breaks
    out before EOF, so byte_capped will still be False when it fires."""
    if plateaued:
        return "plateau"
    if byte_capped:
        return "byte_cap"
    if parse_error:
        return "parse_error"
    return "ok"


def _bump(paths, p, t, value=None):
    rec = paths.get(p)
    if rec is None:
        rec = paths[p] = {"types": set(), "count": 0, "example": None}
        new = True
    else:
        new = False
    rec["types"].add(t)
    rec["count"] += 1
    if rec["example"] is None and value not in (None, ""):
        rec["example"] = str(value)[:60]
    return new


# ---------------------------------------------------------------------------
# JSON schema extraction
# ---------------------------------------------------------------------------

def norm_path(prefix):
    """ijson prefix -> JSONPath-ish string.

    ijson uses the literal token 'item' for array elements, so a genuine object
    key named 'item' is indistinguishable and will render as []. Not observed in
    WEMDE data, but worth knowing.
    """
    if prefix == "":
        return "$"
    out = "$"
    for token in prefix.split("."):
        if token == "item":
            out += "[]"          # leading-token case handles top-level arrays
        else:
            out += "." + token
    return out


def json_paths(fileobj):
    """Stream a JSON document. Returns ({path: info}, plateaued)."""
    paths = {}
    since_new = 0
    plateaued = False
    try:
        for prefix, event, value in ijson.parse(fileobj):
            if event in ("map_key", "end_map", "end_array"):
                continue
            t = {"start_map": "object", "start_array": "array"}.get(event, event)
            new = _bump(paths, norm_path(prefix), t,
                        value if t in SCALAR else None)
            since_new = 0 if new else since_new + 1

            if PLATEAU_EVENTS and since_new > PLATEAU_EVENTS:
                plateaued = True
                break
    except ijson.common.IncompleteJSONError:
        pass  # expected when we hit the byte budget
    return paths, plateaued


# ---------------------------------------------------------------------------
# XML schema extraction
# ---------------------------------------------------------------------------

def _localname(tag):
    """Strip any {namespace} prefix ElementTree prepends."""
    if isinstance(tag, str) and tag.startswith("{"):
        return tag.split("}", 1)[1]
    return str(tag)


def xml_paths(fileobj):
    """Stream an XML document. Returns ({path: info}, plateaued, parse_error).

    Paths look like /root/child, /root/child/@attr, /root/child/#text so they
    stay visually distinct from the JSON $. paths in the same CSV.
    """
    paths, stack = {}, []
    since_new = 0
    plateaued = parse_error = False
    try:
        for event, elem in ET.iterparse(fileobj, events=("start", "end")):
            if event == "start":
                stack.append(_localname(elem.tag))
                p = "/" + "/".join(stack)
                new = _bump(paths, p, "element")
                # attributes are populated by the time 'start' fires
                for k, v in elem.attrib.items():
                    new |= _bump(paths, f"{p}/@{_localname(k)}", "attribute", v)
            else:
                if not stack:
                    continue
                p = "/" + "/".join(stack)
                text = (elem.text or "").strip()
                new = _bump(paths, p + "/#text", "text", text) if text else False
                elem.clear()          # release children, keep memory flat
                stack.pop()

            since_new = 0 if new else since_new + 1
            if PLATEAU_EVENTS and since_new > PLATEAU_EVENTS:
                plateaued = True
                break
    except ET.ParseError:
        parse_error = True            # includes the byte-budget cut
    return paths, plateaued, parse_error


# ---------------------------------------------------------------------------
# Row recording
# ---------------------------------------------------------------------------

def record_paths(paths, dataset, folder, source, member="",
                 kind="json", status="ok"):
    for p in sorted(paths):
        r = paths[p]
        writer.write({
            "dataset": dataset,
            "folder_url": folder,
            "source_file": source,
            "member_file": member,
            "kind": kind,
            "path": p,
            "types": "|".join(sorted(r["types"])),
            "occurrences": r["count"],
            "position": "",
            "read_status": status,
            "example": r["example"] or "",
        })


def record_csv_header(header, dataset, folder, source, member=""):
    for i, col in enumerate(header):
        writer.write({
            "dataset": dataset,
            "folder_url": folder,
            "source_file": source,
            "member_file": member,
            "kind": "csv",
            "path": col.strip(),
            "types": "",
            "occurrences": "",
            "position": i,          # column index, kept separate from occurrences
            "read_status": "ok",
            "example": "",
        })


# ---------------------------------------------------------------------------
# Per-file handling
# ---------------------------------------------------------------------------

def handle_json_url(url, dataset, folder, name):
    with session.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = True
        budget = ByteBudget(resp.raw, MAX_JSON_BYTES)
        paths, plateaued = json_paths(budget)
    status = read_status(budget.truncated, plateaued)
    record_paths(paths, dataset, folder, name, kind="json", status=status)
    print(f"      {len(paths)} JSON paths, {budget.used / 1e6:.1f} MB read [{status}]")


def handle_xml_url(url, dataset, folder, name):
    with session.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        resp.raw.decode_content = True
        budget = ByteBudget(resp.raw, MAX_JSON_BYTES)
        paths, plateaued, perr = xml_paths(budget)
    status = read_status(budget.truncated, plateaued, perr)
    record_paths(paths, dataset, folder, name, kind="xml", status=status)
    print(f"      {len(paths)} XML paths, {budget.used / 1e6:.1f} MB read [{status}]")


def handle_csv_url(url, dataset, folder, name):
    with session.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        chunk = next(resp.iter_content(1 << 20), b"")     # b"" if body is empty
    lines = chunk.decode("utf-8", "replace").splitlines()
    header = next(csv.reader([lines[0]])) if lines else []
    record_csv_header(header, dataset, folder, name)
    print(f"      {len(header)} CSV columns")


def handle_zip_url(url, dataset, folder, name):
    try:
        h = session.head(url, timeout=30, allow_redirects=True)
        size = int(h.headers.get("Content-Length") or 0)
        if size > MAX_ZIP_BYTES:
            print(f"      skipped zip ({size / 1e6:.0f} MB over cap)")
            return
    except requests.RequestException:
        pass

    # mkstemp + try/finally so the temp file is removed even if the DOWNLOAD
    # fails, not just if the unzip fails.
    fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    try:
        with os.fdopen(fd, "wb") as tmp:
            with session.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_content(1 << 20):
                    tmp.write(chunk)

        with zipfile.ZipFile(tmp_path) as zf:
            for member in zf.namelist():
                low = member.lower()
                if low.endswith(".json"):
                    with zf.open(member) as fh:
                        budget = ByteBudget(fh, MAX_JSON_BYTES)
                        paths, plateaued = json_paths(budget)
                    status = read_status(budget.truncated, plateaued)
                    record_paths(paths, dataset, folder, name, member,
                                 kind="json", status=status)
                    print(f"      {member}: {len(paths)} JSON paths [{status}]")
                elif low.endswith(".xml"):
                    with zf.open(member) as fh:
                        budget = ByteBudget(fh, MAX_JSON_BYTES)
                        paths, plateaued, perr = xml_paths(budget)
                    status = read_status(budget.truncated, plateaued, perr)
                    record_paths(paths, dataset, folder, name, member,
                                 kind="xml", status=status)
                    print(f"      {member}: {len(paths)} XML paths [{status}]")
                elif low.endswith(".csv"):
                    with zf.open(member) as fh:
                        first = fh.readline().decode("utf-8", "replace")
                    header = next(csv.reader([first])) if first else []
                    record_csv_header(header, dataset, folder, name, member)
                    print(f"      {member}: {len(header)} CSV columns")
                else:
                    print(f"      {member}: skipped (not csv/json/xml)")
    except zipfile.BadZipFile:
        errors.append(f"bad zip: {url}")
        print("      ERROR: not a valid zip")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def handle_file(name, url, dataset, folder):
    print(f"    -> {name}")
    try:
        low = name.lower()
        if low.endswith(".json"):
            handle_json_url(url, dataset, folder, name)
        elif low.endswith(".xml"):
            handle_xml_url(url, dataset, folder, name)
        elif low.endswith(".csv"):
            handle_csv_url(url, dataset, folder, name)
        elif low.endswith(".zip"):
            handle_zip_url(url, dataset, folder, name)
        else:
            print("      (unsupported extension)")
    except KeyboardInterrupt:
        raise                      # don't swallow Ctrl-C
    except Exception as e:
        errors.append(f"{url}: {e}")
        print(f"      ERROR: {e}")


# ---------------------------------------------------------------------------
# Walk
#
# Still depth-first. Fine for wemde, which is shallow and mostly
# current/previous/schema. If you ever point ROOT_URL at market-data/ root,
# where some branches hold thousands of dated subfolders, switch to a queue so
# MAX_FOLDERS doesn't get spent entirely inside the first branch.
# ---------------------------------------------------------------------------

def walk(url, dataset=None, depth=0):
    if url in visited or len(visited) >= MAX_FOLDERS or depth > MAX_DEPTH:
        return
    visited.add(url)

    print(f"{'  ' * depth}[DIR] {url}")
    try:
        subdirs, files = list_directory(url)
    except requests.RequestException as e:
        errors.append(f"listing {url}: {e}")
        print(f"{'  ' * depth}  ERROR: {e}")
        return
    time.sleep(SLEEP)

    if dataset is None and depth == 1:
        dataset = url.rstrip("/").split("/")[-1]

    # one exemplar per filename pattern
    groups = defaultdict(list)
    for name, furl in files:
        groups[pattern_of(name)].append((name, furl))

    for pat, members in sorted(groups.items()):
        name, furl = members[0]
        print(f"{'  ' * depth}  pattern {pat} ({len(members)} file(s))")
        handle_file(name, furl, dataset or "?", url)

    writer.flush()      # checkpoint after every folder

    for name, surl in subdirs:
        if name.strip("/").lower() in SKIP_FOLDERS:
            print(f"{'  ' * (depth + 1)}[skip] {name}")
            continue
        walk(surl, dataset, depth + 1)


def main():
    global writer
    writer = FieldWriter(OUT_CSV)
    interrupted = False
    try:
        walk(ROOT_URL)
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted - keeping everything written so far.")
    finally:
        writer.flush()
        print("\n--- Summary ---")
        print(f"Folders visited: {len(visited)}")
        print(f"Fields recorded: {writer.count}")
        print(f"Datasets:        {len(writer.datasets)}")
        print(f"Errors:          {len(errors)}")
        for e in errors:
            print(f"  - {e}")
        print(f"\nWrote {OUT_CSV}{' (partial)' if interrupted else ''}")
        writer.close()


if __name__ == "__main__":
    main()