"""Build a self-contained HTML copy of a report notebook (images embedded) for GitHub Pages.
The notebook's first cell becomes the page header - see pop_header().
The page opens in reading mode (one scrolling page); presentation mode shows one
section/subsection per screen - see split_into_pages().

Usage:
    uv run python -m reports.publish [notebook_path] [output_path]

Defaults to report-v{report_version}.ipynb -> docs/index.html.
"""

import base64
import datetime as dt
import html
import re
import sys
from pathlib import Path
from urllib.parse import quote

import mistune
import nbformat
from bs4 import BeautifulSoup, NavigableString
from nbconvert import HTMLExporter

from tools.paths import local_docs_dir

report_version = 6
DETAILS_HEADING = "Deliverables"  # small heading over the front page's "- Label: value" strip
FAVICON_PATH: Path | None = (
    local_docs_dir / "favicon.png"
)  # browser-tab icon (.png/.svg); None = the built-in battery icon
DEFAULT_NOTEBOOK = Path(__file__).parent / f"report-v{report_version}.ipynb"
DEFAULT_OUTPUT = local_docs_dir / "index.html"

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".gif": "image/gif",
}

_IMG_LINK = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------


def _embed_one_image(match: re.Match, notebook_dir: Path) -> str:
    """Replacement for one markdown image match: a base64 data URI if it's a local file, else unchanged."""
    alt, src = match.group(1), match.group(2)
    if src.startswith(("http://", "https://", "data:")):
        return match.group(0)

    image_path = (notebook_dir / src).resolve()
    mime = _MIME_TYPES.get(image_path.suffix.lower())
    if mime is None or not image_path.exists():
        raise FileNotFoundError(f"can't embed image referenced as {src!r} - not found at {image_path}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"![{alt}](data:{mime};base64,{encoded})"


def embed_local_images(nb, notebook_dir: Path) -> None:
    """Replace local image links in markdown cells with base64 data URIs, in place.

    Args:
        nb: Notebook.
        notebook_dir: Folder the image links are relative to.
    """
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = _IMG_LINK.sub(lambda m: _embed_one_image(m, notebook_dir), cell.source)


# --------------------------------------------------------------------------
# Escaped dollars
# --------------------------------------------------------------------------

# A dollar sign MathJax leaves alone: the entity hides it from nbconvert's maths parser, and the
# span (MathJax skips the tex2jax_ignore class) stops MathJax pairing it with another dollar.
_LITERAL_DOLLAR = '<span class="tex2jax_ignore">&#36;</span>'


_LONE_CDOT = re.compile(r"\$\s*\\cdot\s*\$")


def replace_lone_cdots(nb) -> None:
    """Turn a maths span holding only \\cdot ("$\\cdot$") into a plain "·" in markdown cells, in place.

    Used as a separator (e.g. "# D2 $\\cdot$ Batteries Statistics Pack"), it would otherwise show as raw
    "$\\cdot$" wherever the heading is copied as plain text: sidebar, top bar and next/previous buttons.
    Formulas that contain \\cdot alongside other maths are left alone.
    """
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = _LONE_CDOT.sub("·", cell.source)


def protect_escaped_dollars(nb) -> None:
    """Make \\$ in markdown cells show as a plain dollar sign, in place.

    nbconvert drops the backslash, so without this MathJax reads the text between two
    prices ("about \\$115 ... about \\$239") as a formula.
    """
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = cell.source.replace("\\$", _LITERAL_DOLLAR)


# --------------------------------------------------------------------------
# Header (title cell)
# --------------------------------------------------------------------------

_HEADING_LINE = re.compile(r"^#{1,6}\s+(.+?)\s*#*$")
_TAG_SEP = re.compile(r"\s*·\s*")  # "·" between tags ("$\cdot$" is already "·" - see replace_lone_cdots)
_QUOTE_LINE = re.compile(r"^>\s?(.*)$")
_BULLET_LINE = re.compile(r"^[-*+]\s+(.+)$")


def _inline_md(text: str) -> str:
    """Render one line of markdown (bold, links, code...) without the <p>."""
    out = mistune.html(text).strip()
    return out[3:-4] if out.startswith("<p>") and out.endswith("</p>") else out


def pop_header(nb, fallback_title: str) -> dict:
    """Read the page header from the notebook's first cell and remove that cell.

    The cell may hold, in this order (all but the title optional):

        AR-PST · Stage 6 Topic 4-2                          <- tags (one pill each, split at "·")
        # Report title                                      <- title
        > One-sentence summary of what the report answers.  <- lede (description, below the divider)
        The Australian National University                  <- byline (any other line; the first is bold)
        - Data: Dec 2024 – Aug 2026                         <- details strip ("Label: value", label optional)

    A first cell without a heading is left alone.

    Args:
        nb: Notebook.
        fallback_title: Title to use if neither the cell nor the notebook metadata has one.

    Returns:
        {"tags", "title", "lede", "byline", "details"}, where details is a list of (label, value).
    """
    header = {"tags": [], "title": nb.metadata.get("title") or fallback_title, "lede": "", "byline": [], "details": []}
    if not nb.cells or nb.cells[0].cell_type != "markdown":
        return header

    tags, lede, byline, details, title = [], [], [], [], None
    for line in (ln.strip() for ln in nb.cells[0].source.splitlines()):
        if not line:
            continue
        if title is None and (m := _HEADING_LINE.match(line)):
            title = m.group(1)
        elif title is None:
            tags.extend(t for t in _TAG_SEP.split(line) if t)
        elif m := _QUOTE_LINE.match(line):
            lede.append(m.group(1))
        elif m := _BULLET_LINE.match(line):
            label, sep, value = m.group(1).partition(": ")
            details.append((label, value) if sep else ("", label))
        else:
            byline.append(line)

    if title is None:
        print(f"warning: the first cell has no heading, so it isn't the page header - title is {header['title']!r}")
        return header
    nb.cells.pop(0)
    return {
        "tags": [_inline_md(t) for t in tags],
        "title": title.replace("**", "").replace("__", "").strip(),
        "lede": _inline_md(" ".join(lede)) if lede else "",
        "byline": [_inline_md(b) for b in byline],
        "details": [(_inline_md(label) if label else "", _inline_md(value)) for label, value in details],
    }


# --------------------------------------------------------------------------
# Body post-processing: figures, tables, headings -> contents tree
# --------------------------------------------------------------------------

# How many heading levels get automatic numbers: 1 = sections only
# ("Section 1"), 2 = subsections too (1.1, 1.2 ...).
NUMBER_DEPTH = 1

_UNNUMBERED = re.compile(r"\s*\{(?:-|\.unnumbered)\}\s*$")
_NUM_PREFIX = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+")
_NUMERIC = re.compile(r"^[-+−]?[\d,]*\.?\d+\s*%?$")


def _is_blank(node) -> bool:
    """Whether node is whitespace or a <br>."""
    return (isinstance(node, NavigableString) and not node.strip()) or getattr(node, "name", None) == "br"


def _heading_label(h) -> str:
    """Text of a heading, without its anchor link."""
    return "".join(
        s for s in h.find_all(string=True) if not (s.parent.name == "a" and "anchor-link" in s.parent.get("class", []))
    ).strip()


def enhance_body(body: str) -> tuple[str, list[dict]]:
    """Restyle nbconvert's cell HTML and collect the headings.

    Args:
        body: HTML from nbconvert's basic template.

    Returns:
        (body_html, contents_tree), where contents_tree is the nested headings for the sidebar.
    """
    soup = BeautifulSoup(body, "html.parser")

    # Code cells with no output (inputs are excluded) render as empty boxes.
    for cell in soup.select("div.code_cell"):
        if not cell.select(".output_area"):
            cell.decompose()

    # A markdown paragraph holding only images becomes a figure card.
    for p in soup.select(".text_cell_render p"):
        kids = [c for c in p.contents if not _is_blank(c)]
        if kids and all(getattr(c, "name", None) == "img" for c in kids):
            for c in list(p.contents):
                if _is_blank(c):
                    c.extract()
            p.name = "figure"
            p["class"] = ["fig"]
    for img in soup.select(".fig img"):
        img["loading"] = "lazy"

    # Tables: scrollable card, right-align columns that are all numbers.
    for table in soup.find_all("table"):
        table.wrap(soup.new_tag("div", attrs={"class": "table-wrap"}))
        rows = [r.find_all(["td", "th"]) for r in table.select("tbody tr")]
        if not rows:
            continue
        for j in range(max(len(r) for r in rows)):
            col = [r[j] for r in rows if j < len(r)]
            if all(_NUMERIC.match(c.get_text(strip=True)) for c in col):
                for c in col:
                    c["class"] = c.get("class", []) + ["n"]
                header = table.select("thead tr")
                if header and j < len(header[-1].find_all("th")):
                    th = header[-1].find_all("th")[j]
                    th["class"] = th.get("class", []) + ["n"]

    # Headings -> flat list, then nested by level.
    # Numbering is automatic: the first NUMBER_DEPTH heading levels are
    # counted in order (1, 2, 3 / 1.1, 1.2 ...). Numbers typed by hand at
    # those levels are dropped, and "{-}" at the end opts a heading out.
    # If the body has any "#" headings, those are parts (D2, D3 ...): they
    # are not numbered, section numbers restart in each part, sections before
    # the first part (e.g. an executive summary) are not numbered, and each
    # part becomes a collapsible block (see wrap_parts).
    found = soup.find_all(["h1", "h2", "h3", "h4"])
    part_headings = [h for h in found if h.name == "h1"]
    if part_headings:
        top_level = 2
    else:
        top_level = min((int(h.name[1]) for h in found), default=2)
    counters = [0] * (NUMBER_DEPTH + 1)
    flat = []
    in_part = False
    for i, h in enumerate(found):
        if not h.get("id"):
            h["id"] = f"section-{i}"
        level = int(h.name[1])

        if level == 1 and part_headings:
            h["class"] = h.get("class", []) + ["part-title"]
            in_part = True
            counters = [0] * (NUMBER_DEPTH + 1)
            flat.append({"id": h["id"], "level": level, "num": None, "label": _heading_label(h), "children": []})
            continue

        depth = level - top_level + 1
        h["class"] = h.get("class", []) + [f"lvl-{depth}"]
        texts = [
            s for s in h.find_all(string=True) if s.parent.name != "a" or "anchor-link" not in s.parent.get("class", [])
        ]

        unnumbered = False
        if texts and (m := _UNNUMBERED.search(str(texts[-1]))):
            texts[-1].replace_with(str(texts[-1])[: m.start()])
            unnumbered = True
        if part_headings and not in_part:
            unnumbered = True

        num = None
        if depth <= NUMBER_DEPTH:
            if texts and _NUM_PREFIX.match(str(texts[0])):  # hand-typed "1." / "1.2"
                texts[0].replace_with(_NUM_PREFIX.sub("", str(texts[0]), count=1))
        label = _heading_label(h)

        if depth <= NUMBER_DEPTH and not unnumbered:
            counters[depth] += 1
            counters[depth + 1 :] = [0] * (NUMBER_DEPTH - depth)
            num = ".".join(str(c) for c in counters[1 : depth + 1])
            tag = soup.new_tag("span", attrs={"class": "h-num" if depth == 1 else "h-subnum"})
            tag.string = f"Section {num}" if depth == 1 else num
            h.insert(0, tag)

        flat.append({"id": h["id"], "level": level, "num": num, "label": label, "children": []})

    tree: list[dict] = []
    stack: list[dict] = []
    for node in flat:
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()
        (stack[-1]["children"] if stack else tree).append(node)
        stack.append(node)

    wrap_parts(soup, part_headings)
    split_into_pages(soup)
    return str(soup), tree


def wrap_parts(soup, part_headings: list) -> None:
    """Wrap each part (a "#" heading and the cells up to the next one) in a collapsible <details>, in place.

    The heading moves into the <summary>, so clicking it folds the whole part.

    Args:
        soup: Body HTML, whose top-level children are the notebook cells.
        part_headings: The part <h1> tags, in order.
    """
    part_cells = [h.find_parent("div", class_="cell") for h in part_headings]
    next_cells = part_cells[1:] + [None]
    for h, cell, next_cell in zip(part_headings, part_cells, next_cells[: len(part_cells)], strict=True):
        members = []
        node = cell
        while node is not None and node is not next_cell:
            members.append(node)
            node = node.next_sibling

        details = soup.new_tag("details", attrs={"class": "part", "open": ""})
        summary = soup.new_tag("summary")
        chevron = BeautifulSoup(_ICON_CHEVRON, "html.parser")
        chevron.svg["class"] = "part-chev"
        summary.append(chevron)
        summary.append(h.extract())
        details.append(summary)
        part_body = soup.new_tag("div", attrs={"class": "part-body"})
        details.append(part_body)

        cell.insert_before(details)
        for member in members:
            part_body.append(member.extract())

        # The heading cell may hold nothing but the heading.
        if not cell.get_text(strip=True) and not cell.find("img"):
            cell.decompose()


_HEADING_TAGS = ["h1", "h2", "h3", "h4"]


def _leading_headings(cell) -> tuple[list, bool]:
    """The headings a cell starts with, and whether the cell holds nothing but those headings."""
    render = cell.select_one(".text_cell_render")
    if render is None:  # code cell output
        return [], False
    kids = [c for c in render.children if not _is_blank(c)]
    leading = []
    for kid in kids:
        if getattr(kid, "name", None) not in _HEADING_TAGS:
            break
        leading.append(kid)

    later = render.find_all(_HEADING_TAGS)[len(leading) :]
    for h in later:
        print(
            f"warning: heading {_heading_label(h)!r} isn't at the start of its cell, so in presentation mode "
            "it doesn't start a new page - move it into a cell of its own to fix"
        )
    return leading, len(leading) == len(kids)


def split_into_pages(soup) -> None:
    """Group the cells into <section class="page"> blocks for presentation mode, in place.

    A heading starts a new page, except that headings with nothing under them (e.g. a section
    title straight before its first subsection) stay on the same page as what follows. Each page
    gets data-heading = the id of its last starting heading, which names it in the sidebar and
    the next/previous buttons. Reading mode shows all pages, so it looks the same as before.

    Args:
        soup: Body HTML, after wrap_parts.
    """
    for container in [soup] + soup.select(".part-body"):
        cells = [c for c in container.children if getattr(c, "name", None) == "div" and "cell" in c.get("class", [])]
        page = None
        page_has_content = False
        for cell in cells:
            leading, heading_only = _leading_headings(cell)
            if page is None or (leading and page_has_content):
                page = soup.new_tag("section", attrs={"class": "page"})
                cell.insert_before(page)
                page_has_content = False
            page.append(cell.extract())
            if leading:
                page["data-heading"] = leading[-1]["id"]
            page_has_content = page_has_content or not heading_only


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

_ICON_SIDEBAR = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2.5"/>'
    '<path d="M9 4v16"/></svg>'
)
_ICON_CHEVRON = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>'
)
_ICON_UP = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5"/><path d="m5 12 7-7 7 7"/></svg>'
)


# Browser-tab icon: a battery about 70% charged, in the light-mode accent blue.
_ICON_FAVICON = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect x="2" y="8" width="24" height="16" rx="3.5" fill="none" stroke="#2350C4" stroke-width="2.5"/>'
    '<rect x="27" y="13" width="3" height="6" rx="1" fill="#2350C4"/>'
    '<rect x="6" y="12" width="11" height="8" rx="1.5" fill="#2350C4"/></svg>'
)


def _favicon_href() -> str:
    """Data URI for the browser-tab icon: FAVICON_PATH if set, else the built-in battery icon."""
    if FAVICON_PATH is None:
        return "data:image/svg+xml," + quote(_ICON_FAVICON)
    mime = _MIME_TYPES.get(FAVICON_PATH.suffix.lower())
    if mime is None:
        raise ValueError(f"FAVICON_PATH must be one of {sorted(_MIME_TYPES)}, got {FAVICON_PATH.name!r}")
    return f"data:{mime};base64,{base64.b64encode(FAVICON_PATH.read_bytes()).decode('ascii')}"


def _render_toc(nodes: list[dict], depth: int = 1) -> str:
    """Nested <ul> table of contents for heading nodes."""
    items = []
    for n in nodes:
        part = " part" if n["level"] == 1 else ""
        num = f'<span class="num">{html.escape(n["num"])}</span>' if n["num"] else ""
        attrs = (
            f'href="#{html.escape(n["id"])}" data-target="{html.escape(n["id"])}" '
            f'data-label="{html.escape(n["label"])}" data-num="{html.escape(n["num"] or "")}"'
        )
        fold = ""
        children = ""
        if n["children"]:
            fold = f'<button class="toc-fold" type="button" aria-expanded="true" aria-label="Fold subsections">{_ICON_CHEVRON}</button>'
            children = _render_toc(n["children"], depth + 1)
        items.append(
            f'<li class="lvl-{depth}{part}"><div class="toc-row"><a {attrs}>{num}<span class="label">{html.escape(n["label"])}</span></a>'
            f"{fold}</div>{children}</li>"
        )
    return f"<ul>{''.join(items)}</ul>"


_MATHJAX = """
<script>window.MathJax = { tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']], processEscapes: true } };</script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-chtml-full.js"></script>
"""

_HAS_MATH = re.compile(r"\$\$|\\\(|\\\[|\$[^\s$][^$\n]*\$")

_CSS = r"""
:root {
  --bg: #F2F5F6; --surface: #FFFFFF; --ink: #17232A; --muted: #566871; --line: #D3DCE0; --code-bg: #E9EEF0;
  --accent: #2350C4; --accent-soft: #E2E9FB; --fig-bg: #FFFFFF;
  --f-display: "Source Serif 4", Georgia, "Times New Roman", serif;
  --f-body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
  --side-w: 290px; --bar-h: 52px;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #1C252A; --surface: #242F35; --ink: #E6EDF0; --muted: #A2B2BA; --line: #38474E; --code-bg: #2B373E;
    --accent: #86A6FF; --accent-soft: #2C3B62; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #1C252A; --surface: #242F35; --ink: #E6EDF0; --muted: #A2B2BA; --line: #38474E; --code-bg: #2B373E;
  --accent: #86A6FF; --accent-soft: #2C3B62; color-scheme: dark;
}
* { box-sizing: border-box; }
html { scroll-padding-top: calc(var(--bar-h) + 20px); }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--f-body); font-size: 16.5px; line-height: 1.6; -webkit-font-smoothing: antialiased; }
a { color: var(--accent); }
button { font: inherit; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* ---------- layout ---------- */
.layout { min-height: 100vh; }
.sidebar { background: var(--surface); border-right: 1px solid var(--line); z-index: 30; overflow: hidden; }
.sidebar-inner { width: var(--side-w); height: 100%; display: flex; flex-direction: column; }
.main { min-width: 0; }

@media (min-width: 961px) {
  .layout { display: grid; grid-template-columns: var(--side-w) minmax(0, 1fr); transition: grid-template-columns .22s ease; }
  .sidebar { position: sticky; top: 0; height: 100vh; transition: visibility 0s linear 0s; }
  html.toc-closed .layout { grid-template-columns: 0px minmax(0, 1fr); }
  html.toc-closed .sidebar { visibility: hidden; border-right-color: transparent; transition: visibility 0s linear .22s; }
  .scrim { display: none; }
}
@media (max-width: 960px) {
  .sidebar { position: fixed; inset: 0 auto 0 0; width: min(86vw, var(--side-w)); transform: translateX(-100%); visibility: hidden;
             transition: transform .22s ease, visibility 0s linear .22s; box-shadow: 0 10px 40px rgb(0 0 0 / .18); }
  .sidebar-inner { width: 100%; }
  html.toc-open .sidebar { transform: none; visibility: visible; transition: transform .22s ease; }
  .scrim { position: fixed; inset: 0; z-index: 25; background: rgb(8 12 14 / .4); opacity: 0; pointer-events: none; transition: opacity .2s; }
  html.toc-open .scrim { opacity: 1; pointer-events: auto; }
}

/* ---------- sidebar ---------- */
.side-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; padding: 18px 14px 14px 20px; border-bottom: 1px solid var(--line); }
.side-head .eyebrow { margin-bottom: 4px; }
.side-title { display: block; font-family: var(--f-display); font-weight: 700; font-size: 15px; line-height: 1.25; text-wrap: balance; color: var(--ink); text-decoration: none; }
.side-title:hover { color: var(--accent); }
.toc { flex: 1; overflow-y: auto; padding: 12px 10px 28px; overscroll-behavior: contain; }
.toc ul { list-style: none; margin: 0; padding: 0; }
.toc ul ul { margin: 2px 0 8px 17px; padding-left: 8px; border-left: 1px solid var(--line); }
.toc li.lvl-1 + li.lvl-1 { margin-top: 4px; }
.toc-row { display: flex; align-items: flex-start; gap: 2px; }
.toc a { flex: 1; min-width: 0; display: flex; gap: 8px; padding: 6px 10px; border-radius: 7px; color: var(--muted); text-decoration: none; font-size: 14px; line-height: 1.35; }
.toc a:hover { background: var(--code-bg); color: var(--ink); }
.toc li.lvl-1 > .toc-row > a { color: var(--ink); font-weight: 700; }
.toc a .num { font-family: var(--f-mono); font-size: 12px; font-weight: 600; color: var(--accent); min-width: 1.2em; padding-top: 1px; }
.toc li > .toc-row > a.in-path { color: var(--accent); }
.toc li > .toc-row > a.active { background: var(--accent-soft); color: var(--accent); }
.toc-fold { flex: none; width: 28px; height: 30px; display: grid; place-items: center; border: 0; background: none; color: var(--muted); border-radius: 6px; cursor: pointer; }
.toc-fold:hover { background: var(--code-bg); color: var(--ink); }
.toc-fold svg { transition: transform .15s; }
.toc li.folded > ul { display: none; }
.toc li.part + li.part { margin-top: 18px; }
.toc li.part > .toc-row > a { font-family: var(--f-mono); font-size: 12px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--accent); }
.toc li.part > ul { margin: 2px 0 0; padding-left: 0; border-left: 0; }
.toc li.part > ul > li > .toc-row > a { color: var(--ink); font-weight: 700; }
.toc li.folded > .toc-row .toc-fold svg { transform: rotate(-90deg); }

/* ---------- top bar ---------- */
.topbar { position: sticky; top: 0; z-index: 20; height: var(--bar-h); display: flex; align-items: center; gap: 12px; padding: 0 16px;
          background: color-mix(in srgb, var(--bg) 86%, transparent); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border-bottom: 1px solid var(--line); }
.icon-btn { flex: none; width: 34px; height: 34px; display: grid; place-items: center; border: 1px solid var(--line); background: var(--surface); color: var(--ink); border-radius: 8px; cursor: pointer; text-decoration: none; }
.icon-btn:hover { border-color: var(--muted); }
.crumb { flex: 1; min-width: 0; font-size: 14px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.crumb .sep { margin: 0 6px; opacity: .6; }
.crumb b { color: var(--ink); font-weight: 700; }
.crumb a { color: inherit; text-decoration: none; }
.crumb a:hover, .crumb a:hover b { color: var(--accent); }
.crumb .num { font-family: var(--f-mono); font-size: 12px; color: var(--accent); margin-right: 6px; }
.progress { position: absolute; left: 0; right: 0; bottom: -1px; height: 2px; background: var(--accent); transform-origin: left; transform: scaleX(0); }

/* ---------- content ---------- */
.content { max-width: 920px; margin: 0 auto; padding: 0 28px 96px; }
.hero { padding: 56px 0 8px; display: grid; gap: 14px; }
.eyebrow { font-family: var(--f-mono); font-size: 11.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.hero h1 { font-family: var(--f-display); font-size: clamp(32px, 5vw, 48px); font-weight: 700; line-height: 1.08; letter-spacing: -.02em; margin: 0 0 14px; text-wrap: pretty; }
/* front page, laid out like a cover slide: tags / title / byline, then a divider, then description + details strip */
.tags { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 10px; }
.tag { font-size: 14px; font-weight: 600; color: var(--accent); background: var(--accent-soft); border-radius: 999px; padding: 4px 12px; }
.byline { display: grid; gap: 4px; font-size: 15.5px; line-height: 1.45; color: var(--muted); }
.byline p { margin: 0; }
.byline p:first-child { font-size: 17.5px; font-weight: 700; color: var(--ink); }
/* divider like the D2/D3 part titles: hairline with a short accent at its start */
.about { position: relative; display: grid; gap: 24px; margin-top: 26px; padding-top: 26px; border-top: 1px solid var(--line); }
.about::before { content: ""; position: absolute; left: 0; top: -1px; width: 64px; height: 2px; background: var(--accent); }
.about:not(:has(*)) { display: none; }
.lede { margin: 0; font-size: 17.5px; line-height: 1.55; color: color-mix(in srgb, var(--ink) 85%, var(--bg)); }
.details-title { margin-bottom: 12px; font-size: 11.5px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.details { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px 28px; margin: 0; }
.details dt { margin-bottom: 2px; font-size: 14px; font-weight: 700; color: var(--accent); }
.details dd { margin: 0; font-size: 16px; line-height: 1.4; color: var(--ink); }
.hero .updated { margin: 10px 0 0; font-size: 13px; color: var(--muted); }
.tags a, .details a { color: inherit; }

.report h1, .report h2, .report h3, .report h4 { font-family: var(--f-display); line-height: 1.2; margin: 0; text-wrap: balance; }
.report h2 { font-size: clamp(25px, 3.4vw, 31px); letter-spacing: -.015em; margin: 64px 0 18px; padding-top: 28px; border-top: 1px solid var(--line); }
.report h2 .h-num { display: block; margin-bottom: 8px; font-family: var(--f-mono); font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }
.report .h-subnum { margin-right: .5em; font-family: var(--f-mono); font-size: .78em; font-weight: 600; color: var(--accent); }
.report h3 { font-size: 21px; margin: 44px 0 12px; }
.report h4 { font-size: 17.5px; margin: 34px 0 10px; }

/* parts (D2, D3 ...): each "#" heading folds everything up to the next one */
details.part { margin: 88px 0 0; }
/* a chapter divider: title over a hairline rule with a short accent at its start, fold arrow on the right */
details.part > summary { position: relative; list-style: none; display: flex; align-items: center; gap: 14px; cursor: pointer;
                         padding: 0 2px 14px 0; border-bottom: 1px solid var(--line); }
details.part > summary::after { content: ""; position: absolute; left: 0; bottom: -1px; width: 64px; height: 2px; background: var(--accent); }
details.part > summary::-webkit-details-marker { display: none; }
details.part > summary h1 { flex: 1; min-width: 0; }
details.part > summary .part-chev { order: 1; flex: none; width: 18px; height: 18px; color: var(--muted); transition: transform .15s, color .15s; }
details.part > summary:hover .part-chev { color: var(--accent); }
details.part:not([open]) > summary .part-chev { transform: rotate(90deg); }
.report h1.part-title { font-size: clamp(28px, 4vw, 36px); letter-spacing: -.02em; }
.part-body > .page:first-child > .cell:first-child .text_cell_render > h2:first-child { margin-top: 36px; padding-top: 0; border-top: 0; }
.report .anchor-link { margin-left: .35em; color: var(--muted); text-decoration: none; font-weight: 400; font-size: .75em; opacity: 0; transition: opacity .15s; }
.report :is(h1, h2, h3, h4):hover .anchor-link { opacity: 1; }
.report p, .report ul, .report ol { margin: 0 0 14px; }
.report li + li { margin-top: 4px; }
.report strong { font-weight: 700; }
.report a { overflow-wrap: anywhere; }
.report hr { border: 0; margin: 24px 0; }
.report blockquote { margin: 18px 0; padding: 10px 16px; border-left: 3px solid var(--accent); background: var(--surface); border-radius: 0 8px 8px 0; color: var(--muted); }
.report code { font-family: var(--f-mono); font-size: .86em; background: var(--code-bg); padding: 1px 5px; border-radius: 4px; }
.report pre { font-family: var(--f-mono); font-size: 13px; line-height: 1.5; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; overflow-x: auto; }
.report pre code { background: none; padding: 0; }

.fig { margin: 22px 0 28px; padding: 10px; display: grid; gap: 10px; background: var(--fig-bg); border: 1px solid var(--line); border-radius: 12px; }
.fig img, .output_png img, .output_jpeg img { display: block; max-width: 100%; height: auto; margin: 0 auto; cursor: zoom-in; }
.output_png, .output_jpeg, .output_svg { margin: 22px 0 28px; padding: 10px; background: var(--fig-bg); border: 1px solid var(--line); border-radius: 12px; }

.table-wrap { overflow-x: auto; margin: 18px 0 26px; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }
.report table { border-collapse: collapse; width: 100%; font-size: 14.5px; font-variant-numeric: tabular-nums; }
.report th, .report td { padding: 9px 14px; text-align: left; border-bottom: 1px solid var(--line); white-space: nowrap; }
.report th { font-size: 12.5px; font-weight: 700; color: var(--muted); background: var(--code-bg); }
.report tbody tr:last-child td { border-bottom: 0; }
.report tbody tr:hover td { background: color-mix(in srgb, var(--accent-soft) 45%, transparent); }
.report .n { text-align: right; }

.foot { margin-top: 80px; padding-top: 18px; border-top: 1px solid var(--line); font-family: var(--f-mono); font-size: 12px; color: var(--muted); }

/* ---------- presentation mode: one section/subsection per screen (html.paged) ---------- */
.view-switch { flex: none; display: flex; padding: 2px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }
.view-switch button { border: 0; background: none; color: var(--muted); font-size: 13px; padding: 4px 10px; border-radius: 6px; cursor: pointer; }
.view-switch button:hover { color: var(--ink); }
.view-switch button[aria-pressed="true"] { background: var(--accent-soft); color: var(--accent); font-weight: 600; }
.page-count, .pager { display: none; }
html.paged .page-count { display: block; flex: none; font-family: var(--f-mono); font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
html.paged :is(.hero, .report .page):not(.is-current) { display: none; }
/* the content fills the screen, so the pager sits at the bottom even on a short page */
html.paged .content { display: flex; flex-direction: column; min-height: calc(100vh - var(--bar-h)); min-height: calc(100dvh - var(--bar-h)); padding-bottom: 16px; }
html.paged .hero { flex: 1; align-content: center; }
html.paged .report { padding-bottom: 40px; }
/* each new page fades in (skipped if the reader's system asks for reduced motion) */
@keyframes page-fade { from { opacity: 0; } }
@media (prefers-reduced-motion: no-preference) { html.paged .content.page-fade { animation: page-fade .3s ease-out; } }
html.paged details.part { margin-top: 0; }
html.paged details.part.starts-here { margin-top: 40px; }
html.paged details.part:not(.holds-current) { display: none; }
html.paged details.part:not(.starts-here) > summary { display: none; }
html.paged details.part > summary { cursor: default; }
html.paged details.part > summary .part-chev { display: none; }
html.paged .page > .cell:first-child .text_cell_render > :is(h2, h3, h4):first-child { margin-top: 48px; padding-top: 0; border-top: 0; }
html.paged .foot { display: none; }
html.paged .pager { display: flex; justify-content: space-between; gap: 16px; margin-top: auto; padding-top: 10px; border-top: 1px solid var(--line); }
.pager-btn { display: flex; align-items: baseline; gap: 8px; min-width: 0; max-width: 48%; padding: 4px 8px; background: none; border: 0; border-radius: 6px; color: var(--muted); font-size: 13.5px; cursor: pointer; }
.pager-btn:hover { color: var(--accent); background: var(--code-bg); }
.pager-btn:disabled { visibility: hidden; }
.pager-btn .dir { flex: none; }
.pager-btn .t { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

/* ---------- image zoom ---------- */
.lightbox { position: fixed; inset: 0; z-index: 60; display: none; overflow: auto; background: rgb(8 12 14 / .9); cursor: zoom-out; }
.lightbox.open { display: flex; }
.lightbox img { margin: auto; max-width: 96vw; max-height: 94vh; background: #fff; border-radius: 6px; cursor: zoom-in; }
.lightbox.actual img { max-width: none; max-height: none; cursor: zoom-out; }

@media (max-width: 640px) {
  body { font-size: 16px; }
  .content { padding: 0 16px 72px; }
  .hero { padding-top: 36px; }
  .topbar { padding: 0 12px; }
  .crumb .parent { display: none; }
  .fig { padding: 6px; }
}
@media print {
  .sidebar, .topbar, .scrim, .lightbox { display: none !important; }
  .layout { display: block; }
  .report h2 { break-before: page; }
  details.part > summary .part-chev { display: none; }
}
"""

_JS = r"""
(function () {
  var root = document.documentElement;
  var mobile = window.matchMedia('(max-width: 960px)');
  var KEY = 'report-toc-closed';
  var toggles = document.querySelectorAll('[data-toc-toggle]');

  function isOpen() { return mobile.matches ? root.classList.contains('toc-open') : !root.classList.contains('toc-closed'); }
  function sync() { var o = isOpen(); toggles.forEach(function (b) { b.setAttribute('aria-expanded', String(o)); }); }
  function setOpen(open) {
    if (mobile.matches) { root.classList.toggle('toc-open', open); }
    else {
      root.classList.toggle('toc-closed', !open);
      try { localStorage.setItem(KEY, open ? '0' : '1'); } catch (e) {}
    }
    sync();
  }
  toggles.forEach(function (b) { b.addEventListener('click', function () { setOpen(!isOpen()); }); });
  document.querySelector('.scrim').addEventListener('click', function () { setOpen(false); });
  mobile.addEventListener('change', function () { root.classList.remove('toc-open'); sync(); });
  sync();

  // fold / unfold subsections
  function setFolded(li, folded) {
    li.classList.toggle('folded', folded);
    var b = li.querySelector(':scope > .toc-row > .toc-fold');
    if (!b) return;
    b.setAttribute('aria-expanded', String(!folded));
    b.setAttribute('aria-label', folded ? 'Unfold subsections' : 'Fold subsections');
  }
  // The <details> of a part's sidebar entry, or null for an ordinary section.
  function partFor(li) {
    if (!li.classList.contains('part')) return null;
    var h = document.getElementById(li.querySelector(':scope > .toc-row > a').dataset.target);
    return h && h.closest('details.part');
  }
  document.querySelectorAll('.toc-fold').forEach(function (b) {
    b.addEventListener('click', function () {
      var li = b.closest('li');
      var part = partFor(li);
      // in presentation mode, closing the part would hide the page on screen, so only the sidebar folds
      if (part && !paged()) part.open = !part.open;  // its toggle event folds the sidebar entry
      else setFolded(li, !li.classList.contains('folded'));
    });
  });
  // a part opened or closed in the page (or from the sidebar) -> fold its sidebar entry to match
  document.querySelectorAll('details.part').forEach(function (d) {
    var h = d.querySelector(':scope > summary h1[id]');
    var a = h && document.querySelector('.toc a[data-target="' + CSS.escape(h.id) + '"]');
    if (!a) return;
    d.addEventListener('toggle', function () { setFolded(a.closest('li'), !d.open); });
  });

  // scroll spy + breadcrumb + progress
  var toc = document.querySelector('.toc');
  var crumb = document.querySelector('.crumb');
  var progress = document.querySelector('.progress');
  var defaultCrumb = crumb.textContent;
  var links = Array.prototype.slice.call(document.querySelectorAll('.toc a[data-target]'));
  var items = links.map(function (a) { return { a: a, el: document.getElementById(a.dataset.target) }; })
                   .filter(function (it) { return it.el; });
  var current;

  // a link into a collapsed part opens that part first
  function openPartFor(el) {
    var part = el && el.closest('details.part');
    if (part && !part.open) part.open = true;
  }
  links.forEach(function (a) {
    a.addEventListener('click', function (e) {
      var el = document.getElementById(a.dataset.target);
      if (paged()) {
        e.preventDefault();
        var i = pageOf(el);
        if (i >= 0 && i !== pageIdx) showPage(i);
      } else openPartFor(el);
      if (mobile.matches) setOpen(false);
    });
  });
  // Some heading ids hold %-escapes themselves (e.g. an en dash), so try the hash as written and decoded.
  function hashTarget() {
    if (!location.hash) return null;
    var raw = location.hash.slice(1), decoded = raw;
    try { decoded = decodeURIComponent(raw); } catch (e) {}
    return document.getElementById(raw) || document.getElementById(decoded);
  }
  window.addEventListener('hashchange', function () {
    var el = hashTarget();
    if (paged()) { var i = pageOf(el); if (i >= 0 && i !== pageIdx) showPage(i); }
    else openPartFor(el);
  });
  if (location.hash) openPartFor(hashTarget());

  function renderCrumb(path) {
    crumb.textContent = '';
    if (!path.length) { crumb.textContent = defaultCrumb; return; }
    var last = path.length - 1;
    path.forEach(function (a, i) {
      if (i) { var s = document.createElement('span'); s.className = 'sep parent'; s.textContent = '›'; crumb.appendChild(s); }
      // each step links to its section, the same as clicking it in the sidebar
      var link = document.createElement('a');
      link.href = a.getAttribute('href');
      if (i !== last) link.className = 'parent';
      link.addEventListener('click', function (e) { e.preventDefault(); a.click(); });
      if (a.dataset.num) { var n = document.createElement('span'); n.className = 'num'; n.textContent = a.dataset.num; link.appendChild(n); }
      var t = document.createElement(i === last ? 'b' : 'span');
      t.textContent = a.dataset.label;
      link.appendChild(t);
      crumb.appendChild(link);
    });
  }

  function update() {
    var act = null;
    if (paged()) {
      // the page's own heading; the progress bar is set by showPage
      var id = pages[pageIdx].dataset.heading;
      for (var k = 0; k < items.length; k++) if (items[k].el.id === id) act = items[k];
    } else {
      var doc = document.documentElement;
      var max = doc.scrollHeight - doc.clientHeight;
      var p = max > 0 ? Math.min(1, doc.scrollTop / max) : 0;
      progress.style.transform = 'scaleX(' + p + ')';

      var line = (parseFloat(getComputedStyle(root).getPropertyValue('--bar-h')) || 52) + 32;
      for (var i = 0; i < items.length; i++) {
        if (!items[i].el.getClientRects().length) continue;  // inside a collapsed part
        if (items[i].el.getBoundingClientRect().top - line <= 0) act = items[i]; else break;
      }
      if (p > 0.995 && items.length) act = items[items.length - 1];
    }
    if (act === current) return;
    current = act;

    links.forEach(function (a) { a.classList.remove('active', 'in-path'); a.removeAttribute('aria-current'); });
    if (!act) { renderCrumb([]); return; }
    act.a.classList.add('active');
    act.a.setAttribute('aria-current', 'location');
    var path = [act.a];
    var li = act.a.closest('li').parentElement.closest('li');
    while (li) {
      var pa = li.querySelector(':scope > .toc-row > a');
      pa.classList.add('in-path');
      path.unshift(pa);
      li = li.parentElement.closest('li');
    }
    renderCrumb(path);

    var r = act.a.getBoundingClientRect(), tr = toc.getBoundingClientRect();
    if (r.top < tr.top + 8 || r.bottom > tr.bottom - 8) toc.scrollTop += r.top - tr.top - tr.height / 3;
  }
  var ticking = false;
  window.addEventListener('scroll', function () {
    if (!ticking) { ticking = true; requestAnimationFrame(function () { update(); ticking = false; }); }
  }, { passive: true });
  window.addEventListener('resize', update);
  document.addEventListener('toggle', update, true);  // a part opened or closed

  // click a figure to view it larger; click again for actual size
  var box = document.querySelector('.lightbox');
  var boxImg = box.querySelector('img');
  document.querySelectorAll('.fig img, .output_png img, .output_jpeg img').forEach(function (img) {
    img.addEventListener('click', function () {
      boxImg.src = img.src; boxImg.alt = img.alt || '';
      box.classList.remove('actual'); box.classList.add('open');
    });
  });
  boxImg.addEventListener('click', function (e) { e.stopPropagation(); box.classList.toggle('actual'); });
  box.addEventListener('click', function () { box.classList.remove('open'); });
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Escape') return;
    if (box.classList.contains('open')) box.classList.remove('open');
    else if (mobile.matches && isOpen()) setOpen(false);
  });

  // presentation mode: the cover (hero) then one <section class="page"> at a time
  var VIEW_KEY = 'report-view';
  var hero = document.querySelector('.hero');
  var pages = [hero].concat(Array.prototype.slice.call(document.querySelectorAll('.report .page')));
  var pageIdx = 0;
  var pageCount = document.querySelector('.page-count');
  var content = document.querySelector('.content');
  var prevBtn = document.querySelector('.pager-btn.prev');
  var nextBtn = document.querySelector('.pager-btn.next');
  var viewBtns = document.querySelectorAll('[data-view]');

  function paged() { return root.classList.contains('paged'); }

  // Index of the page holding el (a part heading -> the part's first page), or -1.
  function pageOf(el) {
    if (!el) return -1;
    if (hero.contains(el)) return 0;
    var p = el.closest('.page');
    if (!p) { var part = el.closest('details.part'); p = part && part.querySelector('.page'); }
    return p ? pages.indexOf(p) : -1;
  }

  function pageLabel(i) {
    if (i === 0) return hero.querySelector('h1').textContent;
    var id = pages[i].dataset.heading;
    var a = id && document.querySelector('.toc a[data-target="' + CSS.escape(id) + '"]');
    return a ? a.dataset.label : 'Page ' + (i + 1);
  }

  function setPagerButton(btn, i) {
    var ok = i >= 0 && i < pages.length;
    btn.disabled = !ok;
    btn.querySelector('.t').textContent = ok ? pageLabel(i) : '';
  }

  function showPage(i) {
    pageIdx = i;
    content.classList.remove('page-fade');
    void content.offsetWidth;  // restart the fade-in animation
    content.classList.add('page-fade');
    pages.forEach(function (p, j) { p.classList.toggle('is-current', j === i); });
    document.querySelectorAll('details.part').forEach(function (d) {
      var holds = d.contains(pages[i]);
      d.classList.toggle('holds-current', holds);
      d.classList.toggle('starts-here', holds && d.querySelector('.page') === pages[i]);
      if (holds) d.open = true;
    });
    pageCount.textContent = (i + 1) + ' / ' + pages.length;
    progress.style.transform = 'scaleX(' + (pages.length > 1 ? i / (pages.length - 1) : 1) + ')';
    setPagerButton(prevBtn, i - 1);
    setPagerButton(nextBtn, i + 1);
    // keep the page in the address bar, so a refresh or a shared link opens it
    var id = pages[i].dataset.heading;
    history.replaceState(null, '', id ? '#' + encodeURIComponent(id) : location.pathname + location.search);
    window.scrollTo({ top: 0, behavior: 'instant' });
    update();
  }

  function go(step) {
    var i = pageIdx + step;
    if (i >= 0 && i < pages.length) showPage(i);
  }

  function setView(toPaged, remember) {
    var here = current && current.el;  // keep the reader on the same section
    root.classList.toggle('paged', toPaged);
    viewBtns.forEach(function (b) { b.setAttribute('aria-pressed', String((b.dataset.view === 'pages') === toPaged)); });
    if (remember) { try { localStorage.setItem(VIEW_KEY, toPaged ? 'pages' : 'scroll'); } catch (e) {} }
    current = undefined;
    if (toPaged) { showPage(Math.max(0, pageOf(here))); return; }
    if (here) here.scrollIntoView({ behavior: 'instant' }); else window.scrollTo({ top: 0, behavior: 'instant' });
    update();
  }

  viewBtns.forEach(function (b) {
    b.addEventListener('click', function () { if ((b.dataset.view === 'pages') !== paged()) setView(!paged(), true); });
  });
  // the sidebar title and the up arrow go back to the start: the cover page, or the top in reading mode
  document.querySelectorAll('[data-to-start]').forEach(function (a) {
    a.addEventListener('click', function (e) {
      if (mobile.matches) setOpen(false);
      if (!paged()) return;
      e.preventDefault();
      if (pageIdx !== 0) showPage(0);
    });
  });
  prevBtn.addEventListener('click', function () { go(-1); });
  nextBtn.addEventListener('click', function () { go(1); });
  document.addEventListener('keydown', function (e) {
    if (!paged() || (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft')) return;
    if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || box.classList.contains('open')) return;
    if (e.target.closest && e.target.closest('input, textarea, select, [contenteditable]')) return;
    e.preventDefault();
    go(e.key === 'ArrowRight' ? 1 : -1);
  });
  // a part's title bar would fold away the page on screen
  document.querySelectorAll('details.part > summary').forEach(function (s) {
    s.addEventListener('click', function (e) { if (paged()) e.preventDefault(); });
  });

  // printing always gives the whole report
  var printedPaged = false;
  window.addEventListener('beforeprint', function () {
    printedPaged = paged();
    if (printedPaged) setView(false, false);
    document.querySelectorAll('details.part').forEach(function (d) { d.open = true; });
  });
  window.addEventListener('afterprint', function () { if (printedPaged) setView(true, false); });

  // start: the <head> script already chose the view, to avoid a flash of the other one
  viewBtns.forEach(function (b) { b.setAttribute('aria-pressed', String((b.dataset.view === 'pages') === paged())); });
  if (paged()) showPage(Math.max(0, pageOf(hashTarget())));
  else update();
})();
"""

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%%TITLE%%</title>
<link rel="icon" href="%%FAVICON%%">
<script>try { if (localStorage.getItem('report-toc-closed') === '1') document.documentElement.classList.add('toc-closed'); } catch (e) {}
try { if (localStorage.getItem('report-view') === 'pages') document.documentElement.classList.add('paged'); } catch (e) {}</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>%%CSS%%</style>
%%MATHJAX%%
</head>
<body>
<div class="layout">
  <aside class="sidebar" id="sidebar" aria-label="Contents">
    <div class="sidebar-inner">
      <div class="side-head">
        <div>
          <div class="eyebrow">Contents</div>
          <a class="side-title" href="#top" data-to-start title="Back to the start">%%TITLE%%</a>
        </div>
      </div>
      <nav class="toc">%%TOC%%</nav>
    </div>
  </aside>
  <div class="scrim"></div>
  <div class="main">
    <header class="topbar">
      <button class="icon-btn" type="button" data-toc-toggle aria-controls="sidebar" aria-label="Show or hide contents" title="Contents">%%ICON_SIDEBAR%%</button>
      <div class="crumb" aria-live="polite">%%TITLE%%</div>
      <span class="page-count"></span>
      <div class="view-switch" role="group" aria-label="View">
        <button type="button" data-view="scroll" aria-pressed="true" title="The whole report on one scrolling page">Reading</button>
        <button type="button" data-view="pages" aria-pressed="false" title="One section per page (← → keys to move)">Presentation</button>
      </div>
      <a class="icon-btn" href="#top" data-to-start aria-label="Back to the start" title="Back to the start">%%ICON_UP%%</a>
      <div class="progress"></div>
    </header>
    <main class="content" id="top">
      <header class="hero">
        %%TAGS%%
        <h1>%%TITLE%%</h1>
        %%BYLINE%%
        <div class="about">
          %%LEDE%%
          %%DETAILS%%
        </div>
        <p class="updated">Updated %%DATE%%</p>
      </header>
      <article class="report">
%%BODY%%
      </article>
      <nav class="pager" aria-label="Pages">
        <button class="pager-btn prev" type="button" aria-label="Previous page"><span class="dir">←</span><span class="t"></span></button>
        <button class="pager-btn next" type="button" aria-label="Next page"><span class="t"></span><span class="dir">→</span></button>
      </nav>
      <footer class="foot">Built from %%SOURCE%% on %%DATE%%</footer>
    </main>
  </div>
</div>
<div class="lightbox" role="dialog" aria-label="Figure, enlarged"><img alt=""></div>
<script>%%JS%%</script>
</body>
</html>
"""


def render_page(header: dict, body: str, tree: list[dict], source_name: str, has_math: bool) -> str:
    """Wrap the report body in the full page: header, sidebar contents and styles.

    Args:
        header: From pop_header.
        body: From enhance_body.
        tree: Headings, from enhance_body.
        source_name: Notebook file name, shown in the footer.
        has_math: Whether to load MathJax.

    Returns:
        The page HTML.
    """
    today = dt.date.today()
    date_str = f"{today.day} {today:%b %Y}"
    title = header["title"]
    replacements = {
        "%%CSS%%": _CSS,
        "%%JS%%": _JS,
        "%%MATHJAX%%": _MATHJAX if has_math else "",
        "%%TOC%%": _render_toc(tree),
        "%%ICON_SIDEBAR%%": _ICON_SIDEBAR,
        "%%ICON_UP%%": _ICON_UP,
        "%%FAVICON%%": _favicon_href(),
        "%%TAGS%%": (
            '<div class="tags">' + "".join(f'<span class="tag">{t}</span>' for t in header["tags"]) + "</div>"
            if header["tags"]
            else ""
        ),
        "%%LEDE%%": f'<p class="lede">{header["lede"]}</p>' if header["lede"] else "",
        "%%BYLINE%%": (
            '<div class="byline">' + "".join(f"<p>{b}</p>" for b in header["byline"]) + "</div>"
            if header["byline"]
            else ""
        ),
        "%%DETAILS%%": (
            '<div class="details-block">'
            f'<div class="details-title">{html.escape(DETAILS_HEADING)}</div>'
            '<dl class="details">'
            + "".join(
                f"<div>{f'<dt>{label}</dt>' if label else ''}<dd>{value}</dd></div>"
                for label, value in header["details"]
            )
            + "</dl></div>"
            if header["details"]
            else ""
        ),
        "%%SOURCE%%": html.escape(source_name),
        "%%DATE%%": date_str,
        "%%TITLE%%": html.escape(title),
    }
    page = _PAGE
    for key, value in replacements.items():
        if key != "%%BODY%%":
            page = page.replace(key, value)
    # Body last, so nothing inside the report is mistaken for a placeholder.
    return page.replace("%%BODY%%", body)


def build_html(notebook_path: Path, output_path: Path) -> None:
    """Build the self-contained HTML page for a report notebook.

    Args:
        notebook_path: Report notebook.
        output_path: HTML file to write.
    """
    nb = nbformat.read(notebook_path, as_version=4)
    embed_local_images(nb, notebook_path.parent)
    replace_lone_cdots(nb)
    header = pop_header(nb, fallback_title=notebook_path.stem)
    protect_escaped_dollars(nb)
    header_text = " ".join([*header["tags"], header["lede"], *header["byline"], *(v for _, v in header["details"])])
    has_math = bool(_HAS_MATH.search(header_text)) or any(
        c.cell_type == "markdown" and _HAS_MATH.search(c.source) for c in nb.cells
    )

    exporter = HTMLExporter(
        template_name="basic", exclude_input=True, exclude_input_prompt=True, exclude_output_prompt=True
    )
    body, _ = exporter.from_notebook_node(nb)
    body, tree = enhance_body(body)

    page = render_page(header, body, tree, notebook_path.name, has_math)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")


def main():
    notebook_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NOTEBOOK
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    build_html(notebook_path, output_path)
    print(f"wrote {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
