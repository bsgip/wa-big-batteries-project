"""Build a self-contained HTML copy of a report notebook, for GitHub Pages.

Manual step, run by hand whenever a report is ready to (re)publish - no CI,
no watching for pushes. Markdown image links in the report notebooks (e.g.
`![](../../data/plots/2025-08-25_day.png)`) point at data/, which is
gitignored, so a plain `jupyter nbconvert --to html` would keep those same
relative links and break once the output is served from docs/ on its own.
This embeds every local image as a base64 data URI first, so the resulting
HTML has no dependency on data/ existing at all - copy/paste-safe, and safe
to commit to docs/ for Pages to serve.

Usage:
    uv run python -m reports.publish [notebook_path] [output_path]

Defaults to report-v2.ipynb -> docs/index.html.
"""

import base64
import re
import sys
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

from tools.paths import repo_docs_dir

DEFAULT_NOTEBOOK = Path(__file__).parent / "report-v2.ipynb"
DEFAULT_OUTPUT = repo_docs_dir / "index.html"

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".gif": "image/gif",
}

_IMG_LINK = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def _embed_one_image(match: re.Match, notebook_dir: Path) -> str:
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
    """Rewrite every markdown-cell image link that points at a local file
    into a base64 data URI, in place. http(s) links and existing data URIs
    are left untouched."""
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = _IMG_LINK.sub(lambda m: _embed_one_image(m, notebook_dir), cell.source)


# nbconvert's default template renders figures at their native pixel size
# (e.g. ~2400px wide for a 12in/200dpi plot) with no container width limit,
# so on a normal screen they overflow both the window and each other. This
# caps images to their container and gives the notebook body a readable
# max-width, the same way the report reads in JupyterLab's own preview.
_EXTRA_CSS = """
<style>
  .jp-Notebook { max-width: 900px; margin: 0 auto; padding: 2rem 1.5rem; }
  .jp-RenderedMarkdown img { max-width: 100%; height: auto; display: block; margin: 1rem auto; }
</style>
"""


def build_html(notebook_path: Path, output_path: Path) -> None:
    nb = nbformat.read(notebook_path, as_version=4)
    embed_local_images(nb, notebook_path.parent)

    exporter = HTMLExporter(exclude_input=True, exclude_input_prompt=True, exclude_output_prompt=True)
    body, _ = exporter.from_notebook_node(nb)
    body = body.replace("</head>", f"{_EXTRA_CSS}</head>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(body)


def main():
    notebook_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NOTEBOOK
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    build_html(notebook_path, output_path)
    print(f"wrote {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
