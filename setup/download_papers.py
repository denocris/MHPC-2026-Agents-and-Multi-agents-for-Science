"""
setup/download_papers.py
========================

Download a small corpus of arXiv papers on the 2D Ising model for the
Day 1 RAG module (Module 1.2).

Runs two searches:

  * "2D Ising model Monte Carlo"
  * "Ising model critical exponents"

and keeps the union of the top results, de-duplicating by arXiv ID.
PDFs are saved under ``data/papers/`` and a sidecar JSON metadata file
is written to ``data/papers/metadata.json``.

Usage
-----
From the repo root, with the course venv activated::

    python setup/download_papers.py                 # default: 10 papers
    python setup/download_papers.py --max-papers 8
    python setup/download_papers.py --out-dir some/other/dir

The script is idempotent: already-downloaded PDFs are skipped. A final
summary reports how many papers are now in the corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any


# --- Search queries covering both the simulation and the theory angle ---
#
# We use arXiv's field-qualified syntax (``ti:``/``abs:``) so that the
# query terms have to appear in the title or abstract, not just somewhere
# in the full text. Without this, a generic query like
# "2D Ising model Monte Carlo" matches hundreds of thousands of papers
# that merely *mention* Monte Carlo.
DEFAULT_QUERIES: tuple[str, ...] = (
    'ti:"Ising model" AND (abs:"Monte Carlo" OR abs:"Metropolis" OR abs:"Wolff")',
    'abs:"Ising model" AND abs:"critical exponents"',
)
DEFAULT_MAX_PAPERS = 10
DEFAULT_OUT_DIR = Path("data/papers")


log = logging.getLogger("download_papers")


def _slugify(text: str, maxlen: int = 80) -> str:
    """Turn a paper title into a filesystem-safe slug for the PDF filename."""
    text = re.sub(r"\s+", "_", text.strip())
    text = re.sub(r"[^A-Za-z0-9_\-\.]", "", text)
    return text[:maxlen] or "paper"


def _short_id(result: Any) -> str:
    """Return the short arXiv id (e.g. '1707.05392') without version suffix."""
    sid = result.get_short_id()
    # strip trailing "vN" version tag so duplicates dedupe correctly
    return re.sub(r"v\d+$", "", sid)


def _id_to_filename_safe(sid: str) -> str:
    """Make an arXiv id safe to embed in a single filename.

    Old-style ids like ``physics/0306182`` would otherwise be treated as
    a subdirectory by ``arxiv.Result.download_pdf``.
    """
    return sid.replace("/", "_")


def download_papers(
    queries: tuple[str, ...] = DEFAULT_QUERIES,
    max_papers: int = DEFAULT_MAX_PAPERS,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> list[dict[str, Any]]:
    """Run the searches, download PDFs, write metadata, return the metadata list."""
    try:
        import arxiv
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The 'arxiv' package is not installed. "
            "Run `pip install -r setup/requirements.txt` first."
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Writing PDFs to %s/", out_dir)

    # Pull roughly enough candidates to reach max_papers after de-duping.
    per_query = max(3, max_papers)
    client = arxiv.Client(page_size=per_query, delay_seconds=3.0, num_retries=3)

    seen: dict[str, Any] = {}
    # Run every query, dedupe by arXiv id, and keep going until we have
    # ``max_papers`` distinct papers. This way a slightly weaker first
    # query still gets topped up by the second query.
    for q in queries:
        if len(seen) >= max_papers:
            break
        log.info("Searching arXiv: %r", q)
        search = arxiv.Search(
            query=q,
            max_results=per_query,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for r in client.results(search):
            sid = _short_id(r)
            if sid in seen:
                continue
            seen[sid] = r
            if len(seen) >= max_papers:
                break

    if not seen:
        raise SystemExit("arXiv search returned no results — network issue?")

    log.info("Selected %d unique papers. Downloading PDFs...", len(seen))

    metadata: list[dict[str, Any]] = []
    for sid, r in seen.items():
        slug = _slugify(r.title)
        safe_id = _id_to_filename_safe(sid)
        pdf_name = f"{safe_id}_{slug}.pdf"
        pdf_path = out_dir / pdf_name

        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            log.info("  [skip] %s (already present)", pdf_name)
        else:
            try:
                # arxiv.Result.download_pdf writes to dirpath/filename
                r.download_pdf(dirpath=str(out_dir), filename=pdf_name)
                log.info("  [ok]   %s", pdf_name)
                time.sleep(1.0)  # be polite to arXiv
            except Exception as exc:  # noqa: BLE001
                log.warning("  [fail] %s: %s", pdf_name, exc)
                continue

        metadata.append(
            {
                "arxiv_id": sid,
                "title": r.title.strip().replace("\n", " "),
                "authors": [a.name for a in r.authors],
                "abstract": r.summary.strip().replace("\n", " "),
                "url": r.entry_id,
                "published": r.published.date().isoformat() if r.published else None,
                "pdf_path": str(pdf_path.relative_to(out_dir.parent.parent))
                if out_dir.is_absolute() is False
                else str(pdf_path),
            }
        )

    meta_path = out_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    log.info("Wrote metadata for %d papers to %s", len(metadata), meta_path)
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--max-papers",
        type=int,
        default=DEFAULT_MAX_PAPERS,
        help="Max number of papers to keep (after de-duping). Default: 10.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory for PDFs + metadata.json. Default: data/papers",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=None,
        help="Override the default search queries (can be given multiple times).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="More logging."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    queries = tuple(args.query) if args.query else DEFAULT_QUERIES
    metadata = download_papers(
        queries=queries, max_papers=args.max_papers, out_dir=args.out_dir
    )

    # Final report
    pdfs = sorted(p for p in args.out_dir.glob("*.pdf"))
    print()
    print(f"Corpus summary ({args.out_dir}):")
    print(f"  PDFs on disk : {len(pdfs)}")
    print(f"  metadata.json: {len(metadata)} entries")
    for entry in metadata:
        print(f"   - {entry['arxiv_id']}: {entry['title'][:80]}")

    if len(pdfs) < 5:
        log.error(
            "Only %d PDFs downloaded — the course RAG module needs >= 5. "
            "Check your network connection and retry.",
            len(pdfs),
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
