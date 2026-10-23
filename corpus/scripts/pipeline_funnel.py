"""Count papers at every stage of the SciCo dataset pipeline.

Produces a funnel table showing how many papers survive each stage,
so the gap between (ACL Anthology metadata) → (PDFs downloaded) →
(per-parser output) → (Gemini affiliations) → (ROR-linked) →
(+CSO topics) → (+bibmeta) is no longer a mystery. Drop the resulting
table into the reproducibility appendix.

Counts are by distinct paper-id (stem of filename, with known
suffixes stripped). The script reports per-stage size, drop from the
previous stage, and the set-difference paper IDs (sample of 10) so
debugging the drops is one grep away.

Usage (on the cluster):
    python scripts/pipeline_funnel.py \\
        --anthology-dir /share/moritz.staudinger/acl-anthology \\
        --pdfs-dir      /share/moritz.staudinger/acl_pdfs \\
        --grobid-dir    /home/$USER/data/acl_pdfs_parsed \\
        --pymu-dir      /share/moritz.staudinger/acl_data/acl_pdfs_pymu_llm \\
        --hybrid-dir    /home/$USER/data/acl_pdfs_hybrid \\
        --gemini-dir    /share/moritz.staudinger/acl_data/gemini_affiliations \\
        --ror-dir       /share/moritz.staudinger/acl_data/ror_parsed_affiliations_90 \\
        --topics-dir    /share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics \\
        --final-dir     /share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta \\
        --out reports/pipeline_funnel.md

All dir args are optional — any stage you omit (or that doesn't exist
on this host) is skipped and noted in the report.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger("pipeline_funnel")

KNOWN_SUFFIXES = {
    ".pdf", ".json", ".jsonl", ".xml", ".tei", ".tei.xml",
    ".md", ".txt", ".html", ".xhtml", ".gz",
}


def _strip_id(name: str) -> str:
    """Strip known suffixes (possibly compound, e.g. .tei.xml.gz) from a
    filename, preserving dots inside the anthology ID itself."""
    while True:
        p = Path(name)
        sfx = p.suffix.lower()
        if sfx and sfx in KNOWN_SUFFIXES:
            name = p.stem
        else:
            return name


def _ids_in_dir(d: Optional[Path], glob: str = "*") -> Optional[Set[str]]:
    """Return set of paper IDs in directory (None if dir is None or
    doesn't exist)."""
    if d is None or not d.is_dir():
        return None
    return {_strip_id(fp.name) for fp in d.glob(glob) if fp.is_file()}


def _anthology_ids(anthology_dir: Path) -> Optional[Set[str]]:
    """Walk the anthology XML metadata and emit paper IDs.

    The anthology repo layout:
        anthology-data/  or  data/xml/<venue>.xml
    Each <paper id="N"> nested under <volume id="V">, and the full
    ID is built per the anthology's known naming convention. We try
    BOTH lxml and Python's stdlib, falling back to a regex sweep if
    neither produces anything plausible (defensive — anthology layout
    has evolved across years)."""
    if not anthology_dir.is_dir():
        return None
    ids: Set[str] = set()
    # Prefer the import-anthology bindings if available
    try:
        from acl_anthology import Anthology  # type: ignore
        anth = Anthology(anthology_dir)
        for paper in anth.papers():
            ids.add(str(paper.full_id))
        if ids:
            return ids
    except Exception as e:
        logger.debug("acl_anthology import failed: %s", e)

    # Fallback: regex over XML files (best-effort)
    import re
    xml_files = list(anthology_dir.rglob("*.xml"))
    paper_re = re.compile(r'<paper\s+id="([^"]+)"', re.IGNORECASE)
    volume_re = re.compile(r'<volume\s+id="([^"]+)"', re.IGNORECASE)
    for fp in xml_files:
        try:
            txt = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        venue_year = fp.stem
        cur_vol: Optional[str] = None
        # Lightweight stream over the file
        for m in re.finditer(r'<(volume|paper)\b[^>]*id="([^"]+)"',
                             txt, re.IGNORECASE):
            tag, val = m.group(1).lower(), m.group(2)
            if tag == "volume":
                cur_vol = val
            elif tag == "paper" and cur_vol:
                ids.add(f"{venue_year}-{cur_vol}.{val}")
    return ids or None


def _format_row(stage: str, n: int, prev_n: Optional[int]) -> str:
    if prev_n is None or prev_n == 0:
        delta = "—"
        pct = "—"
    else:
        d = n - prev_n
        delta = f"{d:+,}"
        pct = f"{100 * n / prev_n:.1f}%"
    return f"| {stage} | {n:,} | {delta} | {pct} |"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anthology-dir", type=Path, default=None,
                    help="Path to acl-anthology git clone (for stage 1).")
    ap.add_argument("--pdfs-dir", type=Path, default=None)
    ap.add_argument("--grobid-dir", type=Path, default=None)
    ap.add_argument("--pymu-dir", type=Path, default=None)
    ap.add_argument("--hybrid-dir", type=Path, default=None)
    ap.add_argument("--gemini-dir", type=Path, default=None)
    ap.add_argument("--ror-dir", type=Path, default=None)
    ap.add_argument("--topics-dir", type=Path, default=None)
    ap.add_argument("--final-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("reports/pipeline_funnel.md"))
    ap.add_argument("--sample-diffs", type=int, default=10,
                    help="How many sample paper IDs to show for each "
                         "stage-to-stage drop.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s")

    stages: List[Tuple[str, Optional[Set[str]], str]] = []  # (label, ids, dir)

    if args.anthology_dir:
        logger.info("scanning anthology metadata at %s", args.anthology_dir)
        ids = _anthology_ids(args.anthology_dir)
        stages.append(("1. ACL Anthology metadata", ids,
                       str(args.anthology_dir)))

    for label, d, glob in [
        ("2. PDFs downloaded",          args.pdfs_dir,   "*.pdf"),
        ("3a. GROBID parses (TEI)",     args.grobid_dir, "*.xml"),
        ("3b. PyMu+LLM parses (MD)",    args.pymu_dir,   "*.md"),
        ("3c. Hybrid OCR parses (JSONL)", args.hybrid_dir, "*.jsonl"),
        ("4. Gemini affiliation JSONs", args.gemini_dir, "*.json"),
        ("5. ROR-linked (>=0.90)",      args.ror_dir,    "*.json"),
        ("6. + CSO topics",             args.topics_dir, "*.json"),
        ("7. + bibmeta (FINAL)",        args.final_dir,  "*.json"),
    ]:
        if d is None:
            continue
        logger.info("counting %s ...", d)
        ids = _ids_in_dir(d, glob)
        stages.append((label, ids, str(d)))

    if not stages:
        raise SystemExit("no stage dirs provided — pass at least one --*-dir")

    # ------------ build the markdown report ------------
    lines: List[str] = []
    lines.append("# Pipeline funnel\n")
    lines.append("Per-stage paper counts for the SciCo dataset pipeline.")
    lines.append("Each row is the set of distinct paper IDs present at "
                 "that stage. The `Δ from previous` and `survival %` "
                 "columns track how much each stage drops vs the row "
                 "above it.\n")
    lines.append("| Stage | Papers | Δ from previous | Survival % |")
    lines.append("|---|---:|---:|---:|")

    prev_n: Optional[int] = None
    prev_ids: Optional[Set[str]] = None
    drop_samples: List[Tuple[str, str, List[str]]] = []
    for label, ids, where in stages:
        if ids is None:
            lines.append(f"| {label} | _dir not found_ | — | — |")
            continue
        n = len(ids)
        lines.append(_format_row(label, n, prev_n))
        if prev_ids is not None and prev_n is not None:
            dropped = sorted(prev_ids - ids)
            if dropped:
                drop_samples.append(
                    (label, f"({len(dropped):,} IDs dropped)",
                     dropped[:args.sample_diffs]))
        prev_ids, prev_n = ids, n

    lines.append("")
    lines.append("## Per-stage paths\n")
    for label, ids, where in stages:
        present = "" if ids is not None else "  _(dir not found on this host)_"
        lines.append(f"- **{label}** — `{where}`{present}")

    if drop_samples:
        lines.append("\n## Sample paper IDs dropped at each stage\n")
        lines.append("Useful for grepping the source-of-drop. The list "
                     "is set-difference against the immediately previous "
                     "stage, capped at 10 examples.\n")
        for label, summary, sample in drop_samples:
            lines.append(f"### Dropped going into **{label}** {summary}\n")
            for pid in sample:
                lines.append(f"- `{pid}`")
            lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", args.out)

    # Also print to stdout for the slurm log
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
