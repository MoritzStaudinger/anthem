"""Export a CSV ready for manual affiliation annotation.

One row per (paper, author). Each row pre-fills Gemini's extracted
affiliations (institution name, country, ROR, institution type) so the
annotator only needs to mark correct/incorrect and provide corrections
where Gemini got it wrong — instead of extracting from scratch.

The annotation task collapses to verification + correction:
    - `correct` column: Y / N / partial
    - `gold_*` columns: filled in only when `correct` != Y
    - `notes`: free text

Combine with scripts/sample_ground_truth.py to scope to a specific sample
(e.g., 720 papers across 1980-2025).

Usage:
    # Export all annotated papers
    python scripts/export_authors_for_annotation.py \\
        --affiliations-dir /share/.../affiliations \\
        --out annotation_template.csv

    # Restrict to a sampled subset (CSV from sample_ground_truth.py)
    python scripts/export_authors_for_annotation.py \\
        --affiliations-dir /share/.../affiliations \\
        --sample-csv sampled_truth_1980_2025.csv \\
        --out annotation_template.csv

    # Restrict to a year range
    python scripts/export_authors_for_annotation.py \\
        --affiliations-dir /share/.../affiliations \\
        --years 1980-2000 \\
        --out annotation_template_pre2000.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _load_sample_ids(sample_csv: Path) -> Set[str]:
    """Read an anthology_id set from a sample CSV (output of sample_ground_truth.py)."""
    ids: Set[str] = set()
    with sample_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            aid = (row.get("anthology_id") or row.get("paper_id") or "").strip()
            if aid:
                ids.add(aid)
    return ids


def _parse_year_range(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    if not spec:
        return None
    parts = spec.split("-")
    if len(parts) != 2:
        raise ValueError(f"--years must be like '1980-2000', got {spec!r}")
    return int(parts[0]), int(parts[1])


def _iter_papers(
    affiliations_dir: Path,
    keep_ids: Optional[Set[str]],
    year_range: Optional[Tuple[int, int]],
) -> Iterable[Dict[str, Any]]:
    for fp in sorted(affiliations_dir.glob("*.json")):
        if fp.stem.endswith(".enriched"):
            # Prefer the source JSON; the enriched mirror has same author/aff data.
            continue
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("skipping unreadable %s", fp.name)
            continue
        pid = d.get("paper_id") or fp.stem
        if keep_ids is not None and pid not in keep_ids:
            continue
        y = d.get("year")
        if year_range and isinstance(y, int):
            if not (year_range[0] <= y <= year_range[1]):
                continue
        yield d


def _gemini_affiliation_summary(aff: Dict[str, Any]) -> Dict[str, str]:
    """Compact Gemini-extracted view of one affiliation entry."""
    return {
        "clean_institution_name": aff.get("clean_institution_name") or "",
        "raw_affiliation":        aff.get("raw_affiliation") or "",
        "country":                aff.get("country") or "",
        "country_code":           aff.get("country_code") or "",
        "ror_id":                 aff.get("ror_id") or "",
        "institution_type":       aff.get("institution_type") or "",
        "confidence":             str(aff.get("confidence") or ""),
    }


def _flatten_author(
    paper: Dict[str, Any],
    author_idx: int,
    author: Dict[str, Any],
) -> Dict[str, Any]:
    """One CSV row for one (paper, author)."""
    affs = author.get("affiliations") or []
    aff_summaries = [_gemini_affiliation_summary(a) for a in affs]

    # Compose the "system_*" pre-filled columns as semicolon-joined lists
    # so multi-affiliation authors fit in one cell. The annotator can
    # split them or just edit inline.
    sys_insts = " ; ".join(a["clean_institution_name"] for a in aff_summaries if a["clean_institution_name"])
    sys_raws  = " ; ".join(a["raw_affiliation"]        for a in aff_summaries if a["raw_affiliation"])
    sys_ccs   = " ; ".join(a["country_code"]           for a in aff_summaries if a["country_code"])
    sys_ctry  = " ; ".join(a["country"]                for a in aff_summaries if a["country"])
    sys_rors  = " ; ".join(a["ror_id"]                 for a in aff_summaries if a["ror_id"])
    sys_types = " ; ".join(a["institution_type"]       for a in aff_summaries if a["institution_type"])

    return {
        # --- paper-level ---
        "paper_id":   paper.get("paper_id") or "",
        "year":       paper.get("year") or "",
        "venue":      (paper.get("venue") or "")[:90],
        "title":      paper.get("title") or "",
        "url":        paper.get("url") or f"https://aclanthology.org/{paper.get('paper_id','')}/",
        # --- author identity ---
        "author_position": author_idx,
        "author_name":     author.get("name") or "",
        "author_markers":  ",".join(str(m) for m in (author.get("affiliation_markers") or [])),
        # --- Gemini-extracted (PRE-FILLED, to verify) ---
        "system_institutions": sys_insts,
        "system_raw_affiliations": sys_raws,
        "system_countries":    sys_ctry,
        "system_country_codes": sys_ccs,
        "system_ror_ids":      sys_rors,
        "system_institution_types": sys_types,
        "system_n_affiliations": len(aff_summaries),
        # --- annotator fills these ---
        "correct":           "",     # Y / N / partial
        "gold_institutions": "",     # only if correct != Y
        "gold_countries":    "",
        "gold_ror_ids":      "",
        "notes":             "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--affiliations-dir", type=Path, required=True,
                   help="Directory of SciCo affiliations JSONs.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output CSV path.")
    p.add_argument("--sample-csv", type=Path, default=None,
                   help="Optional CSV from sample_ground_truth.py — restricts "
                        "export to those anthology_ids.")
    p.add_argument("--years", type=str, default=None,
                   help="Optional year range filter, e.g. '1980-2000'.")
    p.add_argument("--skip-empty-authors", action="store_true",
                   help="Skip authors with no name (rare, from parser errors).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    keep_ids = _load_sample_ids(args.sample_csv) if args.sample_csv else None
    if keep_ids is not None:
        logger.info("restricting to %d sampled anthology_ids", len(keep_ids))
    year_range = _parse_year_range(args.years)
    if year_range:
        logger.info("restricting to years %d..%d (inclusive)", *year_range)

    rows: List[Dict[str, Any]] = []
    n_papers = 0
    n_skipped_papers = 0
    n_skipped_authors = 0
    for paper in _iter_papers(args.affiliations_dir, keep_ids, year_range):
        n_papers += 1
        authors = paper.get("authors") or []
        if not authors:
            n_skipped_papers += 1
            continue
        for i, author in enumerate(authors, start=1):
            if args.skip_empty_authors and not (author.get("name") or "").strip():
                n_skipped_authors += 1
                continue
            rows.append(_flatten_author(paper, i, author))

    if not rows:
        logger.error("no rows to write (check --affiliations-dir, --sample-csv, --years)")
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "wrote %d author-rows from %d papers (%d papers without authors, %d empty authors) → %s",
        len(rows), n_papers, n_skipped_papers, n_skipped_authors, args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
