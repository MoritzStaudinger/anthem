"""Aggregate per-year statistics from the bibmeta-enriched corpus.

Walks <corpus-dir>/*.json once and emits a long-form CSV with one row
per year containing the counts and dispersion measures needed for
yearly trend plots with error bands:

    year, n_papers,
    authors_mean, authors_std, authors_median, authors_p25, authors_p75,
    insts_mean,   insts_std,   insts_median,   insts_p25,   insts_p75

Where:
  - "authors" = number of author entries on the paper.
  - "insts"   = number of DISTINCT ROR-linked institutions on the
                paper (deduplicated per paper, matching the convention
                used in corpus_stats.py).

Run on the cluster directly (no GPU needed, ~10-15 min):

    python scripts/aggregate_yearly_stats.py \\
        --corpus-dir /share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta \\
        --out reports/yearly_stats.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger("yearly_stats")


def _safe_int(v):
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _stats(xs: List[float]):
    """Return (mean, std, median, p25, p75) for a list of numbers.
    Uses sample standard deviation (ddof=1) when len > 1."""
    if not xs:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    n = len(xs)
    mean = sum(xs) / n
    if n > 1:
        var = sum((x - mean) ** 2 for x in xs) / (n - 1)
        std = math.sqrt(var)
    else:
        std = 0.0
    s = sorted(xs)
    def quant(q):
        if n == 1:
            return s[0]
        pos = (n - 1) * q
        lo, hi = int(pos), min(int(pos) + 1, n - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)
    return (mean, std, quant(0.5), quant(0.25), quant(0.75))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--corpus-dir", type=Path,
        default=Path("/share/moritz.staudinger/acl_data/"
                     "ror_parsed_affiliations_90_with_topics_and_bibmeta"))
    ap.add_argument("--out", type=Path,
                    default=Path("reports/yearly_stats.csv"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s")

    if not args.corpus_dir.is_dir():
        raise SystemExit(f"corpus dir not found: {args.corpus_dir}")

    files = sorted(args.corpus_dir.glob("*.json"))
    logger.info("scanning %d JSONs", len(files))

    # year -> list of (n_authors, n_distinct_institutions)
    by_year_authors: Dict[int, List[int]] = defaultdict(list)
    by_year_insts: Dict[int, List[int]] = defaultdict(list)

    n_papers_total = 0
    for i, fp in enumerate(files):
        if i and i % 10000 == 0:
            logger.info("  %d / %d processed", i, len(files))
        try:
            d = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        year = _safe_int(d.get("year"))
        if year is None:
            continue
        n_papers_total += 1

        authors = d.get("authors") or []
        n_authors = len(authors)

        # Count DISTINCT ROR-linked institutions on this paper
        seen_ror: set = set()
        for author in authors:
            for aff in (author.get("affiliations") or []):
                if not isinstance(aff, dict):
                    continue
                ror = (aff.get("ror_id") or "").strip()
                if ror:
                    seen_ror.add(ror)
        n_insts = len(seen_ror)

        by_year_authors[year].append(n_authors)
        by_year_insts[year].append(n_insts)

    logger.info("dropped %d files without a usable year",
                len(files) - n_papers_total)

    # Sort and write CSV
    years = sorted(set(by_year_authors) | set(by_year_insts))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "year", "n_papers",
            "authors_mean", "authors_std",
            "authors_median", "authors_p25", "authors_p75",
            "insts_mean", "insts_std",
            "insts_median", "insts_p25", "insts_p75",
        ])
        for y in years:
            a = by_year_authors[y]
            i = by_year_insts[y]
            am, as_, amed, ap25, ap75 = _stats(a)
            im, is_, imed, ip25, ip75 = _stats(i)
            w.writerow([y, len(a),
                        f"{am:.4f}", f"{as_:.4f}",
                        f"{amed:.4f}", f"{ap25:.4f}", f"{ap75:.4f}",
                        f"{im:.4f}", f"{is_:.4f}",
                        f"{imed:.4f}", f"{ip25:.4f}", f"{ip75:.4f}"])
    logger.info("wrote %s (%d years, %d papers total)",
                args.out, len(years), n_papers_total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
