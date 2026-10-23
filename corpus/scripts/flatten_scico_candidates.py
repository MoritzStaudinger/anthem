"""Flatten scico_candidates.jsonl into a curation-friendly CSV.

Output columns (curation columns first, then metadata):
    id, keep, notes,
    category, persona, depth,
    query,
    uni1, uni1_ror, uni2, uni2_ror,
    country, country_b,
    topic, cso_url, closure_size,
    year_start, year_end,
    target_paper_id, target_paper_title, target_paper_year,
    cso_closure_sample

Usage:
    python scripts/flatten_scico_candidates.py
    python scripts/flatten_scico_candidates.py --in path/to/candidates.jsonl --out path/to/review.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


DEFAULT_IN = Path(__file__).resolve().parent.parent / "scico_candidates.jsonl"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "scico_candidates_review.csv"


FIELDNAMES = [
    "id", "keep", "notes",
    "category", "persona", "depth",
    "gold_paper_count",          # populated by enrich_question_gold_counts.py
    "query",
    "uni1", "uni1_ror", "uni2", "uni2_ror",
    "country", "country_b",
    "topic", "cso_url", "closure_size",
    "year_start", "year_end",
    "target_paper_id", "target_paper_title", "target_paper_year",
    "cso_closure_sample",
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN,
                   help=f"Input JSONL (default: {DEFAULT_IN})")
    p.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT,
                   help=f"Output CSV (default: {DEFAULT_OUT})")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit rows for testing.")
    args = p.parse_args(argv)

    if not args.in_path.exists():
        print(f"ERROR: input not found: {args.in_path}", file=sys.stderr)
        return 2

    with args.in_path.open() as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    cats = Counter((r.get("classifier_gold") or {}).get("question_category") for r in rows)
    print("Per-category counts:")
    for c, n in cats.most_common():
        print(f"  {c}: {n}")
    print(f"Total: {len(rows)}")

    # Useful sanity: year-window distribution
    ranges = Counter((
        (r.get("classifier_gold") or {}).get("year_start"),
        (r.get("classifier_gold") or {}).get("year_end"),
    ) for r in rows)
    print(f"\nUnique year-window pairs: {len(ranges)}")
    print("Top 10:")
    for pair, n in ranges.most_common(10):
        print(f"  {pair[0]}-{pair[1]}: {n}")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for r in rows:
            cg = r.get("classifier_gold") or {}
            tm = r.get("topic_metadata") or {}
            paper = r.get("paper") or {}
            closure = tm.get("cso_closure") or []
            gold_ids = r.get("gold_paper_ids") or []
            w.writerow({
                "id": r.get("id", ""),
                "keep": "",
                "notes": "",
                "category": cg.get("question_category", ""),
                "persona": r.get("persona", ""),
                "depth": r.get("depth", ""),
                "gold_paper_count": r.get("gold_paper_count", ""),
                "query": r.get("query", ""),
                "uni1": cg.get("uni1") or "",
                "uni1_ror": cg.get("uni1_ror") or "",
                "uni2": cg.get("uni2") or "",
                "uni2_ror": cg.get("uni2_ror") or "",
                "country": cg.get("country") or "",
                # New: for cross_country_method rows; empty for everything else.
                "country_b": cg.get("country_b") or "",
                "topic": cg.get("topic") or "",
                "cso_url": tm.get("cso_url") or "",
                "closure_size": tm.get("cso_closure_size") or "",
                "year_start": cg.get("year_start") or "",
                "year_end": cg.get("year_end") or "",
                "target_paper_id": gold_ids[0] if gold_ids else "",
                "target_paper_title": (paper.get("title") or "")[:120],
                "target_paper_year": paper.get("year") or "",
                "cso_closure_sample": "; ".join(closure[:8]),
            })

    print(f"\nwrote {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
