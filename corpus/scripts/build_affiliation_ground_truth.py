"""Build a single ground_truth.csv combining the old SciCo gold (99
papers) and the new manual annotation (683 papers, ROR-linked), in the
exact schema the existing eval_affiliations.py expects:

    anthology_id, year, url, authors_raw_json, authors_normalized_json

`authors_normalized_json` is a JSON list of authors:

    [
      {
        "name": "...",
        "affiliations": [
          {"ror_id": "...", "name": "...", "country": "...", "country_code": "..."}
        ]
      },
      ...
    ]

When a paper is in BOTH sources, new annotation wins (more recent + more
rows per author for multi-affiliation cases). Old ground_truth.csv
remains intact; this script just produces a merged copy.

Usage:
    python scripts/build_affiliation_ground_truth.py \\
        --new-csv  reports/annotation_template.ror_linked.csv \\
        --old-csv  ~/Comparer_flask/data/datastructure/ground_truth.csv \\
        --out      reports/ground_truth_merged.csv \\
        --exclusions configs/annotation_exclusions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

NULL_TOKENS = {"!NULL", "NULL", "NA", "N/A", "#NULL!"}


def _cell(row: Dict[str, str], key: str) -> str:
    return (row.get(key) or "").strip()


def _is_null(s: str) -> bool:
    return s.upper() in NULL_TOKENS


def _year_of(paper_id: str) -> Optional[int]:
    m = re.match(r"(\d{4})", paper_id)
    if m:
        return int(m.group(1))
    m = re.match(r"^[A-Z](\d{2})", paper_id)
    if m:
        yy = int(m.group(1))
        return 1900 + yy if yy >= 60 else 2000 + yy
    return None


def _split(v: str) -> List[str]:
    """Split a pipe-separated cell into clean parts."""
    return [x.strip() for x in (v or "").split("|") if x.strip()] if v else []


def _url_for(pid: str) -> str:
    return f"https://aclanthology.org/{pid}/"


def load_exclusions(path: Optional[Path]) -> Set[str]:
    if not path or not path.exists():
        return set()
    out: Set[str] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("paper_id") or "").strip()
            if pid:
                out.add(pid)
    return out


def papers_from_new(
    new_csv: Path,
    excluded: Set[str],
) -> Dict[str, Dict[str, Any]]:
    """Convert the per-author annotation CSV to per-paper records in
    SciCo ground_truth schema."""
    with new_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    by_paper: Dict[str, Dict[Tuple[int, str], List[Dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        pid = _cell(r, "paper_id")
        if not pid or pid in excluded:
            continue
        try:
            pos = int(_cell(r, "author_position") or 0)
        except ValueError:
            pos = 0
        name = _cell(r, "author_name")
        by_paper[pid][(pos, name)].append(r)

    out: Dict[str, Dict[str, Any]] = {}
    for pid, author_groups in by_paper.items():
        authors_normalized: List[Dict[str, Any]] = []
        authors_raw: List[Dict[str, Any]] = []
        for (pos, name), aff_rows in sorted(author_groups.items()):
            if _is_null(name) or not name:
                continue
            norm_affs: List[Dict[str, Any]] = []
            raw_affs: List[Dict[str, Any]] = []
            for ar in aff_rows:
                raw_aff = _cell(ar, "raw affiliation")
                raw_city = _cell(ar, "raw city")
                raw_country = _cell(ar, "raw country")
                if _is_null(raw_aff):
                    continue
                if not raw_aff and not raw_country:
                    continue
                # Raw side: what the annotator typed verbatim
                raw_affs.append({
                    "affiliation": raw_aff,
                    "city": raw_city or None,
                    "country": raw_country or None,
                })
                # Normalized side: ROR-linked. One affiliation per ROR match
                # (multi-match → multiple normalized entries).
                ror_ids = _split(_cell(ar, "ror_id"))
                ror_names = _split(_cell(ar, "ror_name"))
                ror_ccs = _split(_cell(ar, "ror_country_code"))
                if ror_ids:
                    for i, rid in enumerate(ror_ids):
                        norm_affs.append({
                            "ror_id": rid,
                            "name": ror_names[i] if i < len(ror_names) else raw_aff,
                            "country": raw_country or None,
                            "country_code": ror_ccs[i] if i < len(ror_ccs) else None,
                        })
                else:
                    # No ROR match — use raw values
                    norm_affs.append({
                        "ror_id": None,
                        "name": raw_aff,
                        "country": raw_country or None,
                        "country_code": None,
                    })
            authors_normalized.append({
                "name": name,
                "affiliations": norm_affs,
            })
            authors_raw.append({
                "name": name,
                "affiliations": raw_affs,
            })
        out[pid] = {
            "anthology_id": pid,
            "year": str(_year_of(pid) or ""),
            "url": _url_for(pid),
            "authors_raw_json": json.dumps(authors_raw, ensure_ascii=False),
            "authors_normalized_json": json.dumps(authors_normalized, ensure_ascii=False),
            "_source": "new_annotation",
        }
    return out


def papers_from_old(old_csv: Path) -> Dict[str, Dict[str, Any]]:
    """Load the existing Comparer_flask ground_truth.csv as-is."""
    out: Dict[str, Dict[str, Any]] = {}
    with old_csv.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pid = (r.get("anthology_id") or "").strip()
            if not pid:
                continue
            out[pid] = {
                "anthology_id": pid,
                "year": (r.get("year") or "").strip(),
                "url": (r.get("url") or _url_for(pid)).strip(),
                "authors_raw_json": r.get("authors_raw_json") or "[]",
                "authors_normalized_json": r.get("authors_normalized_json") or "[]",
                "_source": "old_ground_truth",
            }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--new-csv", type=Path, required=True,
                   help="ROR-linked annotation CSV.")
    p.add_argument("--old-csv", type=Path, required=True,
                   help="Existing Comparer_flask ground_truth.csv.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output merged ground_truth CSV.")
    p.add_argument("--exclusions", type=Path,
                   default=Path("configs/annotation_exclusions.csv"),
                   help="Excluded paper IDs (one per line + reason).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    excluded = load_exclusions(args.exclusions)
    logger.info("loaded %d exclusion entries", len(excluded))

    new_papers = papers_from_new(args.new_csv, excluded)
    old_papers = papers_from_old(args.old_csv)
    logger.info("new: %d papers | old: %d papers", len(new_papers), len(old_papers))

    overlap = set(new_papers) & set(old_papers)
    logger.info("overlap (new wins): %d papers", len(overlap))

    # Merge: new wins on overlap
    merged: Dict[str, Dict[str, Any]] = {}
    merged.update(old_papers)
    merged.update(new_papers)

    # Drop excluded entries
    for pid in list(merged):
        if pid in excluded:
            del merged[pid]

    # Write
    fieldnames = ["anthology_id", "year", "url",
                  "authors_raw_json", "authors_normalized_json", "_source"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for pid in sorted(merged):
            w.writerow(merged[pid])

    # Stats
    sources = Counter(r["_source"] for r in merged.values())
    by_decade: Counter = Counter()
    for pid in merged:
        y = _year_of(pid)
        if y:
            by_decade[(y // 10) * 10] += 1

    print()
    print(f"=== Merged ground truth ===")
    print(f"  Output: {args.out}")
    print(f"  Total papers: {len(merged)}")
    print(f"  Excluded (dropped): {len(excluded)}")
    print()
    print(f"  By source:")
    for src, c in sources.items():
        print(f"    {src:<22} {c:>6}")
    print()
    print(f"  Per-decade:")
    for d in sorted(by_decade):
        print(f"    {d}s   {by_decade[d]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
