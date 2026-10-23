"""Merge the new manual annotation (per-author CSV) and the old SciCo
ground_truth.csv (99 papers, ROR-normalized JSON) into a unified
per-paper gold directory.

The output dir mirrors the SciCo per-paper-JSON schema, so it drops
straight into `eval_scico.py --affiliations-dir <out_dir>` and replaces
the full Gemini-extracted corpus with verified human gold for the
papers we've actually annotated. This is the non-circularity fix for
retrieval evaluation.

Each output file is `<paper_id>.json` with:
  - paper_id, year, url, venue, abstract     (copied from enriched SciCo)
  - topics.cso.union                          (copied from enriched SciCo)
  - authors[].affiliations[]                  (REPLACED with verified gold)
  - _provenance.gold_source                   ("new_annotation" / "old_ground_truth" / "merged")

When a paper appears in BOTH sources, "new_annotation" wins (more recent),
and any extra affiliations from the old source are appended with a
`_provenance.note`.

Usage:
    python scripts/merge_gold.py \\
        --new-csv     /path/to/annotation_template.cleaned.csv \\
        --old-csv     /path/to/ground_truth.csv \\
        --enriched-dir /path/to/ror_parsed_affiliations_90_with_topics \\
        --out-dir     /path/to/gold_affiliations
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

ANN_FIELDS = ("raw affiliation", "raw city", "raw country")
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


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_new_annotation(path: Path) -> Dict[str, Dict[str, Any]]:
    """Returns {paper_id -> {authors: [...], excluded_papers: set}}."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    logger.info("new annotation: %d author-rows", len(rows))

    # Group by (paper_id, author_position)
    by_paper: Dict[str, Dict[Tuple[int, str], List[Dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    excluded_paper_ids: Set[str] = set()
    for r in rows:
        pid = _cell(r, "paper_id")
        if not pid:
            continue
        if _cell(r, "excluded").upper() == "Y":
            excluded_paper_ids.add(pid)
            continue
        try:
            pos = int(_cell(r, "author_position") or 0)
        except ValueError:
            pos = 0
        name = _cell(r, "author_name")
        by_paper[pid][(pos, name)].append(r)

    out: Dict[str, Dict[str, Any]] = {}
    for pid, author_groups in by_paper.items():
        authors_list: List[Dict[str, Any]] = []
        for (pos, name), aff_rows in sorted(author_groups.items()):
            if _is_null(name):
                # author wasn't extractable — skip but record at paper level
                continue
            affs: List[Dict[str, Any]] = []
            for ar in aff_rows:
                raw_aff = _cell(ar, "raw affiliation")
                raw_city = _cell(ar, "raw city")
                raw_country = _cell(ar, "raw country")
                if _is_null(raw_aff) or _is_null(raw_country):
                    continue
                if not raw_aff and not raw_country:
                    continue
                # Prefer ROR-linked canonical name when present. ROR-link
                # script can produce multiple matches per affiliation
                # (e.g., "LIMSI & CNRS" → 2 ROR IDs); we expand those into
                # separate affiliation entries in the SciCo schema.
                ror_ids = _cell(ar, "ror_id").split(" | ") if _cell(ar, "ror_id") else []
                ror_names = _cell(ar, "ror_name").split(" | ") if _cell(ar, "ror_name") else []
                ror_ccs = _cell(ar, "ror_country_code").split(" | ") if _cell(ar, "ror_country_code") else []
                ror_confs = _cell(ar, "ror_confidence").split(" | ") if _cell(ar, "ror_confidence") else []
                # Filter empties
                ror_ids = [x for x in ror_ids if x]
                if ror_ids:
                    for i, rid in enumerate(ror_ids):
                        affs.append({
                            "clean_institution_name": ror_names[i] if i < len(ror_names) else raw_aff,
                            "raw_affiliation": raw_aff,
                            "city": raw_city or None,
                            "country": raw_country or None,
                            "country_code": ror_ccs[i] if i < len(ror_ccs) else None,
                            "ror_id": rid,
                            "ror_confidence": float(ror_confs[i]) if i < len(ror_confs) and ror_confs[i] else None,
                            "institution_type": None,
                        })
                else:
                    # No ROR match — fall back to raw user-typed values
                    affs.append({
                        "clean_institution_name": raw_aff,
                        "raw_affiliation": raw_aff,
                        "city": raw_city or None,
                        "country": raw_country or None,
                        "country_code": None,
                        "ror_id": None,
                        "ror_confidence": None,
                        "institution_type": None,
                    })
            if affs or name:
                authors_list.append({
                    "name": name,
                    "affiliation_markers": [],
                    "affiliations": affs,
                })
        out[pid] = {"authors": authors_list}

    # Mark excluded papers separately so they can be reported and skipped
    return {"papers": out, "excluded": excluded_paper_ids}


def load_old_ground_truth(path: Path) -> Dict[str, Dict[str, Any]]:
    """Reads the old SciCo ground_truth.csv (99 rows, ROR-normalized JSON).
    Returns {paper_id -> {authors: [...]}}."""
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    logger.info("old ground_truth: %d rows", len(rows))

    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        pid = (r.get("anthology_id") or "").strip()
        if not pid:
            continue
        try:
            authors_json = json.loads(r.get("authors_normalized_json") or "[]")
        except json.JSONDecodeError:
            continue
        # Normalize key naming to match SciCo per-paper-JSON schema
        normalized_authors = []
        for a in authors_json:
            affs = []
            for aff in (a.get("affiliations") or []):
                affs.append({
                    "clean_institution_name": aff.get("name") or aff.get("clean_institution_name"),
                    "city": aff.get("city"),
                    "country": aff.get("country"),
                    "country_code": aff.get("country_code"),
                    "ror_id": aff.get("ror_id"),
                    "institution_type": aff.get("institution_type"),
                })
            normalized_authors.append({
                "name": a.get("name"),
                "affiliation_markers": a.get("affiliation_markers") or [],
                "affiliations": affs,
            })
        out[pid] = {"authors": normalized_authors}
    return out


# ---------------------------------------------------------------------------
# Merge + write
# ---------------------------------------------------------------------------

def load_enriched_paper(enriched_dir: Path, paper_id: str) -> Optional[Dict[str, Any]]:
    fp = enriched_dir / f"{paper_id}.json"
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def merge_and_write(
    new_papers: Dict[str, Dict[str, Any]],
    old_papers: Dict[str, Dict[str, Any]],
    enriched_dir: Path,
    out_dir: Path,
    excluded: Set[str],
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_pids = set(new_papers) | set(old_papers)
    stats = {
        "new_only": 0, "old_only": 0, "both": 0,
        "skipped_excluded": 0, "missing_enriched": 0, "written": 0,
        "per_decade_papers": Counter(),
        "per_source_papers": Counter(),
    }

    for pid in sorted(all_pids):
        if pid in excluded:
            stats["skipped_excluded"] += 1
            continue

        # Pull the enriched base (topics, abstract, year, venue, ...)
        base = load_enriched_paper(enriched_dir, pid)
        if base is None:
            stats["missing_enriched"] += 1
            # We can still emit a minimal record without topics — but the
            # eval needs topics.cso.union to compute closure matches. Skip
            # for now and log it.
            logger.warning("no enriched JSON for %s; skipping (no topics)", pid)
            continue

        # Decide which annotation wins
        in_new, in_old = (pid in new_papers), (pid in old_papers)
        if in_new and in_old:
            source = "merged"
            stats["both"] += 1
            authors = new_papers[pid]["authors"]
            note = "primary=new_annotation; old_ground_truth available but not used"
        elif in_new:
            source = "new_annotation"
            stats["new_only"] += 1
            authors = new_papers[pid]["authors"]
            note = None
        else:
            source = "old_ground_truth"
            stats["old_only"] += 1
            authors = old_papers[pid]["authors"]
            note = None

        merged = dict(base)
        merged["authors"] = authors
        prov = dict(merged.get("_provenance") or {})
        prov["gold_source"] = source
        if note:
            prov["gold_note"] = note
        merged["_provenance"] = prov

        out_path = out_dir / f"{pid}.json"
        out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        stats["written"] += 1
        stats["per_source_papers"][source] += 1
        y = _year_of(pid)
        if y is not None:
            stats["per_decade_papers"][(y // 10) * 10] += 1

    return stats


def print_stats(stats: Dict[str, Any]) -> None:
    print()
    print(f"=== Merged gold set ===")
    print(f"  Written papers:           {stats['written']}")
    print(f"  Skipped (excluded):       {stats['skipped_excluded']}")
    print(f"  Skipped (no enriched):    {stats['missing_enriched']}")
    print()
    print(f"=== By source ===")
    print(f"  New annotation only:      {stats['new_only']}")
    print(f"  Old ground_truth only:    {stats['old_only']}")
    print(f"  Both (new wins):          {stats['both']}")
    print()
    print(f"=== Per-decade (papers in merged set) ===")
    for d in sorted(stats["per_decade_papers"]):
        c = stats["per_decade_papers"][d]
        print(f"  {d}s:  {c}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--new-csv", type=Path, required=True,
                   help="Cleaned annotation CSV (post-cleanup_annotation.py).")
    p.add_argument("--old-csv", type=Path, required=True,
                   help="Old SciCo ground_truth.csv (99 papers, ROR-normalized).")
    p.add_argument("--enriched-dir", type=Path, required=True,
                   help="SciCo per-paper enriched JSONs (with topics.cso.union).")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for merged per-paper gold JSONs.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    new = load_new_annotation(args.new_csv)
    old = load_old_ground_truth(args.old_csv)

    overlap = set(new["papers"]) & set(old)
    logger.info("new: %d papers, old: %d papers, overlap: %d papers",
                len(new["papers"]), len(old), len(overlap))

    stats = merge_and_write(
        new_papers=new["papers"],
        old_papers=old,
        enriched_dir=args.enriched_dir,
        out_dir=args.out_dir,
        excluded=new["excluded"],
    )
    print_stats(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
