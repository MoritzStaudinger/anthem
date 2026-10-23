"""Cleanup + audit pipeline for the manual affiliation-annotation CSV.

Does five things, all idempotent:

1. **Normalize.** Backfill `year` from `paper_id` where empty; trim
   whitespace; canonicalize `!NULL` spellings to exactly `!NULL`.

2. **Apply exclusion list.** Reads configs/annotation_exclusions.csv
   (paper_id, reason) and marks listed papers' rows with `excluded=Y`
   plus the reason in the `notes` column. Use this for papers where the
   affiliations exist on the PDF but cannot be reliably normalized
   (e.g., institution name doesn't map to any ROR entity).

3. **Report cleanup needs.** Lists papers/rows that are still empty
   (not annotated, not marked NULL, not excluded) so the annotator can
   close them out.

4. **Print coverage stats.** Per-decade + overall, with breakdown of
   annotated / NULL / excluded / still-pending.

5. **Optionally write a balanced subset.** Pass --balanced-out PATH and
   --per-decade N to emit a derivative CSV with N papers per decade
   (random subsample, seeded), useful for headline tables in the paper.
   Original full file stays intact.

Usage:
    python scripts/cleanup_annotation.py \\
        --input  annotation_template.csv \\
        --output annotation_template.cleaned.csv \\
        --exclusions configs/annotation_exclusions.csv \\
        --balanced-out annotation_balanced_100perdecade.csv --per-decade 100

The exclusions config lives at configs/annotation_exclusions.csv with
columns: paper_id, reason. One paper per line.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
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
    """Extract year from paper_id. Modern (2020.acl-main.1) and legacy
    (P80-1001 / J96-1001 / etc.) both supported."""
    m = re.match(r"(\d{4})", paper_id)
    if m:
        return int(m.group(1))
    m = re.match(r"^[A-Z](\d{2})", paper_id)
    if m:
        yy = int(m.group(1))
        return 1900 + yy if yy >= 60 else 2000 + yy
    return None


# ---------------------------------------------------------------------------
# Exclusion list
# ---------------------------------------------------------------------------

def load_exclusions(path: Optional[Path]) -> Dict[str, str]:
    """Returns {paper_id -> reason}."""
    if not path or not path.exists():
        return {}
    out: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pid = (row.get("paper_id") or "").strip()
            reason = (row.get("reason") or "").strip()
            if pid:
                out[pid] = reason or "(no reason given)"
    return out


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_row(row: Dict[str, str], exclusions: Dict[str, str]) -> Dict[str, str]:
    """In-place-style normalization: backfill year, canonicalize NULL,
    trim, apply exclusion list, propagate author-level NULL to affiliation
    fields. Returns the row."""
    pid = (row.get("paper_id") or "").strip()
    row["paper_id"] = pid

    # Year backfill
    if not _cell(row, "year"):
        y = _year_of(pid)
        if y is not None:
            row["year"] = str(y)

    # Canonicalize NULL tokens
    for k in ANN_FIELDS + ("author_name",):
        v = _cell(row, k)
        if _is_null(v):
            row[k] = "!NULL"

    # If the author itself was un-annotateable, propagate !NULL to all
    # affiliation fields so the row is classified as 'null' (the cleanup
    # report otherwise saw it as 'empty' / forgotten).
    if _cell(row, "author_name") == "!NULL":
        for k in ANN_FIELDS:
            if not _cell(row, k):
                row[k] = "!NULL"

    # Whitespace trim on every string column
    for k, v in list(row.items()):
        if isinstance(v, str):
            row[k] = v.strip()

    # Apply exclusion list
    if pid in exclusions:
        if "excluded" in row:
            row["excluded"] = "Y"
        else:
            # add the column inline; CSV writer will pick it up via fieldnames
            row["excluded"] = "Y"
        existing_notes = _cell(row, "notes")
        excl_note = f"[excluded] {exclusions[pid]}"
        if excl_note not in existing_notes:
            row["notes"] = (existing_notes + " | " + excl_note).strip(" |") if existing_notes else excl_note

    return row


# ---------------------------------------------------------------------------
# Row classification
# ---------------------------------------------------------------------------

def classify_row(row: Dict[str, str]) -> str:
    """Return one of: 'annotated', 'null', 'excluded', 'empty'."""
    if _cell(row, "excluded").upper() == "Y":
        return "excluded"
    vals = [_cell(row, k) for k in ANN_FIELDS]
    if any(_is_null(v) for v in vals):
        return "null"
    if any(v for v in vals):
        return "annotated"
    return "empty"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(rows: List[Dict[str, str]]) -> None:
    n = len(rows)
    statuses = Counter(classify_row(r) for r in rows)

    print(f"\n=== Author-affiliation rows ({n} total) ===")
    for status in ("annotated", "null", "excluded", "empty"):
        c = statuses.get(status, 0)
        pct = 100 * c / n if n else 0
        print(f"  {status:<10} {c:>6}  ({pct:>5.1f}%)")

    # Per-paper completion
    papers: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        papers[r["paper_id"]].append(r)

    paper_status: Dict[str, Counter] = {}
    for pid, rs in papers.items():
        paper_status[pid] = Counter(classify_row(r) for r in rs)

    def _classify_paper(c: Counter) -> str:
        if c.get("excluded", 0) == sum(c.values()):
            return "all_excluded"
        if c.get("annotated", 0) + c.get("null", 0) + c.get("excluded", 0) == sum(c.values()):
            return "complete"
        if c.get("annotated", 0) or c.get("null", 0):
            return "partial"
        return "untouched"

    p_buckets = Counter(_classify_paper(c) for c in paper_status.values())
    total_papers = len(papers)
    print(f"\n=== Papers ({total_papers} distinct) ===")
    for status in ("complete", "partial", "untouched", "all_excluded"):
        c = p_buckets.get(status, 0)
        pct = 100 * c / total_papers if total_papers else 0
        print(f"  {status:<14} {c:>5}  ({pct:>5.1f}%)")

    # Per-decade
    by_decade: Dict[int, Counter] = defaultdict(Counter)
    by_decade_papers: Dict[int, Set[str]] = defaultdict(set)
    for r in rows:
        y = _year_of(r["paper_id"])
        if y is None:
            continue
        d = (y // 10) * 10
        by_decade[d][classify_row(r)] += 1
        by_decade_papers[d].add(r["paper_id"])

    print(f"\n=== Per-decade ===")
    print(f"  {'decade':<8}{'papers':>8}{'rows':>6}{'annot':>8}{'null':>6}{'excl':>6}{'empty':>7}")
    for d in sorted(by_decade):
        c = by_decade[d]
        print(f"  {d}s     {len(by_decade_papers[d]):>8}{sum(c.values()):>6}"
              f"{c.get('annotated',0):>8}{c.get('null',0):>6}"
              f"{c.get('excluded',0):>6}{c.get('empty',0):>7}")

    # Action items
    untouched = [pid for pid, c in paper_status.items() if _classify_paper(c) == "untouched"]
    partial   = [pid for pid, c in paper_status.items() if _classify_paper(c) == "partial"]
    empties = [(r["paper_id"], r.get("author_name", "")) for r in rows
               if classify_row(r) == "empty"]

    if untouched or partial or empties:
        print(f"\n=== Cleanup actions ===")
    if untouched:
        print(f"  Untouched papers ({len(untouched)}):")
        for p in untouched[:20]:
            print(f"    - {p}")
        if len(untouched) > 20:
            print(f"    ... + {len(untouched)-20} more")
    if partial:
        print(f"  Partially annotated papers ({len(partial)}):")
        for p in partial[:20]:
            print(f"    - {p}")
        if len(partial) > 20:
            print(f"    ... + {len(partial)-20} more")
    if empties:
        print(f"  Empty rows ({len(empties)}):")
        for pid, name in empties[:10]:
            print(f"    - {pid} / {name}")
        if len(empties) > 10:
            print(f"    ... + {len(empties)-10} more")

    if not (untouched or partial or empties):
        print(f"\n=== Cleanup actions ===\n  None — annotation is complete. ✓")


# ---------------------------------------------------------------------------
# Balanced subset
# ---------------------------------------------------------------------------

def write_balanced_subset(
    rows: List[Dict[str, str]],
    out_path: Path,
    per_decade: int,
    seed: int = 42,
) -> None:
    """Pick `per_decade` papers per decade (random sample, seeded) and
    write their rows to `out_path`. If a decade has fewer papers, take
    all of them."""
    by_decade_papers: Dict[int, List[str]] = defaultdict(list)
    seen = set()
    for r in rows:
        pid = r["paper_id"]
        if pid in seen:
            continue
        y = _year_of(pid)
        if y is None:
            continue
        d = (y // 10) * 10
        by_decade_papers[d].append(pid)
        seen.add(pid)

    rng = random.Random(seed)
    kept: Set[str] = set()
    for d, pids in by_decade_papers.items():
        if len(pids) > per_decade:
            chosen = rng.sample(pids, per_decade)
        else:
            chosen = pids
            logger.info("decade %ds has only %d papers (under %d target)",
                        d, len(pids), per_decade)
        kept.update(chosen)

    kept_rows = [r for r in rows if r["paper_id"] in kept]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(kept_rows)
    print(f"\n=== Balanced subset ===")
    print(f"  wrote {len(kept_rows)} rows ({len(kept)} papers) to {out_path}")
    print(f"  per-decade counts:")
    bal_by_decade: Dict[int, int] = defaultdict(int)
    for pid in kept:
        y = _year_of(pid)
        if y is not None:
            bal_by_decade[(y // 10) * 10] += 1
    for d in sorted(bal_by_decade):
        print(f"    {d}s: {bal_by_decade[d]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", type=Path, required=True,
                   help="Annotated CSV (post-ODS export).")
    p.add_argument("--output", type=Path, default=None,
                   help="Cleaned CSV output. If omitted, only reports stats.")
    p.add_argument("--exclusions", type=Path,
                   default=Path("configs/annotation_exclusions.csv"),
                   help="CSV with columns 'paper_id, reason' for excluded papers.")
    p.add_argument("--balanced-out", type=Path, default=None,
                   help="If set, also write a per-decade balanced subset here.")
    p.add_argument("--per-decade", type=int, default=100,
                   help="Papers per decade in the balanced subset (default 100).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load
    with args.input.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        original_fields = reader.fieldnames or []
    logger.info("loaded %d rows from %s", len(rows), args.input)

    exclusions = load_exclusions(args.exclusions)
    logger.info("loaded %d exclusion entries from %s",
                len(exclusions), args.exclusions)

    # Add 'excluded' column if missing
    fieldnames = list(original_fields)
    if "excluded" not in fieldnames:
        # Insert before notes if present, otherwise at end
        try:
            i = fieldnames.index("notes")
            fieldnames.insert(i, "excluded")
        except ValueError:
            fieldnames.append("excluded")

    # Normalize every row
    for row in rows:
        normalize_row(row, exclusions)
        for k in fieldnames:
            row.setdefault(k, "")

    # Report
    report(rows)

    # Write cleaned CSV
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\n=== Cleaned CSV ===")
        print(f"  wrote {len(rows)} rows ({len(fieldnames)} columns) to {args.output}")

    # Balanced subset (optional)
    if args.balanced_out:
        write_balanced_subset(rows, args.balanced_out, args.per_decade, args.seed)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
