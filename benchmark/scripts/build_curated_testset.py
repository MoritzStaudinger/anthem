"""Filter a candidate JSONL + curation CSV → curated testset.

Treats the CSV's `keep` column as the source of truth for which rows
make the cut. The CSV is the curator-friendly flat view; the JSONL has
the full structured `classifier_gold` + `topic_metadata.cso_closure`
that the eval needs.

This is a 2-input → 1-output filter, idempotent: re-run any time the
CSV's `keep` column changes, and the testset regenerates with the new
selection (the old file is backed up first).

Usage:
    python scripts/build_curated_testset.py
        # defaults: --csv  ../acl_processor/scico_candidates_review.csv
        #           --jsonl ../acl_processor/scico_candidates_counts.jsonl
        #           --out   data/eval/scico_testset.jsonl

    python scripts/build_curated_testset.py \\
        --csv path/to/review.csv \\
        --jsonl path/to/counts.jsonl \\
        --out  data/eval/scico_testset.jsonl
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

DEFAULT_CSV   = Path("../acl_processor/scico_candidates_review.csv")
DEFAULT_JSONL = Path("../acl_processor/scico_candidates_counts.jsonl")
DEFAULT_OUT   = Path("data/eval/scico_testset.jsonl")


def _keep(value: str | None) -> bool:
    """Strict: only 'Y' / 'y' counts as kept. Anything else (N, blank,
    typos like 'PL' or '99') is dropped. Easy to relax later if needed."""
    return (value or "").strip().upper() == "Y"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv",   type=Path, default=DEFAULT_CSV,
                   help=f"Curation CSV with a `keep` column. Default: {DEFAULT_CSV}")
    p.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL,
                   help=f"Full-schema candidates JSONL. Default: {DEFAULT_JSONL}")
    p.add_argument("--out",   type=Path, default=DEFAULT_OUT,
                   help=f"Output testset JSONL. Default: {DEFAULT_OUT}")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip auto-backup of an existing --out file.")
    args = p.parse_args(argv)

    if not args.csv.is_file():
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        return 2
    if not args.jsonl.is_file():
        print(f"ERROR: JSONL not found: {args.jsonl}", file=sys.stderr)
        return 2

    # --- 1. read CSV keep marks ---
    csv_rows = list(csv.DictReader(args.csv.open()))
    kept_ids = {r["id"] for r in csv_rows if _keep(r.get("keep"))}
    typos = [r for r in csv_rows
             if (r.get("keep") or "").strip()
             and (r["keep"] or "").strip().upper() not in ("Y", "N")]
    print(f"CSV: {len(csv_rows)} rows  → kept={len(kept_ids)}  "
          f"typos_dropped={len(typos)}")
    if typos:
        for r in typos:
            print(f"  typo dropped: id={r.get('id','?')!r}  keep={r.get('keep')!r}",
                  file=sys.stderr)

    # --- 2. filter JSONL to kept ids, preserving CSV order for stability ---
    csv_order = [r["id"] for r in csv_rows if _keep(r.get("keep"))]
    by_id: dict[str, dict] = {}
    with args.jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = rec.get("id")
            if rid in kept_ids:
                by_id[rid] = rec
    missing_in_jsonl = sorted(kept_ids - set(by_id))
    if missing_in_jsonl:
        print(f"WARN: {len(missing_in_jsonl)} ids marked Y but missing in JSONL:",
              file=sys.stderr)
        for rid in missing_in_jsonl[:10]:
            print(f"  {rid}", file=sys.stderr)
    curated = [by_id[rid] for rid in csv_order if rid in by_id]
    print(f"JSONL: {len(by_id)} matched rows → curated set: {len(curated)}")

    # --- 3. sanity checks the eval needs ---
    cats = Counter(((r.get("classifier_gold") or {}).get("question_category")
                    or "(missing)") for r in curated)
    print("\nPer-category breakdown of curated set:")
    for c, n in cats.most_common():
        print(f"  {c:<40} {n}")
    missing_cg = sum(1 for r in curated
                     if not (r.get("classifier_gold") or {}).get("question_category"))
    missing_tm = sum(1 for r in curated
                     if not (r.get("topic_metadata") or {}).get("cso_closure"))
    if missing_cg or missing_tm:
        print(f"\nWARN: {missing_cg} rows missing classifier_gold.question_category",
              file=sys.stderr)
        print(f"WARN: {missing_tm} rows missing topic_metadata.cso_closure",
              file=sys.stderr)

    # --- 4. write, backing up the previous file ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists() and not args.no_backup:
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = args.out.with_suffix(args.out.suffix + f".{stamp}.bak")
        shutil.copy2(args.out, backup)
        print(f"\nBacked up old testset → {backup}")
    with args.out.open("w", encoding="utf-8") as fh:
        for r in curated:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(curated)} rows → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
