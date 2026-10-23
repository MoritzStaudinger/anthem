"""Annotate scico_candidates.jsonl with the size of each question's
gold paper set, computed against the live affiliations corpus.

Adds two fields per row:
    gold_paper_count : int    number of papers that satisfy the row's
                              classifier_gold filter (topic ∩ inst/country
                              ∩ year window, per compute_gold_paper_ids)
    gold_paper_ids   : List[str]  the actual paper ids (kept for audit;
                              normally empty in the question file because
                              eval recomputes at runtime — populated here
                              so the curation CSV can list the IDs without
                              re-running the corpus walk)

Curation rule of thumb: rows with gold_paper_count < 5 are usually too
sparse to give a defensible reviewer answer; drop or relax their
constraints. The generator's depth_check already enforces ≥3 papers
at (inst, topic) — this pass is the post-hoc verification across all
gold filters (incl. year window and country) which the depth_check
doesn't cover.

Usage:
    python scripts/enrich_question_gold_counts.py \\
        --in  data/eval/scico_candidates.jsonl \\
        --out data/eval/scico_candidates.jsonl \\
        --affiliations-dir /share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics

Output is the same JSONL with one extra integer field per row. Safe to
re-run idempotently (it overwrites the count each time).
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

# Reuse the runtime gold logic so this script can't drift from eval.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend.eval.gold import compute_gold_paper_ids, load_affiliations_corpus  # noqa: E402

logger = logging.getLogger("enrich_gold_counts")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", dest="out_path", type=Path, required=True)
    p.add_argument("--affiliations-dir", type=Path, required=True,
                   help="Per-paper affiliations dir (the same one eval_scico "
                        "uses for --affiliations-dir).")
    p.add_argument("--store-ids", action="store_true",
                   help="Also save the full gold_paper_ids list per row. "
                        "Bloats the JSONL — keep off unless you want the "
                        "curation CSV to show the actual papers.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    if not args.in_path.exists():
        raise SystemExit(f"input not found: {args.in_path}")
    if not args.affiliations_dir.is_dir():
        raise SystemExit(f"affiliations-dir not found: {args.affiliations_dir}")

    t0 = time.time()
    corpus = load_affiliations_corpus(args.affiliations_dir)
    logger.info("loaded %d corpus papers in %.1fs",
                len(corpus), time.time() - t0)

    with args.in_path.open() as f:
        rows = [json.loads(l) for l in f if l.strip()]
    logger.info("loaded %d question rows", len(rows))

    cat_count_buckets: Dict[str, Counter] = {}
    sparse_rows: List[Dict[str, Any]] = []

    t1 = time.time()
    for i, row in enumerate(rows):
        gold = compute_gold_paper_ids(row, corpus)
        row["gold_paper_count"] = len(gold)
        if args.store_ids:
            row["gold_paper_ids"] = sorted(gold)
        # Per-category bucket histogram for the summary print
        cat = (row.get("classifier_gold") or {}).get("question_category") or "?"
        bucket = (
            "0" if len(gold) == 0 else
            "1-2" if len(gold) <= 2 else
            "3-4" if len(gold) <= 4 else
            "5-9" if len(gold) <= 9 else
            "10-24" if len(gold) <= 24 else
            "25-99" if len(gold) <= 99 else
            "100+"
        )
        cat_count_buckets.setdefault(cat, Counter())[bucket] += 1
        if len(gold) < 5:
            sparse_rows.append({
                "id": row.get("id"),
                "category": cat,
                "gold_count": len(gold),
                "query": (row.get("query") or "")[:100],
            })
        if (i + 1) % 25 == 0:
            logger.info("  scored %d/%d rows", i + 1, len(rows))
    logger.info("scored %d rows in %.1fs", len(rows), time.time() - t1)

    # ---- summary ----
    print(f"\n=== gold-count distribution per category ===")
    buckets = ["0", "1-2", "3-4", "5-9", "10-24", "25-99", "100+"]
    print(f"{'category':<38} " + " ".join(f"{b:>5}" for b in buckets) + "  total")
    for cat in sorted(cat_count_buckets):
        ctr = cat_count_buckets[cat]
        row_total = sum(ctr.values())
        row_strs = [f"{ctr.get(b, 0):>5}" for b in buckets]
        print(f"{cat:<38} " + " ".join(row_strs) + f"  {row_total:>5}")

    if sparse_rows:
        print(f"\n=== {len(sparse_rows)} rows with gold_paper_count < 5 ===")
        print("(consider marking keep=N or relaxing constraints during curation)")
        for r in sparse_rows[:25]:
            print(f"  [{r['gold_count']:>2}] {r['category']:<28} {r['id']:<40} {r['query']}")
        if len(sparse_rows) > 25:
            print(f"  ... {len(sparse_rows) - 25} more")
    else:
        print("\n✓ every row has at least 5 gold papers")

    # Atomic-ish write
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.in_path.resolve() == args.out_path.resolve():
        backup = args.out_path.with_suffix(args.out_path.suffix + ".precount.bak")
        shutil.copy2(args.in_path, backup)
        print(f"\n# backed up original to {backup}")

    tmp = args.out_path.with_suffix(args.out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.out_path)
    print(f"# wrote {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
