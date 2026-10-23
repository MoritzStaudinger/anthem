
"""Report CSO label coverage over an enriched SciCo corpus.

For every distinct CSO label that appears in any paper's `topics.cso.union`,
counts the number of papers tagged with it. Sorted by frequency.

Also prints summary stats:
  - papers with abstract / no abstract
  - papers with ≥1 CSO label / 0 CSO labels
  - distribution of CSO labels per paper
  - per-decade coverage

Usage:
    python scripts/topic_coverage_report.py \\
        --affiliations-dir <enriched-dir> \\
        --out reports/coverage.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--affiliations-dir", type=Path, required=True,
                   help="Directory of enriched per-paper JSONs.")
    p.add_argument("--out", type=Path, default=None,
                   help="Optional JSONL output: one line per CSO label with count + sample paper_ids.")
    p.add_argument("--top", type=int, default=100,
                   help="Print the top N labels by count (default 100). All labels still written to --out.")
    p.add_argument("--min-count", type=int, default=1,
                   help="Hide labels with fewer than this many hits from the printed table.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    files = sorted(args.affiliations_dir.glob("*.json"))
    if not files:
        logger.error("no JSONs under %s", args.affiliations_dir)
        return 2
    logger.info("scanning %d JSONs", len(files))

    label_counts: Counter = Counter()
    label_papers: Dict[str, list] = defaultdict(list)
    union_size_dist: Counter = Counter()
    by_decade: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"n": 0, "with_abstract": 0, "with_topics": 0}
    )
    n_total = 0
    n_no_abstract = 0
    n_no_topics = 0
    n_with_topics = 0

    for fp in files:
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        n_total += 1
        abstract_ok = bool((d.get("abstract") or "").strip())
        if not abstract_ok:
            n_no_abstract += 1
        union = (d.get("topics") or {}).get("cso", {}).get("union") or []
        union = [str(x).lower() for x in union]
        union_size_dist[len(union)] += 1
        if union:
            n_with_topics += 1
            pid = d.get("paper_id") or fp.stem
            for lab in set(union):
                label_counts[lab] += 1
                if len(label_papers[lab]) < 5:
                    label_papers[lab].append(pid)
        else:
            n_no_topics += 1
        # Per-decade
        y = d.get("year")
        if isinstance(y, int):
            decade = (y // 10) * 10
            by_decade[decade]["n"] += 1
            if abstract_ok:
                by_decade[decade]["with_abstract"] += 1
            if union:
                by_decade[decade]["with_topics"] += 1

    # ---- Print summary -------------------------------------------------
    print()
    print(f"=== Summary ({n_total} papers) ===")
    print(f"  Papers with no abstract:    {n_no_abstract}  ({100*n_no_abstract/n_total:.1f}%)")
    print(f"  Papers with ≥1 CSO label:   {n_with_topics}  ({100*n_with_topics/n_total:.1f}%)")
    print(f"  Papers with 0 CSO labels:   {n_no_topics}  ({100*n_no_topics/n_total:.1f}%)")
    print()

    print("Union-size distribution (papers with N CSO labels):")
    for n in sorted(union_size_dist):
        c = union_size_dist[n]
        bar = "█" * min(50, c // max(1, n_total // 500))
        print(f"  {n:3d} labels: {c:7d}  {bar}")
    print()

    print(f"Top {args.top} CSO labels by paper count (≥ {args.min_count}):")
    print(f"{'count':>8}  {'pct':>6}  CSO label")
    print("-" * 70)
    shown = 0
    for lab, cnt in label_counts.most_common():
        if cnt < args.min_count:
            break
        if shown >= args.top:
            break
        pct = 100 * cnt / n_total
        print(f"{cnt:>8}  {pct:>5.1f}%  {lab}")
        shown += 1
    print(f"\n(total distinct CSO labels: {len(label_counts)})")
    print()

    print("Per-decade coverage:")
    print(f"  {'decade':<8} {'n':>8} {'w/abstract':>12} {'w/topics':>10} {'topic_cov%':>11}")
    for decade in sorted(by_decade):
        s = by_decade[decade]
        cov = 100 * s["with_topics"] / s["n"] if s["n"] else 0
        print(f"  {decade}s    {s['n']:>8} {s['with_abstract']:>12} {s['with_topics']:>10} {cov:>10.1f}%")

    # ---- Write detailed JSONL -----------------------------------------
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as fh:
            for lab, cnt in label_counts.most_common():
                fh.write(json.dumps({
                    "cso_label": lab,
                    "n_papers": cnt,
                    "sample_paper_ids": label_papers[lab],
                }, ensure_ascii=False) + "\n")
        logger.info("wrote %d label entries to %s", len(label_counts), args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
