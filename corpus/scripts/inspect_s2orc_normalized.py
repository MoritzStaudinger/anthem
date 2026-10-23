"""Diagnose why s2orc predictions are empty.

Prints, for a handful of representative papers (recent + old), what
the normalised JSON actually contains: paper_id, affiliations list,
provenance (s2orc vs s2_papers_only), authors count.

The aff_compare pipeline reads `affiliations` from these files; if
it's always empty, this tells us why — either normalize never wrote
the field, or the source paper had no S2ORC fulltext, or normalize
ran on the wrong input dir.

Usage:
    python scripts/inspect_s2orc_normalized.py
    python scripts/inspect_s2orc_normalized.py \\
        --norm-dir /share/moritz.staudinger/acl_data/s2orc_normalized
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--norm-dir", type=Path,
                   default=Path("/share/moritz.staudinger/acl_data/s2orc_normalized"))
    args = p.parse_args()

    aff_dir = args.norm_dir / "affiliations"
    if not aff_dir.is_dir():
        aff_dir = args.norm_dir
    if not aff_dir.is_dir():
        raise SystemExit(f"missing {aff_dir}")

    all_files = sorted(aff_dir.glob("*.json"))
    print(f"total normalized JSONs: {len(all_files):,}")
    print(f"sample dir: {aff_dir}")
    print()

    # Aggregate over ALL files: how many have non-empty affiliations?
    n_total = 0
    n_with_affs = 0
    by_source = Counter()
    for fp in all_files:
        try:
            d = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        n_total += 1
        if d.get("affiliations"):
            n_with_affs += 1
        src = ((d.get("_provenance") or {}).get("source")) or "(unknown)"
        by_source[src] += 1

    print("=" * 60)
    print(f"files scanned       : {n_total:,}")
    print(f"with non-empty affs : {n_with_affs:,}  ({100 * n_with_affs / n_total:.1f}%)")
    print()
    print("breakdown by _provenance.source:")
    for s, n in by_source.most_common():
        print(f"  {s:<24}{n:>8}")
    print()

    # Show 3 modern + 3 old per-paper samples
    def _sample(prefix_glob: str, n: int = 3):
        for fp in sorted(aff_dir.glob(prefix_glob))[:n]:
            try:
                d = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
            except json.JSONDecodeError:
                continue
            src = ((d.get("_provenance") or {}).get("source")) or "?"
            affs = d.get("affiliations") or []
            n_authors = len(d.get("authors") or [])
            preview = (affs[0] if affs else "(empty)")
            if isinstance(preview, str) and len(preview) > 70:
                preview = preview[:70] + "…"
            print(f"  {fp.stem:<32}  src={src:<18} "
                  f"n_authors={n_authors:>3}  n_affs={len(affs):>3}  "
                  f"first_aff={preview!r}")

    print("=" * 60)
    print("modern sample (2024.acl-*):")
    _sample("2024.acl-long.*.json", 3)
    _sample("2024.acl-main.*.json", 3)
    print()
    print("modern sample (2023.emnlp-*):")
    _sample("2023.emnlp-main.*.json", 3)
    print()
    print("older sample (P19-*):")
    _sample("P19-*.json", 3)
    print()
    print("very old sample (2001.* / W94-*):")
    _sample("2001.*.json", 3)
    _sample("W94-*.json", 3)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
