"""Year/decade distribution of S2ORC fulltext coverage over the ACL corpus.

For each paper in <s2orc_normalized>/affiliations/, derive its year from
the ACL Anthology ID (modern `2024.acl-long.123` or legacy `P12-1010`
formats) and report:

  - per-decade total ACL papers / # with S2ORC fulltext / % coverage
  - per-year totals for recent decades (2010+)

This tells us whether S2ORC's missing fulltext is concentrated in
specific eras (likely pre-2010, where S2 didn't always ingest scanned
proceedings) — which determines whether the `s2orc` row in the
comparison harness is a recent-papers-only number vs a corpus-wide one.

Usage:
    python scripts/s2orc_year_distribution.py
    python scripts/s2orc_year_distribution.py \\
        --norm-dir /share/.../s2orc_normalized \\
        --s2orc-dir /share/.../s2orc_raw/s2orc
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

_OLD_ID = re.compile(r"^[A-Z](\d\d)-\d+")


def year_from_id(pid: str) -> int | None:
    """Modern: '2024.acl-long.123' -> 2024.
    Legacy: 'P12-1010' -> 2012, 'W94-0001' -> 1994, 'C65-1001' -> 1965.

    Legacy ACL IDs use 2-digit years. Real legacy NN values are either
    65-99 (1965-1999) or 00-19 (2000-2019); the new format took over
    in 2020. So: NN >= 65 → 19NN, otherwise 20NN."""
    if not pid:
        return None
    if pid[:4].isdigit():
        return int(pid[:4])
    m = _OLD_ID.match(pid)
    if m:
        nn = int(m.group(1))
        return 1900 + nn if nn >= 65 else 2000 + nn
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--norm-dir", type=Path,
                    default=Path("/share/moritz.staudinger/acl_data/s2orc_normalized"),
                    help="Directory with affiliations/<acl_id>.json from s2orc_normalize.py")
    ap.add_argument("--s2orc-dir", type=Path,
                    default=Path("/share/moritz.staudinger/acl_data/s2orc_raw/s2orc"),
                    help="Directory with raw S2ORC <acl_id>.json fulltext records")
    args = ap.parse_args()

    aff_dir = args.norm_dir / "affiliations"
    if not aff_dir.is_dir():
        raise SystemExit(f"missing {aff_dir}")
    if not args.s2orc_dir.is_dir():
        raise SystemExit(f"missing {args.s2orc_dir}")

    s2orc_ids = {p.stem for p in args.s2orc_dir.glob("*.json")}
    print(f"S2ORC fulltext records found: {len(s2orc_ids):,}\n")

    by_decade_total = Counter()
    by_decade_ft = Counter()
    by_year_total = Counter()
    by_year_ft = Counter()
    no_year = 0

    for f in aff_dir.glob("*.json"):
        pid = f.stem
        y = year_from_id(pid)
        if y is None:
            no_year += 1
            continue
        dec = f"{y // 10 * 10}s"
        by_decade_total[dec] += 1
        by_year_total[y] += 1
        if pid in s2orc_ids:
            by_decade_ft[dec] += 1
            by_year_ft[y] += 1

    print(f"{'decade':<8} {'total':>7} {'fulltext':>9} {'coverage':>9}")
    print("-" * 36)
    for dec in sorted(by_decade_total):
        tot = by_decade_total[dec]
        ft = by_decade_ft[dec]
        pct = 100 * ft / tot if tot else 0
        print(f"{dec:<8} {tot:>7,} {ft:>9,} {pct:>8.1f}%")
    print()
    print(f"papers with un-parseable year: {no_year}")

    print()
    print("Year-by-year for 2010+:")
    print(f"{'year':<6} {'total':>7} {'fulltext':>9} {'coverage':>9}")
    print("-" * 34)
    for y in sorted(by_year_total):
        if y < 2010:
            continue
        tot = by_year_total[y]
        ft = by_year_ft[y]
        pct = 100 * ft / tot if tot else 0
        print(f"{y:<6} {tot:>7,} {ft:>9,} {pct:>8.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
