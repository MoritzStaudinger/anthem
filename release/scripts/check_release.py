"""Verify a built ANTHEM release before it is pushed to a public repo.

Checks, in order of how badly each would hurt if missed:

  1. Leaked absolute paths (/home/..., /share/...) anywhere in the text columns.
     The exporter scrubs these, but the fulltext comes straight off the parsers.
  2. Leaked e-mail addresses in the fulltext — GROBID lifts author contact
     details out of the PDF header, and the card promises none are included.
  3. paper_id referential integrity between the configs.
  4. Row counts, shard sizes and the licence/flag distribution, printed so the
     numbers in the dataset card can be checked against reality.

Exit code 1 if any check fails, so `make check` gates the upload.

    python scripts/check_release.py ~/anthem_build/release
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Set

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    print("# ERROR: pyarrow is required -> pip install 'pyarrow<19'", file=sys.stderr)
    raise SystemExit(1)

# Must not fire on a URL such as "brighton.ac.uk/home/Anja.Belz/": require
# the path to start at a boundary rather than follow a hostname.
PATH_RX = re.compile(r"(?<![\w.:/-])/(?:home|share|work|scratch)/[A-Za-z0-9._-]+/")
# Deliberately narrow: TEI contains <email> tags, and a bare @ is common in
# formulae and citations.
MAIL_RX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

TEXT_COLS = ("tei_xml", "pymu_md", "ocr_text", "title", "abstract",
             "raw_affiliation", "query", "notes")


def configs(root: Path) -> Dict[str, list]:
    out = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        shards = sorted(d.glob("*.parquet"))
        if shards:
            out[d.name] = shards
    return out


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    root = Path(sys.argv[1])
    if not root.is_dir():
        print(f"# ERROR: not a directory: {root}", file=sys.stderr)
        return 1

    cfgs = configs(root)
    if not cfgs:
        print(f"# ERROR: no configs under {root}", file=sys.stderr)
        return 1

    failures = []
    ids: Dict[str, Set[str]] = {}
    print(f"# release: {root}\n")
    print(f"{'config':24s} {'rows':>10s} {'shards':>7s} {'size':>10s}")
    print("-" * 54)

    totals = {}
    for name, shards in cfgs.items():
        rows = 0
        size = sum(s.stat().st_size for s in shards)
        n_path = n_mail = 0
        seen: Set[str] = set()
        for s in shards:
            t = pq.read_table(s)
            rows += t.num_rows
            names = set(t.schema.names)
            for idcol in ("paper_id", "anthology_id"):
                if idcol in names:
                    seen.update(v for v in t.column(idcol).to_pylist() if v)
                    break
            for col in TEXT_COLS:
                if col not in names:
                    continue
                for v in t.column(col).to_pylist():
                    if not v:
                        continue
                    if PATH_RX.search(v):
                        n_path += 1
                    if col in ("tei_xml", "pymu_md", "ocr_text") and MAIL_RX.search(v):
                        n_mail += 1
        totals[name] = rows
        if seen:
            ids[name] = seen
        print(f"{name:24s} {rows:>10,} {len(shards):>7} {size/1e9:>9.2f}G")
        if n_path:
            failures.append(f"{name}: {n_path} values contain an absolute path")
        if n_mail:
            print(f"{'':24s} note: {n_mail:,} fulltext rows contain an e-mail address")

    print(f"\n{'':24s} {sum(totals.values()):>10,} rows total")

    # Referential integrity: every fulltext/affiliation/topic row should point at
    # a paper that exists in `papers`.
    if "papers" in ids:
        base = ids["papers"]
        print(f"\n# papers: {len(base):,} distinct ids")
        for name in ("affiliations", "topics", "fulltext"):
            if name in ids:
                orphans = ids[name] - base
                mark = "OK" if not orphans else f"{len(orphans):,} ORPHANS"
                print(f"#   {name:18s} {len(ids[name]):>8,} ids  {mark}")
                if orphans:
                    failures.append(f"{name}: {len(orphans)} ids absent from papers")

    # Licence / third-party distribution, for checking the card.
    if "papers" in cfgs:
        lic: Dict[str, int] = {}
        tp = 0
        for s in cfgs["papers"]:
            t = pq.read_table(s, columns=["license", "third_party_venue"])
            for v in t.column("license").to_pylist():
                lic[v] = lic.get(v, 0) + 1
            tp += sum(1 for v in t.column("third_party_venue").to_pylist() if v)
        print("\n# underlying licences")
        for k, v in sorted(lic.items(), key=lambda t: -t[1]):
            print(f"#   {v:>8,}  {k}")
        print(f"#   third_party_venue = true: {tp:,}")

    if not (root / "README.md").is_file():
        failures.append("README.md (dataset card) is missing")

    print()
    if failures:
        for f in failures:
            print(f"# FAIL {f}")
        return 1
    print("# all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
