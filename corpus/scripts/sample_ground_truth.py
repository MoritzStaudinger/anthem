#!/usr/bin/env python3
"""
Sample random ACL Anthology papers per year (e.g., 2021-2025) reproducibly via a seed.

Supports:
  1) Local files in a directory (recursively). Example filenames may include:
       2021.acl-short.1.pdf
       2021.emnlp-main.1.pdf
       https___aclanthology_org_2021_acl-short_1_.pdf
  2) A text file with one URL/ID per line:
       https://aclanthology.org/2021.acl-short.1/
       2021.emnlp-main.1

Outputs:
  - CSV of sampled items (year, anthology_id, source)
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# Modern ACL Anthology IDs (2020+): "2021.acl-short.1", "2021.emnlp-main.123"
ANTHOLOGY_ID_RE = re.compile(r"\b(20\d{2})\.([a-z0-9-]+)\.(\d+)\b", re.IGNORECASE)
# Legacy ACL Anthology IDs (pre-2020): "P80-1001", "W99-0101", "J96-1001"
LEGACY_ID_RE = re.compile(r"\b([A-Z])(\d{2})-(\d{4})\b")


def _legacy_year(yy: int) -> int:
    # ACL Anthology started in 1965; map 65-99 -> 19xx, 00-19 -> 20xx.
    return 1900 + yy if yy >= 60 else 2000 + yy


@dataclass(frozen=True)
class Item:
    year: int
    anthology_id: str
    source: str  # file path or URL/line


def extract_anthology_id(text: str) -> Optional[tuple[str, int]]:
    """
    Extracts the first ACL Anthology ID from a string.
    Returns (normalized_id, year) or None if not found.
    Modern: "2021.acl-short.1" -> ("2021.acl-short.1", 2021)
    Legacy: "P80-1001"         -> ("P80-1001", 1980)
    """
    m = ANTHOLOGY_ID_RE.search(text)
    if m:
        year, venue, number = m.group(1), m.group(2), m.group(3)
        return f"{year}.{venue}.{number}", int(year)
    m = LEGACY_ID_RE.search(text)
    if m:
        prefix, yy, number = m.group(1), int(m.group(2)), m.group(3)
        return f"{prefix}{m.group(2)}-{number}", _legacy_year(yy)
    return None


def iter_files(root: Path, exts: set[str]) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and (not exts or p.suffix.lower().lstrip(".") in exts):
            yield p


def load_from_directory(root: Path, exts: set[str]) -> list[Item]:
    items: list[Item] = []
    for f in iter_files(root, exts):
        parsed = extract_anthology_id(str(f))
        if not parsed:
            continue
        aid, year = parsed
        items.append(Item(year=year, anthology_id=aid, source=str(f)))
    return items


def load_from_textfile(path: Path) -> list[Item]:
    items: list[Item] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parsed = extract_anthology_id(line)
        if not parsed:
            continue
        aid, year = parsed
        items.append(Item(year=year, anthology_id=aid, source=f"{path}:{line_no}:{line}"))
    return items


def sample_per_year(
    items: list[Item],
    years: list[int],
    k: int,
    seed: int,
    k_early: Optional[int] = None,
    k_early_cutoff: Optional[int] = None,
) -> list[Item]:
    rng = random.Random(seed)
    sampled: list[Item] = []

    # Dedupe by anthology_id (keep first source)
    seen: set[str] = set()
    deduped: list[Item] = []
    for it in items:
        if it.anthology_id in seen:
            continue
        seen.add(it.anthology_id)
        deduped.append(it)

    for y in years:
        k_y = k_early if (k_early is not None and k_early_cutoff is not None and y < k_early_cutoff) else k
        pool = [it for it in deduped if it.year == y]
        if not pool:
            print(f"[WARN] Year {y}: no matching items found.")
            continue
        if len(pool) < k_y:
            print(f"[WARN] Year {y}: only {len(pool)} items available; sampling all of them.")
            sampled.extend(pool)
        else:
            sampled.extend(rng.sample(pool, k_y))

    # Sort output for readability
    return sorted(sampled, key=lambda x: (x.year, x.anthology_id))


def write_csv(out_path: Path, sampled: list[Item]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "anthology_id", "source", "url"])
        for it in sampled:
            url = f"https://aclanthology.org/{it.anthology_id}/"
            w.writerow([it.year, it.anthology_id, it.source, url])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reproducibly sample random ACL Anthology papers per year (e.g., 2021–2025)."
    )
    ap.add_argument(
        "--dir",
        type=Path,
        help="Directory to scan recursively for files whose path/name contains ACL Anthology IDs.",
    )
    ap.add_argument(
        "--urls",
        type=Path,
        help="Text file containing one ACL Anthology URL or ID per line.",
    )
    ap.add_argument("--seed", type=int, required=True, help="Random seed for reproducible sampling.")
    ap.add_argument("--k", type=int, default=20, help="Number of papers to sample per year (default: 20).")
    ap.add_argument(
        "--k-early",
        type=int,
        default=None,
        help="Alternate k for years strictly before --k-early-cutoff (e.g., 10 for pre-2000).",
    )
    ap.add_argument(
        "--k-early-cutoff",
        type=int,
        default=None,
        help="Year cutoff (exclusive). Years strictly less than this use --k-early instead of --k.",
    )
    ap.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(range(1980, 2026)),
        help="Years to sample (default: 1980..2025).",
    )
    ap.add_argument(
        "--ext",
        type=str,
        default="pdf",
        help="File extension(s) to include when scanning --dir, comma-separated (default: pdf). Use '*' for all.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("sampled_acl_papers.csv"),
        help="Output CSV path (default: sampled_acl_papers.csv).",
    )

    args = ap.parse_args()

    if not args.dir and not args.urls:
        ap.error("Provide at least one of --dir or --urls.")

    exts: set[str]
    if args.ext.strip() == "*" or args.ext.strip() == "":
        exts = set()
    else:
        exts = {e.strip().lower().lstrip(".") for e in args.ext.split(",") if e.strip()}

    items: list[Item] = []
    if args.dir:
        items.extend(load_from_directory(args.dir, exts))
    if args.urls:
        items.extend(load_from_textfile(args.urls))

    if not items:
        raise SystemExit(
            "No ACL Anthology IDs found. Ensure your filenames/paths or URL lines contain patterns like "
            "'2021.acl-short.1' or '2021.emnlp-main.1'."
        )

    if (args.k_early is None) != (args.k_early_cutoff is None):
        ap.error("--k-early and --k-early-cutoff must be used together.")

    sampled = sample_per_year(
        items, args.years, args.k, args.seed,
        k_early=args.k_early, k_early_cutoff=args.k_early_cutoff,
    )
    write_csv(args.out, sampled)

    # Print a short summary
    counts = {y: 0 for y in args.years}
    for it in sampled:
        if it.year in counts:
            counts[it.year] += 1

    print("\nSampled counts:")
    for y in args.years:
        print(f"  {y}: {counts[y]}")
    print(f"\nWrote: {args.out.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
