"""Dump the full top-level structure of a few S2ORC records to find
where the annotations actually live in this release.

Hypothesis under test: in newer S2ORC releases, content.annotations is
deprecated or relocated. The 97% of files with empty annotations might
have the data under a different key — `parsed`, `unified_annotations`,
`s2orc_metadata`, or somewhere else entirely.

Usage:
    python scripts/inspect_s2orc_raw_full.py
    python scripts/inspect_s2orc_raw_full.py --n 5
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _shape_summary(v, depth=0, max_depth=3) -> str:
    """Short human-readable type/size of a value, recursing into dicts/lists."""
    if depth >= max_depth:
        return f"{type(v).__name__}(…)"
    if isinstance(v, dict):
        keys = list(v.keys())
        if len(keys) > 8:
            keys = keys[:8] + ["..."]
        return f"dict({len(v)} keys: {keys})"
    if isinstance(v, list):
        return f"list[{len(v)}]" + (
            f" first={_shape_summary(v[0], depth+1, max_depth)}" if v else ""
        )
    if isinstance(v, str):
        return f"str(len={len(v)})" + (
            f" preview={v[:80]!r}" if len(v) < 200 else ""
        )
    if v is None:
        return "None"
    return f"{type(v).__name__}={v!r}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--s2orc-dir", type=Path,
                   default=Path("/share/moritz.staudinger/acl_data/s2orc_raw/s2orc"))
    p.add_argument("--n", type=int, default=4,
                   help="How many records to dump (mix of with/without annotations).")
    args = p.parse_args()

    files = sorted(args.s2orc_dir.glob("*.json"))
    print(f"raw s2orc files: {len(files):,}\n")

    # Categorize files by whether content.annotations is populated
    with_anns: list = []
    without_anns: list = []
    random.seed(42)
    random.shuffle(files)
    for fp in files:
        try:
            d = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        anns = (d.get("content") or {}).get("annotations") or {}
        if anns and any(anns.values()):
            with_anns.append((fp, d))
        else:
            without_anns.append((fp, d))
        if len(with_anns) >= args.n and len(without_anns) >= args.n:
            break

    def _dump(label, items):
        for fp, d in items[:args.n]:
            print("=" * 78)
            print(f"{label}: {fp.name}")
            print("=" * 78)
            # Top-level structure
            print(f"top-level keys ({len(d)}):")
            for k in sorted(d.keys()):
                print(f"  {k:<24} {_shape_summary(d[k], depth=0, max_depth=2)}")
            # Show content.* in detail
            content = d.get("content") or {}
            if isinstance(content, dict):
                print(f"content.* ({len(content)} keys):")
                for k in sorted(content.keys()):
                    print(f"  content.{k:<18} {_shape_summary(content[k], depth=0, max_depth=2)}")
            print()

    print("\n" + "#" * 78)
    print("# RECORDS WITH non-empty content.annotations (expected schema)")
    print("#" * 78 + "\n")
    _dump("annotated", with_anns)

    print("\n" + "#" * 78)
    print("# RECORDS WITHOUT content.annotations (the 97% case — what's there?)")
    print("#" * 78 + "\n")
    _dump("unannotated", without_anns)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
