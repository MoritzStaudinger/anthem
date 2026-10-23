"""Diagnose why _extract_affiliations_from_s2orc returns empty on
98% of fulltext papers.

Reads a sample of raw S2ORC records (output of s2orc_download.py s2orc)
and reports:
  - which annotation fields are populated per record
  - what type each annotation is (list of dicts vs JSON string vs None)
  - whether `authoraffiliation` specifically is present and non-empty
  - aggregate stats across the whole s2orc_raw/s2orc directory

If `authoraffiliation` is mostly empty but `author` is populated, S2ORC
didn't parse affiliations for those papers (data-side gap). If
`authoraffiliation` exists under a different key name, we have a
schema-mismatch bug to fix in s2orc_normalize.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def _ann_shape(ann: Any) -> str:
    """Describe what an annotation value actually is."""
    if ann is None:
        return "None"
    if isinstance(ann, str):
        try:
            parsed = json.loads(ann)
        except json.JSONDecodeError:
            return f"str(len={len(ann)}, not-json)"
        if isinstance(parsed, list):
            return f"str→list[{len(parsed)}]"
        return f"str→{type(parsed).__name__}"
    if isinstance(ann, list):
        return f"list[{len(ann)}]"
    return type(ann).__name__


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--s2orc-dir", type=Path,
                   default=Path("/share/moritz.staudinger/acl_data/s2orc_raw/s2orc"))
    p.add_argument("--sample", type=int, default=5,
                   help="How many sample records to dump in detail.")
    args = p.parse_args()

    files = sorted(args.s2orc_dir.glob("*.json"))
    print(f"raw s2orc files: {len(files):,}")
    print(f"dir: {args.s2orc_dir}")
    print()

    # ---- Aggregate: which annotation fields are populated, how often ----
    ann_key_present = Counter()
    ann_key_nonempty = Counter()
    n_total = 0
    for fp in files:
        try:
            d = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        n_total += 1
        content = d.get("content") or {}
        anns = content.get("annotations") or {}
        if not isinstance(anns, dict):
            continue
        for k, v in anns.items():
            ann_key_present[k] += 1
            # "non-empty" = parseable list-of-dicts with len>0
            non_empty = False
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    non_empty = isinstance(parsed, list) and len(parsed) > 0
                except json.JSONDecodeError:
                    pass
            elif isinstance(v, list):
                non_empty = len(v) > 0
            if non_empty:
                ann_key_nonempty[k] += 1

    print("=" * 70)
    print(f"scanned {n_total} records")
    print()
    print(f"{'annotation key':<28}{'present':>10}{'non-empty':>12}{'% non-empty':>14}")
    print("-" * 64)
    for k in sorted(ann_key_present, key=lambda x: -ann_key_present[x]):
        p_count = ann_key_present[k]
        ne_count = ann_key_nonempty[k]
        pct = (100 * ne_count / n_total) if n_total else 0
        print(f"  {k:<26}{p_count:>10}{ne_count:>12}{pct:>13.1f}%")
    print()

    # ---- Detailed sample dumps ----
    random.seed(0)
    sample = random.sample(files, min(args.sample, len(files)))
    for fp in sample:
        print("=" * 70)
        print(f"FILE: {fp.name}")
        try:
            d = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError as e:
            print(f"  ERROR: {e}")
            continue
        cid = d.get("corpusid") or d.get("corpusId")
        ext = d.get("externalids") or d.get("externalIds") or {}
        acl_id = ext.get("ACL")
        title_ann = (d.get("content") or {}).get("annotations", {}).get("title")
        print(f"  corpusid={cid}  acl={acl_id}  title_ann={_ann_shape(title_ann)}")

        anns = (d.get("content") or {}).get("annotations") or {}
        if not isinstance(anns, dict):
            print(f"  content.annotations is not a dict — type={type(anns).__name__}")
            continue
        print("  annotations:")
        for k, v in anns.items():
            print(f"    {k:<22} {_ann_shape(v)}")

        # If authoraffiliation looks populated, show first span resolved
        aa = anns.get("authoraffiliation")
        text = (d.get("content") or {}).get("text") or ""
        if isinstance(aa, str):
            try:
                aa = json.loads(aa)
            except json.JSONDecodeError:
                aa = []
        if isinstance(aa, list) and aa:
            print(f"  first authoraffiliation span text:")
            for a in aa[:3]:
                try:
                    s, e = int(a["start"]), int(a["end"])
                    print(f"    [{s}:{e}] {text[s:e]!r}")
                except (KeyError, TypeError, ValueError):
                    print(f"    (bad span: {a!r})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
