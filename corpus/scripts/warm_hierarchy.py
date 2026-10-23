"""Warm the ROR hierarchy cache for any gold/pred ROR ids missing from it.

Reads the base ror_hierarchy_cache.json, finds every ROR id used by gold or by
the target pipeline predictions in merged_per_paper.jsonl that is absent from the
cache, fetches its parent/child/related relationships from the ROR v2 org-by-id
endpoint, and writes a MERGED cache to a NEW file (base is left untouched).

This makes the offline hierarchy-aware F1 match the pipeline's RorResolver-backed
numbers exactly (the resolver fetches missing relationships on demand; offline we
must pre-fetch them).

Usage:
    python scripts/warm_hierarchy.py
    # then re-run emit_task2_counts / relink_strategy_compare with
    #   --hierarchy-cache reports/ror_hierarchy_cache.warmed.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from relink_strategy_compare import norm_ror  # noqa: E402

ORG_BY_ID = "https://api.ror.org/v2/organizations/"


def fetch_rel(bare: str, timeout: int):
    for attempt in range(6):
        try:
            req = urllib.request.Request(ORG_BY_ID + bare,
                                         headers={"User-Agent": "acl-warm-hier/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                org = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 or 500 <= e.code < 600:
                time.sleep(2 ** attempt); continue
            return {"parent": [], "child": [], "related": []}
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt); continue
    else:
        return {"parent": [], "child": [], "related": []}
    buckets = {"parent": [], "child": [], "related": []}
    for rel in (org.get("relationships") or []):
        rtype = (rel.get("type") or "").lower()
        rid = norm_ror(rel.get("id") or "")
        if rid and rtype in buckets:
            buckets[rtype].append(rid)
    return buckets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", type=Path,
                    default=REPO / "reports/pipeline_comparison/merged_per_paper.jsonl")
    ap.add_argument("--base", type=Path, default=REPO / "reports/ror_hierarchy_cache.json")
    ap.add_argument("--pipelines", nargs="+",
                    default=["gemini25_combined", "claude_sonnet_combined", "gpt5nano_combined"])
    ap.add_argument("--out", type=Path, default=REPO / "reports/ror_hierarchy_cache.warmed.json")
    ap.add_argument("--rate-hz", type=float, default=5.0)
    ap.add_argument("--timeout", type=int, default=20)
    args = ap.parse_args()

    hier = json.loads(args.base.read_text()) if args.base.exists() else {}
    ids = set()
    for line in args.merged.open():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        for x in ((d.get("gold") or {}).get("ror_ids") or []):
            b = norm_ror(x)
            if b:
                ids.add(b)
        for p in args.pipelines:
            e = (d.get("predictions") or {}).get(p) or {}
            if isinstance(e, dict) and not e.get("error"):
                for x in (e.get("ror_ids") or []):
                    b = norm_ror(x)
                    if b:
                        ids.add(b)
    missing = sorted(i for i in ids if i not in hier)
    print(f"# {len(ids)} unique ids | {len(missing)} missing from base cache", flush=True)
    delay = 1.0 / args.rate_hz if args.rate_hz > 0 else 0.0
    for i, b in enumerate(missing):
        hier[b] = fetch_rel(b, args.timeout)
        if delay:
            time.sleep(delay)
        if (i + 1) % 25 == 0:
            print(f"# warmed {i+1}/{len(missing)}", flush=True)
    args.out.write_text(json.dumps(hier, ensure_ascii=False))
    print(f"# wrote merged hierarchy cache ({len(hier)} entries) -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
