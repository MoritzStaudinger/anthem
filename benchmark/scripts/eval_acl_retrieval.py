"""Evaluate the local ACL retrieval tool against a small testset.

This is the deterministic, free, retrieval-only tier — no LLM calls. For
each row in --testset (JSONL), call search_local_acl_papers(query, ...)
and compute recall@k against the expected_paper_ids list. Reports per-row
hits and an aggregate mean recall@k.

Schema (one JSON object per line):
    {
      "id": "acl-001",
      "query": "<natural language question>",
      "expected_paper_ids": ["2024.acl-long.123", "..."],
      "year_start": 2020,            // optional
      "year_end": 2024,              // optional
      "institution": "TU Wien",      // optional
      "tags": ["..."],               // optional
      "notes": "..."                 // optional
    }

Rows with empty expected_paper_ids are skipped from recall aggregation
(they print as "[no gold] retrieved N papers"), so you can check what the
tool returns while you're still curating gold labels.

Examples
--------
    python scripts/eval_acl_retrieval.py
    python scripts/eval_acl_retrieval.py --testset data/eval/acl_testset.jsonl --k 15
    python scripts/eval_acl_retrieval.py --k 5 --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.rag.agent_tools import (  # noqa: E402
    reset_evidence_buffer,
    search_local_acl_papers,
)


def _load_testset(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def _retrieved_ids(records: List[Dict[str, Any]]) -> List[str]:
    out = []
    for r in records:
        pid = r.get("paper_id") or ""
        if not pid and r.get("title"):
            pid = r["title"]  # fallback if metadata had no paper_id
        if pid:
            out.append(pid)
    return out


def _recall(retrieved: List[str], expected: List[str]) -> float:
    if not expected:
        return float("nan")
    expected_set = set(expected)
    hits = sum(1 for pid in retrieved if pid in expected_set)
    return hits / len(expected_set)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--testset",
        default="data/eval/acl_testset.jsonl",
        help="JSONL test cases (default: data/eval/acl_testset.jsonl).",
    )
    p.add_argument("--k", type=int, default=15, help="Top-k papers to retrieve (default: 15).")
    p.add_argument("-v", "--verbose", action="store_true", help="Print every retrieved paper id.")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    testset_path = Path(args.testset)
    if not testset_path.exists():
        print(f"Testset not found: {testset_path}", file=sys.stderr)
        return 2

    rows = _load_testset(testset_path)
    if not rows:
        print(f"Empty testset: {testset_path}", file=sys.stderr)
        return 2

    print(f"# ACL retrieval eval — {len(rows)} cases, k={args.k}\n")

    recalls: List[float] = []
    no_gold = 0
    for row in rows:
        rid = row.get("id", "?")
        query = row.get("query", "")
        expected = row.get("expected_paper_ids") or []

        reset_evidence_buffer()
        records = search_local_acl_papers(
            query=query,
            year_start=row.get("year_start"),
            year_end=row.get("year_end"),
            institution=row.get("institution"),
            samples=args.k,
        )

        if records and records[0].get("error"):
            print(f"[{rid}] ERROR: {records[0]['error']}")
            return 1

        retrieved = _retrieved_ids(records)

        if not expected:
            no_gold += 1
            print(f"[{rid}] no gold — retrieved {len(retrieved)} papers")
            if args.verbose:
                for pid in retrieved:
                    print(f"    {pid}")
            continue

        r = _recall(retrieved, expected)
        recalls.append(r)
        hits = sum(1 for pid in retrieved if pid in set(expected))
        print(f"[{rid}] recall@{args.k} = {r:.2f}  ({hits}/{len(expected)} hits)")
        if args.verbose:
            expected_set = set(expected)
            for pid in retrieved:
                marker = "✓" if pid in expected_set else " "
                print(f"    {marker} {pid}")
            missing = expected_set - set(retrieved)
            for pid in missing:
                print(f"    ✗ {pid} (missing)")

    print()
    if recalls:
        mean = sum(recalls) / len(recalls)
        print(f"# mean recall@{args.k}: {mean:.3f} over {len(recalls)} graded cases")
    if no_gold:
        print(f"# {no_gold} case(s) had no gold labels — fill expected_paper_ids in {testset_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
