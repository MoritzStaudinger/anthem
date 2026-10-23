"""Print just the question + answer pairs from an answers.jsonl, no
records noise. Defaults to first 3 rows; pass --all for everything.

Usage:
    python scripts/show_answers.py <answers_dir>                 # first 3
    python scripts/show_answers.py <answers_dir> --all           # all rows
    python scripts/show_answers.py <answers_dir> --id <row_id>   # one row
    python scripts/show_answers.py <answers_dir> --limit 10      # first N
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=3)
    ap.add_argument("--id", default=None, help="Show only this row id")
    ap.add_argument("--testset", type=Path,
                    default=Path("data/eval/scico_testset.jsonl"),
                    help="Used to recover the question text")
    args = ap.parse_args()

    f = args.dir / "answers.jsonl"
    if not f.is_file():
        print(f"ERROR: {f} not found", file=sys.stderr)
        return 1

    # Load questions from testset
    questions = {}
    if args.testset.is_file():
        with args.testset.open() as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if r.get("id"):
                        questions[r["id"]] = r.get("query", "")
                except json.JSONDecodeError:
                    continue

    shown = 0
    with f.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = r.get("id", "")
            if args.id and rid != args.id:
                continue
            if not args.all and not args.id and shown >= args.limit:
                break
            q = questions.get(rid, "(question not in testset)")
            s = (r.get("summary_text") or "").strip()
            err = r.get("error")
            n_rec = len(r.get("records") or [])
            print("=" * 80)
            print(f"ID: {rid}")
            print(f"Q : {q}")
            print(f"-- {n_rec} records retrieved" + (f"  ERROR: {err}" if err else ""))
            print()
            print(s if s else "(empty answer)")
            print()
            shown += 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
