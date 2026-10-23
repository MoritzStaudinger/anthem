"""Dedupe rows in answers.jsonl by `id` — keep the LAST occurrence
(most recent regeneration). Backs up the original to answers.jsonl.bak.

Usage:
    python scripts/dedupe_answers.py <answers_dir>
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: dedupe_answers.py <answers_dir>", file=sys.stderr)
        return 2
    d = Path(sys.argv[1])
    f = d / "answers.jsonl"
    if not f.is_file():
        print(f"ERROR: {f} not found", file=sys.stderr)
        return 1

    # Keep the LAST occurrence of each id (most recent run wins)
    by_id: dict = {}
    n_total = 0
    with f.open() as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                r = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            n_total += 1
            rid = r.get("id")
            if rid is None:
                continue
            by_id[rid] = line if line.endswith("\n") else line + "\n"

    shutil.copy2(f, f.with_suffix(".jsonl.bak"))
    with f.open("w", encoding="utf-8") as fh:
        for line in by_id.values():
            fh.write(line)

    print(f"{d.name}: total_in={n_total} unique_ids={len(by_id)} "
          f"dropped_duplicates={n_total - len(by_id)}  "
          f"backup -> answers.jsonl.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
