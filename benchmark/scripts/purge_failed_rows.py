"""Purge failed/empty rows from a dir's answers.jsonl in place.

A row is "failed" when ANY of:
  - row.error is set
  - row.summary_text is empty/whitespace
  - row.records is empty

Survivors are written back to answers.jsonl. Prints what was kept/dropped.
Original file is moved to answers.jsonl.bak before rewriting.

Usage:
    python scripts/purge_failed_rows.py <answers_dir>
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: purge_failed_rows.py <answers_dir>", file=sys.stderr)
        return 2
    d = Path(sys.argv[1])
    f = d / "answers.jsonl"
    if not f.is_file():
        print(f"ERROR: {f} not found", file=sys.stderr)
        return 1

    keep_lines = []
    n_total = n_err = n_empty_ans = n_empty_rec = 0
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
            if r.get("error"):
                n_err += 1
                continue
            if not (r.get("summary_text") or "").strip():
                n_empty_ans += 1
                continue
            if not r.get("records"):
                n_empty_rec += 1
                continue
            keep_lines.append(line if line.endswith("\n") else line + "\n")

    shutil.copy2(f, f.with_suffix(".jsonl.bak"))
    with f.open("w", encoding="utf-8") as fh:
        fh.writelines(keep_lines)

    n_kept = len(keep_lines)
    n_dropped = n_total - n_kept
    print(f"{d.name}: total={n_total} kept={n_kept} dropped={n_dropped} "
          f"(err={n_err} empty_summary={n_empty_ans} empty_records={n_empty_rec}) "
          f"  backup -> answers.jsonl.bak")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
