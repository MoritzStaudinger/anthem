"""Per-dir eval health table.

Walks reports/eval_scico/* and reports, per dir:
  - total rows in answers.jsonl
  - healthy rows (no error, non-empty summary_text, non-empty records)
  - completion of each downstream artifact (judge_report, ir_metrics, ragas_*)

A dir is "ready to judge" when total == EXPECTED and healthy >= ~EXPECTED.

Usage:
    python scripts/eval_status.py
    python scripts/eval_status.py --expected 99
    python scripts/eval_status.py --root reports/eval_scico --min-healthy 90
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _row_health(rec):
    if rec.get("error"):
        return "err"
    if not (rec.get("summary_text") or "").strip():
        return "empty_ans"
    if not rec.get("records"):
        return "empty_rec"
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("reports/eval_scico"))
    ap.add_argument("--expected", type=int, default=99,
                    help="Expected number of rows per dir (default 99).")
    ap.add_argument("--min-healthy", type=int, default=None,
                    help="If set, mark dirs as READY only when healthy >= this. "
                         "Defaults to --expected (i.e. fully clean).")
    args = ap.parse_args()

    min_healthy = args.min_healthy if args.min_healthy is not None else args.expected

    if not args.root.is_dir():
        print(f"ERROR: {args.root} not found")
        return 1

    rows = []
    for d in sorted(args.root.iterdir()):
        if not d.is_dir():
            continue
        f = d / "answers.jsonl"
        if not f.is_file():
            continue
        total = ok = err = empty_ans = empty_rec = 0
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            h = _row_health(rec)
            if h == "ok": ok += 1
            elif h == "err": err += 1
            elif h == "empty_ans": empty_ans += 1
            elif h == "empty_rec": empty_rec += 1

        # Downstream artifacts
        has_judge = (d / "judge_report.json").is_file()
        has_judge_nobert = (d / "judge_report_nobert.json").is_file()
        has_ir = (d / "ir_metrics.jsonl").is_file()
        # RAGAS completion: per-judge file row counts. Format as "done/total"
        # where done = files with >= expected rows, total = files present.
        ragas_files = sorted(d.glob("ragas_*.jsonl"))
        ragas_done = 0
        for rf in ragas_files:
            try:
                with rf.open() as fh:
                    n = sum(1 for _ in fh)
            except OSError:
                n = 0
            if n >= args.expected:
                ragas_done += 1
        n_ragas = len(ragas_files)
        ragas_status = f"{ragas_done}/{n_ragas}" if n_ragas else "0/0"

        ready = (total == args.expected) and (ok >= min_healthy)
        rows.append({
            "dir": d.name, "total": total, "ok": ok, "err": err,
            "empty_ans": empty_ans, "empty_rec": empty_rec,
            "judge": "Y" if has_judge else "-",
            "judge_nobert": "Y" if has_judge_nobert else "-",
            "ir": "Y" if has_ir else "-",
            "ragas_n": n_ragas, "ragas_status": ragas_status, "ready": ready,
        })

    if not rows:
        print(f"(no answer dirs found under {args.root})")
        return 0

    # ---- Print table ----
    print(f"# Per-dir status  (expected={args.expected} rows; "
          f"ready = total==expected AND ok>={min_healthy})\n")
    fmt = "{ready:5s}  {dir:55s}  total={total:3d}  ok={ok:3d}  err={err:3d}  empty_ans={empty_ans:3d}  empty_rec={empty_rec:3d}  judge={judge}  no-bert={judge_nobert}  ir={ir}  ragas={ragas_status:5s}"
    print(fmt.format(
        ready="STATE", dir="DIR", total=0, ok=0, err=0,
        empty_ans=0, empty_rec=0, judge="J", judge_nobert="N",
        ir="I", ragas_status="done/total").replace("  0", "  -"))
    print("-" * 180)
    for r in rows:
        r["ready"] = "READY" if r["ready"] else "  -  "
        print(fmt.format(**r))

    # ---- Summary ----
    n_ready = sum(1 for r in rows if r["ready"] == "READY")
    n_incomplete = sum(1 for r in rows if r["total"] != args.expected)
    n_unhealthy = sum(1 for r in rows
                      if r["total"] == args.expected and r["ok"] < min_healthy)
    print(f"\n# Summary: {n_ready} READY, {n_incomplete} incomplete (total != "
          f"{args.expected}), {n_unhealthy} unhealthy (total == {args.expected} but "
          f"ok < {min_healthy})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
