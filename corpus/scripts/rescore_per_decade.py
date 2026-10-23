"""Per-decade ror_hier F1 from the relinked merged file.

Pairs each paper_id with its year (from reports/ground_truth_merged.csv)
and aggregates TP/FP/FN per (pipeline, decade) on the hierarchy-aware
ROR track.

Output: reports/comparison_per_decade.relinked.csv (long-form rows) +
prints a Markdown-friendly grid to stdout.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

logger = logging.getLogger("rescore_dec")


def _bare(rid: str) -> str:
    if not rid:
        return ""
    return rid.strip().rstrip("/").split("/")[-1].lower()


def _eq(bare: str, hier: Dict[str, Dict[str, list]]) -> Set[str]:
    if not bare:
        return set()
    h = hier.get(bare) or {}
    return {bare, *(h.get("parent") or []), *(h.get("child") or [])}


def _score_hier(pred: List[str], gold: List[str],
                hier: Dict[str, Dict[str, list]]) -> Tuple[int, int, int]:
    pred_b = [_bare(x) for x in pred if x]
    gold_b = [_bare(x) for x in gold if x]
    pred_eq = [(b, _eq(b, hier)) for b in pred_b]
    consumed = [False] * len(pred_eq)
    tp = 0; fn = 0
    for g in gold_b:
        matched = False
        for i, (pb, eq) in enumerate(pred_eq):
            if consumed[i]:
                continue
            if g == pb or g in eq or pb in _eq(g, hier):
                consumed[i] = True
                tp += 1
                matched = True
                break
        if not matched:
            fn += 1
    fp = sum(1 for c in consumed if not c)
    return tp, fp, fn


def _f1(tp: int, fp: int, fn: int) -> float:
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * P * R / (P + R) if (P + R) else 0.0


_OLD_ID = re.compile(r"^[A-Z](\d\d)-\d+")  # "W01-0001", "P85-1010"


def _year_from_id(pid: str) -> int | None:
    """Fallback when ground-truth CSV doesn't list the paper.

    Legacy ACL IDs use 2-digit years. NN >= 65 → 1965-1999, otherwise
    20NN (covers 2000-2019; the new YYYY.venue format took over in 2020).
    """
    if not pid:
        return None
    if pid[:4].isdigit():
        return int(pid[:4])
    m = _OLD_ID.match(pid)
    if m:
        nn = int(m.group(1))
        return 1900 + nn if nn >= 65 else 2000 + nn
    return None


def _decade(year: int | None) -> str:
    if not year:
        return "unknown"
    return f"{(year // 10) * 10}s"


def _load_year_map(gt_csv: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not gt_csv.is_file():
        logger.warning("no ground-truth CSV at %s — deriving year from paper_id", gt_csv)
        return out
    with gt_csv.open() as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("anthology_id") or "").strip()
            try:
                y = int(row.get("year") or "")
            except ValueError:
                continue
            if pid:
                out[pid] = y
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", type=Path,
                    default=Path("reports/merged_per_paper.relinked.jsonl"))
    ap.add_argument("--hierarchy", type=Path,
                    default=Path("reports/ror_hierarchy_cache.json"))
    ap.add_argument("--gold-csv", type=Path,
                    default=Path("reports/ground_truth_merged.csv"))
    ap.add_argument("--out-csv", type=Path,
                    default=Path("reports/comparison_per_decade.relinked.csv"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    hier = json.loads(args.hierarchy.read_text()) if args.hierarchy.is_file() else {}
    if not hier:
        logger.warning("no hierarchy cache — ror_hier will degenerate to strict ROR")
    year_map = _load_year_map(args.gold_csv)
    logger.info("loaded year map for %d papers from %s", len(year_map), args.gold_csv)

    # per (pipeline, decade): tp/fp/fn + papers count
    agg: Dict[Tuple[str, str], List[int]] = defaultdict(lambda: [0, 0, 0, 0])
    # per (pipeline): set of decades it has any predictions for (for the row order)
    pipelines: Set[str] = set()
    decades_seen: Set[str] = set()
    papers_per_decade: Dict[str, int] = defaultdict(int)
    counted_paper: Set[Tuple[str, str]] = set()  # (paper_id, decade) so we don't double-count

    with args.merged.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pid = d.get("paper_id") or ""
            y = year_map.get(pid) or _year_from_id(pid)
            dec = _decade(y)
            decades_seen.add(dec)
            if (pid, dec) not in counted_paper:
                papers_per_decade[dec] += 1
                counted_paper.add((pid, dec))
            gold_ror = d.get("gold", {}).get("ror_ids") or []
            for ppid, pred in (d.get("predictions") or {}).items():
                pipelines.add(ppid)
                pred_ror = pred.get("ror_ids") or []
                tp, fp, fn = _score_hier(pred_ror, gold_ror, hier)
                a = agg[(ppid, dec)]
                a[0] += tp; a[1] += fp; a[2] += fn; a[3] += 1

    # Order decades chronologically; "unknown" last.
    decades = sorted([d for d in decades_seen if d != "unknown"
                      and d.endswith("s") and d[:-1].isdigit()],
                     key=lambda s: int(s[:-1]))
    if "unknown" in decades_seen:
        decades.append("unknown")

    # Sort pipelines by overall covered-only hier F1 (sum across decades)
    overall = {}
    for ppid in pipelines:
        tp = sum(agg[(ppid, dec)][0] for dec in decades)
        fp = sum(agg[(ppid, dec)][1] for dec in decades)
        fn = sum(agg[(ppid, dec)][2] for dec in decades)
        overall[ppid] = (tp, fp, fn, _f1(tp, fp, fn))
    order = sorted(pipelines, key=lambda p: -overall[p][3])

    # Long-form CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w") as fh:
        w = csv.writer(fh)
        w.writerow(["pipeline_id", "decade", "tp", "fp", "fn", "papers", "f1"])
        for ppid in order:
            for dec in decades:
                tp, fp, fn, papers = agg[(ppid, dec)]
                w.writerow([ppid, dec, tp, fp, fn, papers,
                            f"{_f1(tp, fp, fn):.3f}"])
            w.writerow([ppid, "overall", *overall[ppid][:3],
                        sum(agg[(ppid, dec)][3] for dec in decades),
                        f"{overall[ppid][3]:.3f}"])
    logger.info("wrote %s", args.out_csv)

    # Markdown grid
    print()
    print("| pipeline | " + " | ".join(decades) + " | overall |")
    print("|---|" + "---:|" * (len(decades) + 1))
    for ppid in order:
        cells = []
        for dec in decades:
            tp, fp, fn, papers = agg[(ppid, dec)]
            if papers == 0:
                cells.append("—")
            else:
                cells.append(f"{_f1(tp, fp, fn):.3f}")
        ov = overall[ppid][3]
        cells.append(f"**{ov:.3f}**")
        print(f"| `{ppid}` | " + " | ".join(cells) + " |")
    print(f"| _papers/decade_ | " +
          " | ".join(str(papers_per_decade[d]) for d in decades) +
          f" | {sum(papers_per_decade.values())} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
