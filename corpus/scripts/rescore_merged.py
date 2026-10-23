"""Recompute the comparison_summary.csv table from a (re-linked) merged
JSONL. Computes raw / ror / ror_hier / name tracks.

ror_hier is hierarchy-aware: a predicted ROR id matches the gold id
when (a) it's the same id OR (b) it's a parent/child in the ROR
hierarchy. Needs `reports/ror_hierarchy_cache.json` (build with
`scripts/fetch_ror_hierarchy.py`); falls back to ror_hier == ror with
a warning when missing.

Usage:
    python scripts/rescore_merged.py \\
        --merged reports/merged_per_paper.relinked.jsonl \\
        --out    reports/comparison_summary.relinked.csv \\
        --hierarchy reports/ror_hierarchy_cache.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

logger = logging.getLogger("rescore")


def _f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F


def _norm(s: str | None) -> str:
    if not s:
        return ""
    return " ".join(s.strip().lower().split())


def _score_track(pred: List[str], gold: List[str], casefold: bool = True
                 ) -> Tuple[int, int, int]:
    """Multiset overlap with optional lowercase normalisation."""
    if casefold:
        pred_c = Counter(_norm(x) for x in pred if x)
        gold_c = Counter(_norm(x) for x in gold if x)
    else:
        pred_c = Counter(x for x in pred if x)
        gold_c = Counter(x for x in gold if x)
    tp = sum((pred_c & gold_c).values())
    fp = sum((pred_c - gold_c).values())
    fn = sum((gold_c - pred_c).values())
    return tp, fp, fn


def _bare(rid: str) -> str:
    if not rid:
        return ""
    return rid.strip().rstrip("/").split("/")[-1].lower()


def _equivalence_set(bare: str, hier: Dict[str, Dict[str, List[str]]]) -> Set[str]:
    """{self} ∪ parents ∪ children. `related` is excluded — it's a much
    looser link (successor orgs, sister labs) that inflates FPs."""
    if not bare:
        return set()
    h = hier.get(bare) or {}
    out = {bare}
    out.update(h.get("parent") or [])
    out.update(h.get("child") or [])
    return out


def _score_ror_hier(pred_ror: List[str], gold_ror: List[str],
                    hier: Dict[str, Dict[str, List[str]]]) -> Tuple[int, int, int]:
    """Hierarchy-aware multiset matching on bare ROR ids.

    Greedy bipartite: for each gold id, mark a TP if any unconsumed
    pred id is in its equivalence set (self + parents + children).
    Remaining preds are FPs, remaining golds are FNs. Multiset
    semantics: duplicates in either side consume one slot each.
    """
    pred_bare = [_bare(x) for x in pred_ror if x]
    gold_bare = [_bare(x) for x in gold_ror if x]

    # For each pred id, the set of ids it can match.
    pred_eq = [(b, _equivalence_set(b, hier)) for b in pred_bare]

    consumed = [False] * len(pred_eq)
    tp = 0
    fn = 0
    for g in gold_bare:
        matched = False
        for i, (pb, eq) in enumerate(pred_eq):
            if consumed[i]:
                continue
            if g == pb or g in eq or pb in _equivalence_set(g, hier):
                consumed[i] = True
                tp += 1
                matched = True
                break
        if not matched:
            fn += 1
    fp = sum(1 for c in consumed if not c)
    return tp, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path,
                    default=Path("reports/merged_per_paper.relinked.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("reports/comparison_summary.relinked.csv"))
    ap.add_argument("--hierarchy", type=Path,
                    default=Path("reports/ror_hierarchy_cache.json"))
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    hier: Dict[str, Dict[str, List[str]]] = {}
    if args.hierarchy.is_file():
        hier = json.loads(args.hierarchy.read_text())
        with_rels = sum(1 for v in hier.values()
                        if v.get("parent") or v.get("child"))
        logger.info("loaded hierarchy cache: %d ids (%d with parent/child)",
                    len(hier), with_rels)
    else:
        logger.warning("no hierarchy cache at %s — ror_hier will equal ror",
                       args.hierarchy)

    # pipeline_id -> track -> [tp_covered, fp_covered, fn_covered,
    #                          tp_inclusive, fp_inclusive, fn_inclusive,
    #                          papers_covered, papers_total]
    agg = {}

    def _bump(pid, track, tp, fp, fn, covered):
        a = agg.setdefault(pid, {}).setdefault(track,
            [0, 0, 0, 0, 0, 0, 0, 0])
        # inclusive always counts
        a[3] += tp; a[4] += fp; a[5] += fn
        if covered:
            a[0] += tp; a[1] += fp; a[2] += fn

    pipeline_papers_total: Dict[str, int] = {}
    pipeline_papers_covered: Dict[str, int] = {}

    with args.merged.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            gold = d.get("gold") or {}
            gold_raw  = gold.get("affiliations_raw") or []
            gold_ror  = gold.get("ror_ids") or []
            gold_name = gold.get("names") or []
            for pid, pred in (d.get("predictions") or {}).items():
                pipeline_papers_total[pid] = pipeline_papers_total.get(pid, 0) + 1
                p_raw  = pred.get("affiliations_raw") or []
                p_ror  = pred.get("ror_ids") or []
                p_name = pred.get("names") or []
                covered = bool(p_raw)
                if covered:
                    pipeline_papers_covered[pid] = pipeline_papers_covered.get(pid, 0) + 1
                for track, p, g, casefold in (
                    ("raw",  p_raw,  gold_raw,  True),
                    ("ror",  p_ror,  gold_ror,  False),
                    ("name", p_name, gold_name, True),
                ):
                    tp, fp, fn = _score_track(p, g, casefold=casefold)
                    _bump(pid, track, tp, fp, fn, covered)
                if hier:
                    tp, fp, fn = _score_ror_hier(p_ror, gold_ror, hier)
                else:
                    tp, fp, fn = _score_track(p_ror, gold_ror, casefold=False)
                _bump(pid, "ror_hier", tp, fp, fn, covered)

    rows = []
    for pid in sorted(agg):
        total = pipeline_papers_total.get(pid, 0)
        cov = pipeline_papers_covered.get(pid, 0)
        cov_pct = 100.0 * cov / total if total else 0.0
        for track, a in agg[pid].items():
            cP, cR, cF = _f1(a[0], a[1], a[2])
            iP, iR, iF = _f1(a[3], a[4], a[5])
            rows.append({
                "pipeline_id": pid, "track": track,
                "coverage_pct": f"{cov_pct:.1f}",
                "covered_precision": f"{cP:.3f}",
                "covered_recall":    f"{cR:.3f}",
                "covered_f1":        f"{cF:.3f}",
                "inclusive_precision": f"{iP:.3f}",
                "inclusive_recall":    f"{iR:.3f}",
                "inclusive_f1":        f"{iF:.3f}",
                "tp_covered": a[0], "fp_covered": a[1], "fn_covered": a[2],
                "tp_inclusive": a[3], "fp_inclusive": a[4], "fn_inclusive": a[5],
                "papers_covered": cov, "papers_total": total,
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {args.out} ({len(rows)} rows)")

    # Print headline leaderboard
    by_pipe = {}
    for r in rows:
        by_pipe.setdefault(r["pipeline_id"], {})[r["track"]] = r
    order = sorted(by_pipe.keys(),
                   key=lambda p: -float(by_pipe[p]["ror"]["covered_f1"]))
    print()
    print(f"{'pipeline':<38} {'cov%':>5}   {'raw':>5}    {'ror':>5}    {'hier':>5}    {'name':>5}")
    print("-" * 90)
    for p in order:
        d = by_pipe[p]
        cov   = float(d["ror"]["coverage_pct"])
        f_raw = float(d["raw"]["covered_f1"])
        f_ror = float(d["ror"]["covered_f1"])
        f_hir = float(d.get("ror_hier", d["ror"])["covered_f1"])
        f_nm  = float(d["name"]["covered_f1"])
        print(f"{p:<38} {cov:>5.1f}   {f_raw:>5.3f}    {f_ror:>5.3f}    "
              f"{f_hir:>5.3f}    {f_nm:>5.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
