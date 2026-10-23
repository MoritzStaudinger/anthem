"""Score the four ROR strategies against the HUMAN-annotated gold.

Replaces the circular auto-linked comparison: reads correct_ror from the
adjudication sheet (agreements pre-filled + disagreements hand-annotated),
builds a per-string gold, and scores each strategy per-paper (F1_hier, exact)
and per-string (top-1 accuracy for the argmax rules), all hierarchy-aware and
consistent with the 'highest parent' annotation policy.

Also emits per-paper counts for a paired bootstrap of the strategy differences.

Usage:
  python scripts/score_relink_vs_human.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Set

from openpyxl import load_workbook

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from relink_strategy_compare import (load_gold, string_ror_set, norm_ror,  # noqa: E402
                                     make_expand, score_ror_hier, STRATEGIES)

# auto-linked (circular) F1_hier from the earlier run, for the inflation contrast
AUTO_F1HIER = {"current": 0.905, "chosen_legacy": 0.894,
               "single_argmax": 0.703, "single_chosen": 0.742}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def load_human_gold(sheet: Path):
    wb = load_workbook(sheet)
    ws = wb.active
    hdr = [c.value for c in ws[2]]
    I = {h: i for i, h in enumerate(hdr)}
    gold: Dict[str, Set[str]] = {}
    unknown = 0
    for r in range(3, ws.max_row + 1):
        s = ws.cell(r, I["string"] + 1).value
        cr = ws.cell(r, I["correct_ror"] + 1).value
        if not s:
            continue
        key = str(s).lower()
        if cr in (None, "") or str(cr).strip().lower() in ("none", "na", "-"):
            gold[key] = set()  # no confident match -> "no ROR" IS the gold (links = FP)
            continue
        crs = str(cr).strip()
        gold[key] = {norm_ror(g) for g in crs.replace(",", ";").split(";") if norm_ror(g)}
    return gold, unknown


def prf(tp, fp, fn):
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    return P, R, (2 * P * R / (P + R) if (P + R) else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sheet", type=Path,
                    default=REPO / "reports/relink_compare/adjudication_sheet.xlsx")
    ap.add_argument("--gold-csv", type=Path,
                    default=Path("/home/moritz/Downloads/11688_Singing_from_the_Same_AN_Data/"
                                 "sample_data/affiliation_ground_truth.csv"))
    ap.add_argument("--items-cache", type=Path, default=REPO / "reports/ror_items_raw_cache.json")
    ap.add_argument("--hierarchy-cache", type=Path,
                    default=REPO / "reports/ror_hierarchy_cache.warmed.json")
    ap.add_argument("--min-score", type=float, default=0.7)
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports/relink_compare")
    args = ap.parse_args()

    human, unknown = load_human_gold(args.sheet)
    papers, uniq = load_gold(args.gold_csv)
    cache = json.loads(args.items_cache.read_text())
    hier = json.loads(args.hierarchy_cache.read_text())
    expand = make_expand(hier, False)
    annotated = [s for s in uniq if s.lower() in human]
    print(f"# unique strings {len(uniq)} | annotated {len(annotated)} | unknown {unknown}",
          flush=True)

    variants = list(STRATEGIES.keys())
    # per-paper F1 accumulators + per-string top-1 accuracy
    agg = {v: {"htp": 0, "hfp": 0, "hfn": 0, "etp": 0, "efp": 0, "efn": 0,
               "s_ok": 0, "s_n": 0, "s_linked": 0} for v in variants}
    per_paper_counts = {v: [] for v in variants}

    # per-string top-1 accuracy (single-pick rules are meaningful; report all)
    for v in variants:
        for s in annotated:
            g = human[s.lower()]
            pred = string_ror_set(v, s, cache, args.min_score)
            agg[v]["s_n"] += 1
            if pred:
                agg[v]["s_linked"] += 1
            # correct: gold empty & pred empty, OR pred hier-matches some gold
            if not g:
                ok = (len(pred) == 0)
            else:
                ok = any(expand(p) & expand(x) for p in pred for x in g)
            agg[v]["s_ok"] += int(ok)

    # per-paper F1_hier / exact
    for pid, p in papers.items():
        # gold set from human annotation (skip unknown strings)
        gold = set()
        for s in p["raw_strings"]:
            k = s.lower()
            if k in human:
                gold |= human[k]
        for v in variants:
            pred = set()
            for s in p["raw_strings"]:
                pred |= string_ror_set(v, s, cache, args.min_score)
            tp, fp, fn = score_ror_hier(gold, pred, expand)
            agg[v]["htp"] += tp; agg[v]["hfp"] += fp; agg[v]["hfn"] += fn
            agg[v]["etp"] += len(gold & pred); agg[v]["efp"] += len(pred - gold)
            agg[v]["efn"] += len(gold - pred)
            per_paper_counts[v].append((pid, tp, fp, fn))

    # per-paper counts for bootstrap
    counts_path = args.out_dir / "relink_counts_human_hier.jsonl"
    rows = {pid: {"paper_id": pid, "counts": {}} for pid in papers}
    for v in variants:
        for pid, tp, fp, fn in per_paper_counts[v]:
            rows[pid]["counts"][v] = [tp, fp, fn]
    counts_path.write_text("\n".join(json.dumps(r) for r in rows.values()) + "\n")

    # render
    md = []
    md.append("# ROR strategies vs HUMAN gold (non-circular)\n")
    md.append(f"- Gold: hand-annotated `correct_ror` ({len(annotated)}/{len(uniq)} strings) "
              f"| min_score {args.min_score} | hierarchy-aware")
    md.append("")
    md.append("| Strategy | ROR src | F1_hier | P | R | F1_exact | string acc | 95% CI | "
              "string cov | (auto-gold F1_hier) |")
    md.append("|---|---|---:|---:|---:|---:|---:|---|---:|---:|")
    for v in variants:
        a = agg[v]
        hP, hR, hF = prf(a["htp"], a["hfp"], a["hfn"])
        _, _, eF = prf(a["etp"], a["efp"], a["efn"])
        acc = a["s_ok"] / a["s_n"] if a["s_n"] else float("nan")
        lo, hi = wilson(a["s_ok"], a["s_n"])
        cov = a["s_linked"] / a["s_n"] if a["s_n"] else float("nan")
        md.append(f"| {v} | {STRATEGIES[v][0]} | **{hF:.3f}** | {hP:.3f} | {hR:.3f} "
                  f"| {eF:.3f} | {acc:.3f} | [{lo:.3f}, {hi:.3f}] | {cov:.3f} "
                  f"| {AUTO_F1HIER.get(v,'?')} |")
    md.append("")
    md.append("_string acc = top-pick hierarchy-match vs human gold (Wilson CI). The last "
              "column is the earlier circular auto-gold F1_hier — the drop is the inflation "
              "that the manual gold removes._")
    md.append(f"\nPer-paper counts for bootstrap: `{counts_path}`")
    out = args.out_dir / "relink_vs_human.md"
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n# wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
