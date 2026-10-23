"""Inter-judge agreement for RAGAS scores across judge models.

Reads ragas_<judge>.jsonl files from one or more answer dirs and reports, on the
questions BOTH judges scored: per-judge mean faithfulness / relevancy, and
pairwise agreement — Pearson r, Spearman rho, mean-abs-diff, and Cohen's kappa
on a faithfulness pass/fail binarisation (>= --threshold).

Answers the rebuttal question "does the RAGAS conclusion hold under a second
judge?" for the matched Gemini-2.5-Pro pair (pass both dirs to pool the 198 items).

Usage:
    python scripts/ragas_judge_agreement.py \
        --answers-dirs reports/eval_scico/agent_gemini_2_5_pro \
                       reports/eval_scico/pipeline_gemini_2_5_pro_topk5 \
        --primary qwen_qwen3_5_397b_a17b --second gpt_4o_mini
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _load(dir_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """dir -> judge_slug -> id -> {faithfulness, answer_relevancy} (scored only)."""
    out: Dict[str, Dict[str, float]] = {}
    for fp in glob.glob(str(dir_path / "ragas_*.jsonl")):
        slug = os.path.basename(fp)[len("ragas_"):-len(".jsonl")]
        rows = {}
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            f = r.get("faithfulness"); a = r.get("answer_relevancy")
            if r.get("id") and (isinstance(f, (int, float)) or isinstance(a, (int, float))):
                rows[r["id"]] = {"faithfulness": f, "answer_relevancy": a}
        out[slug] = rows
    return out


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def _rank(v: List[float]) -> List[float]:
    idx = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            r[idx[k]] = avg
        i = j + 1
    return r


def _spearman(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    return _pearson(_rank(xs), _rank(ys))


def _kappa(a: List[int], b: List[int]) -> float:
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa1 = sum(a) / n; pb1 = sum(b) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if (1 - pe) else float("nan")


def _paired(items, jA, jB, metric):
    xs, ys = [], []
    for (d, i) in items:
        va = jA.get((d, i), {}).get(metric)
        vb = jB.get((d, i), {}).get(metric)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            xs.append(float(va)); ys.append(float(vb))
    return xs, ys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers-dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--judges", nargs="*", default=None,
                    help="Judge slugs to compare (default: all present in every dir).")
    ap.add_argument("--primary", default=None, help="Primary judge slug to feature.")
    ap.add_argument("--second", default=None, help="Second judge slug to feature.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Faithfulness pass threshold for Cohen's kappa.")
    ap.add_argument("--out", type=Path,
                    default=Path("reports/eval_scico/ragas_judge_agreement.md"))
    args = ap.parse_args()

    # per (dir,id) flattened per judge
    perdir = {str(d): _load(d) for d in args.answers_dirs}
    # judges present in ALL dirs (so a pair is comparable across the pool)
    judge_sets = [set(v.keys()) for v in perdir.values()]
    common = set.intersection(*judge_sets) if judge_sets else set()
    judges = args.judges or sorted(common)
    # flatten: judge_slug -> {(dir,id): scores}
    flat: Dict[str, Dict[Tuple[str, str], Dict[str, float]]] = {j: {} for j in judges}
    for d, jmap in perdir.items():
        for j in judges:
            for i, sc in (jmap.get(j) or {}).items():
                flat[j][(d, i)] = sc

    md = []
    md.append("# RAGAS inter-judge agreement\n")
    md.append(f"- Dirs: {', '.join(str(d) for d in args.answers_dirs)}")
    md.append(f"- Judges compared: {', '.join(judges)}")
    md.append(f"- Kappa on faithfulness pass/fail at threshold {args.threshold}")
    md.append("")

    # Per-judge means (over each judge's own scored items, pooled across dirs)
    md.append("## Per-judge means (pooled)")
    md.append("| Judge | n | mean faithfulness | mean relevancy |")
    md.append("|---|---:|---:|---:|")
    for j in judges:
        items = list(flat[j].keys())
        fs = [flat[j][k]["faithfulness"] for k in items
              if isinstance(flat[j][k].get("faithfulness"), (int, float))]
        rs = [flat[j][k]["answer_relevancy"] for k in items
              if isinstance(flat[j][k].get("answer_relevancy"), (int, float))]
        mf = sum(fs) / len(fs) if fs else float("nan")
        mr = sum(rs) / len(rs) if rs else float("nan")
        md.append(f"| {j} | {len(items)} | {mf:.3f} | {mr:.3f} |")
    md.append("")

    # Pairwise agreement (all pairs, or featured pair first)
    pairs = []
    if args.primary and args.second:
        pairs.append((args.primary, args.second))
    for i in range(len(judges)):
        for j in range(i + 1, len(judges)):
            p = (judges[i], judges[j])
            if p not in pairs and (p[1], p[0]) not in pairs:
                pairs.append(p)

    md.append("## Pairwise agreement — faithfulness")
    md.append("| Judge A | Judge B | n | Pearson r | Spearman ρ | mean|Δ| | Cohen's κ |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for a, b in pairs:
        if a not in flat or b not in flat:
            continue
        common_items = set(flat[a]) & set(flat[b])
        xs, ys = _paired(common_items, flat[a], flat[b], "faithfulness")
        if not xs:
            continue
        mad = sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)
        ba = [1 if x >= args.threshold else 0 for x in xs]
        bb = [1 if y >= args.threshold else 0 for y in ys]
        md.append(f"| {a} | {b} | {len(xs)} | {_pearson(xs,ys):.3f} | {_spearman(xs,ys):.3f} "
                  f"| {mad:.3f} | {_kappa(ba,bb):.3f} |")
    md.append("")

    md.append("## Pairwise agreement — answer relevancy")
    md.append("| Judge A | Judge B | n | Pearson r | Spearman ρ | mean|Δ| |")
    md.append("|---|---|---:|---:|---:|---:|")
    for a, b in pairs:
        if a not in flat or b not in flat:
            continue
        common_items = set(flat[a]) & set(flat[b])
        xs, ys = _paired(common_items, flat[a], flat[b], "answer_relevancy")
        if not xs:
            continue
        mad = sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)
        md.append(f"| {a} | {b} | {len(xs)} | {_pearson(xs,ys):.3f} | {_spearman(xs,ys):.3f} "
                  f"| {mad:.3f} |")
    md.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n# wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
