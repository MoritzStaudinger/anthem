"""Generic paired bootstrap over per-paper micro-F1 counts.

Consumes a per-paper counts JSONL (one row per paper) of the form:
    {"paper_id": "...", "counts": {"<system>": [tp, fp, fn], ...}}
where every row lists the SAME set of systems (so the bootstrap is paired: one
resample of paper indices is applied to all systems at once).

For each of B resamples it draws paper indices with replacement (via a single
multinomial weight vector per draw — mathematically identical to index sampling
but vectorised), re-aggregates micro TP/FP/FN per system, and computes
    micro-F1 = 2*TP / (2*TP + FP + FN).
This matches how the headline numbers are computed (micro over papers), NOT an
average of per-paper F1s.

Reports, per system: point F1 (on the un-resampled data) + 95% percentile CI.
Per requested pair (a,b): the CI on the DIFFERENCE F1(a) - F1(b), the fraction
of draws with a>b, and whether the 95% CI excludes 0 — the statistically correct
answer to "could a bootstrap erase the gap."

Deterministic: fixed --seed so the CI is reproducible for the paper.

Usage:
    python scripts/bootstrap_paired.py --counts reports/bootstrap/task2_hier.jsonl \
        --pairs gemini25_combined-claude_sonnet_combined \
                gemini25_combined-gpt5nano_combined \
                claude_sonnet_combined-gpt5nano_combined
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_counts(path: Path, systems: Optional[List[str]]):
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    all_systems = sorted({s for r in rows for s in r["counts"].keys()})
    systems = systems or all_systems
    P, S = len(rows), len(systems)
    tp = np.zeros((P, S)); fp = np.zeros((P, S)); fn = np.zeros((P, S))
    pids = []
    missing = 0
    for i, r in enumerate(rows):
        pids.append(r["paper_id"])
        for j, s in enumerate(systems):
            c = r["counts"].get(s)
            if c is None:
                missing += 1
                continue
            tp[i, j], fp[i, j], fn[i, j] = c
    if missing:
        print(f"# WARNING: {missing} (paper,system) cells missing — treated as 0/0/0",
              file=sys.stderr)
    return pids, systems, tp, fp, fn


def micro_f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / denom, 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts", type=Path, required=True)
    ap.add_argument("--systems", nargs="*", default=None,
                    help="Subset/order of systems (default: all in the file).")
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="Pairs 'a-b' for difference CIs (default: all pairs).")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--ci", type=float, default=95.0)
    ap.add_argument("--label", default=None, help="Title for the report.")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/bootstrap"))
    args = ap.parse_args()

    pids, systems, tp, fp, fn = load_counts(args.counts, args.systems)
    P, S = len(pids), len(systems)
    B = args.draws
    lo_q = (100 - args.ci) / 2
    hi_q = 100 - lo_q

    # Point estimates (weights = 1 per paper).
    point = {s: float(micro_f1(tp[:, j].sum(), fp[:, j].sum(), fn[:, j].sum()))
             for j, s in enumerate(systems)}

    # Paired bootstrap: one multinomial weight vector per draw (row sums to P),
    # applied to every system's per-paper counts simultaneously.
    rng = np.random.default_rng(args.seed)
    W = rng.multinomial(P, np.full(P, 1.0 / P), size=B).astype(np.float64)  # B x P
    TP = W @ tp; FP = W @ fp; FN = W @ fn                                    # B x S
    F1 = micro_f1(TP, FP, FN)                                                # B x S

    sys_ci = {}
    for j, s in enumerate(systems):
        lo, hi = np.percentile(F1[:, j], [lo_q, hi_q])
        sys_ci[s] = (point[s], float(lo), float(hi))

    # Pairs.
    if args.pairs:
        pairs = []
        for pr in args.pairs:
            a, b = pr.split("-", 1) if "-" in pr else (None, None)
            if a in systems and b in systems:
                pairs.append((a, b))
            else:
                print(f"# WARNING: pair '{pr}' not both in systems {systems}",
                      file=sys.stderr)
    else:
        pairs = list(itertools.combinations(systems, 2))

    idx = {s: j for j, s in enumerate(systems)}
    pair_rows = []
    for a, b in pairs:
        diff = F1[:, idx[a]] - F1[:, idx[b]]
        dpt = point[a] - point[b]
        lo, hi = np.percentile(diff, [lo_q, hi_q])
        p_gt = float(np.mean(diff > 0))
        excludes0 = bool((lo > 0) or (hi < 0))
        pair_rows.append((a, b, dpt, float(lo), float(hi), p_gt, excludes0))

    # ---- render ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.counts.stem
    md = []
    title = args.label or f"Paired bootstrap — {stem}"
    md.append(f"# {title}\n")
    md.append(f"- counts: `{args.counts}` | papers: {P} | systems: {S}")
    md.append(f"- draws: {B} | seed: {args.seed} | CI: {args.ci:.0f}% "
              "(percentile) | metric: micro-F1")
    md.append("")
    md.append(f"## Per-system micro-F1 with {args.ci:.0f}% CI")
    md.append("")
    md.append("| System | F1 | CI low | CI high |")
    md.append("|---|---:|---:|---:|")
    for s in sorted(systems, key=lambda s: point[s], reverse=True):
        pt, lo, hi = sys_ci[s]
        md.append(f"| {s} | {pt:.4f} | {lo:.4f} | {hi:.4f} |")
    md.append("")
    md.append(f"## Paired difference CIs  (a − b)")
    md.append("")
    md.append("| a | b | ΔF1 | CI low | CI high | P(a>b) | 95% CI excludes 0 |")
    md.append("|---|---|---:|---:|---:|---:|:---:|")
    for a, b, dpt, lo, hi, p_gt, ex0 in pair_rows:
        md.append(f"| {a} | {b} | {dpt:+.4f} | {lo:+.4f} | {hi:+.4f} | {p_gt:.3f} "
                  f"| {'**yes**' if ex0 else 'no'} |")
    md.append("")
    md.append("_ΔF1>0 with a CI that excludes 0 ⇒ the gap is robust to resampling "
              "the papers. A CI spanning 0 ⇒ 'comparable within uncertainty'._")

    out = args.out_dir / f"bootstrap_{stem}.md"
    out.write_text("\n".join(md))
    (args.out_dir / f"bootstrap_{stem}.json").write_text(json.dumps({
        "counts": str(args.counts), "papers": P, "draws": B, "seed": args.seed,
        "ci": args.ci,
        "per_system": {s: {"f1": sys_ci[s][0], "lo": sys_ci[s][1], "hi": sys_ci[s][2]}
                       for s in systems},
        "pairs": [{"a": a, "b": b, "delta": dpt, "lo": lo, "hi": hi,
                   "p_a_gt_b": p_gt, "excludes_zero": ex0}
                  for a, b, dpt, lo, hi, p_gt, ex0 in pair_rows],
    }, indent=2))
    print("\n" + "\n".join(md), flush=True)
    print(f"\n# wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
