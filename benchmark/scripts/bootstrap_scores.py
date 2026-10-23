"""Paired bootstrap CIs + significance for per-question benchmark scores.

Unlike bootstrap_paired.py (micro-F1 from per-paper TP/FP/FN counts), this
bootstraps the MEAN of per-question scores (macro), which is how the retrieval
set-level table and the RAGAS columns are aggregated. It resamples the 99
questions with replacement (paired: the same resample applied to every system),
recomputes each system's mean, and reports 95% CIs and pairwise difference CIs
(a difference whose CI excludes 0 is significant).

Sources:
  --source rescore : per-question set-level scores from the failures-as-zero
                     rescore (reports/eval_scico/rescore/rescore_per_question.jsonl);
                     --metric one of F1_set / P_set / R_set. Failures already
                     score 0 in that file, so this is the honest macro CI.
  --source ragas   : per-question RAGAS scores from ragas_<judge>.jsonl in each
                     system dir; --metric faithfulness / answer_relevancy.

Deterministic (--seed). Questions = the ids present for every selected system
(so the pairing is well-defined); with --source rescore that is the scoreable
universe, with --source ragas it is the ids all systems' judge scored.

Usage:
  python scripts/bootstrap_scores.py --source rescore --metric F1_set \
      --systems pipeline_gemini_2_5_pro_topk5 agent_claude_sonnet_4_6 agent_gpt_5_mini
  python scripts/bootstrap_scores.py --source ragas --metric faithfulness \
      --judge qwen_qwen3_5_397b_a17b --systems pipeline_gemini_2_5_pro_topk5 ...
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def _load_rescore(path: Path, metric: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        v = r.get(metric)
        if r.get("system") and r.get("id") and isinstance(v, (int, float)):
            out.setdefault(r["system"], {})[r["id"]] = float(v)
    return out


def _load_ragas(reports_dir: Path, systems: List[str], judge: str,
                metric: str) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for s in systems:
        fp = reports_dir / s / f"ragas_{judge}.jsonl"
        if not fp.exists():
            continue
        rows = {}
        for line in fp.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            v = r.get(metric)
            if r.get("id") and isinstance(v, (int, float)):
                rows[r["id"]] = float(v)
        out[s] = rows
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["rescore", "ragas"], required=True)
    ap.add_argument("--metric", required=True,
                    help="F1_set/P_set/R_set (rescore) or faithfulness/answer_relevancy (ragas)")
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--reports-dir", type=Path, default=Path("reports/eval_scico"))
    ap.add_argument("--rescore-jsonl", type=Path,
                    default=Path("reports/eval_scico/rescore/rescore_per_question.jsonl"))
    ap.add_argument("--judge", default="qwen_qwen3_5_397b_a17b")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--ci", type=float, default=95.0)
    ap.add_argument("--pairs", nargs="*", default=None, help="'a-b' pairs (default all).")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.source == "rescore":
        data = _load_rescore(args.rescore_jsonl, args.metric)
    else:
        data = _load_ragas(args.reports_dir, args.systems, args.judge, args.metric)

    systems = [s for s in args.systems if s in data and data[s]]
    missing = [s for s in args.systems if s not in systems]
    if missing:
        print(f"# WARNING: no data for {missing}", flush=True)
    # questions present for EVERY selected system (well-defined pairing)
    common = set.intersection(*[set(data[s].keys()) for s in systems]) if systems else set()
    ids = sorted(common)
    P, S = len(ids), len(systems)
    if P == 0 or S < 1:
        print("# ERROR: no common questions / systems", flush=True)
        return 1

    V = np.array([[data[s][q] for s in systems] for q in ids])  # P x S
    point = V.mean(axis=0)

    rng = np.random.default_rng(args.seed)
    W = rng.multinomial(P, np.full(P, 1.0 / P), size=args.draws).astype(np.float64)  # B x P
    means = (W @ V) / P  # B x S
    lo_q = (100 - args.ci) / 2
    hi_q = 100 - lo_q

    md = []
    md.append(f"# Bootstrap CIs — {args.source} / {args.metric}\n")
    md.append(f"- questions (common to all systems): {P} | draws: {args.draws} | "
              f"seed: {args.seed} | {args.ci:.0f}% percentile CI")
    if args.source == "ragas":
        md.append(f"- judge: {args.judge}")
    md.append("")
    md.append(f"## Per-system {args.metric} with {args.ci:.0f}% CI")
    md.append(f"| System | {args.metric} | CI low | CI high |")
    md.append("|---|---:|---:|---:|")
    order = sorted(range(S), key=lambda j: -point[j])
    ci = {}
    for j in order:
        lo, hi = np.percentile(means[:, j], [lo_q, hi_q])
        ci[systems[j]] = (float(point[j]), float(lo), float(hi))
        md.append(f"| {systems[j]} | {point[j]:.4f} | {lo:.4f} | {hi:.4f} |")
    md.append("")

    idx = {s: j for j, s in enumerate(systems)}
    if args.pairs:
        pairs = [tuple(p.split("-", 1)) for p in args.pairs if "-" in p]
    else:
        pairs = list(itertools.combinations(systems, 2))
    md.append("## Pairwise differences (a − b)")
    md.append("| a | b | Δ | CI low | CI high | P(a>b) | 95% CI excludes 0 |")
    md.append("|---|---|---:|---:|---:|---:|:---:|")
    for a, b in pairs:
        if a not in idx or b not in idx:
            continue
        diff = means[:, idx[a]] - means[:, idx[b]]
        d = point[idx[a]] - point[idx[b]]
        lo, hi = np.percentile(diff, [lo_q, hi_q])
        p_gt = float(np.mean(diff > 0))
        ex0 = bool(lo > 0 or hi < 0)
        md.append(f"| {a} | {b} | {d:+.4f} | {lo:+.4f} | {hi:+.4f} | {p_gt:.3f} "
                  f"| {'**yes**' if ex0 else 'no'} |")
    md.append("")

    out = args.out or Path(f"reports/eval_scico/bootstrap_{args.source}_{args.metric}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n# wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
