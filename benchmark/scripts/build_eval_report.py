"""Build a paper-ready markdown report of eval results.

Filters:
  - Only pipeline_* and agent_* dirs (closed_* + smoke/test excluded)
  - Only dirs with >= --min-ok healthy rows (default 50) in answers.jsonl
  - LLM-judge cells with n < --min-judge-n (default 30) are excluded

Writes:
  reports/eval_scico/_eval_report.md

Usage:
    python scripts/build_eval_report.py
    python scripts/build_eval_report.py --min-ok 50 --min-judge-n 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_jsonl(p: Path):
    if not p.is_file():
        return []
    out = []
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _ok_count(p: Path) -> int:
    n = 0
    for r in _load_jsonl(p):
        if r.get("error"):
            continue
        if not (r.get("summary_text") or "").strip():
            continue
        if not r.get("records"):
            continue
        n += 1
    return n


def _ir_summary(p: Path):
    """Return the LAST summary row (in case the file has multiple from
    repeated runs in append mode — newer summaries have the richer
    schema with rougeL/bertscore keys)."""
    last = None
    for r in _load_jsonl(p):
        if r.get("type") == "summary":
            last = r
    if last is None:
        return 0, {}
    return last.get("n_scored", 0), last.get("aggregate", {})


def _ragas_stats(p: Path):
    faith = []; rel = []
    for r in _load_jsonl(p):
        f = r.get("faithfulness"); rl = r.get("answer_relevancy")
        if isinstance(f, (int, float)): faith.append(f)
        if isinstance(rl, (int, float)): rel.append(rl)
    return (
        len(faith),
        (sum(faith) / len(faith)) if faith else None,
        (sum(rel) / len(rel)) if rel else None,
    )


def _fmt(v, prec=3):
    if v is None or (isinstance(v, float) and (v != v)):  # NaN
        return "—"
    if isinstance(v, (int, float)):
        return f"{v:.{prec}f}"
    return str(v)


# Only show the Qwen-3.5-397b judge in the report (matches the category
# report). Other judges' files remain on disk but aren't surfaced here.
JUDGE_DISPLAY = {
    "qwen_qwen3_5_397b_a17b": "qwen-3.5-397b",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("reports/eval_scico"))
    ap.add_argument("--min-ok", type=int, default=50,
                    help="Drop dirs with fewer than this many healthy rows.")
    ap.add_argument("--min-judge-n", type=int, default=30,
                    help="Drop judge cells with fewer than this many scored rows.")
    ap.add_argument("--out", type=Path,
                    default=Path("reports/eval_scico/_eval_report.md"))
    args = ap.parse_args()

    # Collect + filter dirs
    dirs = []
    for d in sorted(args.root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if not (name.startswith("agent_") or name.startswith("pipeline_")):
            continue
        if name.endswith("_smoke") or name.endswith("_test") or "smoke" in name:
            continue
        af = d / "answers.jsonl"
        if not af.is_file():
            continue
        ok = _ok_count(af)
        if ok < args.min_ok:
            continue
        dirs.append((name, d, ok))

    if not dirs:
        print(f"No dirs survived the >= {args.min_ok}-ok filter.")
        return 0

    md = []
    md.append(f"# Eval report\n")
    md.append(f"_Filter: dirs with ≥ {args.min_ok} healthy rows; "
              f"judge cells with ≥ {args.min_judge_n} scored rows._\n")
    md.append(f"_Generated from `{args.root}`._\n")

    # ---- Table 1: Retrieval + NLG ----
    md.append("\n## 1. Retrieval + NLG (set-level + rank-aware + ROUGE-L + BERTScore)\n")
    md.append("| Dir | ok | n_ir | P_set | R_set | F1_set | P@5 | nDCG@15 | R@15 | MRR | rougeL_evi | bert_evi |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, d, ok in dirs:
        n_ir, a = _ir_summary(d / "ir_metrics.jsonl")
        md.append(
            f"| {name} | {ok} | {n_ir} | "
            f"{_fmt(a.get('P_set'))} | {_fmt(a.get('R_set'))} | {_fmt(a.get('F1_set'))} | "
            f"{_fmt(a.get('P@5'))} | {_fmt(a.get('nDCG@15'))} | {_fmt(a.get('R@15'))} | {_fmt(a.get('RR'))} | "
            f"{_fmt(a.get('rougeL_vs_evidence'))} | {_fmt(a.get('bertscore_f1_vs_evidence'))} |"
        )

    md.append("\n_Note: `rougeL_vs_gold` / `bertscore_f1_vs_gold` excluded — testset's "
              "`gold_summary` is empty for these rows. Negative `bert_evi` reflects "
              "metadata-only evidence (no abstracts in records); not a bug._\n")

    # ---- Table 2: LLM-as-judge ----
    md.append("\n## 2. LLM-as-judge (RAGAS faithfulness / answer_relevancy)\n")
    md.append(f"_Cells with n < {args.min_judge_n} excluded._\n")
    md.append("| Dir | Judge | n | Faithfulness | Relevancy |")
    md.append("|---|---|---:|---:|---:|")
    for name, d, _ in dirs:
        for slug, display in JUDGE_DISPLAY.items():
            rf = d / f"ragas_{slug}.jsonl"
            if not rf.is_file():
                continue
            n, faith, rel = _ragas_stats(rf)
            if n < args.min_judge_n:
                continue
            md.append(f"| {name} | {display} | {n} | {_fmt(faith)} | {_fmt(rel)} |")

    # ---- Headline aggregates ----
    md.append("\n## 3. Cross-judge faithfulness means (per dir, judges with n ≥ "
              f"{args.min_judge_n})\n")
    md.append("| Dir | mean faithfulness | n judges | judges used |")
    md.append("|---|---:|---:|---|")
    for name, d, _ in dirs:
        vals = []
        names_used = []
        for slug, display in JUDGE_DISPLAY.items():
            rf = d / f"ragas_{slug}.jsonl"
            if not rf.is_file():
                continue
            n, faith, _ = _ragas_stats(rf)
            if n < args.min_judge_n or faith is None:
                continue
            vals.append(faith)
            names_used.append(display)
        if not vals:
            continue
        mean = sum(vals) / len(vals)
        md.append(f"| {name} | {_fmt(mean)} | {len(vals)} | {', '.join(names_used)} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"# wrote {args.out}  ({len(dirs)} dirs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
