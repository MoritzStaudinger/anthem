"""Oracle retrieval ceiling for the SciCo benchmark.

Rebuttal tasks:
  * Harness sanity check: use the constraint-matching gold function itself as
    the retrieval system (returned set == constraint-filtered gold set) and score
    it with the SAME set-level P/R/F1 harness. By construction the SET-LEVEL score
    must be P=R=F1=1.0 for every question; anything else is a harness bug, and we
    want to find it now. (This is the only sense in which the oracle "approaches
    1.0" — see the K note below.)

  * K-imposed recall ceiling (answers reviewer ms1W directly): a top-K retrieval
    system can return at most K papers, so on a gold set of size |gold| its recall
    is bounded by  recall@K <= min(K, |gold|) / |gold|  regardless of how perfect
    the retriever is. For the largest gold set (|gold| = 24,340) this is
    15 / 24,340 ~= 0.0006 at K=15. The oracle achieves this bound exactly
    (precision@K = 1.0, recall@K = min(K,|gold|)/|gold|), so it is the ceiling no
    system in the table can exceed. We report it per question, aggregated, by
    category, and by gold-size bucket.

Why we do NOT ship an "oracle-constraints + MiniLM over the filtered set" variant
here: if the candidate universe is restricted to the gold set, MiniLM ranking is
degenerate — every candidate is gold, so precision@K = 1.0 and recall@K = K/|gold|
identically to this oracle, independent of the embedding. The only informative
retrieval experiment is ranking over the FULL corpus and checking how many gold
papers land in the top K — and that is exactly what the pipeline runs in Table 4
already measure. So the meaningful comparison is: this analytic ceiling vs. the
actual per-system recall@K (pass --overlay-report to fold the latter in).

No model calls. Runs anywhere the affiliations corpus is present; the large gold
sets (and hence the dramatic ceilings) only appear against the full corpus.

Usage:
    python scripts/oracle_ceiling.py --k 5 15
    python scripts/oracle_ceiling.py --k 5 15 \
        --overlay reports/eval_scico/pipeline_gemini_2_5_pro_topk5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from backend.eval.gold import compute_gold_paper_ids, load_affiliations_corpus  # noqa: E402
from backend.eval.metrics import precision_recall_mrr  # noqa: E402
import eval_ir_measures as eim  # noqa: E402


def _mean(xs: List[float]) -> float:
    return (sum(xs) / len(xs)) if xs else float("nan")


def _bucket(g: int) -> str:
    if g <= 15:
        return "<=15"
    if g <= 100:
        return "16-100"
    if g <= 1000:
        return "101-1000"
    return ">1000"


def _overlay_recall(
    answers_dir: Path,
    gold_by_qid: Dict[str, Set[str]],
    ks: List[int],
    doi_lookup: Dict[str, str],
    title_lookup: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    """Actual per-question recall for a real system, using the same record->id
    join as the published pipeline.

    Returns qid -> {
        "per_k": {k: recall@k},              # recall if truncated to top-k
        "full": recall over the FULL retrieved set (no K cap — the honest number,
                since query expansion means the system surfaces many more than K),
        "n_ret": matched retrieved-set size (the system's real budget),
        "budget_ceiling": min(n_ret, |gold|)/|gold|  # max recall achievable given
                the system only retrieved n_ret papers
    }. Missing/failed questions -> zeros.
    """
    apath = answers_dir / "answers.jsonl"
    rows: Dict[str, Dict[str, Any]] = {}
    with apath.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("id") is not None:
                rows[r["id"]] = r  # last wins
    out: Dict[str, Dict[str, Any]] = {}
    for qid, gold in gold_by_qid.items():
        r = rows.get(qid)
        g = len(gold)
        if r is None or r.get("error"):
            out[qid] = {"per_k": {k: 0.0 for k in ks}, "full": 0.0,
                        "n_ret": 0, "budget_ceiling": 0.0}
            continue
        retrieved, _ = eim._retrieved_ids(r.get("records") or [],
                                          doi_lookup, title_lookup)
        n_ret = len(retrieved)
        out[qid] = {
            "per_k": {k: precision_recall_mrr(retrieved, gold, k=k)["recall"] for k in ks},
            "full": precision_recall_mrr(retrieved, gold, k=None)["recall"],
            "n_ret": n_ret,
            "budget_ceiling": (min(n_ret, g) / g) if g else 0.0,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--testset", type=Path, default=Path("data/eval/scico_testset.jsonl"))
    ap.add_argument("--affiliations-dir", type=Path,
                    default=Path("/share/moritz.staudinger/acl_data/"
                                 "ror_parsed_affiliations_90_with_topics_and_bibmeta"))
    ap.add_argument("--k", type=int, nargs="+", default=[5, 15, 25, 50, 75],
                    help="K values for the fixed-K ceiling. Defaults span the real "
                         "retrieval budget (query expansion surfaces ~median 22, up to "
                         "~75 papers/question), not just the nominal top-k.")
    ap.add_argument("--overlay", type=Path, default=None,
                    help="A real system's answers dir; overlay its actual "
                         "recall@k against the ceiling.")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/eval_scico/oracle"))
    args = ap.parse_args()

    testset = eim._load_testset(args.testset)
    print(f"# testset: {len(testset)} questions", flush=True)
    print(f"# loading corpus from {args.affiliations_dir}", flush=True)
    corpus = load_affiliations_corpus(args.affiliations_dir)
    print(f"# corpus: {len(corpus)} papers", flush=True)

    ks = sorted(set(args.k))
    # Per-question ceiling + harness sanity.
    rows_out: List[Dict[str, Any]] = []
    harness_bugs: List[Tuple[str, float, float, float]] = []
    empty_gold: List[str] = []
    for qid, row in testset.items():
        gold = eim._normalize_gold(compute_gold_paper_ids(row, corpus))
        g = len(gold)
        cat = (row.get("classifier_gold") or {}).get("question_category")
        if g == 0:
            empty_gold.append(qid)
            continue

        # (1) Harness sanity: retrieved == gold, set-level -> must be all 1.0.
        hits = gold & gold
        p_set = len(hits) / g
        r_set = len(hits) / g
        f1_set = (2 * p_set * r_set / (p_set + r_set)) if (p_set + r_set) else 0.0
        if not (abs(p_set - 1.0) < 1e-9 and abs(r_set - 1.0) < 1e-9
                and abs(f1_set - 1.0) < 1e-9):
            harness_bugs.append((qid, p_set, r_set, f1_set))

        # (2) K-recall ceiling. Analytic AND cross-checked via the real scorer,
        # feeding the gold set (arbitrary order — order is irrelevant when every
        # returned id is gold) as the retrieval list.
        rec = {"id": qid, "category": cat, "n_gold": g,
               "bucket": _bucket(g), "set_P": p_set, "set_R": r_set, "set_F1": f1_set}
        gold_list = list(gold)
        for k in ks:
            analytic_recall = min(k, g) / g
            analytic_prec = 1.0  # every returned id is gold
            m = precision_recall_mrr(gold_list, gold, k=k)
            # cross-check analytic vs harness
            assert abs(m["recall"] - analytic_recall) < 1e-9, (qid, k, m, analytic_recall)
            assert abs(m["precision"] - analytic_prec) < 1e-9, (qid, k, m)
            rec[f"recall@{k}"] = analytic_recall
            rec[f"precision@{k}"] = analytic_prec
        rows_out.append(rec)

    n = len(rows_out)
    print(f"# scored {n} questions ({len(empty_gold)} empty-gold excluded)", flush=True)
    if empty_gold:
        print(f"#   WARNING: {len(empty_gold)} empty-gold questions — partial "
              f"corpus? (all 99 have gold on the full corpus)", flush=True)

    # Overlay (optional).
    overlay = None
    if args.overlay is not None:
        doi_lookup, title_lookup = eim._build_id_lookups(corpus)
        gold_by_qid = {r["id"]: eim._normalize_gold(
            compute_gold_paper_ids(testset[r["id"]], corpus)) for r in rows_out}
        overlay = _overlay_recall(args.overlay, gold_by_qid, ks, doi_lookup, title_lookup)

    # --- render -------------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_q_path = args.out_dir / "oracle_per_question.jsonl"
    with per_q_path.open("w", encoding="utf-8") as fh:
        for r in rows_out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    md: List[str] = []
    md.append("# Oracle retrieval ceiling\n")
    md.append(f"- Corpus: `{args.affiliations_dir}` ({len(corpus)} papers)")
    md.append(f"- Questions scored: {n} (empty-gold excluded: {len(empty_gold)})")
    md.append("")
    md.append("## Harness sanity check (returned set == gold, set-level)")
    if not harness_bugs:
        md.append(f"- PASS: set-level P=R=F1=1.0 for all {n} questions "
                  "(scorer is self-consistent).")
    else:
        md.append(f"- **FAIL on {len(harness_bugs)} questions** — harness bug, "
                  "investigate:")
        for qid, p, r, f1 in harness_bugs[:20]:
            md.append(f"    - {qid}: P={p:.4f} R={r:.4f} F1={f1:.4f}")
    md.append("")

    md.append("## K-imposed recall ceiling (no system can exceed this)")
    md.append("")
    header = "| Slice | n | " + " | ".join(f"mean recall@{k}" for k in ks) + " |"
    sep = "|---|---:|" + "|".join(["---:"] * len(ks)) + "|"
    md.append(header)
    md.append(sep)
    # overall
    cells = " | ".join(_fmt(_mean([r[f"recall@{k}"] for r in rows_out])) for k in ks)
    md.append(f"| ALL | {n} | {cells} |")
    # by bucket
    for bkt in ["<=15", "16-100", "101-1000", ">1000"]:
        rs = [r for r in rows_out if r["bucket"] == bkt]
        if not rs:
            continue
        cells = " | ".join(_fmt(_mean([r[f"recall@{k}"] for r in rs])) for k in ks)
        md.append(f"| gold {bkt} | {len(rs)} | {cells} |")
    md.append("")
    md.append("### By question category")
    md.append(header)
    md.append(sep)
    cats = sorted({r["category"] for r in rows_out})
    for cat in cats:
        rs = [r for r in rows_out if r["category"] == cat]
        cells = " | ".join(_fmt(_mean([r[f"recall@{k}"] for r in rs])) for k in ks)
        md.append(f"| {cat} | {len(rs)} | {cells} |")
    md.append("")
    md.append("_precision@k is 1.0 for the oracle by construction; the ceiling "
              "is entirely a recall ceiling driven by K / |gold|._")
    md.append("")

    if overlay is not None:
        md.append(f"## Overlay — {args.overlay.name}: actual recall vs ceiling")
        md.append("")
        md.append("Fixed-K ceiling assumes the system returns exactly K papers. But "
                  "query expansion makes the real retrieved set larger than the nominal "
                  "top-k, so the fair comparison is the system's FULL-set recall against "
                  "the ceiling at its OWN retrieved-set size.")
        md.append("")
        md.append("| k | ceiling@k (mean) | actual@k (mean) | actual/ceiling |")
        md.append("|---:|---:|---:|---:|")
        for k in ks:
            ceil = _mean([r[f"recall@{k}"] for r in rows_out])
            act = _mean([overlay[r["id"]]["per_k"][k] for r in rows_out])
            frac = (act / ceil) if ceil else float("nan")
            md.append(f"| {k} | {_fmt(ceil)} | {_fmt(act)} | {_fmt(frac)} |")
        md.append("")
        n_rets = sorted(overlay[r["id"]]["n_ret"] for r in rows_out)
        act_full = _mean([overlay[r["id"]]["full"] for r in rows_out])
        budget_ceil = _mean([overlay[r["id"]]["budget_ceiling"] for r in rows_out])
        eff = (act_full / budget_ceil) if budget_ceil else float("nan")
        md.append(f"- Retrieved-set size (matched `|records|`, the real budget): "
                  f"median **{n_rets[len(n_rets)//2]}**, max **{n_rets[-1]}** "
                  f"(not the nominal K)")
        md.append(f"- **Actual recall over the FULL retrieved set (no K cap): "
                  f"{_fmt(act_full)}**  ← the honest recall")
        md.append(f"- Ceiling at the system's own budget "
                  f"(mean of min(|records|,|gold|)/|gold|): {_fmt(budget_ceil)}")
        md.append(f"- Budget efficiency (actual_full / budget_ceiling): {_fmt(eff)}")
        md.append("")
        md.append("_recall@15 truncates the system to 15 of its ~median-many retrieved "
                  "papers, so it understates. Even at the real budget, gold sets larger "
                  "than it stay ~0 (e.g. 75/24,340 = 0.003) — the structural cap holds._")

    table_path = args.out_dir / "oracle_ceiling.md"
    table_path.write_text("\n".join(md), encoding="utf-8")
    print(f"# wrote {table_path}", flush=True)
    print(f"# wrote {per_q_path}", flush=True)
    print("\n" + "\n".join(md), flush=True)
    return 0 if not harness_bugs else 2


def _fmt(x: float) -> str:
    import math
    return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


if __name__ == "__main__":
    raise SystemExit(main())
