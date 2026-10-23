"""Standard IR-metric pass over saved eval_scico answers via ir_measures.

Reads $answers_dir/answers.jsonl (any run of eval_scico.py with
--answers-out), joins each row to the gold paper-id set via the same
compute_gold_paper_ids function the in-house metrics use, and scores
with ir_measures (nDCG, MAP, P, R, MRR, Bpref at multiple k values).

Writes $answers_dir/ir_metrics_<k_list>.jsonl with per-row scores plus
an aggregate summary at the end. Joins back to other reports on `id`.

This is purely additive — does NOT modify answers.jsonl, the in-house
metrics, or RAGAS reports. Run after generation; idempotent across
re-runs (resume support: skips ids already in the output file).

CAVEAT — ranking semantics:
  The `records` list per row is collected_evidence from agent_tools.py:
  deduplicated by DOI/title, in encounter order across the agent's tool
  calls. This is not a classical retrieval-rank list. We use the
  encounter order as the ranking (descending integer scores
  `len(records) - i`), so nDCG@k / MRR / etc. reflect "did relevant
  papers come up early in the agent's exploration" — not pure retrieval
  rank fidelity. For deterministic-retrieval runs (eval_acl_retrieval.py
  output), the same script applies and the ranking IS true similarity
  order; the interpretation is then classical.

Usage:
    python scripts/eval_ir_measures.py \\
        --answers-dir reports/eval_scico/agent_claude_sonnet_4_6

    # Custom k values + different testset
    python scripts/eval_ir_measures.py \\
        --answers-dir reports/eval_scico/agent_gemini_2_5_pro \\
        --k 5 10 15 30 100

    # Sweep all completed agent dirs
    for d in reports/eval_scico/agent_*; do
        [ -f "$d/answers.jsonl" ] || continue
        python scripts/eval_ir_measures.py --answers-dir "$d"
    done
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)


def _load_testset(path: Path) -> Dict[str, Dict[str, Any]]:
    """Return id -> testset row (needed for compute_gold_paper_ids)."""
    out: Dict[str, Dict[str, Any]] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("id"):
                out[row["id"]] = row
    return out


def _load_done_ids(out_path: Path) -> Set[str]:
    """Resume support: ids whose IR scores are already saved."""
    done: Set[str] = set()
    if not out_path.exists():
        return done
    with out_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rec = json.loads(line)
                if "id" in rec:
                    done.add(rec["id"])
            except json.JSONDecodeError:
                continue
    return done


_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "doi:")
_TITLE_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def _normalize_doi(s: Any) -> str:
    if not s:
        return ""
    s = str(s).strip().lower()
    for prefix in _DOI_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s.rstrip(".,;) ")


def _normalize_title(s: Any) -> str:
    if not s:
        return ""
    s = str(s).strip().lower()
    s = _TITLE_PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def _normalize_paper_id(s: Any) -> str:
    if not s:
        return ""
    return str(s).strip().lower()


def _build_id_lookups(
    corpus: List[Dict[str, Any]],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build (doi → paper_id, normalized_title → paper_id) maps from corpus.

    paper_ids in the corpus are already canonical (e.g. '2021.acl-long.541');
    we lowercase them for consistency with the agent-side normalization."""
    doi_to_pid: Dict[str, str] = {}
    title_to_pid: Dict[str, str] = {}
    for p in corpus:
        pid = _normalize_paper_id(p.get("paper_id"))
        if not pid:
            continue
        doi = _normalize_doi(p.get("doi"))
        if doi:
            doi_to_pid.setdefault(doi, pid)
        title = _normalize_title(p.get("title"))
        if title:
            title_to_pid.setdefault(title, pid)
    return doi_to_pid, title_to_pid


def _retrieved_ids(
    records: List[Dict[str, Any]],
    doi_lookup: Dict[str, str],
    title_lookup: Dict[str, str],
) -> Tuple[List[str], int]:
    """Pull ordered paper IDs from the evidence buffer. Encounter order
    is preserved — see module docstring for ranking semantics.

    Returns (ordered_paper_ids, n_unmatched). An evidence record matches
    when (a) it carries `paper_id` directly (post-fix runs) OR (b) its
    DOI normalizes into doi_lookup OR (c) its title normalizes into
    title_lookup."""
    out: List[str] = []
    seen: Set[str] = set()
    unmatched = 0
    for r in records or []:
        if not isinstance(r, dict):
            continue
        pid = _normalize_paper_id(r.get("paper_id"))
        if not pid:
            doi = _normalize_doi(r.get("DOI") or r.get("doi"))
            if doi:
                pid = doi_lookup.get(doi, "")
        if not pid:
            title = _normalize_title(r.get("title"))
            if title:
                pid = title_lookup.get(title, "")
        if not pid:
            unmatched += 1
            continue
        if pid in seen:
            continue
        seen.add(pid)
        out.append(pid)
    return out, unmatched


def _normalize_gold(gold: Set[str]) -> Set[str]:
    """Normalize gold paper IDs to the same casing as _retrieved_ids."""
    return {_normalize_paper_id(g) for g in gold if g}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--answers-dir", type=Path, required=True,
                   help="Dir containing answers.jsonl")
    p.add_argument("--testset", type=Path,
                   default=Path("data/eval/scico_testset.jsonl"),
                   help="Testset path (needed to compute gold per row)")
    p.add_argument("--affiliations-dir", type=Path,
                   default=Path("/share/moritz.staudinger/acl_data/"
                                "ror_parsed_affiliations_90_with_topics_and_bibmeta"),
                   help="Enriched-affiliations dir for gold computation")
    p.add_argument("--k", type=int, nargs="+", default=[5, 10, 15, 30],
                   help="k values to report (default: 5 10 15 30)")
    p.add_argument("--out-name", default=None,
                   help="Output filename (default: ir_metrics.jsonl)")
    p.add_argument("--no-resume", action="store_true",
                   help="Overwrite the output file instead of appending.")
    p.add_argument("--bertscore", action="store_true",
                   help="Also compute BERTScore (heavy: downloads roberta-large, "
                        "GPU recommended). Off by default.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    answers_path = args.answers_dir / "answers.jsonl"
    if not answers_path.exists():
        print(f"ERROR: {answers_path} not found", file=sys.stderr)
        return 1
    if not args.testset.exists():
        print(f"ERROR: testset {args.testset} not found", file=sys.stderr)
        return 1

    try:
        from ir_measures import calc_aggregate, iter_calc, nDCG, MAP, P, R, RR, Bpref
    except ImportError as e:
        print(f"ERROR: ir_measures not installed ({e}). "
              "pip install 'ir_measures>=0.3'", file=sys.stderr)
        return 1

    try:
        from rouge_score import rouge_scorer
    except ImportError as e:
        print(f"ERROR: rouge_score not installed ({e}). "
              "pip install 'rouge_score>=0.1.2'", file=sys.stderr)
        return 1

    _rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    # BERTScore is optional — heavy (downloads roberta-large, GPU-friendly).
    bert_scorer = None
    if args.bertscore:
        try:
            from bert_score import BERTScorer  # type: ignore
        except ImportError as e:
            print(f"ERROR: bert-score not installed ({e}). "
                  "pip install 'bert-score>=0.3.13'", file=sys.stderr)
            return 1
        print("# loading BERTScore (roberta-large, may download ~440MB)",
              flush=True)
        bert_scorer = BERTScorer(model_type="roberta-large", lang="en",
                                 rescale_with_baseline=True, batch_size=16)

    from backend.eval.gold import compute_gold_paper_ids, load_affiliations_corpus  # noqa: E402

    print(f"# loading testset {args.testset}", flush=True)
    testset = _load_testset(args.testset)
    print(f"# testset has {len(testset)} rows", flush=True)

    print(f"# loading affiliations corpus from {args.affiliations_dir}",
          flush=True)
    corpus = load_affiliations_corpus(args.affiliations_dir)
    print("# building DOI / title → paper_id lookups", flush=True)
    doi_lookup, title_lookup = _build_id_lookups(corpus)
    print(f"# lookups: {len(doi_lookup)} DOIs, {len(title_lookup)} titles",
          flush=True)

    out_name = args.out_name or "ir_metrics.jsonl"
    out_path = args.answers_dir / out_name

    done: Set[str] = set() if args.no_resume else _load_done_ids(out_path)
    if done:
        print(f"# resume: {len(done)} rows already scored in {out_path.name}",
              flush=True)

    # Build the metric list — every k value × every k-metric, plus the
    # k-free aggregate metrics. ir_measures wants metric objects.
    metric_objs = []
    for k in args.k:
        metric_objs.append(nDCG @ k)
        metric_objs.append(P @ k)
        metric_objs.append(R @ k)
    metric_objs.append(MAP)
    metric_objs.append(RR)
    metric_objs.append(Bpref)

    qrels: Dict[str, Dict[str, int]] = {}
    run: Dict[str, Dict[str, float]] = {}
    row_skipped: List[Tuple[str, str]] = []
    # Cache (row, ts_row) per id for NLG metrics computed after the loop.
    _row_cache: Dict[str, Tuple[Dict[str, Any], Dict[str, Any]]] = {}

    mode = "w" if args.no_resume else "a"
    new_ids: List[str] = []
    n_no_id = 0
    n_already_done = 0

    with answers_path.open() as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = row.get("id")
            if rid is None:
                n_no_id += 1
                continue
            if rid in done:
                n_already_done += 1
                continue
            if row.get("error"):
                row_skipped.append((rid, f"agent error: {row.get('error')}"))
                continue

            ts_row = testset.get(rid)
            if not ts_row:
                row_skipped.append((rid, "no row in testset"))
                continue

            gold = _normalize_gold(compute_gold_paper_ids(ts_row, corpus))
            if not gold:
                row_skipped.append((rid, "empty gold"))
                continue

            retrieved, unmatched = _retrieved_ids(
                row.get("records") or [], doi_lookup, title_lookup
            )
            if not retrieved:
                row_skipped.append((
                    rid,
                    f"no matched records ({unmatched} unmatched)",
                ))
                continue

            # qrels: docid -> relevance (binary, 1 for gold)
            qrels[rid] = {pid: 1 for pid in gold}
            # run: docid -> score (descending integer for encounter rank)
            n = len(retrieved)
            run[rid] = {pid: float(n - i) for i, pid in enumerate(retrieved)}
            new_ids.append(rid)
            # Cache the row + question for later NLG scoring
            _row_cache[rid] = (row, ts_row)

    print(f"# {len(new_ids)} new rows to score, {len(row_skipped)} skipped, "
          f"{n_already_done} already done, {n_no_id} had no id", flush=True)
    if row_skipped:
        from collections import Counter
        reasons = Counter(_short_reason(r) for _, r in row_skipped)
        print("# skip reasons:", flush=True)
        for r, n in reasons.most_common():
            print(f"#   {n:>4}  {r}", flush=True)
        print("# first skipped rows (up to 10):", flush=True)
        for rid, reason in row_skipped[:10]:
            print(f"#   {rid}: {reason}", flush=True)

    if not new_ids:
        print("# nothing to score — exiting", flush=True)
        return 0

    # Per-row scores (rank-aware metrics via ir_measures)
    per_row: Dict[str, Dict[str, float]] = {rid: {} for rid in new_ids}
    for m in iter_calc(metric_objs, qrels, run):
        per_row[m.query_id][str(m.measure)] = float(m.value)

    # Aggregate (mean across new rows only — caller can compute global
    # aggregates from the per-row output, but we surface a sanity number)
    agg = calc_aggregate(metric_objs, qrels, run)

    # Set-level (rank-free) metrics: precision/recall/F1 computed over
    # the FULL retrieved set, not a top-k slice. These are the honest
    # primary metrics for this eval because the persisted `records` list
    # is encounter-order, not a global similarity rank — so @k numbers
    # over it are not classical rank-aware retrieval metrics.
    set_per_row: Dict[str, Dict[str, float]] = {}
    for rid in new_ids:
        retrieved_set = set(run[rid].keys())
        gold_set = set(qrels[rid].keys())
        hits = retrieved_set & gold_set
        n_ret = len(retrieved_set)
        n_gold = len(gold_set)
        p = (len(hits) / n_ret) if n_ret else 0.0
        r = (len(hits) / n_gold) if n_gold else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        set_per_row[rid] = {
            "P_set": p, "R_set": r, "F1_set": f1,
        }

    def _nanmean(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return (sum(xs) / len(xs)) if xs else float("nan")

    set_agg = {
        "P_set":  _nanmean([set_per_row[rid]["P_set"]  for rid in new_ids]),
        "R_set":  _nanmean([set_per_row[rid]["R_set"]  for rid in new_ids]),
        "F1_set": _nanmean([set_per_row[rid]["F1_set"] for rid in new_ids]),
    }

    # ROUGE-L + (optional) BERTScore — NLG metrics on summary_text.
    #   *_vs_gold     : compare answer to testset.gold_summary
    #   *_vs_evidence : compare answer to concatenated record text
    #                   (title + authors + year — abstracts are stripped
    #                    from records, so this is metadata-only and
    #                    intentionally a proxy)
    def _evidence_text(records: List[Dict[str, Any]]) -> str:
        parts = []
        for r in records or []:
            if not isinstance(r, dict):
                continue
            t = (r.get("title") or "").strip()
            a = r.get("authors") or []
            if isinstance(a, list):
                a = ", ".join(x for x in a if x)
            y = r.get("publication_year") or r.get("year") or ""
            if t:
                parts.append(f"{t}. {a} ({y}).".strip())
        return "\n".join(parts)

    nlg_per_row: Dict[str, Dict[str, float]] = {}
    # Pre-collect string pairs for batched BERTScore (more efficient than
    # one-call-per-row on GPU).
    bs_preds_gold: List[str] = []
    bs_refs_gold:  List[str] = []
    bs_preds_evi:  List[str] = []
    bs_refs_evi:   List[str] = []
    bs_idx_gold:   List[str] = []
    bs_idx_evi:    List[str] = []

    for rid in new_ids:
        row, ts_row = _row_cache[rid]
        pred = (row.get("summary_text") or "").strip()
        gold = (ts_row.get("gold_summary") or "").strip()
        evi  = _evidence_text(row.get("records") or [])

        d: Dict[str, float] = {}
        if pred and gold:
            d["rougeL_vs_gold"] = float(
                _rouge.score(gold, pred)["rougeL"].fmeasure
            )
            bs_preds_gold.append(pred); bs_refs_gold.append(gold); bs_idx_gold.append(rid)
        else:
            d["rougeL_vs_gold"] = float("nan")

        if pred and evi:
            d["rougeL_vs_evidence"] = float(
                _rouge.score(evi, pred)["rougeL"].fmeasure
            )
            bs_preds_evi.append(pred); bs_refs_evi.append(evi); bs_idx_evi.append(rid)
        else:
            d["rougeL_vs_evidence"] = float("nan")

        nlg_per_row[rid] = d

    if bert_scorer is not None:
        if bs_preds_gold:
            print(f"# BERTScore vs_gold: scoring {len(bs_preds_gold)} pairs",
                  flush=True)
            _, _, f1_gold = bert_scorer.score(bs_preds_gold, bs_refs_gold)
            for i, rid in enumerate(bs_idx_gold):
                nlg_per_row[rid]["bertscore_f1_vs_gold"] = float(f1_gold[i].item())
        if bs_preds_evi:
            print(f"# BERTScore vs_evidence: scoring {len(bs_preds_evi)} pairs",
                  flush=True)
            _, _, f1_evi = bert_scorer.score(bs_preds_evi, bs_refs_evi)
            for i, rid in enumerate(bs_idx_evi):
                nlg_per_row[rid]["bertscore_f1_vs_evidence"] = float(f1_evi[i].item())
        # Fill NaN for rows that had no usable text on one side
        for rid in new_ids:
            nlg_per_row[rid].setdefault("bertscore_f1_vs_gold", float("nan"))
            nlg_per_row[rid].setdefault("bertscore_f1_vs_evidence", float("nan"))

    nlg_keys = sorted({k for d in nlg_per_row.values() for k in d.keys()})
    nlg_agg = {k: _nanmean([nlg_per_row[rid].get(k, float("nan")) for rid in new_ids])
               for k in nlg_keys}

    with out_path.open(mode, encoding="utf-8") as fout:
        for rid in new_ids:
            rec = {
                "id": rid,
                "n_retrieved": len(run.get(rid, {})),
                "n_gold": len(qrels.get(rid, {})),
                **per_row[rid],
                **set_per_row[rid],
                **nlg_per_row[rid],
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # Append a summary line — type=summary so consumers can filter.
        summary = {
            "type": "summary",
            "answers_dir": str(args.answers_dir),
            "n_scored": len(new_ids),
            "n_skipped": len(row_skipped),
            "skipped_reasons": _count_reasons(row_skipped),
            "aggregate": {
                **{str(m): float(v) for m, v in agg.items()},
                **set_agg,
                **nlg_agg,
            },
        }
        fout.write(json.dumps(summary, ensure_ascii=False) + "\n")
        fout.flush()

    print(f"# wrote {len(new_ids)} per-row + 1 summary to {out_path}",
          flush=True)
    print("# rank-aware aggregate (mean across new rows):", flush=True)
    for m, v in agg.items():
        print(f"#   {m}: {v:.4f}", flush=True)
    print("# set-level aggregate (full retrieved set, no rank):", flush=True)
    for k, v in set_agg.items():
        print(f"#   {k}: {v:.4f}", flush=True)
    print("# NLG aggregate (ROUGE-L + optional BERTScore):", flush=True)
    for k, v in nlg_agg.items():
        print(f"#   {k}: {v:.4f}", flush=True)
    if row_skipped:
        print(f"# skipped {len(row_skipped)} rows:", flush=True)
        for rid, reason in row_skipped[:10]:
            print(f"#   {rid}: {reason}", flush=True)
        if len(row_skipped) > 10:
            print(f"#   ... +{len(row_skipped) - 10} more", flush=True)
    return 0


def _short_reason(reason: str) -> str:
    """Collapse parameterized reasons into stable buckets for counting."""
    if reason.startswith("agent error"):
        return "agent_error"
    if reason.startswith("no matched records"):
        return "no_matched_records"
    return reason


def _count_reasons(skipped: List[Tuple[str, str]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for _, r in skipped:
        bucket = _short_reason(r)
        out[bucket] = out.get(bucket, 0) + 1
    return out


if __name__ == "__main__":
    raise SystemExit(main())
