"""Build a first cross-(mode × model) table of retrieval + ROUGE-L
metrics from the SciCo eval answers.jsonl files written so far.

No LLM calls, no GPU. Reads:
  - reports/eval_scico/<mode>_<model_tag>/answers.jsonl
  - data/eval/scico_testset.jsonl   (for gold_paper_ids + gold_summary)
and emits:
  - reports/eval_scico/_first_table.md  (also stdout)

What's covered:
  - Retrieval: precision@k, recall@k, MRR (using each row's `records`
    against the testset's `gold_paper_ids`).
  - Generation: ROUGE-L F1 of `summary_text` vs `gold_summary`, and
    ROUGE-L of `summary_text` vs concatenated evidence (the retrieved
    records' titles+abstracts). Both computed for closed_book / pipeline
    / agent modes — for closed_book mode evidence is empty so the
    rougeL_vs_evidence column is NaN.

Coverage column: # rows in answers.jsonl / 99 testset rows. Lets you
see partial / timed-out runs at a glance (the DeepSeek pipeline run
that hit the 4 h wall before resume was added will show < 99).

Usage:
  python scripts/build_first_eval_table.py
  python scripts/build_first_eval_table.py --root reports/eval_scico --k 10
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

# Standardized metric libraries — reviewer-defensible numbers:
#   ir_measures wraps pytrec_eval (the de-facto IR-metric reference impl)
#   rouge_score is google-research's ROUGE implementation (Porter stemming
#     by default, matches Lin 2004 reference behavior)
# Both are pinned in dev-requirements.txt.
try:
    from ir_measures import calc_aggregate, P, R, nDCG  # noqa: E402
    from rouge_score import rouge_scorer  # noqa: E402
except ImportError as _e:
    raise SystemExit(
        f"ERROR: missing metric library ({_e}). "
        "Install with: pip install 'ir_measures>=0.3' 'rouge_score>=0.1.2'"
    )

_ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# Put the repo root on sys.path so `from backend.eval.gold import ...`
# works when this script is invoked as `python scripts/build_first_eval_table.py`
# (without that, only the scripts/ dir is on sys.path).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


# ACL Anthology DOI prefixes. Stripping these from a DOI gives the
# anthology paper_id, which is what compute_gold_paper_ids() returns.
# Pre-2020 papers use 10.3115/...; post-2020 use 10.18653/v1/... .
_ACL_DOI_PREFIXES = (
    "10.18653/v1/",
    "10.3115/",
    "https://doi.org/10.18653/v1/",
    "https://doi.org/10.3115/",
    "http://doi.org/10.18653/v1/",
    "http://doi.org/10.3115/",
    "doi.org/10.18653/v1/",
    "doi.org/10.3115/",
)


def _normalize_to_paper_id(rec: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of an ACL anthology paper_id from a record.

    Why this exists: pipeline-mode records set `paper_id` explicitly
    (scripts/eval_scico.py:427), but agent-mode records only carry
    `DOI`/`url`/`title` (backend/rag/agent_tools.py:_record_evidence).
    Without normalisation, the join key against gold paper_ids never
    matches for agent rows, and precision/recall round to zero even
    when retrieval was actually good.

    Match order:
      1) explicit paper_id field
      2) DOI/url stripped of any ACL prefix
      3) title (last-resort, will only match precomputed-from-title gold)
    """
    pid = rec.get("paper_id")
    if pid:
        return str(pid).strip()
    for k in ("DOI", "doi", "url"):
        v = (rec.get(k) or "").strip()
        if not v:
            continue
        vl = v.lower()
        for pre in _ACL_DOI_PREFIXES:
            if vl.startswith(pre):
                # Preserve the original case after the prefix — ACL paper_ids
                # are case-insensitive in practice but we don't want to munge.
                return v[len(pre):]
        # If the URL is an aclanthology.org link, extract from path:
        #   https://aclanthology.org/2023.acl-long.123/
        if "aclanthology.org/" in vl:
            tail = v.rstrip("/").rsplit("aclanthology.org/", 1)[-1]
            if tail:
                return tail
        # Last resort for non-ACL DOIs: use the DOI verbatim.
        return v
    title = (rec.get("title") or "").strip()
    return title or None


def _retrieved_ids(records: List[Dict[str, Any]]) -> List[str]:
    """Ordered, dedup'd retrieval list. Order preserved (MRR needs it)."""
    out: List[str] = []
    seen = set()
    for r in records or []:
        rid = _normalize_to_paper_id(r)
        if not rid:
            continue
        if rid in seen:
            continue
        seen.add(rid)
        out.append(rid)
    return out


def _pr_ndcg(retrieved: List[str], gold: List[str], k: int) -> Dict[str, float]:
    """precision@k, recall@k, nDCG@k via ir_measures (pytrec_eval backend).

    Binary relevance; nDCG uses pytrec_eval's standard log2 discount and
    handles |gold| < k normalization correctly. Returns the same key shape
    as the previous homegrown implementation so callers are unchanged.
    """
    if not gold:
        return {"precision": float("nan"), "recall": float("nan"),
                "ndcg": float("nan")}
    top_k = retrieved[:k]
    if not top_k:
        return {"precision": 0.0, "recall": 0.0, "ndcg": 0.0}
    # ir_measures wants qrels {qid: {docid: rel}} and run {qid: {docid: score}}.
    # Descending integer scores preserve the retrieved ordering.
    qrels = {"q": {pid: 1 for pid in gold}}
    n = len(top_k)
    run = {"q": {pid: float(n - i) for i, pid in enumerate(top_k)}}
    agg = calc_aggregate([P @ k, R @ k, nDCG @ k], qrels, run)
    by_name = {str(m): float(v) for m, v in agg.items()}
    return {
        "precision": by_name.get(f"P@{k}", float("nan")),
        "recall":    by_name.get(f"R@{k}", float("nan")),
        "ndcg":      by_name.get(f"nDCG@{k}", float("nan")),
    }


def _log2(x: float) -> float:
    return math.log(x, 2)


_WORD = re.compile(r"\w+", re.UNICODE)


def _tok(s: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(s or "")]


def _lcs_len(a: List[str], b: List[str]) -> int:
    """Length of the longest common subsequence — O(len(a)*len(b))
    space-efficient single-row DP. Good enough for ~few-hundred-token
    summaries."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    curr = [0] * (len(b) + 1)
    for i, x in enumerate(a, start=1):
        for j, y in enumerate(b, start=1):
            if x == y:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(curr[j - 1], prev[j])
        prev, curr = curr, prev
        for j in range(len(b) + 1):
            curr[j] = 0
    return prev[-1]


def _rougeL_f1(hyp: str, ref: str) -> float:
    """Token-level ROUGE-L F1 via google-research's rouge_score package
    (Porter stemming, matches Lin 2004 reference behavior)."""
    if not (hyp and hyp.strip()) or not (ref and ref.strip()):
        return float("nan")
    return _ROUGE_SCORER.score(ref, hyp)["rougeL"].fmeasure


def _evidence_text(records: List[Dict[str, Any]]) -> str:
    """Concatenate retrieved evidence (title + abstract) for
    rougeL_vs_evidence."""
    parts: List[str] = []
    for rec in records or []:
        title = rec.get("title") or ""
        abstract = rec.get("abstract") or rec.get("summary") or ""
        if title or abstract:
            parts.append(f"{title}\n{abstract}".strip())
    return "\n\n".join(parts)


def _nanmean(xs: List[float]) -> float:
    vals = [x for x in xs if x is not None and not math.isnan(x)]
    return mean(vals) if vals else float("nan")


def _fmt(v: float) -> str:
    return "  nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


_SKIP_DIR_PATTERNS = ("_smoke", "_test")
_SKIP_DIR_NAMES = {"pipeline_smoke"}


def _parse_dir_name(name: str) -> Optional[Tuple[str, str]]:
    """Recognize <mode>_<model_tag> dirs.

    Handles multiple historical naming conventions:
      - closed_book_* / closed_*   → mode=closed_book
      - pipeline_*                  → mode=pipeline
      - agent_* / answers_*         → mode=agent
        (answers_* is the eval_scico_answer.sbatch phase-1 convention)
      - test_* / *_smoke / *_test   → skipped (smoke/test runs)
    """
    if name in _SKIP_DIR_NAMES or name.startswith("test_"):
        return None
    if any(name.endswith(suf) for suf in _SKIP_DIR_PATTERNS):
        return None
    if name.startswith("closed_book_"):
        return "closed_book", name[len("closed_book_"):]
    if name.startswith("closed_"):
        return "closed_book", name[len("closed_"):]
    if name.startswith("pipeline_"):
        return "pipeline", name[len("pipeline_"):]
    if name.startswith("agent_"):
        return "agent", name[len("agent_"):]
    if name.startswith("answers_"):
        return "agent", name[len("answers_"):]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path,
                    default=Path("reports/eval_scico"),
                    help="Root containing <mode>_<model>/answers.jsonl dirs.")
    ap.add_argument("--testset", type=Path,
                    default=Path("data/eval/scico_testset.jsonl"))
    ap.add_argument("--k", type=int, default=20,
                    help="Top-k cutoff. Default 20 since gold sets are large "
                         "(median ~20-80 papers per row, max in the thousands "
                         "for field_evolution questions). k=10 caps recall too "
                         "aggressively against the bigger gold sets.")
    ap.add_argument("--out-md", type=Path,
                    default=Path("reports/eval_scico/_first_table.md"))
    ap.add_argument("--gold-paper-ids", type=Path, default=None,
                    help="Precomputed {row_id: [paper_id]} JSON. Skip this "
                         "and pass --affiliations-dir instead to compute gold "
                         "on the fly against the (evolving) corpus — same path "
                         "eval_scico.py uses.")
    ap.add_argument("--affiliations-dir", type=Path,
                    default=Path("/share/moritz.staudinger/acl_data/"
                                 "ror_parsed_affiliations_90_with_topics_and_bibmeta"),
                    help="Bibmeta affiliations corpus. Loaded once, then "
                         "compute_gold_paper_ids() is called per testset row "
                         "to derive gold paper_ids from the current corpus "
                         "snapshot. Pass empty string to disable (retrieval "
                         "metrics will be NaN).")
    args = ap.parse_args()

    testset = _load_jsonl(args.testset)
    if not testset:
        raise SystemExit(f"testset empty/missing: {args.testset}")
    testset_by_id: Dict[str, Dict[str, Any]] = {r["id"]: r for r in testset if r.get("id")}
    print(f"# testset: {len(testset_by_id)} rows from {args.testset}")

    # Filter ablations: each one applies a different SUBSET of gold's
    # criteria, so we can see "if only year mattered, would retrieval
    # match?" vs "if only topic mattered?" — and see which dimension
    # the retriever is good vs bad at.
    #
    # "overall" matches compute_gold_paper_ids exactly. The others are
    # local subsets computed in this script (gold.py doesn't expose a
    # mask param).
    FILTER_SETS: List[Tuple[str, frozenset]] = [
        ("year",     frozenset({"year"})),
        ("topic",    frozenset({"topic"})),
        ("inst",     frozenset({"institution"})),
        ("country",  frozenset({"country"})),
        ("overall",  frozenset({"year", "topic", "institution", "country"})),
    ]

    def _gold_for_row(row: Dict[str, Any], corpus: list,
                      filters_active: frozenset) -> set:
        """Mirrors backend.eval.gold.compute_gold_paper_ids but applies
        only the filters in filters_active. Used for the per-filter
        ablation breakdown."""
        cg = row.get("classifier_gold") or {}
        tm = row.get("topic_metadata") or {}
        cat = cg.get("question_category")
        uni1 = cg.get("uni1")
        uni2 = cg.get("uni2")
        y_start = cg.get("year_start")
        y_end = cg.get("year_end")
        closure = {str(x).lower() for x in (tm.get("cso_closure") or [])}
        insts_filter = {x for x in (uni1, uni2) if x}
        countries_filter: set = set()
        if cat == "cross_country_method":
            cc1 = (cg.get("country") or "").upper()
            cc2 = (cg.get("country_b") or "").upper()
            countries_filter = {c for c in (cc1, cc2) if c}
        # If a requested filter has no input (e.g. topic with no closure,
        # institution for field_evolution), skip that filter — empty gold
        # would be misleading.
        gold: set = set()
        for p in corpus:
            if "year" in filters_active:
                py = p["year"]
                if py is not None:
                    if y_start is not None and py < y_start:
                        continue
                    if y_end is not None and py > y_end:
                        continue
            if "topic" in filters_active and closure:
                if not (p["cso_union"] & closure):
                    continue
            if "institution" in filters_active:
                if cat in {"overview_university", "university_country_comparison"}:
                    if uni1 and uni1 not in p["institutions"]:
                        continue
                elif cat == "university_to_university_comparison":
                    if insts_filter and not (p["institutions"] & insts_filter):
                        continue
                # field_evolution / cross_country_method: no inst filter applies
            if "country" in filters_active and countries_filter:
                if not (p.get("countries", set()) & countries_filter):
                    continue
            gold.add(p["paper_id"])
        return gold

    # Resolve gold paper_ids per testset row. Two paths, in priority order:
    #   1) --gold-paper-ids precomputed JSON (fastest, repeatable)
    #   2) --affiliations-dir + on-the-fly compute_gold_paper_ids() (matches
    #      eval_scico.py's runtime path — slower the first time the corpus
    #      loads, but correct against the evolving corpus.)
    gold_by_rid: Dict[str, List[str]] = {}
    # (row_id, filter_name) -> list[paper_id], for the ablation breakdown
    gold_by_rid_filter: Dict[Tuple[str, str], List[str]] = {}
    if args.gold_paper_ids is not None:
        if not args.gold_paper_ids.is_file():
            raise SystemExit(f"--gold-paper-ids file not found: {args.gold_paper_ids}")
        gold_by_rid = {rid: list(pids) for rid, pids in
                       json.loads(args.gold_paper_ids.read_text()).items()}
        n_gold_nonempty = sum(1 for v in gold_by_rid.values() if v)
        print(f"# loaded gold for {n_gold_nonempty}/{len(gold_by_rid)} rows from "
              f"{args.gold_paper_ids}")
        print("# (per-filter ablation needs --affiliations-dir; precomputed "
              "file only has the 'overall' filter)")
    elif args.affiliations_dir and str(args.affiliations_dir).strip():
        if not args.affiliations_dir.is_dir():
            print(f"WARN: --affiliations-dir {args.affiliations_dir} not "
                  f"found; retrieval metrics will be NaN.")
        else:
            print(f"# loading affiliations corpus from {args.affiliations_dir} "
                  f"(this can take 1-3 min) ...")
            # Lazy imports so --help is fast and so we don't pay this cost
            # when the user explicitly disables gold.
            from backend.eval.gold import (
                compute_gold_paper_ids, load_affiliations_corpus,
            )
            corpus = load_affiliations_corpus(args.affiliations_dir)
            print(f"# corpus loaded: {len(corpus)} papers; computing gold per "
                  f"row × {len(FILTER_SETS)} filter sets ...")
            for r in testset:
                rid = r.get("id")
                if not rid:
                    continue
                # The "overall" filter set must match compute_gold_paper_ids
                # exactly — defer to the canonical implementation to avoid
                # subtle drift if gold.py ever adds a new criterion.
                gold_by_rid[rid] = sorted(compute_gold_paper_ids(r, corpus))
                for fname, fset in FILTER_SETS:
                    if fname == "overall":
                        gold_by_rid_filter[(rid, fname)] = gold_by_rid[rid]
                    else:
                        gold_by_rid_filter[(rid, fname)] = sorted(
                            _gold_for_row(r, corpus, fset))
            sizes = [len(v) for v in gold_by_rid.values() if v]
            print(f"# overall gold sizes: nonempty={len(sizes)}/{len(gold_by_rid)} "
                  f"min={min(sizes) if sizes else 0} "
                  f"med={sorted(sizes)[len(sizes)//2] if sizes else 0} "
                  f"max={max(sizes) if sizes else 0}")
    else:
        print("# no gold source configured — retrieval metrics will be NaN.")

    run_dirs: List[Path] = []
    if not args.root.is_dir():
        raise SystemExit(f"root not found: {args.root}")
    for child in sorted(args.root.iterdir()):
        if child.is_dir() and (child / "answers.jsonl").is_file():
            run_dirs.append(child)
    if not run_dirs:
        raise SystemExit(f"no <mode>_<model>/answers.jsonl found under {args.root}")
    print(f"# found {len(run_dirs)} run dirs under {args.root}")

    # Per-(mode, model, category, filter_name) → list of NDCG@k scores
    # (one per row that had non-empty gold for that filter). Aggregated
    # at the end into the breakdown tables. Category "all" rolls up
    # rows of any category.
    ablation: Dict[Tuple[str, str, str, str], List[float]] = {}

    table: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        parsed = _parse_dir_name(run_dir.name)
        if parsed is None:
            print(f"# skipping {run_dir.name} — name doesn't match <mode>_<model>")
            continue
        mode, model_tag = parsed
        answers = _load_jsonl(run_dir / "answers.jsonl")
        n_total = len(testset_by_id)
        if not answers:
            print(f"# {run_dir.name}: empty answers.jsonl, skipping")
            continue
        # Lazy import — only paid for once per script invocation.
        try:
            from backend.eval.metrics import phantom_citation_rate
        except ImportError:
            phantom_citation_rate = None  # type: ignore

        ps, rs, ndcgs, rl_ev = [], [], [], []
        phantom_rates: List[float] = []
        retrieved_sizes: List[int] = []
        gold_sizes_seen: List[int] = []
        for a in answers:
            rid = a.get("id")
            row = testset_by_id.get(rid)
            if row is None:
                continue
            records = a.get("records") or []
            retrieved_sizes.append(len(records))
            gold_ids = gold_by_rid.get(rid, [])
            if gold_ids:
                gold_sizes_seen.append(len(gold_ids))
            retr = _retrieved_ids(records)
            r = _pr_ndcg(retr, gold_ids, args.k)
            ps.append(r["precision"])
            rs.append(r["recall"])
            ndcgs.append(r["ndcg"])

            # Per-filter ablation. Skip if we don't have per-filter gold
            # (i.e., user supplied --gold-paper-ids only, no corpus).
            if gold_by_rid_filter:
                cat = (row.get("classifier_gold") or {}).get("question_category") or "unknown"
                for fname, _ in FILTER_SETS:
                    g = gold_by_rid_filter.get((rid, fname), [])
                    if not g:
                        continue
                    rs_filt = _pr_ndcg(retr, g, args.k)
                    ndcg_v = rs_filt["ndcg"]
                    if isinstance(ndcg_v, float) and math.isnan(ndcg_v):
                        continue
                    ablation.setdefault((mode, model_tag, cat, fname), []).append(ndcg_v)
                    ablation.setdefault((mode, model_tag, "all", fname), []).append(ndcg_v)

            summary = a.get("summary_text") or ""
            ev = _evidence_text(records)
            if ev:
                rl_ev.append(_rougeL_f1(summary, ev))
            if phantom_citation_rate is not None and summary:
                ph = phantom_citation_rate(summary, records)
                if not (isinstance(ph["phantom_rate"], float) and math.isnan(ph["phantom_rate"])):
                    phantom_rates.append(ph["phantom_rate"])
        gold_sorted = sorted(gold_sizes_seen)
        gold_med = gold_sorted[len(gold_sorted) // 2] if gold_sorted else float("nan")
        gold_p90 = (gold_sorted[int(len(gold_sorted) * 0.9)] if gold_sorted else float("nan"))
        table.append({
            "mode": mode,
            "model": model_tag,
            "n": len(answers),
            "n_total": n_total,
            "mean_retrieved": (sum(retrieved_sizes) / len(retrieved_sizes)) if retrieved_sizes else 0,
            "gold_med": gold_med,
            "gold_p90": gold_p90,
            "p@k": _nanmean(ps),
            "r@k": _nanmean(rs),
            "ndcg@k": _nanmean(ndcgs),
            "rougeL_evidence": _nanmean(rl_ev) if rl_ev else float("nan"),
            "phantom_rate": _nanmean(phantom_rates) if phantom_rates else float("nan"),
        })

    # Order by mode (closed_book → pipeline → agent) then by model
    mode_order = {"closed_book": 0, "pipeline": 1, "agent": 2}
    table.sort(key=lambda r: (mode_order.get(r["mode"], 99), r["model"]))

    header = (
        f"# SciCo eval — first table (retrieval @k={args.k}, ROUGE-L, phantom rate)\n\n"
        f"Generated from `{args.root}` against `{args.testset}`. No LLM\n"
        f"judges, no BERTScore — local metrics only. Gold paper_ids\n"
        f"computed on the fly from the affiliations corpus via the same\n"
        f"`compute_gold_paper_ids()` path eval_scico.py uses.\n\n"
        f"Columns:\n"
        f"- **cov**: rows answered / {len(testset_by_id)} total testset rows.\n"
        f"- **n_ret**: mean records returned per row. Pipeline handlers cap\n"
        f"  around 10-30; agent's evidence buffer caps at 80.\n"
        f"- **gold med / p90**: median and 90th percentile of gold-set size\n"
        f"  for rows answered. Mean was misleading because `field_evolution`\n"
        f"  rows have 1-3k-paper gold sets, dragging it up. Median + p90\n"
        f"  shows the typical-vs-tail split: a median of 30 with p90 of\n"
        f"  2000 means most questions are tight but field-evolution rows\n"
        f"  cap recall@{args.k} at single digits regardless of retrieval quality.\n"
        f"- **p@{args.k} / r@{args.k}**: standard precision/recall at cutoff.\n"
        f"- **ndcg@{args.k}**: position-aware metric (binary relevance, log2\n"
        f"  discount). Better than MRR for this task because gold sets have\n"
        f"  many correct papers — MRR ignores all hits after the first.\n"
        f"- **rougeL_vs_ev**: ROUGE-L F1 of summary_text vs concatenated\n"
        f"  retrieved evidence (title+abstract). Proxy for grounding.\n"
        f"- **phantom**: fraction of [N] citations in the answer whose\n"
        f"  reference doesn't appear in the evidence buffer. Lower is\n"
        f"  better; 0 = fully grounded. NaN if the model didn't emit\n"
        f"  inline references.\n\n"
        f"| mode | model | cov | n_ret | gold med | gold p90 | p@{args.k} | "
        f"r@{args.k} | ndcg@{args.k} | rougeL_vs_ev | phantom |\n"
        f"|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    body_lines = []
    for r in table:
        cov = f"{r['n']}/{r['n_total']}"
        n_ret = f"{r['mean_retrieved']:.1f}" if r['mean_retrieved'] else "  0"
        gold_med = ("nan" if isinstance(r['gold_med'], float) and math.isnan(r['gold_med'])
                    else f"{int(r['gold_med'])}")
        gold_p90 = ("nan" if isinstance(r['gold_p90'], float) and math.isnan(r['gold_p90'])
                    else f"{int(r['gold_p90'])}")
        body_lines.append(
            f"| {r['mode']} | {r['model']} | {cov} | {n_ret} | {gold_med} | "
            f"{gold_p90} | {_fmt(r['p@k'])} | {_fmt(r['r@k'])} | "
            f"{_fmt(r['ndcg@k'])} | {_fmt(r['rougeL_evidence'])} | "
            f"{_fmt(r['phantom_rate'])} |"
        )

    md = header + "\n".join(body_lines) + "\n"

    # ------------------------------------------------------------------
    # Per-(mode × model) × (question_category × filter) breakdown
    # ------------------------------------------------------------------
    if ablation:
        # Categories present in the data (preserve canonical order).
        canonical_cats = [
            "overview_university",
            "university_to_university_comparison",
            "cross_country_method",
            "university_country_comparison",
            "field_evolution",
            "all",
        ]
        cats_present = [c for c in canonical_cats
                        if any(k[2] == c for k in ablation)]
        # Short col labels so the table fits.
        cat_label = {
            "overview_university": "ovw_uni",
            "university_to_university_comparison": "u2u",
            "cross_country_method": "x_country",
            "university_country_comparison": "uni_country",
            "field_evolution": "field_ev",
            "all": "ALL",
        }
        mm_pairs = sorted({(k[0], k[1]) for k in ablation},
                          key=lambda x: (mode_order.get(x[0], 99), x[1]))

        # Per-category testset counts (denominator for the "what's possible"
        # context block).
        cat_testset_n: Dict[str, int] = {}
        for r in testset:
            cat = (r.get("classifier_gold") or {}).get("question_category") or "unknown"
            cat_testset_n[cat] = cat_testset_n.get(cat, 0) + 1
        cat_testset_n["all"] = sum(v for k, v in cat_testset_n.items() if k != "all")

        # Per-(category, filter) max evaluable: rows that have non-empty
        # gold under that filter. A cell can never report more than this
        # denominator — when n_answered = n_max the (mode, model) is fully
        # scored for that combo.
        cat_filter_n_max: Dict[Tuple[str, str], int] = {}
        for r in testset:
            rid = r.get("id")
            if not rid:
                continue
            cat = (r.get("classifier_gold") or {}).get("question_category") or "unknown"
            for fname, _ in FILTER_SETS:
                if gold_by_rid_filter.get((rid, fname)):
                    cat_filter_n_max[(cat, fname)] = cat_filter_n_max.get((cat, fname), 0) + 1
                    cat_filter_n_max[("all", fname)] = cat_filter_n_max.get(("all", fname), 0) + 1

        breakdown = ["\n\n# Per-category × per-filter NDCG@{} breakdown\n".format(args.k)]
        # Show the testset shape so cell counts have context
        breakdown.append("## Testset distribution\n")
        breakdown.append("| question_category | rows | gold-evaluable rows by filter |")
        breakdown.append("|---|---:|:---|")
        for cat in cats_present:
            n_rows = cat_testset_n.get(cat, 0)
            per_filter = "  ".join(
                f"{fname}={cat_filter_n_max.get((cat, fname), 0)}"
                for fname, _ in FILTER_SETS)
            breakdown.append(f"| {cat} | {n_rows} | {per_filter} |")
        breakdown.append("")
        breakdown.append(
            "`gold-evaluable` columns show how many of those rows have\n"
            "non-empty gold under that filter (e.g. field_evolution has no\n"
            "institution filter → inst=0). Each cell below shows\n"
            "`mean_ndcg (n_answered/n_max)`. When n_answered < n_max the\n"
            "(mode, model) hasn't covered the full category yet."
        )
        breakdown.append(
            "Each cell = mean NDCG@{} of retrieval against the gold subset\n"
            "that applies ONLY the named filter (year / topic / inst / country)\n"
            "instead of the full conjunction. The `overall` filter matches\n"
            "`compute_gold_paper_ids()` exactly. Empty-gold rows are excluded\n"
            "per cell, so column ns can differ.\n\n"
            "Reading the columns:\n"
            "- **year** alone: ~5-25% of corpus per row → loose upper bound on\n"
            "  what retrieval could match if year were the only constraint.\n"
            "- **topic** alone: every paper with a matching CSO label in the\n"
            "  question's closure. Tells you how well embedding similarity\n"
            "  tracks the CSO ontology.\n"
            "- **inst** alone: every paper from the named institution(s).\n"
            "  Tests whether the retriever's substring institution filter\n"
            "  is hitting the right papers.\n"
            "- **country** alone (cross_country_method only): every paper\n"
            "  with at least one author in either country.\n"
            "- **overall**: all filters AND-combined. The conjunction is\n"
            "  always smaller than any single-filter set, so overall NDCG\n"
            "  ≤ each of the single-filter columns is structurally expected.\n".format(args.k)
        )
        for fname, _ in FILTER_SETS:
            breakdown.append(f"\n## filter = {fname.upper()}\n")
            hdr = ["mode", "model"] + [cat_label[c] for c in cats_present]
            breakdown.append("| " + " | ".join(hdr) + " |")
            breakdown.append("|" + "|".join(["---"] * 2 + ["---:"] * len(cats_present)) + "|")
            for mode, model in mm_pairs:
                row_cells = [mode, model]
                for cat in cats_present:
                    vals = ablation.get((mode, model, cat, fname))
                    n_max = cat_filter_n_max.get((cat, fname), 0)
                    if not vals:
                        # Distinguish "no rows possible" (n_max=0) from
                        # "rows possible but none answered" (n_max>0).
                        if n_max == 0:
                            row_cells.append("  -")
                        else:
                            row_cells.append(f"  -  (0/{n_max})")
                    else:
                        mean_v = sum(vals) / len(vals)
                        row_cells.append(f"{mean_v:.3f} ({len(vals)}/{n_max})")
                breakdown.append("| " + " | ".join(row_cells) + " |")
        md += "\n".join(breakdown) + "\n"

    print()
    print(md)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    print(f"# wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
