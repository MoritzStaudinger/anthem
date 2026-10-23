"""Runtime gold-set computation for the SciCo evaluation.

Each question row in `data/eval/scico_testset.jsonl` carries the question +
classifier targets + CSO topic closure, but NOT the gold paper-id list
itself — that's computed here against whichever ACL Anthology snapshot is
on disk at eval time, so the same question file stays valid as new
proceedings are added.

Three things needed per question:
    classifier_gold.{uni1, uni2, year_start, year_end, question_category}
    topic_metadata.cso_closure       # the CSO labels that count as "on topic"
    paper.paper_id (paper_vs_field)  # the input paper, excluded from gold

Plus a directory of enriched paper JSONs (the SciCo corpus snapshot) with
`topics.cso.union` populated.

The corpus is loaded once and reused across all questions — single pass
over ~120k JSONs takes ~30 seconds, then per-question gold computation is
~50ms.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Corpus loading — compact per-paper records keyed for fast filtering
# ---------------------------------------------------------------------------

def _extract_institutions(meta: Dict[str, Any]) -> Set[str]:
    insts: Set[str] = set()
    for author in meta.get("authors") or []:
        for aff in (author.get("affiliations") or []):
            name = (aff.get("clean_institution_name") or "").strip()
            if name:
                insts.add(name)
    return insts


def _extract_countries(meta: Dict[str, Any]) -> Set[str]:
    """Set of ISO 3166-1 alpha-2 country codes attached to any author
    affiliation on this paper. Used by the `cross_country_method` gold
    filter to find all ACL papers with at least one affiliation in one
    of the chosen countries.
    """
    ccs: Set[str] = set()
    for author in meta.get("authors") or []:
        for aff in (author.get("affiliations") or []):
            cc = (aff.get("country_code") or "").strip().upper()
            if cc:
                ccs.add(cc)
    return ccs


def load_affiliations_corpus(affiliations_dir: Path) -> List[Dict[str, Any]]:
    """Walk the enriched-affiliations dir once; return per-paper records
    pre-extracted for fast filtering at gold-set computation time."""
    t0 = time.time()
    files = sorted(affiliations_dir.glob("*.json"))
    out: List[Dict[str, Any]] = []
    for fp in files:
        try:
            d = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        union = (d.get("topics") or {}).get("cso", {}).get("union") or []
        year = d.get("year")
        # doi / title kept for downstream ID-bridging when joining against
        # agent-evidence records that lack paper_id (eval_ir_measures.py
        # builds DOI→paper_id and title→paper_id lookups from these).
        out.append({
            "paper_id": d.get("paper_id") or fp.stem,
            "year": int(year) if isinstance(year, int) else None,
            "cso_union": {str(u).lower() for u in union},
            "institutions": _extract_institutions(d),
            "countries": _extract_countries(d),
            "doi": (d.get("doi") or d.get("DOI") or "") or "",
            "title": d.get("title") or d.get("paper_title") or "",
        })
    logger.info(
        "loaded %d affiliations records from %s in %.1fs",
        len(out), affiliations_dir, time.time() - t0,
    )
    return out


# ---------------------------------------------------------------------------
# Per-question gold set
# ---------------------------------------------------------------------------

_INST_CATEGORIES = {
    "overview_university",
    "university_comparison",
    "university_country_comparison",
}


def compute_gold_paper_ids(
    row: Dict[str, Any],
    corpus: List[Dict[str, Any]],
) -> Set[str]:
    """Build the gold paper-id set for one question.

    Semantics per question_category:
        overview_university / university_comparison / university_country_comparison
            : papers from uni1 about closure(topic) inside year_window
        university_to_university_comparison
            : papers from uni1 ∪ uni2 about closure(topic) inside year_window
        cross_country_method
            : papers with at least one affiliation in `country` ∪ `country_b`,
              about closure(topic) inside year_window. The LLM answer
              compares method distribution across the two; the gold paper
              set is the union of papers from both sides.
        field_evolution
            : all papers about closure(topic) inside year_window
        paper_vs_field
            : all papers about closure(topic) inside year_window,
              excluding the target paper itself

    Returns an empty set if the row lacks the metadata needed (e.g., no
    CSO closure, or unknown category).
    """
    cg = row.get("classifier_gold") or {}
    tm = row.get("topic_metadata") or {}
    closure = {str(x).lower() for x in (tm.get("cso_closure") or [])}
    if not closure:
        return set()
    cat = cg.get("question_category")
    uni1 = cg.get("uni1")
    uni2 = cg.get("uni2")
    y_start = cg.get("year_start")
    y_end = cg.get("year_end")
    target_pid = (row.get("paper") or {}).get("paper_id")
    insts_filter = {x for x in (uni1, uni2) if x}

    # Country filter used by cross_country_method. Both codes live in
    # dedicated schema fields (`country`, `country_b`); we uppercase for
    # robustness to lowercase entries.
    countries_filter: Set[str] = set()
    if cat == "cross_country_method":
        cc1 = (cg.get("country") or "").upper()
        cc2 = (cg.get("country_b") or "").upper()
        countries_filter = {c for c in (cc1, cc2) if c}

    gold: Set[str] = set()
    for p in corpus:
        # Year filter (skip if paper's year is unknown — be conservative
        # and only filter on papers with a known year).
        py = p["year"]
        if py is not None:
            if y_start is not None and py < y_start:
                continue
            if y_end is not None and py > y_end:
                continue
        # Topic filter: any CSO label on the paper in the closure?
        if not (p["cso_union"] & closure):
            continue
        # Per-category filters
        if cat in _INST_CATEGORIES:
            if uni1 not in p["institutions"]:
                continue
        elif cat == "university_to_university_comparison":
            if not (p["institutions"] & insts_filter):
                continue
        elif cat == "cross_country_method":
            if not countries_filter:
                continue
            if not (p.get("countries", set()) & countries_filter):
                continue
        # field_evolution + paper_vs_field: no institution/country filter.
        gold.add(p["paper_id"])

    if cat == "paper_vs_field" and target_pid:
        gold.discard(target_pid)
    return gold


# ---------------------------------------------------------------------------
# Sanity helpers — useful for log lines + the per-question size breakdown
# ---------------------------------------------------------------------------

def summarize_gold_sizes(rows_with_gold: List[Tuple[Dict[str, Any], Set[str]]]) -> Dict[str, Any]:
    """Aggregate stats on gold-set sizes across a question set."""
    sizes = [len(g) for _, g in rows_with_gold]
    if not sizes:
        return {"n": 0}
    sizes_sorted = sorted(sizes)
    return {
        "n": len(sizes),
        "min": sizes_sorted[0],
        "p25": sizes_sorted[len(sizes_sorted) // 4],
        "median": sizes_sorted[len(sizes_sorted) // 2],
        "p75": sizes_sorted[3 * len(sizes_sorted) // 4],
        "max": sizes_sorted[-1],
        "mean": sum(sizes) / len(sizes),
        "zero_gold": sum(1 for s in sizes if s == 0),
    }
