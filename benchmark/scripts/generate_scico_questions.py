"""Generate the full SciCo eval question set from the corpus itself.

Replaces the legacy CIKM-style import. The legacy 97 questions were
templated against a different (OpenAlex+CORE+arXiv) corpus; their
institutions / topics may not actually be retrievable in *this* corpus.
This script instead derives bindings from the corpus on disk, applies a
MiniLM coverage filter so every (institution, topic) pair is one the
local index can answer, and emits a balanced set across all personas
and categories — capped at the configured budget (default 100).

Personas / categories
---------------------
HEAD tier (overview / breadth):
  - overview_university        — "What did X contribute to T <ywin>?"
  - university_comparison      — "What did X contribute to T <ywin>
                                  compared to other universities?"
  - university_country_comp.   — "...compared to other universities in C?"
  - university_to_university   — "...compared to X2?"

POSTDOC tier (topical / synthesis):
  - field_evolution            — "How has T research evolved <ywin>?"
  - paper_vs_field             — "How does this paper compare to other work on T?"
                                 (a specific corpus paper is attached)

PHD tier (methodology / fine detail):
  - phd_methodology            — LLM-written, requires reading the body.
                                 Generated separately by
                                 scripts/generate_phd_questions.py and
                                 invoked from here when --skip-phd is off.

Coverage filter
---------------
For each candidate (institution, topic) we check the SciCo abstracts
index actually returns ≥ `--min-coverage` papers for that pair (top-k
abstracts retrieval restricted by the institution's affiliations_string).
Pairs that fail the filter are dropped.

Re-runs
-------
Each run overwrites scico_testset.jsonl (with backup). Re-run after the
corpus changes and the question set syncs.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Use Comparer_flask's loader when available (full Flask backend on
# developer laptop), otherwise fall back to a stdlib-only loader that works
# anywhere the enriched JSONs live (e.g., on the cluster).
try:
    from backend.rag.scico_indexing import load_scico_papers  # noqa: E402
except Exception:
    # Catch BaseException-derived issues (ImportError, ModuleNotFoundError,
    # TypeError from new-syntax in llama_index on older Pythons, etc.) so
    # the script still runs in a minimal env that doesn't have the full
    # Comparer_flask backend.
    def load_scico_papers(scico_root):
        """Standalone loader. Reads <paper_id>.json files either from
        `<scico_root>/affiliations/` (Comparer_flask layout) or directly
        from `<scico_root>/` (cluster layout)."""
        root = Path(scico_root)
        aff_dir = root / "affiliations"
        if not aff_dir.is_dir():
            aff_dir = root
        out = []
        for jp in sorted(aff_dir.glob("*.json")):
            try:
                out.append(json.loads(jp.read_text(encoding="utf-8", errors="replace")))
            except json.JSONDecodeError:
                continue
        return out


# ---------------------------------------------------------------------------
# Main-conference + Findings filter (mirrors generate_phd_questions.py)
# ---------------------------------------------------------------------------

import re as _re

_MAIN_FINDINGS_MODERN_RE = _re.compile(
    r"^\d{4}\.("
    r"acl-(long|short)|emnlp-main|naacl-(long|short)|"
    r"eacl-(long|short)|aacl-main|tacl|cl-\d|"
    r"findings-(acl|emnlp|naacl|eacl|aacl)"
    r")\.",
    _re.IGNORECASE,
)
_MAIN_FINDINGS_LEGACY_RE = _re.compile(r"^[PDNEJQH]\d{2}-", _re.IGNORECASE)


def _is_main_or_findings(paper_id: str) -> bool:
    """True iff paper_id is from an *ACL main conference or Findings."""
    return bool(
        _MAIN_FINDINGS_MODERN_RE.match(paper_id)
        or _MAIN_FINDINGS_LEGACY_RE.match(paper_id)
    )


def load_anthology_years(anthology_repo: Path) -> Dict[str, int]:
    """Build {paper_id -> publication_year} from the cloned acl-org/acl-anthology
    repo (its data/xml/ directory). Years come from each volume's <meta>/<year>
    element — the authoritative source, not paper_id heuristics."""
    import xml.etree.ElementTree as ET

    xml_dir = anthology_repo / "data" / "xml"
    if not xml_dir.is_dir():
        xml_dir = anthology_repo
    out: Dict[str, int] = {}
    for xml_file in sorted(xml_dir.glob("*.xml")):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue
        root = tree.getroot()
        collection_id = root.attrib.get("id") or xml_file.stem
        for volume in root.findall("volume"):
            vol_id = volume.attrib.get("id") or ""
            # Per-volume year lives in <meta>/<year>
            meta = volume.find("meta")
            year: Optional[int] = None
            if meta is not None:
                y_el = meta.find("year")
                if y_el is not None and (y_el.text or "").strip().isdigit():
                    year = int(y_el.text.strip())
            if year is None:
                continue
            vol_full = f"{collection_id}-{vol_id}" if vol_id else collection_id
            for paper in volume.findall("paper"):
                pid_local = paper.attrib.get("id") or ""
                paper_id = (f"{vol_full}.{pid_local}"
                            if "." in collection_id
                            else f"{collection_id}-{vol_id}{pid_local}")
                out[paper_id] = year
    return out


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CSO ontology labels picked from the corpus-wide top-100 most-frequent CSO
# tags assigned to ACL Anthology papers (see reports/coverage_full.jsonl).
# Selection criteria:
#   - ≥ ~2000 papers in the SciCo corpus (enough density for ~20+ questions)
#   - actual NLP/CL subarea, not a generic umbrella term
#   - non-redundant with siblings in the list
# These are real CSO ontology nodes; closure matching at question-time pulls
# in their descendants (e.g., "machine translation" closure includes
# "neural machine translation", "statistical machine translation", etc.).
# Topic names match the CSO labels as they appear in the corpus (so the
# closure-match catches the maximum number of papers). Number after each is
# the empirical corpus-wide paper count from reports/coverage_full.jsonl.
DEFAULT_TOPICS: List[str] = [
    # Tier 1 (≥4000 papers each — major NLP subareas)
    "machine translations",           # 15885
    "large language models",          # 14081
    "question answering",             #  7543
    "named entity recognition",       #  6284
    "word sense disambiguation",      #  5224
    "information retrieval",          #  4897
    "information extraction",         #  4792
    "social media",                   #  4464
    # Tier 2 (~2000-4000 papers — well-defined subtasks)
    "sentiment analysis",             #  3851
    "speech recognition",             #  3869
    "treebanks",                      #  3585
    "knowledge base",                 #  3215
    "spoken dialogue",                #  3120
    "word embedding",                 #  3055
    "wordnet",                        #  3046
    "automatic speech recognition",   #  3009
    "parallel corpora",               #  2947
    "dependency parsing",             #  2852
    "twitter",                        #  2840
    "syntactic parsing",              #  2803
    "text summarization",             #  2682
    "deep learning",                  #  2659
    "dialogue systems",               #  2585
    "relation extraction",            #  2543
    "text classification",            #  2352
    "n-gram language models",         #  2296
    "natural language generation",    #  2241
    "semantic similarity",            #  2066
    "speech synthesis",               #  2052
]

# Default per-category counts; sums to 76 head/postdoc questions, leaving
# 24 slots for the PhD generator → 100-question total budget.
# Both paper_vs_field (14 questions) and PhD methodology (24 questions)
# require the paper's body, so full text is load-bearing in 38/100 rows.
DEFAULT_COUNTS = {
    # 100-question scientometric eval. PhD-tier (Qwen-generated) questions
    # were dropped — they overlapped with the QASPER literature on single-
    # paper QA and weakened the corpus-as-contribution claim. The whole
    # eval is now about cross-paper aggregation queries the SciCo corpus
    # uniquely enables. `university_comparison` was also dropped: its gold
    # computation in backend/eval/gold.py is identical to overview_university
    # (same _INST_CATEGORIES branch), so it added no measurable distinction.
    "overview_university": 16,
    "university_country_comparison": 10,
    "university_to_university_comparison": 18,
    "cross_country_method": 26,   # headline novel category
    "field_evolution": 14,
    "paper_vs_field": 16,
}

PERSONA_OF = {
    "overview_university": "head",
    "university_country_comparison": "head",
    "university_to_university_comparison": "head",
    "cross_country_method": "postdoc",
    "field_evolution": "postdoc",
    "paper_vs_field": "postdoc",
}

# Maps category -> our classifier output type.
NORMALIZED_TYPE = {
    "overview_university": "single_university",
    "university_country_comparison": "university",
    "university_to_university_comparison": "university_university",
    # Country×country topic comparison — dedicated classifier label so
    # per-category eval breakdowns surface this distinctly.
    "cross_country_method": "country_country",
    "field_evolution": "overview",
    "paper_vs_field": "comparison",
}


# ---------------------------------------------------------------------------
# Helpers — corpus facts
# ---------------------------------------------------------------------------

def _affiliations_of(
    paper: Dict[str, Any],
    require_ror: bool = True,
) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """Yield (canonical_institution_name, country_code, ror_id) per
    distinct affiliation. By default, only affiliations with a non-empty
    ROR id are kept — that filters out raw-string fallback names that
    Gemini couldn't normalize and keeps the candidate pool ROR-anchored.
    Set require_ror=False to include un-normalized entries too.
    """
    seen: set = set()
    out = []
    for a in paper.get("authors") or []:
        for x in a.get("affiliations") or []:
            ror_id = (x.get("ror_id") or "").strip() or None
            if require_ror and not ror_id:
                continue
            # Prefer the canonical ROR-linked name; fall back to clean
            # then raw only when require_ror=False is in effect.
            name = x.get("clean_institution_name") or x.get("raw_affiliation")
            if not name:
                continue
            key = ror_id or name
            if key in seen:
                continue
            seen.add(key)
            cc = x.get("country_code")
            out.append((name, cc, ror_id))
    return out


def _corpus_aggregates(
    papers: List[Dict[str, Any]],
    require_ror: bool = True,
) -> Dict[str, Any]:
    """Pre-compute aggregates used by question generators:
        top_institutions    : list of (name, count) sorted by frequency
        institution_country : name -> country_code
        institution_ror     : name -> ror_id   (only when ROR-linked)
        year_min/year_max   : corpus year range
    """
    inst_counts: Counter = Counter()
    inst_country: Dict[str, str] = {}
    inst_ror: Dict[str, str] = {}
    years: List[int] = []
    for p in papers:
        y = p.get("year")
        if isinstance(y, int):
            years.append(y)
        for name, cc, ror_id in _affiliations_of(p, require_ror=require_ror):
            inst_counts[name] += 1
            if cc and name not in inst_country:
                inst_country[name] = cc
            if ror_id and name not in inst_ror:
                inst_ror[name] = ror_id
    return {
        "top_institutions": inst_counts.most_common(),
        "institution_country": inst_country,
        "institution_ror": inst_ror,
        "year_min": min(years) if years else None,
        "year_max": max(years) if years else None,
    }


def _year_window(year_min: Optional[int], year_max: Optional[int]) -> str:
    """Render a year-window phrase ('in 2021', 'between 2015 and 2025', etc.)."""
    if year_min is None or year_max is None:
        return ""
    if year_min == year_max:
        return f"in {year_min}"
    return f"between {year_min} and {year_max}"


def _sample_year_window(
    rng,
    year_min: Optional[int],
    year_max: Optional[int],
) -> Tuple[Optional[int], Optional[int], str]:
    """Sample a per-question year window. Returns (start, end, phrase).

    Distribution biases toward shorter windows — but each bucket is
    clamped to what the corpus span actually allows, so a small corpus
    (e.g., span=2 for 2023-2025) doesn't crash on an empty range:
        ~35% single year
        ~35% short  (1-3 yr)
        ~20% medium (4-10 yr)        — skipped when span < 4
        ~10% long   (>10 yr)         — clamped to span when smaller
    """
    if year_min is None or year_max is None or year_min > year_max:
        return year_min, year_max, ""
    span = year_max - year_min
    if span == 0:
        return year_min, year_max, f"in {year_min}"

    r = rng.random()
    if r < 0.35:
        # single year
        y = rng.randint(year_min, year_max)
        return y, y, f"in {y}"
    elif r < 0.70:
        size = rng.randint(1, min(3, span))
    elif r < 0.90 and span >= 4:
        size = rng.randint(4, min(10, span))
    else:
        # long bucket, or fallback when medium isn't viable
        if span >= 11:
            size = rng.randint(11, span)
        elif span >= 4:
            size = rng.randint(4, span)
        else:
            size = rng.randint(1, span)

    start = rng.randint(year_min, year_max - size)
    end = start + size
    return start, end, f"between {start} and {end}"


# ---------------------------------------------------------------------------
# Coverage filter — does the corpus actually answer (institution, topic)?
# Now driven by the precomputed CSO-label index, not MiniLM retrieval.
# ---------------------------------------------------------------------------

def _cso_closure(ont, topic: str) -> set:
    """BFS over ont.narrowers to collect the topic + all descendants.
    Lower-cased for matching. Works on cso-classifier 3.x and 4.x."""
    topic_l = topic.lower()
    out = {topic_l}
    stack = [topic_l]
    narrowers = ont.narrowers or {}
    while stack:
        node = stack.pop()
        for child in (narrowers.get(node) or []):
            cl = child.lower()
            if cl not in out:
                out.add(cl)
                stack.append(cl)
    return out


CSO_PORTAL = "https://cso.kmi.open.ac.uk/topics"


def _cso_topic_url(topic: str) -> str:
    """Canonical CSO portal URL for a topic name (underscores, lower)."""
    slug = topic.strip().lower().replace(" ", "_")
    return f"{CSO_PORTAL}/{slug}"


def build_topic_metadata(
    closures: Dict[str, set],
) -> Dict[str, Dict[str, Any]]:
    """Per-topic CSO grounding to embed in each question row.

    Embeds the FULL closure (parent + descendants) so the question record
    is self-sufficient for gold computation at eval time — no re-loading
    the CSO ontology needed.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for t, cl in closures.items():
        out[t] = {
            "cso_label": t,
            "cso_url": _cso_topic_url(t),
            "cso_closure_size": len(cl),
            "cso_closure": sorted(cl),   # FULL list, not a sample
        }
    return out


def _detect_cso_version(ont) -> Optional[str]:
    """Try a few common attribute/config paths for the CSO ontology
    version. cso-classifier 3.x stores it on the Config object."""
    try:
        from cso_classifier.config import Config
        return Config().get_cso_version()
    except Exception:
        pass
    for attr in ("version", "cso_version", "_version"):
        v = getattr(ont, attr, None)
        if v:
            return str(v)
    return None


def build_topic_paper_index(
    papers: List[Dict[str, Any]],
    topics: List[str],
) -> Tuple[Dict[str, set],
           Dict[Tuple[str, str], set],
           Dict[str, Dict[str, Any]],
           Optional[str],
           Dict[str, Optional[int]],
           Dict[Tuple[str, str], set]]:
    """Returns:
      papers_by_topic       : topic_name -> set(paper_id)
      papers_by_inst_topic  : (institution, topic) -> set(paper_id)
      topic_metadata        : per-topic CSO provenance dict to embed in rows
      cso_version           : ontology version string
      paper_id_to_year      : paper_id -> int year (None if unknown).
                              Used to clamp per-question year windows to
                              the actual paper intersection, fixing the
                              anachronism bug ('deep learning in 1985').
      papers_by_country_topic : (country_code, topic) -> set(paper_id).
                              Drives the cross-country method-comparison
                              question category.
    """
    from cso_classifier.ontology import Ontology
    ont = Ontology(load_ontology=True)
    cso_version = _detect_cso_version(ont)
    closures = {t: _cso_closure(ont, t) for t in topics}

    papers_by_topic: Dict[str, set] = {t: set() for t in topics}
    papers_by_inst_topic: Dict[Tuple[str, str], set] = {}
    paper_id_to_year: Dict[str, Optional[int]] = {}
    papers_by_country_topic: Dict[Tuple[str, str], set] = {}

    for p in papers:
        pid = p.get("paper_id")
        if not pid:
            continue
        union = (p.get("topics") or {}).get("cso", {}).get("union") or []
        union_lower = {str(u).lower() for u in union}
        if not union_lower:
            continue
        matched_topics = [t for t, cl in closures.items() if union_lower & cl]
        if not matched_topics:
            continue
        year = p.get("year") if isinstance(p.get("year"), int) else None
        paper_id_to_year[pid] = year

        affs = _affiliations_of(p)
        insts = {name for name, _cc, _ror in affs}
        countries = {cc for _n, cc, _r in affs if cc}

        for t in matched_topics:
            papers_by_topic[t].add(pid)
            for inst in insts:
                papers_by_inst_topic.setdefault((inst, t), set()).add(pid)
            for cc in countries:
                papers_by_country_topic.setdefault((cc, t), set()).add(pid)

    topic_metadata = build_topic_metadata(closures)
    if cso_version:
        for t in topic_metadata:
            topic_metadata[t]["cso_version"] = cso_version
    return (papers_by_topic, papers_by_inst_topic, topic_metadata, cso_version,
            paper_id_to_year, papers_by_country_topic)


def make_inst_topic_check(
    papers_by_inst_topic: Dict[Tuple[str, str], set],
    min_papers: int = 3,
):
    """Returns a fast (institution, topic) -> bool callable."""
    def check(institution: str, topic: str) -> bool:
        return len(papers_by_inst_topic.get((institution, topic), ())) >= min_papers
    return check


def _inst_topic_year_range(
    papers_by_inst_topic: Dict[Tuple[str, str], set],
    paper_id_to_year: Dict[str, Optional[int]],
    inst: str,
    topic: str,
    fallback: Tuple[Optional[int], Optional[int]],
) -> Tuple[Optional[int], Optional[int]]:
    """Min/max year of papers actually contributed by `inst` on `topic`.
    Falls back to the global corpus range when the intersection has no
    dated papers (rare — happens only if year backfill failed for those
    specific papers). This is what fixes the anachronism bug: instead of
    sampling a year window from the global 1979–2026 range, sample from
    the years this institution actually published on this topic.
    """
    pids = papers_by_inst_topic.get((inst, topic), set())
    years = [paper_id_to_year[pid] for pid in pids
             if paper_id_to_year.get(pid) is not None]
    if not years:
        return fallback
    return min(years), max(years)


def _inst_pair_topic_year_range(
    papers_by_inst_topic: Dict[Tuple[str, str], set],
    paper_id_to_year: Dict[str, Optional[int]],
    inst1: str, inst2: str, topic: str,
    fallback: Tuple[Optional[int], Optional[int]],
) -> Tuple[Optional[int], Optional[int]]:
    """Min/max year over the union of papers from inst1 ∪ inst2 on topic."""
    pids = (papers_by_inst_topic.get((inst1, topic), set())
            | papers_by_inst_topic.get((inst2, topic), set()))
    years = [paper_id_to_year[pid] for pid in pids
             if paper_id_to_year.get(pid) is not None]
    if not years:
        return fallback
    return min(years), max(years)


def _country_pair_topic_year_range(
    papers_by_country_topic: Dict[Tuple[str, str], set],
    paper_id_to_year: Dict[str, Optional[int]],
    cc1: str, cc2: str, topic: str,
    fallback: Tuple[Optional[int], Optional[int]],
) -> Tuple[Optional[int], Optional[int]]:
    pids = (papers_by_country_topic.get((cc1, topic), set())
            | papers_by_country_topic.get((cc2, topic), set()))
    years = [paper_id_to_year[pid] for pid in pids
             if paper_id_to_year.get(pid) is not None]
    if not years:
        return fallback
    return min(years), max(years)


def _window_density(
    paper_pids: set,
    paper_id_to_year: Dict[str, Optional[int]],
    ys: Optional[int],
    ye: Optional[int],
) -> int:
    """Count papers in paper_pids whose year falls inside [ys, ye]."""
    n = 0
    for pid in paper_pids:
        y = paper_id_to_year.get(pid)
        if y is None:
            continue
        if ys is not None and y < ys:
            continue
        if ye is not None and y > ye:
            continue
        n += 1
    return n


def _sample_year_window_dense(
    rng,
    pids: set,
    paper_id_to_year: Dict[str, Optional[int]],
    fallback: Tuple[Optional[int], Optional[int]],
    *,
    min_papers_in_window: int = 3,
    max_window_years: int = 10,
    max_tries: int = 16,
) -> Optional[Tuple[Optional[int], Optional[int], str]]:
    """Sample a year window via _sample_year_window, retrying until the
    window meets two constraints:
      - contains at least `min_papers_in_window` papers from `pids`
      - spans no more than `max_window_years` years (ye - ys + 1 cap)

    Returns (year_start, year_end, phrase) on success, or None when no
    qualifying window is found. Returning None is preferred over emitting
    a wide-but-sparse fallback — that's what produced the "between 1997
    and 2022" windows reviewers flagged as too broad.
    """
    years = sorted(y for y in (paper_id_to_year.get(p) for p in pids) if y is not None)
    if not years:
        return _sample_year_window(rng, *fallback)
    ymin, ymax = min(years), max(years)
    if len(years) < min_papers_in_window:
        return None
    for _ in range(max_tries):
        ys, ye, ywin_q = _sample_year_window(rng, ymin, ymax)
        if ys is None or ye is None:
            continue
        # Width cap — ye - ys + 1 = inclusive year count
        if (ye - ys + 1) > max_window_years:
            continue
        if _window_density(pids, paper_id_to_year, ys, ye) >= min_papers_in_window:
            return ys, ye, ywin_q
    # No qualifying narrow window — skip rather than emit a wide one.
    return None


# Legacy fallback: kept for callers that bypass the new index path.
def _check_coverage(institution: str, topic: str, min_papers: int = 3, k: int = 25) -> bool:
    """Deprecated MiniLM-based check; kept only for backwards compat."""
    from backend.rag.agent_tools import (
        reset_evidence_buffer,
        search_local_acl_papers,
    )
    reset_evidence_buffer()
    out = search_local_acl_papers(query=topic, institution=institution, samples=k)
    if out and isinstance(out, list) and out[0].get("error"):
        return False
    return len(out) >= min_papers


# ---------------------------------------------------------------------------
# Per-category templates
# ---------------------------------------------------------------------------

def _build_classifier_gold(
    category: str,
    *,
    uni1: Optional[str] = None,
    uni2: Optional[str] = None,
    topic: Optional[str] = None,
    country: Optional[str] = None,
    country_b: Optional[str] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    uni1_ror: Optional[str] = None,
    uni2_ror: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "question_type": NORMALIZED_TYPE[category],
        # Granular generation category — kept (under a clearer name than the
        # old `question_type_legacy`) so per-category eval breakdowns work.
        "question_category": category,
        "uni1": uni1,
        "uni1_ror": uni1_ror,
        "uni2": uni2,
        "uni2_ror": uni2_ror,
        "country": country,
        # Second country code for cross_country_method questions; null for
        # all other categories. Schema mirrors `uni1`/`uni2` so the eval-side
        # gold computation can treat country-pair questions symmetrically.
        "country_b": country_b,
        "topic": topic,
        "year_start": year_start,
        "year_end": year_end,
        "years": None,
    }


def _gen_overview_university(facts, topics, ywin, count, rng, depth_check,
                             papers_by_inst_topic, paper_id_to_year,
                             topic_cap: int = 3) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    insts = [i for i, _ in facts["top_institutions"] if not _looks_junky(i)]
    rng.shuffle(insts)
    used = set()
    topic_usage: Dict[str, int] = {}
    fallback = (facts["year_min"], facts["year_max"])
    for inst in insts:
        if len(out) >= count:
            break
        # Per-topic cap: pick from topics the institution actually
        # qualifies for AND that haven't already filled their share of
        # this category. Without the cap, popular topics dominate because
        # they pass depth_check for many institutions; niche topics never
        # surface.
        topic_pool = [t for t in topics
                      if depth_check(inst, t)
                      and topic_usage.get(t, 0) < topic_cap]
        if not topic_pool:
            continue
        topic = rng.choice(topic_pool)
        key = (inst, topic)
        if key in used:
            continue
        # Sample a year window dense enough to answer (>= 5 papers from
        # this (inst, topic) inside the window). Skip the row if no such
        # window exists — that's the failure mode that produced count=1
        # "MILA on text summarization in 2018"-style sparse questions.
        pids = papers_by_inst_topic.get((inst, topic), set())
        sampled = _sample_year_window_dense(rng, pids, paper_id_to_year, fallback)
        if sampled is None:
            continue
        ys, ye, ywin_q = sampled
        used.add(key)
        topic_usage[topic] = topic_usage.get(topic, 0) + 1
        q = f"What did {inst} contribute to {topic} research {ywin_q}?".strip().rstrip("?") + "?"
        out.append({
            "persona": "head", "depth": "overview",
            "category": "overview_university",
            "query": q,
            "classifier_gold": _build_classifier_gold(
                "overview_university",
                uni1=inst, uni1_ror=facts["institution_ror"].get(inst),
                topic=topic,
                year_start=ys, year_end=ye,
            ),
        })
    return out


def _gen_university_country_comparison(
    facts, topics, ywin, count, rng, depth_check,
    papers_by_inst_topic, paper_id_to_year,
    topic_cap: int = 3,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    inst_country = facts["institution_country"]
    insts = [i for i, _ in facts["top_institutions"] if not _looks_junky(i) and i in inst_country]
    rng.shuffle(insts)
    used = set()
    topic_usage: Dict[str, int] = {}
    fallback = (facts["year_min"], facts["year_max"])
    for inst in insts:
        if len(out) >= count:
            break
        topic_pool = [t for t in topics
                      if depth_check(inst, t)
                      and topic_usage.get(t, 0) < topic_cap]
        if not topic_pool:
            continue
        topic = rng.choice(topic_pool)
        country = inst_country[inst]
        key = (inst, topic, country)
        if key in used:
            continue
        pids = papers_by_inst_topic.get((inst, topic), set())
        sampled = _sample_year_window_dense(rng, pids, paper_id_to_year, fallback)
        if sampled is None:
            continue
        ys, ye, ywin_q = sampled
        used.add(key)
        topic_usage[topic] = topic_usage.get(topic, 0) + 1
        # Render with the full country name; keep the alpha-2 code in
        # classifier_gold for machine matching.
        q = (f"What did {inst} contribute to {topic} research {ywin_q} "
             f"compared to other universities in {_country_label(country)}?")
        out.append({
            "persona": "head", "depth": "overview",
            "category": "university_country_comparison",
            "query": q,
            "classifier_gold": _build_classifier_gold(
                "university_country_comparison",
                uni1=inst, uni1_ror=facts["institution_ror"].get(inst),
                topic=topic, country=country,
                year_start=ys, year_end=ye,
            ),
        })
    return out


def _gen_university_to_university(
    facts, topics, ywin, count, rng, depth_check,
    papers_by_inst_topic, paper_id_to_year,
    topic_cap: int = 3,
) -> List[Dict[str, Any]]:
    """Topic-first pair sampling.

    The previous "draw 2 random institutions, then pick a random topic"
    approach broke down when `--min-coverage` was raised: the joint
    probability that BOTH institutions hit ≥coverage papers on the SAME
    randomly-picked topic dropped low enough that the generator ran out
    of attempts well before filling its quota. Fix: precompute per-topic
    eligible-institution lists once, then sample (topic → two insts) so
    every candidate pair is guaranteed to share a topic both contribute
    to. Also fixes the original Vanderbilt-dominates-uni1 bias that
    motivated the prior random-pair attempt.
    """
    out: List[Dict[str, Any]] = []
    insts_all = [i for i, _ in facts["top_institutions"] if not _looks_junky(i)]
    # Per-topic eligibility — precomputed so the inner loop is O(1) per pair.
    eligible_by_topic: Dict[str, List[str]] = {}
    for t in topics:
        elig = [i for i in insts_all if depth_check(i, t)]
        if len(elig) >= 2:
            eligible_by_topic[t] = elig
    eligible_topics = list(eligible_by_topic.keys())
    if not eligible_topics:
        return out

    used = set()
    topic_usage: Dict[str, int] = {}
    fallback = (facts["year_min"], facts["year_max"])
    max_attempts = count * 50
    attempts = 0
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        # Topic-first, but also cap how many times each topic can appear
        # in this category — otherwise large-vocab topics dominate the
        # output when many institutions are eligible on them.
        topic_pool = [t for t in eligible_topics
                      if topic_usage.get(t, 0) < topic_cap]
        if not topic_pool:
            break
        topic = rng.choice(topic_pool)
        inst1, inst2 = rng.sample(eligible_by_topic[topic], 2)
        key = (inst1, inst2, topic)
        if key in used or (inst2, inst1, topic) in used:
            continue
        # Density check on the UNION of (inst1, topic) ∪ (inst2, topic) papers.
        union_pids = (papers_by_inst_topic.get((inst1, topic), set())
                      | papers_by_inst_topic.get((inst2, topic), set()))
        sampled = _sample_year_window_dense(rng, union_pids, paper_id_to_year, fallback)
        if sampled is None:
            continue
        ys, ye, ywin_q = sampled
        used.add(key)
        topic_usage[topic] = topic_usage.get(topic, 0) + 1
        q = (f"What did {inst1} contribute to {topic} research {ywin_q} "
             f"compared to {inst2}?")
        out.append({
            "persona": "head", "depth": "overview",
            "category": "university_to_university_comparison",
            "query": q,
            "classifier_gold": _build_classifier_gold(
                "university_to_university_comparison",
                uni1=inst1, uni1_ror=facts["institution_ror"].get(inst1),
                uni2=inst2, uni2_ror=facts["institution_ror"].get(inst2),
                topic=topic,
                year_start=ys, year_end=ye,
            ),
        })
    return out


def _gen_cross_country_method(
    facts, topics, count, rng,
    papers_by_country_topic, paper_id_to_year,
    min_papers_per_country: int = 5,
    topic_cap: int = 3,
) -> List[Dict[str, Any]]:
    """Question form:
       "Which approaches to {topic} dominated in {country_a}-affiliated
        ACL papers versus {country_b}-affiliated papers between {y1}–{y2},
        and what does the gap suggest?"

    Constraint: each chosen (country, topic) pair must have at least
    `min_papers_per_country` papers in the corpus so the gold answer is
    actually computable. We then sample a year window that overlaps with
    both countries' active publication on that topic.
    """
    out: List[Dict[str, Any]] = []
    # Build candidate set: (topic, country) -> count
    by_topic: Dict[str, List[str]] = {}
    for (cc, t), pids in papers_by_country_topic.items():
        if len(pids) >= min_papers_per_country and t in topics:
            by_topic.setdefault(t, []).append(cc)

    eligible = [(t, cs) for t, cs in by_topic.items() if len(cs) >= 2]
    rng.shuffle(eligible)
    used = set()
    topic_usage: Dict[str, int] = {}
    fallback = (facts["year_min"], facts["year_max"])
    # Sample with replacement so we can revisit topics under the cap;
    # iterating once misses the topic-cap pattern the other generators use.
    max_attempts = count * 50
    attempts = 0
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        topic_pool = [(t, cs) for t, cs in eligible
                      if topic_usage.get(t, 0) < topic_cap]
        if not topic_pool:
            break
        topic, countries = rng.choice(topic_pool)
        cc_pair = rng.sample(countries, 2)
        cc1, cc2 = sorted(cc_pair)
        key = (cc1, cc2, topic)
        if key in used:
            continue
        used.add(key)
        topic_usage[topic] = topic_usage.get(topic, 0) + 1
        ymin, ymax = _country_pair_topic_year_range(
            papers_by_country_topic, paper_id_to_year, cc1, cc2, topic, fallback)
        ys, ye, ywin_q = _sample_year_window(rng, ymin, ymax)
        # Use full country names ('China', 'the United States') in the
        # question text so reviewers / human readers don't have to decode
        # bare 2-letter codes ('AE', 'LK', 'VN'). The codes stay verbatim
        # in classifier_gold for downstream matching.
        q = (f"Which approaches to {topic} dominated in research from "
             f"{_country_label(cc1)} versus {_country_label(cc2)} {ywin_q}, "
             f"and what does the gap suggest?")
        out.append({
            "persona": "postdoc", "depth": "topical",
            "category": "cross_country_method",
            "query": q,
            "classifier_gold": _build_classifier_gold(
                "cross_country_method",
                topic=topic,
                country=cc1,
                country_b=cc2,
                year_start=ys, year_end=ye,
            ),
        })
    return out


def _gen_field_evolution(facts, topics, ywin, count, rng,
                         papers_by_topic=None,
                         paper_id_to_year=None) -> List[Dict[str, Any]]:
    """Generates one row per topic. Year window varies per row:
       ~30% full corpus span ('between 1984 and 2025')
       ~30% recent half      ('between 2010 and 2025')
       ~25% an era window    (a 10-15 yr slice somewhere in the timeline)
       ~15% a decade ('in the 1990s' / 'in the 2010s')
    Each window is clamped to years where the topic actually has papers,
    so 'evolution of large language models in the 1990s' (empty era)
    never gets emitted.
    """
    out: List[Dict[str, Any]] = []
    chosen_topics = list(topics)
    rng.shuffle(chosen_topics)
    corpus_min = facts["year_min"]
    corpus_max = facts["year_max"]

    def _topic_year_span(topic: str) -> Tuple[int, int]:
        if papers_by_topic is None or paper_id_to_year is None:
            return corpus_min, corpus_max
        pids = papers_by_topic.get(topic) or set()
        years = [paper_id_to_year.get(p) for p in pids]
        years = [y for y in years if isinstance(y, int)]
        if not years:
            return corpus_min, corpus_max
        return min(years), max(years)

    def _sample_evolution_window(topic: str) -> Tuple[int, int, str]:
        ymin, ymax = _topic_year_span(topic)
        if ymin is None or ymax is None or ymax - ymin < 5:
            return ymin, ymax, f"between {ymin} and {ymax}" if ymin else ""
        r = rng.random()
        if r < 0.30:
            return ymin, ymax, f"between {ymin} and {ymax}"
        if r < 0.60:
            # Recent half — capture the modern-deep-learning era
            mid = (ymin + ymax) // 2
            return mid, ymax, f"between {mid} and {ymax}"
        if r < 0.85:
            # A 10-15-year era slice somewhere inside the topic's lifespan
            era_len = rng.randint(10, min(15, ymax - ymin))
            era_start = rng.randint(ymin, ymax - era_len)
            return era_start, era_start + era_len, f"between {era_start} and {era_start + era_len}"
        # Decade window — "in the 1990s", "in the 2010s"
        decades = list(range((ymin // 10) * 10, (ymax // 10) * 10 + 1, 10))
        decade = rng.choice(decades)
        return decade, decade + 9, f"in the {decade}s"

    for topic in chosen_topics[:count]:
        ys, ye, ywin_q = _sample_evolution_window(topic)
        q = f"How has {topic} research evolved {ywin_q}?"
        out.append({
            "persona": "postdoc", "depth": "topical",
            "category": "field_evolution",
            "query": q,
            "classifier_gold": _build_classifier_gold(
                "field_evolution", topic=topic,
                year_start=ys, year_end=ye,
            ),
        })
    return out


def _read_md_fulltext(md_dir: Optional[Path], paper_id: str, max_chars: int = 30000) -> Optional[str]:
    """Read the PyMu_LLM Markdown body for one paper, capped at max_chars
    (typical paper bodies are 20-60KB; LLM context windows handle 30KB easily)."""
    if not md_dir or not paper_id:
        return None
    md_path = md_dir / f"{paper_id}.md"
    if not md_path.exists():
        return None
    try:
        body = md_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    return body[:max_chars] if body else None


def _gen_paper_vs_field(papers, topics, count, rng, md_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Pick papers with rich abstracts AND a non-empty Markdown body (full
    text is load-bearing here — without it the paper-vs-field comparison
    can't actually compare). Papers without an MD body are skipped."""
    out: List[Dict[str, Any]] = []
    candidates = [
        p for p in papers
        if (p.get("abstract") or "").strip() and len(p["abstract"]) > 400
    ]
    rng.shuffle(candidates)
    chosen_topics = list(topics)
    rng.shuffle(chosen_topics)
    skipped_no_md = 0
    for paper in candidates:
        if len(out) >= count:
            break
        pid = paper.get("paper_id")
        fulltext = _read_md_fulltext(md_dir, pid)
        if not fulltext or len(fulltext) < 1500:   # need a real body, not a stub
            skipped_no_md += 1
            continue
        topic = rng.choice(chosen_topics)
        q = f"How does this paper compare to other work on {topic}?"
        # NB: classifier_gold for paper_vs_field intentionally has uni1=None.
        # The query doesn't mention an institution, so the classifier
        # shouldn't extract one. The paper's institutions are still
        # available in the paper.authors block as context metadata.
        out.append({
            "persona": "postdoc", "depth": "topical",
            "category": "paper_vs_field",
            "query": q,
            "classifier_gold": _build_classifier_gold(
                "paper_vs_field", topic=topic,
            ),
            "paper": {
                "title": paper.get("title") or pid,
                "abstract": paper.get("abstract") or "",
                "fulltext": fulltext,
                "doi": paper.get("doi"),
                "year": paper.get("year"),
                "authors": [a.get("name") for a in paper.get("authors") or [] if a.get("name")],
            },
            "_paper_id": pid,
        })
    if skipped_no_md:
        logging.getLogger(__name__).info(
            "paper_vs_field: skipped %d candidates without MD body", skipped_no_md,
        )
    return out


_PAREN_COUNTRY_RE = _re.compile(r"\([A-Z][A-Za-z .']+\)\s*$")


# ISO 3166-1 alpha-2 → human-readable country names. Covers the ~80 most
# common countries we see across ACL-affiliated authors. Falls back to the
# raw code when missing, so a typo here doesn't break generation.
_COUNTRY_NAMES = {
    "AE": "the United Arab Emirates", "AR": "Argentina", "AT": "Austria",
    "AU": "Australia", "BE": "Belgium", "BG": "Bulgaria", "BR": "Brazil",
    "CA": "Canada", "CH": "Switzerland", "CL": "Chile", "CN": "China",
    "CO": "Colombia", "CZ": "the Czech Republic", "DE": "Germany",
    "DK": "Denmark", "EE": "Estonia", "EG": "Egypt", "ES": "Spain",
    "FI": "Finland", "FR": "France", "GB": "the United Kingdom",
    "GR": "Greece", "HK": "Hong Kong", "HR": "Croatia", "HU": "Hungary",
    "ID": "Indonesia", "IE": "Ireland", "IL": "Israel", "IN": "India",
    "IR": "Iran", "IS": "Iceland", "IT": "Italy", "JO": "Jordan",
    "JP": "Japan", "KR": "South Korea", "LK": "Sri Lanka", "LT": "Lithuania",
    "LU": "Luxembourg", "LV": "Latvia", "MA": "Morocco", "MO": "Macau",
    "MT": "Malta", "MX": "Mexico", "MY": "Malaysia", "NG": "Nigeria",
    "NL": "the Netherlands", "NO": "Norway", "NZ": "New Zealand",
    "PE": "Peru", "PH": "the Philippines", "PK": "Pakistan",
    "PL": "Poland", "PT": "Portugal", "QA": "Qatar", "RO": "Romania",
    "RS": "Serbia", "RU": "Russia", "SA": "Saudi Arabia", "SE": "Sweden",
    "SG": "Singapore", "SI": "Slovenia", "SK": "Slovakia", "TH": "Thailand",
    "TR": "Türkiye", "TW": "Taiwan", "UA": "Ukraine", "US": "the United States",
    "VN": "Vietnam", "ZA": "South Africa",
    # Long tail — niche origins for ACL authors. Mostly captured by the
    # `Australia versus MM` case observed in candidate generation.
    "AM": "Armenia", "AZ": "Azerbaijan", "BA": "Bosnia and Herzegovina",
    "BD": "Bangladesh", "BY": "Belarus", "CR": "Costa Rica", "CY": "Cyprus",
    "DZ": "Algeria", "EC": "Ecuador", "ET": "Ethiopia", "GE": "Georgia",
    "GH": "Ghana", "IQ": "Iraq", "KE": "Kenya", "KG": "Kyrgyzstan",
    "KH": "Cambodia", "KZ": "Kazakhstan", "LB": "Lebanon", "MD": "Moldova",
    "MK": "North Macedonia", "MM": "Myanmar", "MN": "Mongolia",
    "NP": "Nepal", "OM": "Oman", "PA": "Panama", "PR": "Puerto Rico",
    "PS": "Palestine", "RW": "Rwanda", "SD": "Sudan", "SN": "Senegal",
    "SY": "Syria", "TN": "Tunisia", "TZ": "Tanzania", "UG": "Uganda",
    "UY": "Uruguay", "UZ": "Uzbekistan", "VE": "Venezuela", "ZM": "Zambia",
    "ZW": "Zimbabwe",
}


def _country_label(cc: Optional[str]) -> str:
    """Render a country code as a human-readable name for question text.
    The classifier_gold side keeps the raw alpha-2 code for matching."""
    if not cc:
        return ""
    cc = cc.strip().upper()
    return _COUNTRY_NAMES.get(cc, cc)

def _looks_junky(institution: str) -> bool:
    """Filter out unhelpful or obviously-broken affiliation strings.

    The source affiliations occasionally contain newline-joined
    multi-institution strings ('Clova AI...\\nNew York University') and
    over-long author-affiliation concatenations; treat those as junk so
    questions reference real, single institutions only.

    Also filters institutions whose ROR-matched name carries a
    parenthesized country/region suffix (e.g., 'Google (Switzerland)',
    'Microsoft (China)') because those are regional subsidiaries that
    (a) often didn't exist in the sampled year window and (b) have a
    non-suffixed canonical sibling that the rest of the corpus uses.
    """
    if not institution:
        return True
    s = institution.strip()
    if "\n" in s or "\r" in s:
        return True
    if len(s) > 80:
        return True
    if s.count(",") > 2:
        return True
    sl = s.lower()
    if len(sl) < 4:
        return True
    # ROR sometimes returns names with regional-subsidiary suffixes
    # ('Google (Switzerland)', 'IBM (United Kingdom)'). Drop them.
    if _PAREN_COUNTRY_RE.search(s):
        return True
    # Non-research charitable / fundraising entities the ROR API
    # occasionally returns as best-match for ambiguous strings.
    junky_substrings = (
        "friends of ", " foundation", " fund", "endowment",
        " ngo", "society for the prevention",
    )
    if any(j in sl for j in junky_substrings):
        return True
    bad_prefixes = (
        "department of", "school of", "faculty of", "the department",
        "1 ", "2 ", "3 ",
    )
    return any(sl.startswith(p) for p in bad_prefixes)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _build_row(
    idx: int,
    gen: Dict[str, Any],
    topic_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Materialize one question record.

    The row is intentionally corpus-snapshot-independent: it carries the
    question + classifier targets + CSO topic grounding (closure +
    version), but NOT the gold paper-id list nor gold count. Eval-time
    code recomputes those against whatever ACL Anthology snapshot is
    being evaluated, so the same question file remains valid as new
    proceedings are added to the corpus.
    """
    rid = f"scico-{gen['category']}-{idx:03d}"
    row = {
        "id": rid,
        "type": "free_form",
        "persona": gen["persona"],
        "depth": gen["depth"],
        "query": gen["query"],
        "gold_summary": "",
        "gold_paper_ids": [],
        "classifier_gold": gen["classifier_gold"],
    }
    if "paper" in gen:
        row["paper"] = gen["paper"]
    if "_paper_id" in gen and gen["_paper_id"]:
        # paper_vs_field: keep the target paper id for traceability of the
        # input. Eval still computes the answer set at runtime.
        row["gold_paper_ids"] = [gen["_paper_id"]]

    topic = (gen.get("classifier_gold") or {}).get("topic")
    if topic and topic_metadata and topic in topic_metadata:
        row["topic_metadata"] = topic_metadata[topic]
    return row


def _maybe_backup(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(f".{stamp}.bak.jsonl")
    shutil.copy2(path, backup)
    return backup


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scico-root", default="data/datastructure",
                   help="SciCo dataset root (default: data/datastructure).")
    p.add_argument("--md-dir", default=None,
                   help="Directory of PyMu_LLM Markdown files (<paper_id>.md). "
                        "MANDATORY for paper_vs_field — the question type requires "
                        "the target paper's body to be meaningful. Candidates "
                        "without an MD body are skipped.")
    p.add_argument("--out", default=str(ROOT / "data" / "eval" / "scico_testset.jsonl"),
                   help="Output JSONL (default: data/eval/scico_testset.jsonl).")
    p.add_argument("--total-budget", type=int, default=100,
                   help="Hard cap on total questions (default: 100).")
    p.add_argument("--oversample", type=float, default=1.0,
                   help="Multiply each category's count to over-generate "
                        "candidates for manual curation. E.g., --oversample 3 "
                        "generates ~300 questions you can review and trim to "
                        "the best 100. Default 1.0 (no oversample).")
    p.add_argument("--main-only", action="store_true",
                   help="Restrict the question-generation corpus to *ACL main "
                        "conferences + Findings. Recommended: questions reference "
                        "substantive research (not workshop/demo papers). "
                        "NOTE: eval-time gold computation is unaffected — that "
                        "still uses the full corpus passed via --affiliations-dir.")
    p.add_argument("--min-year", type=int, default=None,
                   help="Only consider papers published in this year or later "
                        "for question generation. Eval-time gold ignores this.")
    p.add_argument("--max-year", type=int, default=None,
                   help="Only consider papers published in this year or earlier.")
    p.add_argument("--anthology-repo", type=Path,
                   default=Path("/share/moritz.staudinger/acl_data/acl-anthology"),
                   help="Local clone of acl-org/acl-anthology. Used to look up "
                        "publication year from the authoritative metadata when "
                        "the SciCo JSON's `year` field is missing/garbled.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-coverage", type=int, default=5,
                   help="Min papers an (institution, topic) pair must have, "
                        "corpus-wide, before the institution is eligible to "
                        "appear in a templated question about that topic. "
                        "Default 5 — paired with the dense-window sampler "
                        "(≥3 papers *inside* the sampled window AND window "
                        "no wider than 10 years), this opens topic diversity "
                        "while keeping every emitted question grounded in a "
                        "tight time slice. Raise to 10-20 if you want the "
                        "old, more conservative selection.")
    p.add_argument("--no-coverage-filter", action="store_true",
                   help="Skip the MiniLM coverage check (faster but generates "
                        "questions the corpus may not actually answer).")
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    scico_root = Path(args.scico_root)
    papers = load_scico_papers(scico_root)
    if not papers:
        print(f"ERROR: no papers in {scico_root}/affiliations/", file=sys.stderr)
        return 2

    # Filter the question-generation corpus to substantive research only.
    # The eval-time gold computation reads the full corpus separately via
    # eval_scico.py --affiliations-dir, so this filter only affects WHICH
    # (institution, topic, paper) candidates the generators see.
    if args.main_only:
        before = len(papers)
        papers = [p for p in papers if _is_main_or_findings((p.get("paper_id") or ""))]
        print(f"# main+findings filter: {len(papers)}/{before} papers kept "
              f"({100*len(papers)/before:.1f}%)")
        if not papers:
            print(f"ERROR: --main-only left 0 papers in the corpus.", file=sys.stderr)
            return 2

    # Resolve & backfill paper years from the ACL Anthology XML *unconditionally*.
    # Per-paper JSONs frequently have year missing or stored as a non-int string
    # (especially for pre-2020 papers). Without this backfill, _corpus_aggregates
    # silently sees year=None on most of the corpus and computes a span limited
    # to whatever recent slice has clean ints — which then collapses all sampled
    # year windows to that narrow range.
    anth_year: Dict[str, int] = {}
    if args.anthology_repo and args.anthology_repo.exists():
        print(f"# loading anthology years from {args.anthology_repo} ...")
        anth_year = load_anthology_years(args.anthology_repo)
        print(f"# anthology index: {len(anth_year)} paper years")
    else:
        print(f"WARN: --anthology-repo {args.anthology_repo} not found; "
              f"falling back to JSON-only year (corpus year span may be wrong).",
              file=sys.stderr)

    def _year_of(p) -> Optional[int]:
        """Resolve year from (in order): the JSON's year field if it's
        a clean int / stringified int, then the anthology metadata."""
        y = p.get("year")
        if isinstance(y, int):
            return y
        if isinstance(y, str) and y.strip().isdigit():
            return int(y.strip())
        return anth_year.get(p.get("paper_id") or "")

    backfilled = 0
    for p in papers:
        y = _year_of(p)
        if y is not None and p.get("year") != y:
            p["year"] = y
            backfilled += 1
    print(f"# year backfill: {backfilled}/{len(papers)} papers received a "
          f"resolved int year (others had it already or had no resolvable year)")

    # NOTE: --min-year/--max-year apply ONLY to the PhD generator (forwarded
    # via subprocess args further down). Head/postdoc question generation
    # always uses the full backfilled corpus so per-question year windows
    # span the entire ACL Anthology (~1979–present), addressing the reviewer
    # critique that questions were temporally narrow.
    if args.min_year is not None or args.max_year is not None:
        print(f"# year filter [{args.min_year or 'any'}..{args.max_year or 'any'}] "
              f"will be applied to PhD questions only; "
              f"head/postdoc keep the full corpus span.")

    facts = _corpus_aggregates(papers)
    ywin = _year_window(facts["year_min"], facts["year_max"])
    if not ywin:
        ywin = "(all years)"
    print(f"# corpus: {len(papers)} papers, year window: {ywin}, "
          f"{len(facts['top_institutions'])} institutions")

    topics = DEFAULT_TOPICS
    rng = random.Random(args.seed)

    # Always build the topic index so we have CSO metadata to attach to
    # each row, even if --no-coverage-filter skips the depth_check.
    print(f"# building CSO topic index for {len(topics)} topics...")
    (_papers_by_topic, _papers_by_inst_topic, topic_metadata, cso_version,
     _paper_id_to_year, _papers_by_country_topic) = (
        build_topic_paper_index(papers, topics)
    )
    print(f"# CSO ontology version: {cso_version or '(unknown)'}")
    for t in topics:
        print(f"#   {t:<32}  papers={len(_papers_by_topic[t])}")

    if args.no_coverage_filter:
        depth_check = lambda *a, **k: True
    else:
        depth_check = make_inst_topic_check(_papers_by_inst_topic, args.min_coverage)

    # Generate per-category, capped. --oversample N multiplies the target
    # counts so you can over-generate candidates and pick the best 100
    # afterwards by hand.
    oversample = max(1.0, float(args.oversample))
    generated: List[Dict[str, Any]] = []
    # The PhD tier was dropped as a core contribution (the corpus's novelty
    # is in cross-paper scientometric aggregation, not single-paper QA —
    # that's QASPER's domain). Full budget goes to the 6 scientometric
    # categories; total budget is the headline count.
    effective_budget = int(round(args.total_budget * oversample))
    category_total = sum(DEFAULT_COUNTS.values())
    counts = {
        cat: max(1, round(effective_budget * c / category_total))
        for cat, c in DEFAULT_COUNTS.items()
    }
    while sum(counts.values()) > effective_budget:
        biggest = max(counts, key=counts.get)
        counts[biggest] -= 1
    while sum(counts.values()) < effective_budget:
        smallest = min(counts, key=counts.get)
        counts[smallest] += 1
    if oversample > 1.0:
        print(f"# OVERSAMPLE × {oversample:g} — generating {effective_budget} candidates "
              f"(curate to {args.total_budget} after).")
    print(f"# target counts per category: {counts}")

    generated.extend(_gen_overview_university(
        facts, topics, ywin, counts["overview_university"], rng, depth_check,
        _papers_by_inst_topic, _paper_id_to_year))
    print(f"  overview_university: {sum(1 for g in generated if g['category']=='overview_university')}")
    generated.extend(_gen_university_country_comparison(
        facts, topics, ywin, counts["university_country_comparison"], rng, depth_check,
        _papers_by_inst_topic, _paper_id_to_year))
    print(f"  university_country_comparison: {sum(1 for g in generated if g['category']=='university_country_comparison')}")
    generated.extend(_gen_university_to_university(
        facts, topics, ywin, counts["university_to_university_comparison"], rng, depth_check,
        _papers_by_inst_topic, _paper_id_to_year))
    print(f"  university_to_university_comparison: {sum(1 for g in generated if g['category']=='university_to_university_comparison')}")
    generated.extend(_gen_cross_country_method(
        facts, topics, counts["cross_country_method"], rng,
        _papers_by_country_topic, _paper_id_to_year))
    print(f"  cross_country_method: {sum(1 for g in generated if g['category']=='cross_country_method')}")
    generated.extend(_gen_field_evolution(
        facts, topics, ywin, counts["field_evolution"], rng,
        papers_by_topic=_papers_by_topic,
        paper_id_to_year=_paper_id_to_year))
    md_dir_path = Path(args.md_dir) if args.md_dir else None
    if md_dir_path and not md_dir_path.is_dir():
        print(f"WARN: --md-dir {md_dir_path} does not exist; paper_vs_field "
              f"will produce zero questions (full text is mandatory).", file=sys.stderr)
    generated.extend(_gen_paper_vs_field(papers, topics, counts["paper_vs_field"], rng, md_dir=md_dir_path))

    # Materialize rows.
    rows = [_build_row(i, g, topic_metadata) for i, g in enumerate(generated, start=1)]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_backup:
        bk = _maybe_backup(out_path)
        if bk:
            print(f"# backed up existing {out_path.name} -> {bk.name}")
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    persona_counts: Counter = Counter(r["persona"] for r in rows)
    cat_counts: Counter = Counter(r["classifier_gold"]["question_category"] for r in rows)
    print(f"\n# wrote {len(rows)} scientometric questions to {out_path}")
    print(f"# persona breakdown: {dict(persona_counts)}")
    for cat in sorted(cat_counts):
        print(f"#   {cat}: {cat_counts[cat]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
