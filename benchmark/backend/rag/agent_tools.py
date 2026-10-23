"""FunctionTool wrappers around the project's data-source helpers.

The LLM sees each function's docstring as the tool description and the
type-annotated signature as the parameter schema, so docstrings here are
prompt engineering — keep them precise about *when* to use each tool.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

from llama_index.core.tools import FunctionTool

from backend.core.reader import CoreReader
from backend.neo4j.utils import Neo4jReader, search_paper_graph_by_unit
from backend.openalex.utils import (
    get_institution_profile,
    get_ror_id,
    search_paper,
)
from backend.rag.local_acl_indexing import (
    DEFAULT_ACL_PERSIST_DIR,
    load_acl_index,
)
from backend.rag.scico_indexing import (
    DEFAULT_SCICO_ABSTRACTS_PERSIST_DIR,
    DEFAULT_SCICO_FULLTEXT_PERSIST_DIR,
    DEFAULT_SCICO_ROOT,
)
from backend.tuwien.utils import search_tuwien_papers

logger = logging.getLogger(__name__)

# Per-run buffers populated by tools as the agent runs. The agent module
# resets them before each run and reads them out at the end. Sync gunicorn
# workers handle one request at a time, so module-level state is safe.
#
# - profiles: institution-profile dicts, surfaced to the frontend so the
#   chart columns render in agent mode just like in pipeline mode.
# - evidence: slim metadata records of every paper returned by every tool
#   call, so the critic can fact-check citations against the same evidence
#   the agent saw.
_profile_buffer: list = []
_evidence_buffer: list = []


def reset_profile_buffer() -> None:
    _profile_buffer.clear()


def reset_evidence_buffer() -> None:
    _evidence_buffer.clear()


def reset_buffers() -> None:
    reset_profile_buffer()
    reset_evidence_buffer()


def collected_profiles() -> list:
    """Return profiles deduplicated by Institution (last-write-wins).

    The agent typically calls get_university_publications_profile once per
    refinement iteration, so without dedup we'd render the same institution
    as three side-by-side chart columns. The last call usually has the most
    refined args (latest year filter / topic), so keep that one.
    """
    by_institution: dict = {}
    for profile in _profile_buffer:
        key = profile.get("Institution") or "—"
        by_institution[key] = profile
    return list(by_institution.values())


def collected_evidence(max_records: int = 80) -> list:
    """Return retrieved paper records deduplicated by DOI/title.

    Caps the list so the critic's prompt stays bounded. The agent typically
    pulls 25-50 papers per tool call across 1-3 calls, so 80 is generous.
    """
    seen = set()
    out = []
    for rec in _evidence_buffer:
        key = (rec.get("DOI") or rec.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(rec)
        if len(out) >= max_records:
            break
    return out


def _record_evidence(papers: list) -> None:
    """Stash a slim copy of each paper for later cross-checking by the critic."""
    for p in papers:
        if not isinstance(p, dict):
            continue
        affiliations = p.get("affiliations") or []
        if isinstance(affiliations, str):
            affiliations = [affiliations]
        _evidence_buffer.append({
            "paper_id": p.get("paper_id"),
            "title": p.get("title"),
            "authors": (p.get("authors") or [])[:5],
            "publication_year": p.get("publication_year"),
            "DOI": p.get("DOI") or p.get("doi"),
            "url": p.get("url") or p.get("DOI") or p.get("doi") or p.get("openalex_url"),
            "affiliations": affiliations[:5],
        })


def _trim_papers(papers: List[dict], max_abstract_chars: int = 1200) -> List[dict]:
    """Shrink paper records before returning to the agent.

    Full abstracts and author lists chew through the LLM context window quickly
    when 30+ papers come back. The agent only needs enough to identify and cite
    each paper.
    """
    out = []
    for p in papers:
        abstract = (p.get("abstract") or "")[:max_abstract_chars]
        doi = p.get("DOI") or p.get("doi")
        url = p.get("url") or doi or p.get("openalex_url")
        out.append({
            "title": p.get("title"),
            "authors": p.get("authors", [])[:5],
            "publication_year": p.get("publication_year"),
            "DOI": doi,
            "url": url,
            "abstract": abstract,
            "affiliations": p.get("affiliations", [])[:5],
            "countries": p.get("countries", []),
        })
    return out


def search_openalex_by_topic(
    query: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    institution_name: Optional[str] = None,
    samples: int = 25,
) -> List[dict]:
    """Search OpenAlex (~250M papers) for academic publications by topic.

    Returns each paper's title, abstract (truncated), authors, DOI, year, and
    affiliations. Use this for general literature searches and for finding
    output of a specific institution by name.

    Args:
        query: Topic or keyword to search (e.g. "topic modeling",
            "active learning for systematic reviews").
        year_start: Restrict to papers published on/after this year.
        year_end: Restrict to papers published on/before this year.
        institution_name: If given, restrict results to papers affiliated
            with this institution (looked up via ROR).
        samples: Max papers to return (default 25; cap 50).

    Don't use for: TU Wien-only queries (use search_tu_wien_papers) or
    full-text comparisons (use search_core_fulltext).
    """
    samples = max(1, min(samples, 50))
    institution_id = get_ror_id(institution_name) if institution_name else None
    if institution_name and not institution_id:
        return [{"error": f"Could not resolve ROR for {institution_name!r}"}]
    papers = search_paper(
        query,
        institution_id=institution_id,
        year_start=year_start,
        year_end=year_end,
        samples=samples,
    )
    trimmed = _trim_papers(papers)
    _record_evidence(trimmed)
    return trimmed


def list_university_publications(
    university_name: str,
    topic: Optional[str] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    samples: int = 30,
) -> List[dict]:
    """List publications affiliated with a specific university.

    Use this whenever the user asks "what has <university> published / worked
    on / contributed to". Returns each paper's title, authors, year, DOI,
    and affiliations — i.e. the rows you cite in your final answer.

    For aggregated chart-friendly stats (publications-per-year, top authors,
    collaborators) call get_university_publications_profile instead. For
    cross-institution topic searches (no specific university) call
    search_openalex_by_topic.

    Args:
        university_name: e.g. "TU Wien", "University of Aberdeen", "ETH Zurich".
        topic: Optional topic to narrow within the institution's output
            (e.g. "information retrieval"). Omit for the institution's
            general output.
        year_start, year_end: Year range filter.
        samples: Max papers to return (default 30; cap 50).
    """
    samples = max(1, min(samples, 50))
    ror_id = get_ror_id(university_name)
    if not ror_id:
        return [{"error": f"Could not resolve ROR for {university_name!r}"}]
    papers = search_paper(
        topic or "",
        institution_id=ror_id,
        year_start=year_start,
        year_end=year_end,
        samples=samples,
    )
    trimmed = _trim_papers(papers)
    _record_evidence(trimmed)
    return trimmed


def get_university_publications_profile(
    university_name: str,
    topic: str = "",
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
) -> dict:
    """Get a structured publication profile for a university.

    Returns total publications, publications-per-year, top researchers, top
    collaborator institutions, and top-cited papers. Use this when the user
    asks for an institutional overview or when comparing two institutions.

    Args:
        university_name: e.g. "University of Aberdeen", "ETH Zurich".
        topic: Optional topic to focus the profile on (e.g. "information retrieval").
        year_start, year_end: Year range filter.
    """
    ror_id = get_ror_id(university_name)
    if not ror_id:
        return {"error": f"Could not resolve ROR for {university_name!r}"}
    profile = get_institution_profile(
        topic, ror_id, year_start=year_start, year_end=year_end
    )
    # Stash the full profile so the frontend can render charts; return a
    # trimmed copy to the LLM so we don't blow context with per-author-year
    # rows it doesn't need for prose generation.
    _profile_buffer.append(profile)
    # Top-cited papers in the profile are usable evidence for citations.
    for top in (profile.get("Top Cited Papers") or [])[:10]:
        _evidence_buffer.append({
            "title": top.get("Title"),
            "authors": top.get("involved_institution_authors") or [],
            "publication_year": None,
            "DOI": top.get("DOI"),
            "url": top.get("DOI"),
            "affiliations": [profile.get("Institution")] if profile.get("Institution") else [],
        })
    trimmed = dict(profile)
    trimmed.pop("ResearchersWithPublicationsPerYear", None)
    return trimmed


def search_core_fulltext(query: str, max_results: int = 5) -> List[dict]:
    """Search the CORE UK aggregator for full-text papers.

    Returns title, authors, year, DOI, and a *truncated* full text. Use only
    when the user's question requires content beyond abstracts — e.g. comparing
    methodology details of a specific paper to other work.

    Args:
        query: Search query (topic, paper title, or specific terms).
        max_results: Max papers to return (default 5; cap 10 — full texts are large).
    """
    max_results = max(1, min(max_results, 10))
    loader = CoreReader()
    docs = loader.load_data(query, max_results=max_results)
    out = []
    for d in docs:
        meta = getattr(d, "metadata", {}) or {}
        rec = {
            "title": meta.get("title"),
            "authors": meta.get("authors", [])[:5],
            "publication_year": meta.get("publication_year"),
            "DOI": meta.get("doi") or meta.get("DOI"),
            "fulltext_excerpt": (getattr(d, "text", "") or "")[:4000],
        }
        out.append(rec)
        _evidence_buffer.append({
            "title": rec["title"],
            "authors": rec["authors"],
            "publication_year": rec["publication_year"],
            "DOI": rec["DOI"],
            "url": rec["DOI"],
            "affiliations": [],
        })
    return out


def search_tu_wien_papers_tool(
    query: str,
    year_start: Optional[int] = None,
    org_units: Optional[List[int]] = None,
) -> List[dict]:
    """Search TU Wien's Repositum publication archive by keyword.

    Use for any TU Wien-specific question (TU Wien, TU Vienna, Vienna University
    of Technology). Returns publications with title, abstract, authors, year, DOI.

    Args:
        query: Topic or keyword.
        year_start: Restrict to papers from this year onward.
        org_units: Optional list of TISS org-unit IDs to restrict to.
    """
    papers = search_tuwien_papers(
        query, max_results=50, from_year=year_start, org_units=org_units
    )
    for p in papers:
        _evidence_buffer.append({
            "title": p.get("title"),
            "authors": (p.get("authors") or [])[:5],
            "publication_year": p.get("publication_year"),
            "DOI": p.get("DOI") or p.get("doi"),
            "url": p.get("DOI") or p.get("doi"),
            "affiliations": ["TU Wien"],
        })
    return papers


def search_internal_kg_research_unit(
    unit_name: str,
    year: Optional[int] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    samples: int = 50,
) -> List[dict]:
    """Search the internal Neo4j knowledge graph for publications by a TU Wien research unit.

    Use this when the user asks about a TU Wien *research unit* by name —
    e.g. "Data Science", "Knowledge Graphs", "Visual Computing", "Machine Learning" —
    rather than the whole university or a specific person.

    Args:
        unit_name: Substring of the research unit's name (case-insensitive).
        year: Filter to a single year (alternative to year_start/year_end).
        year_start, year_end: Year range filter.
        samples: Max publications to return (default 50; cap 200).
    """
    samples = max(1, min(samples, 200))
    reader = Neo4jReader()
    try:
        papers = search_paper_graph_by_unit(
            reader=reader,
            unit_text=unit_name,
            year=year,
            year_start=year_start,
            year_end=year_end,
            samples=samples,
        )
    finally:
        reader.close()
    for p in papers:
        _evidence_buffer.append({
            "title": p.get("title"),
            "authors": [
                (a.get("author") or {}).get("display_name")
                for a in (p.get("authorships") or [])
                if (a.get("author") or {}).get("display_name")
            ][:5],
            "publication_year": p.get("publication_year"),
            "DOI": p.get("doi"),
            "url": p.get("doi") or p.get("id"),
            "affiliations": [f"TU Wien research unit: {unit_name}"],
        })
    return papers


def search_local_acl_papers(
    query: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    institution: Optional[str] = None,
    country: Optional[str] = None,
    countries: Optional[List[str]] = None,
    samples: int = 10,
) -> List[dict]:
    """Search the local ACL anthology index (curated full texts).

    Use when the question is specifically about NLP / computational
    linguistics work published at ACL venues (ACL, EMNLP, NAACL, EACL,
    Findings, workshops). The index is local and free to query.

    Returns each retrieved paper's title, authors, year, DOI, and
    affiliations. Records the results to the evidence buffer for the
    critic to verify.

    Args:
        query: Topic or research question (e.g. "retrieval-augmented
            generation evaluation").
        year_start, year_end: Restrict by publication year.
        institution: If given, prefer papers whose affiliations contain
            this string (substring, case-insensitive).
        country: ISO 3166-1 alpha-2 country code to filter by (e.g. "US",
            "CN"). Matches papers with at least one affiliation in that
            country.
        countries: Multiple ISO 3166-1 alpha-2 codes (e.g. ["US", "CN"]).
            Use this for country-pair comparison questions — the search
            returns papers affiliated with ANY of the listed countries.
            When set, overrides `country`. Examples of when to use:
              - Cross-country method comparison questions (US vs CN, etc.)
              - Multi-country research collaboration analysis.
        samples: Max papers to return (default 10; cap 50).
    """
    from llama_index.core.vector_stores import (
        FilterCondition, MetadataFilter, MetadataFilters,
    )

    samples = max(1, min(samples, 50))
    try:
        index = load_acl_index(DEFAULT_ACL_PERSIST_DIR)
    except FileNotFoundError as e:
        return [{"error": str(e)}]

    filters: list = []
    if year_start is not None:
        filters.append(
            MetadataFilter(key="publication_year", value=year_start, operator=">=")
        )
    if year_end is not None:
        filters.append(
            MetadataFilter(key="publication_year", value=year_end, operator="<=")
        )
    if institution:
        filters.append(
            MetadataFilter(
                key="affiliations_string", value=institution, operator="contains"
            )
        )
    # Country handling — single code via `country`, or multiple via
    # `countries` (OR'd). `countries` wins when both are supplied.
    country_codes: List[str] = []
    if countries:
        country_codes = [c.strip().upper() for c in countries if c and c.strip()]
    elif country:
        country_codes = [country.strip().upper()]
    if len(country_codes) == 1:
        filters.append(
            MetadataFilter(
                key="countries_string", value=country_codes[0], operator="contains"
            )
        )
    elif len(country_codes) > 1:
        # Build an OR'd group nested into the AND root so that the rest
        # of the filters (year/topic/institution) still AND with the
        # country disjunction.
        cc_or = MetadataFilters(
            filters=[MetadataFilter(key="countries_string", value=cc, operator="contains")
                     for cc in country_codes],
            condition=FilterCondition.OR,
        )
        filters.append(cc_or)
    metadata_filters = MetadataFilters(filters=filters) if filters else None

    # Over-retrieve to absorb dedupe loss — each paper has 2 docs
    # (abstract + people), so both can surface for the same paper.
    retriever = index.as_retriever(
        similarity_top_k=min(samples * 2, 100), filters=metadata_filters
    )
    nodes = retriever.retrieve(query)

    # Dedup by paper_id (chunked nodes may surface the same paper multiple times).
    seen: set = set()
    out: List[dict] = []
    for n in nodes:
        if len(out) >= samples:
            break
        meta = getattr(n, "metadata", None) or getattr(n.node, "metadata", {})
        paper_id = meta.get("paper_id") or meta.get("title")
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        rec = {
            "title": meta.get("title"),
            "authors": (meta.get("authors") or [])[:5],
            "publication_year": meta.get("publication_year"),
            "DOI": meta.get("doi"),
            "url": meta.get("doi") or f"https://aclanthology.org/{paper_id}/",
            "affiliations": (meta.get("affiliations") or [])[:5],
            "paper_id": paper_id,
        }
        out.append(rec)
        _evidence_buffer.append({
            "title": rec["title"],
            "authors": rec["authors"],
            "publication_year": rec["publication_year"],
            "DOI": rec["DOI"],
            "url": rec["url"],
            "affiliations": rec["affiliations"],
        })
    return out


def _retrieve_from_persist_dir(
    persist_dir: str,
    query: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    institution: Optional[str] = None,
    samples: int = 15,
    *,
    fallback_url_template: str = "https://aclanthology.org/{paper_id}/",
) -> List[dict]:
    """Shared retrieval logic for SciCo agent tools.

    Loads the index at ``persist_dir`` (cached after first call), applies
    metadata filters, dedupes hits by paper_id, returns records, and
    pushes them into the evidence buffer.
    """
    from llama_index.core.vector_stores import MetadataFilter, MetadataFilters

    samples = max(1, min(samples, 50))
    try:
        index = load_acl_index(persist_dir)
    except FileNotFoundError as e:
        return [{"error": str(e)}]

    filters: list = []
    if year_start is not None:
        filters.append(MetadataFilter(key="publication_year", value=year_start, operator=">="))
    if year_end is not None:
        filters.append(MetadataFilter(key="publication_year", value=year_end, operator="<="))
    if institution:
        filters.append(MetadataFilter(key="affiliations_string", value=institution, operator="contains"))
    metadata_filters = MetadataFilters(filters=filters) if filters else None

    # Over-retrieve to absorb dedupe loss — each paper has 2 docs
    # (abstract + people) so both can surface for the same paper.
    retriever = index.as_retriever(
        similarity_top_k=min(samples * 2, 100), filters=metadata_filters,
    )
    nodes = retriever.retrieve(query)

    seen: set = set()
    out: List[dict] = []
    for n in nodes:
        if len(out) >= samples:
            break
        meta = getattr(n, "metadata", None) or getattr(n.node, "metadata", {})
        paper_id = meta.get("paper_id") or meta.get("title")
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        fallback_url = fallback_url_template.format(paper_id=paper_id) if paper_id else None
        rec = {
            "title": meta.get("title"),
            "authors": (meta.get("authors") or [])[:5],
            "publication_year": meta.get("publication_year"),
            "DOI": meta.get("doi"),
            "url": meta.get("doi") or meta.get("url") or fallback_url,
            "affiliations": (meta.get("affiliations") or [])[:5],
            "paper_id": paper_id,
        }
        out.append(rec)
        _evidence_buffer.append({
            "title": rec["title"],
            "authors": rec["authors"],
            "publication_year": rec["publication_year"],
            "DOI": rec["DOI"],
            "url": rec["url"],
            "affiliations": rec["affiliations"],
        })
    return out


def drill_into_papers_fulltext(
    query: str,
    paper_ids: List[str],
    samples: int = 8,
    fulltext_dir: str = f"{DEFAULT_SCICO_ROOT}/pymu_llm",
) -> List[dict]:
    """Hierarchical retrieval over the full text of a given candidate set.

    Use this AFTER an abstracts retrieval narrows the search to a handful of
    candidate papers. Reads each paper's full-text markdown from disk,
    splits into chunks, embeds with MiniLM on the fly, and returns the
    top-`samples` most relevant chunks across the union — with their
    parent paper_id, title, authors, year, DOI, affiliations.

    Use for: paper-vs-paper methodology comparisons, evaluation-protocol
    questions, anything where the abstract is too thin and you need the
    paper body of a specific candidate set. Don't use this as the first
    retrieval step — call search_local_acl_papers (abstracts) first to
    pick candidates, then drill here.

    Args:
        query: The specific information need (be detailed; this is matched
            against chunks of paper body text).
        paper_ids: ACL anthology IDs (e.g. ["2021.acl-long.236", ...]) of
            papers to drill into. Capped at 25 per call.
        samples: Max chunks to return (default 8; cap 30).
        fulltext_dir: Directory containing <paper_id>.md files. Defaults
            to the SciCo dataset's pymu_llm/ folder.
    """
    from llama_index.core import Document, VectorStoreIndex
    from llama_index.core.node_parser import TokenTextSplitter

    from backend.rag.local_acl_indexing import _acl_embed_model

    if not paper_ids:
        return [{"error": "drill_into_papers_fulltext requires non-empty paper_ids"}]

    paper_ids = list(paper_ids)[:25]
    samples = max(1, min(samples, 30))
    text_dir = Path(fulltext_dir)
    if not text_dir.is_dir():
        return [{"error": f"fulltext_dir {fulltext_dir!r} does not exist"}]

    # Pull metadata too so retrieved chunks render cleanly as citations.
    aff_dir = Path(DEFAULT_SCICO_ROOT) / "affiliations"

    candidate_docs: List[Document] = []
    missing: List[str] = []
    for pid in paper_ids:
        md_path = text_dir / f"{pid}.md"
        if not md_path.exists():
            missing.append(pid)
            continue
        try:
            text = md_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            missing.append(pid)
            continue

        meta: dict = {"paper_id": pid}
        aff_path = aff_dir / f"{pid}.json"
        if aff_path.exists():
            try:
                aff = json.loads(aff_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                aff = {}
            meta["title"] = aff.get("title") or pid
            meta["publication_year"] = aff.get("year")
            meta["DOI"] = aff.get("doi")
            meta["url"] = aff.get("url") or aff.get("doi") or f"https://aclanthology.org/{pid}/"
            meta["authors"] = [a.get("name") for a in aff.get("authors") or [] if a.get("name")][:5]
            affs = []
            for a in aff.get("authors") or []:
                for x in a.get("affiliations") or []:
                    name = x.get("clean_institution_name") or x.get("raw_affiliation")
                    if name and name not in affs:
                        affs.append(name)
            meta["affiliations"] = affs[:5]
        else:
            meta["title"] = pid
            meta["url"] = f"https://aclanthology.org/{pid}/"

        candidate_docs.append(Document(text=text, metadata=meta))

    if not candidate_docs:
        return [{
            "error": f"no fulltext markdown found for any of {paper_ids[:5]}... "
                     f"(missing: {len(missing)} files)"
        }]

    # Ephemeral in-memory index over the union of candidate chunks.
    splitter = TokenTextSplitter(chunk_size=800, chunk_overlap=120)
    chunk_docs: List[Document] = []
    for d in candidate_docs:
        for chunk in splitter.split_text(d.text):
            chunk_docs.append(Document(text=chunk, metadata=d.metadata))
    if not chunk_docs:
        return [{"error": "all candidate fulltexts were empty after chunking"}]

    index = VectorStoreIndex.from_documents(
        chunk_docs, embed_model=_acl_embed_model(), show_progress=False
    )
    nodes = index.as_retriever(similarity_top_k=samples).retrieve(query)

    out: List[dict] = []
    for n in nodes:
        meta = getattr(n, "metadata", None) or getattr(n.node, "metadata", {})
        pid = meta.get("paper_id")
        text = getattr(n, "text", None) or getattr(n.node, "text", "")
        rec = {
            "paper_id": pid,
            "title": meta.get("title"),
            "authors": meta.get("authors") or [],
            "publication_year": meta.get("publication_year"),
            "DOI": meta.get("DOI"),
            "url": meta.get("url"),
            "affiliations": meta.get("affiliations") or [],
            "chunk_text": text[:2000],  # cap for prompt size
        }
        out.append(rec)
        _evidence_buffer.append({
            "title": rec["title"],
            "authors": rec["authors"],
            "publication_year": rec["publication_year"],
            "DOI": rec["DOI"],
            "url": rec["url"],
            "affiliations": rec["affiliations"],
        })
    return out


def search_scico_fulltext_papers(
    query: str,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    institution: Optional[str] = None,
    samples: int = 12,
) -> List[dict]:
    """Search the local SciCo full-text index for ACL papers.

    Use this when the question needs methodology-level detail — e.g.
    "how does paper X's approach compare to other RAG work", "what
    evaluation protocol did Y use", paper-vs-paper comparisons,
    method-level surveys. The full-text index is larger and slower than
    the abstracts index but actually reads the paper body.

    For institution / topic / overview queries that don't need
    methodology detail, prefer search_local_acl_papers (abstracts).

    Args:
        query: Topic, method, or research question.
        year_start, year_end: Restrict by publication year.
        institution: Optional affiliation substring filter.
        samples: Max papers to return (default 12; cap 50).
    """
    return _retrieve_from_persist_dir(
        DEFAULT_SCICO_FULLTEXT_PERSIST_DIR,
        query,
        year_start=year_start,
        year_end=year_end,
        institution=institution,
        samples=samples,
    )


def build_default_tools() -> List[FunctionTool]:
    """Return the default tool set for the research-comparison agent."""
    return [
        FunctionTool.from_defaults(fn=list_university_publications),
        FunctionTool.from_defaults(fn=get_university_publications_profile),
        FunctionTool.from_defaults(fn=search_openalex_by_topic),
        FunctionTool.from_defaults(fn=search_core_fulltext),
        FunctionTool.from_defaults(
            fn=search_tu_wien_papers_tool, name="search_tu_wien_papers"
        ),
        FunctionTool.from_defaults(fn=search_internal_kg_research_unit),
        FunctionTool.from_defaults(fn=search_local_acl_papers),
        FunctionTool.from_defaults(fn=drill_into_papers_fulltext),
    ]


def build_acl_only_tools() -> List[FunctionTool]:
    """Tool set restricted to the local SciCo dataset — for benchmarks that
    evaluate the agent's retrieval+synthesis on ACL alone.

    Hierarchical pattern: abstracts first (fast), drill into a candidate
    set's full text only when methodology detail is needed.
    """
    return [
        FunctionTool.from_defaults(fn=search_local_acl_papers),
        FunctionTool.from_defaults(fn=drill_into_papers_fulltext),
    ]
