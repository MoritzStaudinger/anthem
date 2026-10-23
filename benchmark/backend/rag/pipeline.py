import json
import logging
from dataclasses import dataclass
from typing import Dict, Any, List

from neo4j import graph

logger = logging.getLogger(__name__)

from backend.core.reader import CoreReader
from backend.openalex.utils import get_most_relevant_institutions_for_queries, get_ror_id, search_paper, \
    get_institution_profile, get_university_name_from_ror
from backend.neo4j.utils import Neo4jReader, search_paper_graph, search_paper_graph_by_unit

from backend.rag.rag_utils import build_index, generate_questions, retrieve, extract_top_universities, build_index_core, \
    summarize_university, institutional_summary, summarize_universities, institutional_summary_structured, \
    compare_universities, compare_papers, overview, paper_answer, summarize, compare_two_universities, \
    fix_output_citations, use_abstracts_universities, use_abstracts_university, non_summarize
from backend.utils.utils import remove_duplicates_by_doi, remove_duplicates_by_doi_dict
from backend.tuwien.utils import search_tuwien_papers, get_tuwien_profile

def stream_fn(msg: str):
    payload = {"update": msg}
    return [f"data: {json.dumps(payload)}\n\n".encode("utf-8")]

@dataclass
class SimpleDocNode:
    text: str
    metadata: Dict[str, Any]

def handle_single_university_wrapper(*args, stream=False, **kwargs):
    if stream:
        return handle_single_university(*args, **kwargs)
    else:
        # Consume the generator and return the final result
        result = b""
        for chunk in handle_single_university(*args, **kwargs):
            result += chunk
        return result


def handle_single_university(data, prompt, online, model, top_k=15, stream_fn=None):
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    queries = data["queries"]
    profiles = []
    all_retrieved_docs = []

    # 1. Data collection and indexing
    if online:
        yield from emit("Fetching ROR ID for the university.")
        ror_id = get_ror_id(data['university_name'])
        logger.debug("ROR ID: %s", ror_id)

        yield from emit("Searching for papers and building index.")
        paper_list = []

        # Profile papers
        profile = get_institution_profile(
            data['topic'], ror_id, queries=queries,
            year_start=data['year_start'], year_end=data['year_end']
        )
        profiles.append(profile)

        # Topic-wide paper sampling
        topic_papers = search_paper(
            data['topic'], institution_id=ror_id,
            year_start=data['year_start'], year_end=data['year_end'],
            samples=100
        )
        paper_list.extend(topic_papers)

        # Query-specific papers
        for query in queries:
            papers = search_paper(
                query, institution_id=ror_id,
                year_start=data['year_start'], year_end=data['year_end'],
                samples=100
            )
            paper_list.extend(papers)

        paper_list = remove_duplicates_by_doi_dict(paper_list)
        yield from emit(f"Found and indexed {len(paper_list)} unique papers.")
        index = build_index(paper_list, llm_model=model)

        # Retrieve top-k papers per query
        for query in queries:
            docs = retrieve(
                query, institution=data['university_name'], top_k=top_k,
                year_start=data['year_start'], year_end=data['year_end'], index=index
            )
            all_retrieved_docs.append(docs)

    else:
        yield from emit("Using offline retrieval.")
        for query in queries:
            docs = retrieve(
                query, institution=data['university_name'], top_k=top_k,
                year_start=data['year_start'], year_end=data['year_end']
            )
            all_retrieved_docs.append(docs)

    total_retrieved = sum(len(docs) for docs in all_retrieved_docs)
    yield from emit(f"Retrieved {total_retrieved} documents across {len(queries)} queries (top {top_k} per query).")

    # 2. Summarization
    yield from emit("Generating summaries.")
    flat_docs = [doc for sublist in all_retrieved_docs for doc in sublist]

    summary = use_abstracts_university(flat_docs) #summarize_university(flat_docs)
    logger.debug("abstracts summary: %s", summary)
    full_summary = institutional_summary(summary, data['university_name'], prompt=prompt)
    logger.debug("institutional summary: %s", full_summary)
    fixed_summary = fix_output_citations(full_summary)
    logger.debug("citation-fixed summary: %s", fixed_summary)

    # Final output
    result = {
        "summary": fixed_summary,
        "profiles": profiles,
        "retrieved_docs_count": total_retrieved,
        "query_count": len(queries),
        "indexed_papers": len(paper_list) if online else None
    }

    yield f"data: {json.dumps(result)}\n\n".encode("utf-8")

def handle_tuwien_wrapper(*args, stream=False, **kwargs):
    """Wrapper to handle streaming vs non-streaming mode."""
    if stream:
        return handle_tuwien(*args, **kwargs)
    else:
        result = b""
        for chunk in handle_tuwien(*args, **kwargs):
            result += chunk
        return result

def handle_tuwien(data, prompt, online=True, model="gpt-4o-mini", top_k=15, stream_fn=None):
    """
    Handle TU Wien-specific queries using Repositum and TISS APIs.
    
    Args:
        data: Query data with topic, queries, year filters
        prompt: User's original prompt
        online: Whether to use online search (always True for TU Wien)
        model: LLM model to use
        top_k: Number of top documents to retrieve per query
        stream_fn: Optional streaming function for progress updates
    """
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)
    
    queries = data.get("queries", [])
    topic = data.get("topic", "")
    year_start = data.get("year_start")
    year_end = data.get("year_end")
    include_researchers = data.get("include_projects", False)  # Reusing this flag
    
    profiles = []
    all_retrieved_docs = []
    
    # 1. Get TU Wien profile
    yield from emit("Fetching TU Wien research data from Repositum...")
    try:
        profile = get_tuwien_profile(
            topic=topic,
            queries=queries,
            year_start=year_start,
            year_end=year_end,
            include_researchers=include_researchers
        )
        profiles.append(profile)
        
        total_papers = len(profile.get("papers", []))
        yield from emit(f"Found {total_papers} publications from TU Wien.")
    except Exception as e:
        yield from emit(f"Error fetching TU Wien data: {str(e)}")
        total_papers = 0
        profile = {
            "Institution": "TU Wien",
            "Publications": 0,
            "papers": []
        }
    
    # 2. Build index and retrieve if we have papers
    if total_papers > 0:
        try:
            yield from emit("Building search index from TU Wien publications...")
            paper_list = profile["papers"]
            index = build_index(paper_list, llm_model=model)
            
            # 3. Retrieve relevant documents for each query
            yield from emit(f"Retrieving top {top_k} documents per query...")
            for query in queries:
                docs = retrieve(
                    query, 
                    institution="TU Wien", 
                    top_k=top_k,
                    year_start=year_start, 
                    year_end=year_end, 
                    index=index
                )
                all_retrieved_docs.append(docs)
            
            total_retrieved = sum(len(docs) for docs in all_retrieved_docs)
            yield from emit(f"Retrieved {total_retrieved} relevant documents.")
            
            # 4. Summarize findings
            yield from emit("Generating summary of TU Wien contributions...")
            flat_docs = [doc for sublist in all_retrieved_docs for doc in sublist]
            
            if flat_docs:
                # Use abstracts as summaries (faster)
                summary = use_abstracts_university(flat_docs)
                
                # Create institutional summary
                full_summary = institutional_summary(summary, "TU Wien", prompt=prompt)
                fixed_summary = fix_output_citations(full_summary)
            else:
                fixed_summary = (
                    "Retrieved documents from TU Wien but could not generate detailed summary. "
                    f"Found {total_papers} publications related to {topic}."
                )
        except Exception as e:
            yield from emit(f"Error processing TU Wien data: {str(e)}")
            fixed_summary = f"Error during processing: {str(e)}"
    else:
        yield from emit("No publications found matching the criteria.")
        fixed_summary = (
            "No publications were found from TU Wien matching the specified criteria. "
            "This could be because:\n"
            "- The Repositum API has limited public access\n"
            "- The topic may not be well-represented in accessible repositories\n"
            "- The year range may be too restrictive\n"
            "- The search terms may need to be adjusted\n\n"
            "Suggestions:\n"
            "- Try broader search terms (e.g., 'computer science' instead of specific algorithms)\n"
            "- Remove or expand the year range\n"
            "- Verify you have network access to repositum.tuwien.at"
        )
    
    # 5. Return results
    result = {
        "summary": fixed_summary,
        "profiles": profiles,
        "retrieved_docs_count": sum(len(docs) for docs in all_retrieved_docs) if all_retrieved_docs else 0,
        "query_count": len(queries),
        "total_papers": total_papers,
        "data_sources": ["TU Wien Repositum"]
    }
    
    yield f"data: {json.dumps(result)}\n\n".encode("utf-8")

def process_input(prompt, paper=None, online=False, model="gpt-4o-mini"):
    """Run the lookup pipeline synchronously and return (summary, profiles).

    Delegates to process_input_streaming with stream_fn=None and consumes the
    SSE chunks looking for the final ``data: {"summary": ..., "profiles": ...}``
    payload. Single source of truth so callers don't need to know which query
    types stream and which return tuples.
    """
    summary = None
    profiles = []
    for chunk in process_input_streaming(
        prompt, paper=paper, online=online, model=model, stream_fn=None
    ):
        text = chunk.decode("utf-8", errors="replace") if isinstance(chunk, (bytes, bytearray)) else chunk
        for line in text.split("\n"):
            line = line.strip()
            if not line.startswith("data:"):
                continue
            body = line[len("data:"):].strip()
            if not body:
                continue
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and "summary" in parsed:
                summary = parsed["summary"]
                profiles = parsed.get("profiles") or profiles
    return summary, profiles

def process_input_streaming(prompt, paper=None, online=False, model="gpt-4o-mini", stream_fn=None):
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    yield from emit("Classifying input")
    data = generate_questions(prompt, paper, llm_model=model)

    data_type = data.get("type")
    yield from emit(f"Detected query type: {data_type}")

    if data_type == "single_university":
        yield from handle_single_university_wrapper(data, prompt, online=online, model=model, stream=True, stream_fn=stream_fn)

    elif data_type == "university":
        yield from handle_university_wrapper(data, prompt, online=online, model=model, stream=True, stream_fn=stream_fn)

    elif data_type == "tuwien":  
        yield from handle_tuwien_wrapper(data, prompt, online=True, model=model, stream=True, stream_fn=stream_fn)

    elif data_type == "university_university":
        yield from handle_university_university_wrapper(data, prompt, online=online, model=model, stream=True, stream_fn=stream_fn)

    elif data_type == "comparison":
        yield from handle_comparison_wrapper(data, prompt, paper, online=online, fulltext=True, model=model, stream=True, stream_fn=stream_fn)

    elif data_type == "paper":
        yield from handle_paper_wrapper(data, prompt, paper, online=online, model=model, stream=True, stream_fn=stream_fn)

    elif data_type == "overview":
        yield from handle_overview_wrapper(data, prompt, online=online, fulltext=True, model=model, stream=True, stream_fn=stream_fn)
    elif data_type == "internal_kg":
        yield from handle_internal_kg_wrapper(data, prompt, online=online, model=model, stream=True, stream_fn=stream_fn)
    elif data_type == "internal_kg_research_unit":
        yield from handle_overview_research_unit_wrapper(data, prompt, online=online, model=model, stream=True, stream_fn=stream_fn)
    else:
        yield from emit(f"Unknown query type: {data_type}")
        raise ValueError(f"Unknown type: {data_type}")


def handle_university_wrapper(*args, stream=False, **kwargs):
    if stream:
        return handle_university(*args, **kwargs)
    else:
        result = b""
        for chunk in handle_university(*args, **kwargs):
            result += chunk
        return result

def handle_university(data, prompt, online=False, model="gpt-4o-mini", stream_fn=None):
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    queries = data["queries"]
    profiles = []

    if not online:
        yield from emit("📄 Retrieving documents (offline mode)...")
        all_results = [
            retrieve(query, top_k=1000, year_start=data['year_start'], year_end=data['year_end'],
                     country=data['country']) for query in queries
        ]
        docs = [item for sublist in all_results for item in sublist]
        docs = remove_duplicates_by_doi(docs)

        yield from emit("📑 Getting top documents for the university...")
        single_university_docs_ml = [
            retrieve(query, institution=data['university_name'], top_k=15,
                     year_start=data['year_start'], year_end=data['year_end']) for query in queries
        ]

        yield from emit("🏫 Extracting top universities...")
        top_universities = extract_top_universities(docs, query=prompt, exclude=data['university_name'])

    else:
        yield from emit("🔍 Getting ROR ID and profile...")
        ror_id = get_ror_id(data['university_name'])
        yield from emit("🌍 Finding relevant institutions...")
        most_relevant_institutions = get_most_relevant_institutions_for_queries(
            queries, year_start=data['year_start'], year_end=data['year_end'],
            country_code=data['country'], limit=3, exclude_ror=ror_id)
        logger.debug("most relevant institutions: %s", most_relevant_institutions)
        yield from emit("Create Research Profile")
        profile = get_institution_profile(data['topic'], ror_id, queries=queries,
                                          year_start=data['year_start'], year_end=data['year_end'])
        profiles.append(profile)

        yield from emit("📄 Searching and indexing papers...")
        paper_list = []
        paper_list.extend(
            search_paper(data['topic'], institution_id=ror_id,
                         year_start=data['year_start'], year_end=data['year_end'], samples=100))

        for query in queries:
            for institution in most_relevant_institutions:
                papers = search_paper(query, institution_id=institution[0],
                                      year_start=data['year_start'], year_end=data['year_end'], samples=100)
                paper_list.extend(papers)

            papers_solo = search_paper(query, institution_id=ror_id,
                                       year_start=data['year_start'], year_end=data['year_end'], samples=100)
            paper_list.extend(papers_solo)

        paper_list = remove_duplicates_by_doi_dict(paper_list)
        index = build_index(paper_list, llm_model=model)

        yield from emit(f"📚 Indexed {len(paper_list)} papers.")

        all_results = [
            retrieve(query, top_k=10, year_start=data['year_start'], year_end=data['year_end'],
                     country=data['country'], index=index) for query in queries
        ]
        docs = [item for sublist in all_results for item in sublist]
        docs = remove_duplicates_by_doi(docs)

        yield from emit("📑 Getting top documents for the university...")
        single_university_docs_ml = [
            retrieve(query, institution=data['university_name'], top_k=10,
                     year_start=data['year_start'], year_end=data['year_end'], index=index) for query in queries
        ]

        yield from emit("🏫 Extracting top universities...")
        top_universities = extract_top_universities(docs, query=prompt, exclude=get_university_name_from_ror(ror_id))

        yield from emit("📄 Getting profiles of top universities...")
        for institution in most_relevant_institutions:
            inst_profile = get_institution_profile(data['topic'], institution[0],
                                                   queries=queries, year_start=data['year_start'], year_end=data['year_end'])
            profiles.append(inst_profile)
    logger.debug("collected %d profiles: %s", len(profiles), profiles)
    yield from emit("🧠 Summarizing the target university...")
    single_university_docs = [item for sublist in single_university_docs_ml for item in sublist]
    university_summary = use_abstracts_university(single_university_docs) #summarize_university(single_university_docs)
    university_over_summary = institutional_summary(university_summary, data['university_name'])

    yield from emit("🧠 Summarizing comparison universities...")
    top_universities = use_abstracts_universities(top_universities) #summarize_universities(top_universities)
    top_universities = institutional_summary_structured(top_universities)

    yield from emit("🆚 Comparing universities...")
    comparison = compare_universities(top_universities, university_over_summary, prompt)
    output = fix_output_citations(comparison)

    retrieved_papers = _nodes_to_records(single_university_docs)
    yield f"data: {json.dumps({'summary': output, 'profiles': profiles, 'retrieved_papers': retrieved_papers})}\n\n".encode("utf-8")



def handle_university_university(data, prompt, top_k=15, online=False, model="gpt-4o-mini", stream_fn=None):
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    queries = data["queries"]
    profiles = []
    total_indexed_papers = 0
    university_docs_ml = []
    comparison_university_docs_ml = []

    if online:
        yield from emit("Fetching ROR IDs for both universities.")
        ror_id = get_ror_id(data['university_name'])
        comp_ror_id = get_ror_id(data['comparison_university_name'])

        yield from emit("Building institutional profiles.")
        profile = get_institution_profile(data['topic'], ror_id, queries=queries,
                                          year_start=data['year_start'], year_end=data['year_end'])
        comp_profile = get_institution_profile(data['topic'], comp_ror_id, queries=queries,
                                               year_start=data['year_start'], year_end=data['year_end'])
        profiles.extend([profile, comp_profile])

        yield from emit("Collecting and indexing papers.")
        paper_list = []

        # Add topic-based papers
        paper_list.extend(search_paper(data['topic'], institution_id=ror_id,
                                       year_start=data['year_start'], year_end=data['year_end'], samples=100))
        paper_list.extend(search_paper(data['topic'], institution_id=comp_ror_id,
                                       year_start=data['year_start'], year_end=data['year_end'], samples=100))

        # Add query-specific papers
        for query in queries:
            paper_list.extend(search_paper(query, institution_id=ror_id,
                                           year_start=data['year_start'], year_end=data['year_end'], samples=100))
            paper_list.extend(search_paper(query, institution_id=comp_ror_id,
                                           year_start=data['year_start'], year_end=data['year_end'], samples=100))

        paper_list = remove_duplicates_by_doi_dict(paper_list)
        total_indexed_papers = len(paper_list)
        yield from emit(f"Indexed {total_indexed_papers} unique papers.")

        index = build_index(paper_list, llm_model=model)

        yield from emit("Retrieving top documents per university and query.")
        university_docs_ml = [
            retrieve(query, institution=data['university_name'], top_k=top_k,
                     year_start=data['year_start'], year_end=data['year_end'], index=index)
            for query in queries
        ]
        comparison_university_docs_ml = [
            retrieve(query, institution=data['comparison_university_name'], top_k=top_k,
                     year_start=data['year_start'], year_end=data['year_end'], index=index)
            for query in queries
        ]

    else:
        yield from emit("Offline document retrieval for both universities.")
        university_docs_ml = [
            retrieve(query, institution=data['university_name'], top_k=top_k,
                     year_start=data['year_start'], year_end=data['year_end'])
            for query in queries
        ]
        comparison_university_docs_ml = [
            retrieve(query, institution=data['comparison_university_name'], top_k=top_k,
                     year_start=data['year_start'], year_end=data['year_end'])
            for query in queries
        ]

    total_docs_univ = sum(len(docs) for docs in university_docs_ml)
    total_docs_comp = sum(len(docs) for docs in comparison_university_docs_ml)
    yield from emit(f"Retrieved {total_docs_univ} docs for {data['university_name']} and {total_docs_comp} for {data['comparison_university_name']}.")

    # ===== Summarization and Comparison =====
    yield from emit(f"Summarizing papers from {data['university_name']}...")
    university_docs = [item for sublist in university_docs_ml for item in sublist]
    university_summary = use_abstracts_university(university_docs) #summarize_university(university_docs)
    university_overview = institutional_summary(university_summary, data['university_name'])

    yield from emit(f"Summarizing papers from {data['comparison_university_name']}.")
    comparison_docs = [item for sublist in comparison_university_docs_ml for item in sublist]
    comparison_summary = use_abstracts_university(comparison_docs) #summarize_university(comparison_docs)
    comparison_overview = institutional_summary(comparison_summary, data['comparison_university_name'])

    yield from emit("Comparing both universities.")
    comparison = compare_two_universities(university_overview, comparison_overview, prompt)
    output = fix_output_citations(comparison)

    final_result = {
        "summary": output,
        "profiles": profiles,
        "retrieved_papers": _nodes_to_records(university_docs + comparison_docs),
        "indexed_papers": total_indexed_papers if online else None,
        "retrieved_docs": {
            data['university_name']: total_docs_univ,
            data['comparison_university_name']: total_docs_comp
        },
        "query_count": len(queries),
        "mode": "online" if online else "offline"
    }

    yield f"data: {json.dumps(final_result)}\n\n".encode("utf-8")

def handle_university_university_wrapper(*args, stream=False, **kwargs):
    if stream:
        return handle_university_university(*args, **kwargs)
    else:
        result = b""
        for chunk in handle_university_university(*args, **kwargs):
            result += chunk
        return result


def _nodes_to_records(docs):
    """Flatten a list of llama_index nodes (or NodeWithScore objects)
    into the per-paper record dict shape eval_scico.py expects.

    Used by every pipeline-mode handler so the eval can recover the
    retrieved paper set for P@k/R@k scoring. Without this the eval sees
    only the summary text and computes records=0 / retrieval metrics=0.
    """
    out = []
    seen = set()
    for d in docs or []:
        if hasattr(d, 'metadata'):
            meta = d.metadata or {}
        elif hasattr(d, 'node'):
            meta = (getattr(d.node, 'metadata', {}) or {})
        elif isinstance(d, dict):
            meta = d.get('metadata') or d
        else:
            continue
        pid = meta.get("paper_id") or meta.get("doi") or meta.get("title")
        if not pid or pid in seen:
            continue
        seen.add(pid)
        out.append({
            "title": meta.get("title"),
            "authors": meta.get("authors") or [],
            "publication_year": meta.get("publication_year") or meta.get("year"),
            "DOI": meta.get("doi"),
            "url": meta.get("doi") or meta.get("url"),
            "affiliations": meta.get("affiliations") or [],
            "paper_id": pid,
        })
    return out


# ---------------------------------------------------------------------------
# SciCo-eval-specific handlers
#
# The two handlers below mirror handle_university and
# handle_university_university but slot in a `country` filter for the
# SciCo testset's cross_country_method and university_country_comparison
# question categories. Both run offline against the local ACL index
# (the SciCo evaluation corpus); the online path is intentionally not
# implemented here because the SciCo experiments don't use OpenAlex.
# ---------------------------------------------------------------------------

def handle_cross_country_method(data, prompt, online=False, top_k=15,
                                model="gpt-4o-mini", stream_fn=None):
    """Compare research methods between two countries on a topic.

    Expected `data` fields:
        country_a, country_b           — ISO 3166-1 alpha-2 codes (e.g. "US", "CN")
        queries                        — list of retrieval queries
        year_start, year_end           — optional year window
    """
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    queries = data["queries"]
    yield from emit(f"Offline retrieval for {data['country_a']} vs {data['country_b']}...")

    country_a_docs_ml = [
        retrieve(q, country=data["country_a"], top_k=top_k,
                 year_start=data.get("year_start"),
                 year_end=data.get("year_end"))
        for q in queries
    ]
    country_b_docs_ml = [
        retrieve(q, country=data["country_b"], top_k=top_k,
                 year_start=data.get("year_start"),
                 year_end=data.get("year_end"))
        for q in queries
    ]

    country_a_docs = [item for sub in country_a_docs_ml for item in sub]
    country_b_docs = [item for sub in country_b_docs_ml for item in sub]
    yield from emit(f"Retrieved {len(country_a_docs)} docs for "
                    f"{data['country_a']} and {len(country_b_docs)} for "
                    f"{data['country_b']}.")

    yield from emit(f"Summarizing {data['country_a']} contributions...")
    sum_a = use_abstracts_university(country_a_docs)
    overview_a = institutional_summary(sum_a, f"papers from {data['country_a']}")

    yield from emit(f"Summarizing {data['country_b']} contributions...")
    sum_b = use_abstracts_university(country_b_docs)
    overview_b = institutional_summary(sum_b, f"papers from {data['country_b']}")

    yield from emit("Synthesizing cross-country comparison...")
    # compare_two_universities is a generic "compare two named blobs"
    # synthesiser; it doesn't care that they're countries here.
    comparison = compare_two_universities(
        {"name": data['country_a'], "institutional_summary": overview_a},
        {"name": data['country_b'], "institutional_summary": overview_b},
        prompt,
    )
    output = fix_output_citations(comparison)

    retrieved_papers = _nodes_to_records(country_a_docs + country_b_docs)
    final = {
        "summary": output,
        "profiles": [],
        "retrieved_papers": retrieved_papers,
        "retrieved_docs": {
            data['country_a']: len(country_a_docs),
            data['country_b']: len(country_b_docs),
        },
        "query_count": len(queries),
        "mode": "offline",
        "question_category": "cross_country_method",
    }
    yield f"data: {json.dumps(final)}\n\n".encode("utf-8")


def handle_cross_country_method_wrapper(*args, stream=False, **kwargs):
    if stream:
        return handle_cross_country_method(*args, **kwargs)
    result = b""
    for chunk in handle_cross_country_method(*args, **kwargs):
        result += chunk
    return result


def handle_university_country_comparison(data, prompt, online=False,
                                         top_k=15, model="gpt-4o-mini",
                                         stream_fn=None):
    """Overview of a university's work on a topic, narrowed to a
    country context (e.g. "what has Stanford published with German co-
    authors on retrieval?"). Filter is AND: papers must mention BOTH
    the institution and the country in their affiliation string.

    Expected `data` fields:
        university_name                — institution
        country                        — ISO 3166-1 alpha-2 code
        queries                        — list of retrieval queries
        year_start, year_end           — optional year window
    """
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    queries = data["queries"]
    yield from emit(f"Offline retrieval for {data['university_name']} ∩ "
                    f"{data['country']}...")

    docs_ml = [
        retrieve(q, institution=data["university_name"],
                 country=data["country"], top_k=top_k,
                 year_start=data.get("year_start"),
                 year_end=data.get("year_end"))
        for q in queries
    ]
    docs = [item for sub in docs_ml for item in sub]
    yield from emit(f"Retrieved {len(docs)} papers.")

    yield from emit("Summarizing per university × country slice...")
    abstracts_summary = use_abstracts_university(docs)
    label = f"{data['university_name']} ({data['country']})"
    overview_text = institutional_summary(abstracts_summary, label, prompt=prompt)
    output = fix_output_citations(overview_text)

    final = {
        "summary": output,
        "profiles": [],
        "retrieved_papers": _nodes_to_records(docs),
        "retrieved_docs": {label: len(docs)},
        "query_count": len(queries),
        "mode": "offline",
        "question_category": "university_country_comparison",
    }
    yield f"data: {json.dumps(final)}\n\n".encode("utf-8")


def handle_university_country_comparison_wrapper(*args, stream=False, **kwargs):
    if stream:
        return handle_university_country_comparison(*args, **kwargs)
    result = b""
    for chunk in handle_university_country_comparison(*args, **kwargs):
        result += chunk
    return result


def handle_comparison_wrapper(*args, stream=False, **kwargs):
    if stream:
        return handle_comparison(*args, **kwargs)
    else:
        result = b""
        for chunk in handle_comparison(*args, **kwargs):
            result += chunk
        return result

def handle_comparison(data, prompt, paper, online=False, fulltext=False, top_k=30, top_k_internal=15, model="gpt-4o-mini", stream_fn=None):
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    queries = data["queries"]
    all_results_ml = []
    retrieved_docs = []
    indexed_docs_count = 0
    mode = "offline"

    if online and fulltext:
        #mode = "online-fulltext"
        yield from emit("Loading full-text documents for topic, paper, and queries.")
        loader = CoreReader()
        full_docs = []

        for content in [data['topic'], paper['title'], paper['abstract']] + queries:
            docs = loader.load_data(content, max_results=top_k)
            full_docs.extend(docs)

        indexed_docs_count = len(full_docs)
        yield from emit(f"Indexed {indexed_docs_count} full-text documents.")
        index = build_index_core(full_docs, llm_model=model)

        yield from emit("Retrieving documents for comparison (fulltext).")
        all_results_ml = [
            retrieve(query, top_k=top_k_internal, year_start=data['year_start'], year_end=data['year_end'], index=index)
            for query in queries
        ]

    elif online and not fulltext:
        mode = "online-metadata"
        yield from emit("Searching metadata-only papers.")
        paper_list = []

        paper_list.extend(search_paper(data['topic'], year_start=data['year_start'], year_end=data['year_end'], samples=100))

        for query in queries:
            paper_list.extend(search_paper(query, year_start=data['year_start'], year_end=data['year_end'], samples=100))

        paper_list = remove_duplicates_by_doi_dict(paper_list)
        indexed_docs_count = len(paper_list)
        yield from emit(f"📚 Indexed {indexed_docs_count} metadata papers.")
        index = build_index(paper_list, llm_model=model)

        yield from emit("Retrieving documents for comparison (metadata)")
        all_results_ml = [
            retrieve(query, top_k=top_k_internal, year_start=data['year_start'], year_end=data['year_end'], index=index)
            for query in queries
        ]
        flat_results = [item for sublist in all_results_ml for item in sublist]
        retrieved_docs = remove_duplicates_by_doi(flat_results)

    else:
        mode = "offline"
        yield from emit("Retrieving documents (offline mode)")
        all_results_ml = [
            retrieve(query, top_k=top_k_internal, year_start=data['year_start'], year_end=data['year_end'])
            for query in queries
        ]
        retrieved_docs = [item for sublist in all_results_ml for item in sublist]

    if not retrieved_docs:
        retrieved_docs = [item for sublist in all_results_ml for item in sublist]

    yield from emit(f"Summarizing {len(retrieved_docs)} retrieved documents.")
    summaries = summarize(retrieved_docs)

    yield from emit("Comparing with target paper")
    comparison = compare_papers(summaries, prompt, paper)
    output = fix_output_citations(comparison)

    result = {
        "summary": output
    }
    logger.debug("comparison output: %s", output)
    yield f"data: {json.dumps(result)}\n\n".encode("utf-8")


def handle_overview(data, prompt, online=False, fulltext=False, top_k=50, model="gpt-4o-mini", stream_fn=None):
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    queries = data["queries"]
    all_results_ml = []
    retrieved_docs = []
    indexed_docs_count = 0

    if online and fulltext:
        mode = "online-fulltext"
        yield from emit("Loading full-text documents for topic and queries")
        loader = CoreReader()
        full_docs = []

        # Load fulltext documents for topic
        full_docs.extend(loader.load_data(data['topic'], max_results=top_k))

        # Load fulltext documents for each query
        for query in queries:
            full_docs.extend(loader.load_data(query, max_results=top_k))

        indexed_docs_count = len(full_docs)
        yield from emit(f"Indexed {indexed_docs_count} full-text documents.")
        index = build_index_core(full_docs, llm_model=model)

        yield from emit("Retrieving documents from full-text index")
        all_results_ml = [
            retrieve(query, top_k=15, year_start=data['year_start'], year_end=data['year_end'], index=index)
            for query in queries
        ]

    elif online and not fulltext:
        mode = "online-metadata"
        yield from emit("Searching metadata-only papers")
        paper_list = []

        # Search papers by topic and queries
        paper_list.extend(search_paper(data['topic'], year_start=data['year_start'], year_end=data['year_end'], samples=100))
        for query in queries:
            paper_list.extend(search_paper(query, year_start=data['year_start'], year_end=data['year_end'], samples=100))

        paper_list = remove_duplicates_by_doi_dict(paper_list)
        indexed_docs_count = len(paper_list)
        yield from emit(f"📚 Indexed {indexed_docs_count} metadata papers.")
        index = build_index(paper_list, llm_model=model)

        yield from emit("Retrieving documents from metadata index")
        all_results_ml = [
            retrieve(query, top_k=15, year_start=data['year_start'], year_end=data['year_end'], index=index)
            for query in queries
        ]

    else:
        mode = "offline"
        yield from emit("Retrieving documents (offline mode)")
        all_results_ml = [
            retrieve(query, top_k=5, year_start=data['year_start'], year_end=data['year_end'])
            for query in queries
        ]

    # Flatten + dedupe by paper_id (or DOI fallback). Without this the
    # 2 queries × top_k=5 = 10 docs gets bloated by near-identical hits
    # across queries; summarize() then makes a redundant LLM call per
    # duplicate. Deduping typically halves the LLM-call count for free.
    flat_docs = [item for sublist in all_results_ml for item in sublist]
    seen_keys = set()
    retrieved_docs = []
    for doc in flat_docs:
        meta = getattr(doc, "metadata", {}) or {}
        key = (meta.get("paper_id") or meta.get("doi") or meta.get("DOI")
               or meta.get("title") or "").strip().lower()
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        retrieved_docs.append(doc)
    yield from emit(f"Summarizing {len(retrieved_docs)} unique documents "
                    f"(from {len(flat_docs)} raw retrievals)")

    summaries = summarize(retrieved_docs)

    yield from emit("📊 Generating overview summary.")
    overview_result = overview(summaries, prompt)
    output = fix_output_citations(overview_result)

    result = {
        "summary": output,
        "retrieved_papers": _nodes_to_records(retrieved_docs),
    }

    yield f"data: {json.dumps(result)}\n\n".encode("utf-8")

def handle_overview_wrapper(*args, stream=False, **kwargs):
    if stream:
        return handle_overview(*args, **kwargs)
    else:
        result = b""
        for chunk in handle_overview(*args, **kwargs):
            result += chunk
        return result

def handle_paper_wrapper(*args, stream=False, **kwargs):
    if stream:
        return handle_paper(*args, **kwargs)
    else:
        result = b""
        for chunk in handle_paper(*args, **kwargs):
            result += chunk
        return result


def handle_paper(data, prompt, paper, online=False, model="gpt-4o-mini", stream_fn=None):
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    answer = paper_answer(prompt, paper)
    output = fix_output_citations(answer)

    result = {
        "summary": output
    }
    # GOOD (what your frontend expects):
    yield f"data: {json.dumps(result)}\n\n".encode("utf-8")

def handle_internal_kg(data, prompt, online=False, model="gpt-4o-mini", stream_fn=None):
    def emit(msg):
        if stream_fn:
            yield from stream_fn(msg)

    queries = data["queries"]
    all_results_ml = []
    retrieved_docs = []
    indexed_docs_count = 0

    if online:
        mode = "online-metadata"
        yield from emit("Searching Neo4j")
        paper_list = []

        # Search papers by topic and queries
        paper_list.extend(
            search_paper_graph(
                graph_client=graph,  # <-- your Neo4j wrapper instance
                query=data["topic"],
                year_start=data["year_start"],
                year_end=data["year_end"],
                samples=100,
            )
        )
        logger.debug("internal_kg paper_list: %s", paper_list)

        for query in queries:
            paper_list.extend(
                search_paper_graph(
                    graph_client=graph,
                    query=query,
                    year_start=data["year_start"],
                    year_end=data["year_end"],
                    samples=100,
                )
            )

        # NOTE: if your graph has no DOI, DOI-based dedup will collapse nothing.
        # Prefer a title+year dedup fallback (see below).
        indexed_docs_count = len(paper_list)
        yield from emit(f"Indexed {indexed_docs_count} metadata papers.")
        index = build_index(paper_list, llm_model=model)

        yield from emit("Retrieving documents from metadata index")
        all_results_ml = [
            retrieve(query, top_k=10, index=index)
            for query in queries]
        yield from emit("completed retrieval")

    yield from emit("Summarizing")
    flat_docs = [doc for sublist in all_results_ml for doc in sublist]
    summaries = summarize(flat_docs)
    yield from emit("Generating overview summary.")
    overview_result = overview(summaries, prompt)
    output = fix_output_citations(overview_result)

    result = {
        "summary": output
    }

    yield f"data: {json.dumps(result)}\n\n".encode("utf-8")


def handle_internal_kg_wrapper(*args, stream=False, **kwargs):
    if stream:
        return handle_internal_kg(*args, **kwargs)
    else:
        result = b""
        for chunk in handle_internal_kg(*args, **kwargs):
            result += chunk
        return result

def handle_overview_research_unit(data, prompt, online=False, model="gpt-4o-mini", stream_fn=None):
    """
    Pipeline: Retrieve publications from Neo4j by Research Unit + timeframe,
    then summarize each doc and produce an overview.
    """

    def emit(msg: str):
        if stream_fn:
            yield from stream_fn(msg)

    indexed_docs_count = 0
    docs_for_summarize: List[SimpleDocNode] = []

    if online:
        yield from emit("Searching Neo4j research-unit publications")

        reader = Neo4jReader()
        try:
            paper_list = search_paper_graph_by_unit(
                reader=reader,
                unit_text=data["research_unit"],
                year=data.get("year"),
                year_start=data.get("year_start"),
                year_end=data.get("year_end"),
                years=data.get("years"),
                samples=200,
                text=data.get("topic"),  # optional title/abstract filter
            )
        finally:
            # Ensure connections are closed even if something fails downstream
            reader.close()

        indexed_docs_count = len(paper_list)
        yield from emit(f"Retrieved {indexed_docs_count} publications; preparing documents for summarization")

        # Convert "work dicts" into nodes expected by summarize()
        for p in paper_list:
            abstract = (p.get("abstract") or "").strip()
            if not abstract:
                continue

            title = p.get("title") or p.get("display_name") or "Untitled"
            year = p.get("publication_year")

            # Authors from OpenAlex-like structure
            authorships = p.get("authorships") or []
            author_names = [
                (a.get("author") or {}).get("display_name")
                for a in authorships
                if (a.get("author") or {}).get("display_name")
            ]
            authors_str = ", ".join(author_names) if author_names else "Unknown authors"

            doi = p.get("doi")  # set by your updated mapping (Publication.id -> doi)
            # If you ever only have p["id"] like "https://doi.org/...", normalize:
            if not doi and p.get("id"):
                s = str(p["id"]).strip()
                s = s.replace("https://doi.org/", "").replace("http://doi.org/", "").replace("doi.org/", "")
                doi = s or None

            docs_for_summarize.append(
                SimpleDocNode(
                    # summarize() uses node.text as "Full Text"; if you only have abstracts,
                    # use abstract as text and also store it in metadata["abstract"].
                    text=abstract,
                    metadata={
                        "title": title,
                        "abstract": abstract,
                        "authors": authors_str,
                        "doi": doi,
                        "publication_year": year,
                    },
                )
            )

        yield from emit(f"Prepared {len(docs_for_summarize)} documents with abstracts for summarization")

    # Summarization phase
    yield from emit("Summarizing")
    summaries = non_summarize(docs_for_summarize)

    # Overview synthesis
    yield from emit("Generating overview summary.")
    overview_result = overview(summaries, prompt, top_k=10)
    output = fix_output_citations(overview_result)

    result = {
        "summary": output,
        "indexed_docs": indexed_docs_count,
        "research_unit": data["research_unit"],
        "summarized_docs": len(docs_for_summarize),
    }

    yield f"data: {json.dumps(result)}\n\n".encode("utf-8")

def handle_overview_research_unit_wrapper(*args, stream=False, **kwargs):
    if stream:
        logger.debug("overview_research_unit args: %s", args)
        return handle_overview_research_unit(*args, **kwargs)
    else:
        result = b""
        for chunk in handle_overview_research_unit(*args, **kwargs):
            result += chunk
        return result