import json
import logging
import os
import re
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai_like import OpenAILike

static_index = None  # Set it at the end of the file

# DON'T set Settings.llm = None at import time. llama-index emits a
# misleading "LLM is explicitly disabled. Using MockLLM." print to stdout
# whenever Settings.llm is None at read time. The agent never reads
# Settings.llm before configure_llm() runs anyway, so leaving the field
# unset (default) keeps the log clean without changing behavior.

logger = logging.getLogger(__name__)

AQUEDUCT_BASE_URL = "https://aqueduct.ai.datalab.tuwien.ac.at/"


def _api_key(name: str) -> str | None:
    """Read an API key from the environment at call time rather than at
    import time. Lets `.env` edits take effect without restarting the
    backend (within sync gunicorn workers, the next request reads fresh)."""
    return os.getenv(name)


def configure_llm(
    model_name: str,
    *,
    chunk_size: int = 8192,
    embedding_model: str = "text-embedding-ada-002",
    temperature: float = 0.0,
):
    """Configure the llama-index global Settings for the requested model.

    Single source of truth for model→provider mapping; previously this logic
    was duplicated across build_index_core, build_index, and generate_questions
    with subtle case-sensitivity drift between copies.

    When ``model_name`` is falsy this is a no-op (preserves current Settings).
    Several call sites — e.g. ``build_index(paper_list)`` inside the handlers —
    rely on ``generate_questions`` having configured ``Settings.llm`` earlier
    in the same request and don't re-pass the model. Unknown model names still
    raise ValueError, which is the property the refactor adds.
    """
    if not model_name:
        return Settings.llm

    Settings.chunk_size = chunk_size

    openai_api_key = _api_key("OPENAI_API_KEY")
    anthropic_api_key = _api_key("ANTHROPIC_API_KEY")
    google_api_key = _api_key("GOOGLE_API_KEY")
    aqueduct_api_key = _api_key("AQUEDUCT_API_KEY")
    # Accept both names: OPEN_ROUTER is what the user added to .env,
    # OPENROUTER_API_KEY is what the pipelines.yaml comments document.
    openrouter_api_key = _api_key("OPEN_ROUTER") or _api_key("OPENROUTER_API_KEY")

    # Generic retry budget — providers vary in rate-limit tightness;
    # 10 attempts gives ~30s of exponential backoff before giving up,
    # which absorbs typical bursts (Aqueduct 30/min, Anthropic ~5 RPM
    # free tier). Llama-index honors max_retries on the OpenAI base
    # client and OpenAILike. Anthropic / GoogleGenAI use `max_retries`
    # in newer versions.
    _RETRY = 10

    if model_name in {"gpt-4o-mini", "gpt-4o", "gpt-5-mini", "gpt-5"}:
        Settings.llm = OpenAI(model=model_name, temperature=temperature, max_tokens=16384,
                              api_key=openai_api_key, max_retries=_RETRY)
        Settings.embed_model = OpenAIEmbedding(model=embedding_model, api_key=openai_api_key)
    elif model_name in {"claude-sonnet-4-6", "claude-haiku-4-5", "claude-opus-4-7"}:
        Settings.llm = Anthropic(model=model_name, temperature=temperature, max_tokens=16384,
                                 api_key=anthropic_api_key, max_retries=_RETRY)
        Settings.embed_model = OpenAIEmbedding(model=embedding_model, max_tokens=16384, api_key=openai_api_key)
    elif model_name == "gemini-2.5-pro":
        # llama-index-llms-google-genai uses the newer google-genai SDK and
        # handles tool-call-only responses without crashing on empty parts,
        # unlike the older llama-index-llms-gemini.
        Settings.llm = GoogleGenAI(
            model="gemini-2.5-pro",
            api_key=google_api_key,
            temperature=temperature,
            max_retries=_RETRY,
        )
        Settings.embed_model = GoogleGenAIEmbedding(model_name="text-embedding-004", api_key=google_api_key)
    elif model_name in {"qwen-3.5-397b", "qwen-3.6-35b"}:
        # Aqueduct's Qwen-3 lineup are reasoning models — keep thinking
        # ON so we get `reasoning_content` for the paper appendix.
        # Tradeoff: first-token latency goes from <2s to 30-120s while
        # the reasoning chain streams. max_tokens raised to 32k to fit
        # reasoning + final answer; timeout raised to 600s to match.
        Settings.llm = OpenAILike(
            model=model_name,
            temperature=temperature,
            max_tokens=32768,
            api_key=aqueduct_api_key,
            api_base=AQUEDUCT_BASE_URL,
            is_chat_model=True,
            is_function_calling_model=True,
            context_window=128000,
            timeout=600,
            max_retries=_RETRY,
            additional_kwargs={
                "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}
            },
        )
        Settings.embed_model = OpenAIEmbedding(model=embedding_model, api_key=openai_api_key)
    elif model_name in {"gemma-4-e2b", "gemma-4-e2b-it"}:
        # Aqueduct's lightweight Gemma. Smaller context, no reasoning
        # toggle to set. Function calling support is shakier than Qwen's
        # — agent retries should cover that.
        Settings.llm = OpenAILike(
            model=model_name,
            temperature=temperature,
            max_tokens=8192,
            api_key=aqueduct_api_key,
            api_base=AQUEDUCT_BASE_URL,
            is_chat_model=True,
            is_function_calling_model=True,
            context_window=32000,
            timeout=360,
            max_retries=_RETRY,
        )
        Settings.embed_model = OpenAIEmbedding(model=embedding_model, api_key=openai_api_key)
    elif "/" in model_name:
        # OpenRouter — OpenAI-compatible gateway, identified by the
        # "vendor/model" naming convention (e.g. deepseek/deepseek-v4-flash,
        # meta-llama/llama-4-..., mistralai/mistral-large-3, etc.). Any
        # model with a slash routes here; the eval just passes the full
        # OpenRouter model id straight through.
        #
        # extra_body.include_reasoning=true asks OpenRouter to surface
        # `reasoning_content` in the response for reasoning models
        # (DeepSeek-R1/V4 reasoning, OpenAI o-series, etc.). For
        # non-reasoning models it's a no-op. The SciCo eval's
        # _ReasoningCaptureHandler picks it up automatically.
        if not openrouter_api_key:
            raise RuntimeError(
                "OPEN_ROUTER / OPENROUTER_API_KEY not set in env — needed "
                f"for OpenRouter model {model_name!r}"
            )
        # include_reasoning=True surfaces reasoning_content but reasoning
        # models on OpenRouter then put their tool-call / structured-JSON
        # output INTO the reasoning trace, leaving content empty. The
        # llama-index OpenAILike parser only reads content → empty
        # responses, missed tool calls, empty classifier JSON. This was
        # the root cause of Qwen-3.6 / DeepSeek failures in agent and
        # pipeline modes. Default to OFF; opt back in for non-tool /
        # non-JSON use cases by setting OPENROUTER_INCLUDE_REASONING=1.
        extra_body: dict = {}
        if os.getenv("OPENROUTER_INCLUDE_REASONING") == "1":
            extra_body["include_reasoning"] = True
        Settings.llm = OpenAILike(
            model=model_name,
            temperature=temperature,
            max_tokens=32768,
            api_key=openrouter_api_key,
            api_base="https://openrouter.ai/api/v1",
            is_chat_model=True,
            is_function_calling_model=True,
            context_window=128000,
            timeout=600,
            max_retries=_RETRY,
            additional_kwargs=(
                {"extra_body": extra_body} if extra_body else {}
            ),
        )
        Settings.embed_model = OpenAIEmbedding(model=embedding_model, api_key=openai_api_key)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    return Settings.llm


def build_index_core(
    data_input,
    chunk_size: int = 8192,
    embedding_model: str = "text-embedding-ada-002",
    llm_model: str = None,
    temperature: float = 0.0,
    show_progress: bool = True,
) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from paper metadata containing abstracts.

    Args:
        data_input: list of documents
        chunk_size (int): Chunk size for document splitting.
        embedding_model (str): Embedding model to use.
        llm_model (str): LLM model to use.
        temperature (float): Temperature setting for the LLM.
        show_progress (bool): Whether to show progress during index construction.

    Returns:
        VectorStoreIndex: The constructed vector store index.
    """
    configure_llm(
        llm_model,
        chunk_size=chunk_size,
        embedding_model=embedding_model,
        temperature=temperature,
    )

    # Build and return the vector index
    index = VectorStoreIndex([])

    all_documents = data_input

    # Initialize the text splitter
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=200)

    batch_size = 20
    for i in range(0, len(all_documents), batch_size):
        batch = all_documents[i:i + batch_size]

        # Split documents into chunks
        split_documents = []
        for doc in batch:
            chunks = text_splitter.split_text(doc.text)
            for chunk in chunks:
                split_documents.append(Document(text=chunk, metadata=doc.metadata))

        # Insert nodes into the index
        index.insert_nodes(split_documents)
    return index

def build_index_acl_persisted(
    documents,  # List[Document]
    persist_dir: str,
    *,
    llm_model: str = "gpt-4o-mini",
    chunk_size: int = 8192,
    embedding_model: str = "text-embedding-ada-002",
    temperature: float = 0.0,
):
    index = build_index_core(
        data_input=documents,
        chunk_size=chunk_size,
        embedding_model=embedding_model,
        llm_model=llm_model,
        temperature=temperature,
        show_progress=True,
    )

    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=persist_dir)
    return index


def load_persisted_index(persist_dir: str):
    storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
    return load_index_from_storage(storage_context)


def build_index(
    data_input: Union[str, List[Dict]],
    chunk_size: int = 8192,
    embedding_model: str = "text-embedding-ada-002",
    llm_model: str = None,
    temperature: float = 0.0,
    show_progress: bool = True
) -> VectorStoreIndex:
    """
    Build a VectorStoreIndex from paper metadata containing abstracts.

    Args:
        data_input (str or List[Dict]): Path to a JSON file or a list of dictionaries containing paper metadata.
        chunk_size (int): Chunk size for document splitting.
        embedding_model (str): Embedding model to use.
        llm_model (str): LLM model to use.
        temperature (float): Temperature setting for the LLM.
        show_progress (bool): Whether to show progress during index construction.

    Returns:
        VectorStoreIndex: The constructed vector store index.
    """
    # Load data from JSON file or use provided list
    if isinstance(data_input, str):
        with open(data_input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = data_input

    documents = []
    for entry in data:
        abstract = entry.get("abstract", "")
        title = entry.get("title", "")
        if abstract and title and len(abstract.strip()) > 20:
            metadata = {
                "title": title,
                "publication_year": entry.get("publication_year", ""),
                "authors": entry.get("authors", []),
                "doi": entry.get("DOI", ""),
                "url": entry.get("url") or entry.get("DOI") or entry.get("openalex_url") or "",
                "affiliations": entry.get("affiliations", []),
                "affiliations_string": ", ".join(a for a in entry.get("affiliations", []) if a),
                "countries": entry.get("countries", []),
                "countries_string": ", ".join(filter(None, entry.get("countries", [])))
            }
            documents.append(Document(text=abstract, metadata=metadata))

    configure_llm(
        llm_model,
        chunk_size=chunk_size,
        embedding_model=embedding_model,
        temperature=temperature,
    )

    # Build and return the vector index
    index = VectorStoreIndex.from_documents(documents, show_progress=show_progress)
    return index


def paper_answer(prompt, paper):
    context = f"{paper.get('title', '')}:{paper.get('abstract', '')} \n\n"
    context = context + f"Full Text:\n{paper.get('fulltext', '')}\n\n"

    answer_prompt = (
        f"{context}"
        f"Given the context above, provide a detailed answer to the posted question: {prompt}  "
        "Do not use any additional knowledge, and only use the information provided above"
        "If you cannot answer the question, state the limiting factors"
    )

    # Generate comparison paragraph with citations
    answer_response = Settings.llm.complete(answer_prompt).text
    logger.debug("paper answer: %s", answer_response.strip())
    return answer_response


def institutional_summary(summaries, institution, prompt=None):
    logger.debug("building institutional summary for %s", institution)
    if not summaries:
        return  # Skip if no summaries

    # Step 1: Build comparison context
    comparison_context = ""
    for s in summaries:
        comparison_context += f"[{s['index']}] {s['summary']}\n\n"

    # Step 2: Build references — use the canonical URL (DOI when available,
    # OpenAlex Work ID otherwise). Note s["doi"] from OpenAlex is already a
    # full URL; don't re-prefix.
    references = "\n".join(
        (
            f"[{institution}_{s['index']}] {s['title']} by {s['authors']}. {s.get('url') or s.get('doi')}"
            if (s.get('url') or s.get('doi'))
            else f"[{institution}_{s['index']}] {s['title']} by {s['authors']} (no link available)"
        )
        for s in summaries
    )
    if prompt:
        institutional_summary = (
            f"{comparison_context}"
            f"Write an overview of the contributions of {institution}, based on the given works above and the given user question:"
            f"User question: {prompt}"            
            "Do not use any additional knowledge, and only use the information provided above. "
            f"Use inline citations like [{institution}_1], [{institution}_2], etc., to refer to them. "
            f"After the paragraph, list the full references you used with the same inline citation style [{institution}_1], authors and DOIs.\n\n"
            f"{references}"
        )
    else:
        institutional_summary = (
            f"{comparison_context}"
            f"Write an overview of the contributions of {institution}, based on the given works above. "
            "Do not use any additional knowledge, and only use the information provided above. "
            f"Use inline citations like [{institution}_1], [{institution}_2], etc., to refer to them. "
            f"After the paragraph, list the full references you used with the same inline citation style [{institution}_1], authors and DOIs.\n\n"
            f"{references}"
        )

    # Step 4: Get LLM response
    institutional_response = Settings.llm.complete(institutional_summary).text.strip()
    logger.debug("institutional response: %s", institutional_response)
    # fixing_prompt = (
    #    f"{institutional_response}"
    #    "Given the response above, please remove all unused references in the end"
    #    "Do not change the text in any other way, and do not rename references"
    # )
    # fixing_response = Settings.llm.complete(fixing_prompt).text
    # print(fixing_response)

    # Step 5: Store response
    return institutional_response


def institutional_summary_structured(structured_output):
    for institution, data in structured_output.items():
        nodes_with_scores = data.get("top_documents", [])

        # Convert NodeWithScore(TextNode) to dict format expected by institutional_summary()
        summaries = []
        for i, nws in enumerate(nodes_with_scores):
            meta = nws.node.metadata
            element = {
                "index": i + 1,
                "title": meta.get("title", ""),
                "authors": meta.get("authors", []),
                "publication_year": meta.get("publication_year", []),
                "doi": meta.get("doi", ""),
                "url": meta.get("url") or meta.get("doi") or meta.get("openalex_url") or "",
                "summary": nws.node.text,
            }
            summaries.append(element)

        data["institutional_summary"] = institutional_summary(summaries, institution)
        logger.debug("structured institution data: %s", data)

    return structured_output

def compare_universities(top_universities, single_university, input_prompt):
    """
    Compare the research contributions of a single university against a set of top universities
    using only the provided institutional summaries.

    Args:
        top_universities (dict): Dictionary with university names as keys and a 'institutional_summary' field in values.
        single_university (dict): A single university's data, just the textual response.
        input_prompt (str): Optional guiding prompt to focus the comparison.

    Returns:
        str: A formatted and citation-corrected comparative analysis.
    """
    comparison_context = "# Summaries of Top Research Institutions:\n"
    for idx, (institution, data) in enumerate(top_universities.items(), start=1):
        comparison_context += f"[{idx}] {data['institutional_summary']}\n\n"

    comparison_context += "# Summary of the Institution to Compare:\n"
    comparison_context += f"[{len(top_universities) + 1}] {single_university}\n\n"

    comparison_prompt = (
        f"{comparison_context}"
        "Based solely on the summaries above, write a comparative analysis discussing how the institution in the second section "
        "compares to the group of institutions in the first section. "
        f"Focus on research strengths, uniqueness, or overlaps as relevant, so you answer the given input question: {input_prompt.strip()} "
        f"Use inline citations like [Harvard_1], [Cambridge_2], etc., to refer to them, similar as they have been used in the context. "
        "Do not use any outside information—only use what is stated above. "
        "After the comparison paragraph, include a list of references matching the inline citation numbers (e.g. Harvard_1, with institution names, authors, and DOIs if available.\n\n"
    )

    comparison_response = Settings.llm.complete(comparison_prompt).text
    return fix_output_citations(comparison_response)


def summarize_universities(structured_institution):
    for institution, data in structured_institution.items():
        top_docs = data["top_documents"]
        summaries = summarize(top_docs)
        data["summary"] = summaries
    return structured_institution
def summarize_university(docs):
    return summarize(docs)

def summarize(documents):
    """Per-doc LLM summarization. Per-call failures are isolated so one
    bad doc doesn't kill the entire row (Claude/OpenAI/Gemini native
    clients raise on content filters / empty completions / truncation;
    we'd lose the whole field_evolution row otherwise)."""
    summaries = []
    for i, node in enumerate(documents, start=1):
        content = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", str(node))
        meta = getattr(node, "metadata", {})
        title = meta.get("title", f"Document {i}")
        abstract = meta.get("abstract", "")
        authors = meta.get("authors", "Unknown authors")
        doi = meta.get("doi", meta.get("DOI", ""))
        url = meta.get("url") or doi or meta.get("openalex_url") or ""
        publication_year = meta.get("publication_year", "")
        prompt = (
            f"Paper Title: {title}\n"
            f"Authors: {authors}\n"
            f"Abstract:\n{abstract}\n\nFull Text:\n{content}\n\n"
            "Summarize the key contributions and findings of the paper above, using only the provided content."
        )
        try:
            summary_text = Settings.llm.complete(prompt).text.strip()
        except Exception as e:
            logger.warning("summarize: LLM call failed for doc %d (%s): %s",
                           i, title[:60], e)
            # Fall back to the abstract so the doc still contributes
            # something to overview(); if the abstract is empty, drop it.
            summary_text = (abstract or "").strip()
            if not summary_text:
                continue
        summaries.append({
            "index": i,
            "title": title,
            "authors": authors,
            "year": publication_year,
            "doi": doi,
            "url": url,
            "summary": summary_text,
        })
    return summaries


def extract_top_universities(
    docs,
    query,
    exclude=None,
    current_year=2025,
    top_k_institutions=3,
    top_docs_per_inst=15,
    country_code=None
):
    """
    Analyze retrieved documents and extract top contributing universities.

    Args:
        docs (list): List of retrieved document objects, each with .text and .metadata
        query (str): The comparison query
        exclude (list or str): Institutions to exclude
        current_year (int): The current year for filtering
        top_k_institutions (int): Number of top institutions to select
        top_docs_per_inst (int): Number of top documents per institution
        country_code (str): Optional country code to filter institutions

    Returns:
        dict: Structured output with top institutions, publication counts, and top documents
    """
    cutoff_year = current_year - 5

    # Step 1: Filter relevant documents (you can add is_relevant logic if needed)
    filtered_docs = docs

    # Step 2: Count publications per institution
    institution_counter = Counter()
    inst_to_docs = defaultdict(list)

    exclude_set = set([exclude]) if isinstance(exclude, str) else set(exclude or [])

    for doc in filtered_docs:
        institutions = doc.metadata.get("affiliations", [])
        countries = doc.metadata.get("countries", [])

        for inst, country in zip(institutions, countries):
            inst_s = inst.strip()
            country_s = (country or "").strip().upper()

            if inst_s in exclude_set:
                continue

            if country_code and country_s != country_code.upper():
                continue

            institution_counter[inst_s] += 1
            inst_to_docs[inst_s].append(doc)

    top_institutions = [inst for inst, _ in institution_counter.most_common(top_k_institutions)]

    # Step 3: Score and rank documents per institution
    embed_model = OpenAIEmbedding()
    query_embedding = embed_model.get_query_embedding(query)

    def score_document(doc, query_emb):
        doc_emb = embed_model.get_text_embedding(doc.text)
        return sum(a * b for a, b in zip(doc_emb, query_emb))  # dot product

    structured_output = {}
    for inst in top_institutions:
        inst_docs = inst_to_docs[inst]
        scored_docs = [(doc, score_document(doc, query_embedding)) for doc in inst_docs]
        top_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:top_docs_per_inst]

        structured_output[inst] = {
            "num_publications": len(inst_docs),
            "top_documents": [doc for doc, _ in top_docs],
        }

    return structured_output

def overview(summaries, prompt, top_k=None):
    if top_k is not None:
        summaries = summaries[:top_k]
    comparison_context = ""
    for s in summaries:
        comparison_context += f"[{s['index']}] {s['summary']}\n\n"

    references = "\n".join(
        (
            f"[{s['index']}] {s['title']} by {s['authors']}. {s.get('url') or s.get('doi')}"
            if (s.get('url') or s.get('doi'))
            else f"[{s['index']}] {s['title']} by {s['authors']} (no link available)"
        )
        for s in summaries
    )

    overview_prompt = (
        f"{comparison_context}"
        "Write a detailed overview using the related works above, to answer the following input question with as much detail as possible: "
        f"{prompt} "
        "##Additional instructions: "
        "- Do not use any additional knowledge, and only use the information provided above"
        "- Use inline citations like [1], [2], etc., to refer to them. "
        "- After the paragraph, list the full references you used with numbers and DOIs.\n\n"
        f"{references}"
    )

    # Generate comparison paragraph with citations. Wrap the LLM call so
    # a provider-side raise (content filter / truncation / empty completion
    # — common on Claude/OpenAI/Gemini for large synthesis prompts) doesn't
    # kill the whole row; fall back to a stub answer keyed off the
    # references so the row still saves with something inspectable.
    try:
        overview_response = Settings.llm.complete(overview_prompt).text
    except Exception as e:
        logger.warning("overview: synthesis LLM call failed: %s", e)
        overview_response = (
            f"[Synthesis failed: {type(e).__name__}. "
            f"{len(summaries)} retrieved papers below were not synthesised.]\n\n"
            f"{references}"
        )

    return fix_output_citations(overview_response)

# --- 6. Compare new paper with inline [n] references and numbered list ---
# new_title = new_papers[0].get("title", "")
# new_authors = new_papers[0].get("authors", "")
# new_abstract = new_papers[0].get("abstract", "")
def compare_papers(summaries, prompt, paper):
    # Compose context + citation references
    comparison_context = f"New Paper Title: {paper.get('title', '')}\nAuthors: {paper.get('authors', '')}\nAbstract: {paper.get('abstract', '')}\n\n"
    comparison_context = comparison_context + f"Full Text:\n{paper.get('fulltext', '')}\n\n"
    comparison_context += "# Summaries of Related Works:\n"
    for s in summaries:
        comparison_context += f"[{s['index']}] {s['summary']}\n\n"

    references = "\n".join(
        (
            f"[{s['index']}] {s['title']} by {s['authors']}. {s.get('url') or s.get('doi')}"
            if (s.get('url') or s.get('doi'))
            else f"[{s['index']}] {s['title']} by {s['authors']} (no link available)"
        )
        for s in summaries
    )

    comparison_prompt = (
        f"{comparison_context}"
        "Given the new paper above, in comparison with the related works, please answer the following question in as much detail as possible: "
        f"{prompt} "
        "Do not use any additional knowledge, and only use the information provided above"
        "Use inline citations like [1], [2], etc., to refer to them. "
        "After the paragraph, list the full references you used with numbers and DOIs.\n\n"
        f"{references}"
    )

    # Generate comparison paragraph with citations
    comparison_response = Settings.llm.complete(comparison_prompt).text


    return fix_output_citations(comparison_response)


def generate_topic_queries(info_dict):
    topic = info_dict.get("topic")
    if not topic:
        return info_dict  # No topic to generate queries for

    prompt = f"""
You are a research assistant tasked with retrieving academic literature.

Given the topic: "{topic}", generate a list of 4 sub-topics, that help to retrieve relevant literature to answer the question.

The topics should be suitable for guiding a literature search.

Return only the list of topics, without numbering or bullets.
"""

    output = Settings.llm.complete(prompt).text.strip()

    # Clean each line to remove numbers, dashes, etc.
    lines = output.splitlines()
    questions = []
    for line in lines:
        clean_line = re.sub(r"^\s*(?:\d+[\).]?|\-|\*)\s*", "", line)  # remove leading "1. ", "- ", etc.
        if clean_line:
            questions.append(clean_line.strip())

    updated_dict = deepcopy(info_dict)
    updated_dict["queries"] = questions
    return updated_dict

def extract_first_json_object(text: str) -> dict:
    text = text.strip()

    # Fast path: already valid JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find first JSON object by scanning for balanced braces
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in output: {text[:200]}")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                return json.loads(candidate)

    raise ValueError(f"Unbalanced braces; cannot extract JSON: {text[:200]}")

def generate_questions(
    prompt,
    paper=None,
    llm_model=None,
    chunk_size: int = 8192,
    temperature: float = 0.0,
    embedding_model: str = "text-embedding-ada-002",
):
    configure_llm(
        llm_model,
        chunk_size=chunk_size,
        embedding_model=embedding_model,
        temperature=temperature,
    )

    current_year = datetime.now().year
    if paper:
        input_prompt = prompt +"\n" + paper.get("title", "") + paper.get("abstract", "")
    else:
        input_prompt = prompt
    generation_prompt = (
        "You are a scientific researcher tasked with analyzing and comparing scientific literature. "
        "You will be given a question posed by fellow researchers. "
        "Your task is to determine whether additional scientific documents are needed to answer the question meaningfully.\n\n"

        "Respond ONLY with a object which has the following structure:\n"
        "{\n"
        '  "type": "<comparison|university|paper|overview|university_university|single_university|country_country|internal_kg|internal_kg_research_unit>",\n'
        '  "university_name": "<university name or null>",\n'
        '  "comparison_university_name": "<university name or null>",\n'
        '  "research_unit": "<research unit or null>",\n'
        '  "country": "<country code or null>",\n'
        '  "country_b": "<second country code or null>",\n'
        f'  "year_start": <integer or null>,  // e.g., {current_year - 5}\n'
        f'  "year_end": <integer or null>,    // e.g., {current_year}\n'
        '   "topic": <topic of interest>, // e.g. Topic Modeling, Reinforcement Learning, Language Generation\n'
        '   "include_projects": <boolean>  // true if query mentions projects, grants, or funding\n'
        "}\n\n"

        "**Explanation of the 'type' field:**\n"
        "- Use 'university' if the goal is to compare research contributions between institutions (e.g., 'What has the University of Aberdeen contributed compared to others?').\n"
        "- Use 'university_university' if the goal is to compare research contributions between exactly two institutions (e.g., 'What has the University of Aberdeen contributed compared to the University of Edinburgh?').\n"
        "- Use 'country_country' if the goal is to compare the body of research between two countries (e.g., 'Which approaches to NER dominated in US-affiliated vs CN-affiliated ACL papers between 2018 and 2022?'). No specific institution name is required.\n"
        "- Use 'internal_kg' if TU Wien is in the query, and no other university is given. The goal is to fall back on internal datasources. For this replace single_university if TU Wien, Technische Universität Wien is mentioned (e.g. What has TU Wien contributed to Information Retrieval in 2024?)"
        "- Use 'single_university' if the goal is to state the contributions of exactly one institution (e.g., 'What has the University of Aberdeen contributed to Information Retrieval over the last 5 years?').\n"
        "- Use 'tuwien' if the query specifically asks about TU Wien, TU Vienna, Vienna University of Technology, or Technische Universität Wien research, publications, or projects.\n"
        "- Use 'comparison' if the goal is to compare the content or impact of a specific paper to other scientific work (e.g., 'How does the paper compare to other relevant literature? + a title and an abstract'\n\n"
        "- Use 'paper' if the goal is to answer a question about the content of a specific paper.\n\n"
        "- Use 'overview' if the goal is to compare the content of scientific work, without any input paper"
        "- Use 'internal_kg' if TU Wien is in the query, and no other university is given. The goal is to fall back on internal datasources. Do this just for TU Wien"
        "- Use 'internal_kg_research_unit' if not a university is given, but a research unit as Data Science, Knowledge Graphs, Visual Computing"

        "**Explanation of the 'university_name' field:**\n"
        "- Use the name of the university or the research institution, if it is given in input question, otherwise return null.\n"
        "- For TU Wien queries, set this to 'TU Wien'.\n"

        "**Explanation of the 'comparison_university_name' field:**\n"
        "- Use this field if more than one name is given, insert the name of the university or the research institution, otherwise return null.\n"

        "**Explanation of the 'research_unit' field:**\n"
        "- Use this field if a research unit is given, otherwise return null.\n"

        "**Explanation of the 'country' field:**\n"
        "- Use this field if a university type is chosen, and the comparison should be limited to a specific country, as specified in the query.\n"
        "- For 'country_country' questions, insert the FIRST country code mentioned in the query.\n"
        "- Insert the country code for the country which is given in the query (e.g., US, GB, AT) otherwise return null.\n"
        "- Set this field to null, if the question does not specify a country.\n"
        "- For TU Wien queries, set this to 'AT'.\n"

        "**Explanation of the 'country_b' field:**\n"
        "- Only populate this field when 'type' is 'country_country'. Insert the SECOND country code mentioned in the query.\n"
        "- Otherwise return null.\n"

        "**Explanation of the 'year_start' and 'year_end' fields:**\n"
        "- Add here the years based on which the query should be filtered, if given in the input query, otherwise return null.\n"
        f"- If the question contains phrases as 'last 5 years', do calculations based on {current_year}"

        "**Explanation of the 'topic' field:**\n"
        "- Use this field to specify the topic under interest, e.g. Topic Modeling, Reinforcement Learning, Language Generation.\n"

        "**Explanation of the 'include_projects' field:**\n"
        "- Set to true if the query mentions projects, grants, funding, or research initiatives.\n"
        "- Set to false for queries focused only on publications.\n"

        f"Here is the input question:\n{input_prompt}"
    )
    output = Settings.llm.complete(generation_prompt).text.strip()
    output = output.replace("```json", "")
    output = output.replace("```", "")
    logger.debug("classifier raw output: %s", output)
    info = extract_first_json_object(output)
    logger.debug("parsed classification: %s", info)
    info_with_queries = generate_topic_queries(info)
    logger.debug("classification with queries: %s", info_with_queries)
    return info_with_queries


def non_summarize(documents):
    summaries = []
    for i, node in enumerate(documents, start=1):
        meta = getattr(node, "metadata", {})
        title = meta.get("title", f"Document {i}")
        abstract = meta.get("abstract", "")
        authors = meta.get("authors", "Unknown authors")
        doi = meta.get("doi", meta.get("DOI", ""))
        url = meta.get("url") or doi or meta.get("openalex_url") or ""
        publication_year = meta.get("publication_year", "")

        summaries.append({
            "index": i,
            "title": title,
            "authors": authors,
            "year": publication_year,
            "doi": doi,
            "url": url,
            "summary": abstract
        })
    return summaries



# --- 4. Retrieve top-K relevant documents ---
def retrieve(query, exclude=None, top_k=5, institution=None, year_start=None, year_end=None, country=None, index=None):
    # Resolve index at call time so callers can swap `static_index`
    # post-import (e.g., the SciCo eval sets it to the local ACL index
    # before invoking process_input).
    if index is None:
        index = static_index
    if index is None:
        raise RuntimeError(
            "retrieve() called with no index — set rag_utils.static_index "
            "or pass `index=` explicitly. For SciCo eval, the local ACL "
            "index is the right one (see backend.rag.local_acl_indexing.load_acl_index)."
        )
    """
    Retrieve documents for a query, optionally filtering by DOI or affiliation.

    Args:
        query (str): The search query.
        exclude (str, optional): DOI to exclude from results.
        top_k (int): Number of top results to retrieve.
        affiliation (str, optional): Affiliation substring to filter (e.g., "TU Wien").

    Returns:
        List of deduplicated retrieved nodes.
    """
    filters = []

    if exclude:
        filters.append(MetadataFilter(key="doi", value=exclude, operator="!="))

    if institution:
        filters.append(MetadataFilter(key="affiliations_string", value=institution, operator="contains"))

    if year_start:
        filters.append(MetadataFilter(key="publication_year", value=year_start, operator=">="))

    if year_end:
        filters.append(MetadataFilter(key="publication_year", value=year_end, operator="<="))

    if country:
        filters.append(MetadataFilter(key="countries_string", value=country, operator="contains"))

    metadata_filters = MetadataFilters(filters=filters) if filters else None

    # Get retriever with optional metadata filters
    retriever = index.as_retriever(similarity_top_k=top_k, filters=metadata_filters)

    # LLM-enhanced query engine (used internally, but not needed for plain retrieval here)
    #citation_engine = CitationQueryEngine(retriever=retriever, llm=Settings.llm)

    retrieved_nodes = retriever.retrieve(query)

    #for i, item in enumerate(retrieved_nodes):
    #    print("retrieved_node:", i, " title: ", item.metadata.get('title'))

    # Deduplicate by DOI or title
    unique = {}
    for node in retrieved_nodes:
        meta = getattr(node, "metadata", {})
        key = meta.get("doi") or meta.get("title")
        if key and key not in unique:
            unique[key] = node

    return list(unique.values())

def compare_two_universities(university, compare_university, prompt):
    """
    Compare the research contributions of two specific universities using only their institutional summaries.

    Args:
        university (dict): Dictionary with keys 'name' and 'institutional_summary'.
        compare_university (dict): Dictionary with keys 'name' and 'institutional_summary'.
        prompt (str): Optional guiding prompt to focus the comparison.

    Returns:
        str: A formatted and citation-corrected comparative analysis.
    """
    context = "# Summary of the First Institution:\n"
    context += f"{university}\n\n"

    context += "# Summary of the Second Institution:\n"
    context += f"{compare_university}\n\n"

    comparison_prompt = (
        f"{context}"
        "Based solely on the summaries above, write a comparative analysis discussing how the second institution "
        "compares to the first one. Focus on research strengths, uniqueness, or overlaps as relevant. "
        "Especially try to answer the following information need given by the user:"
        f"{prompt.strip()} "
        f"Use inline citations like [Harvard_1], [Cambridge_2], etc., to refer to them, similar as they have been used in the context. "
        "Do not use any outside information—only use what is stated above. "
        "After the comparison paragraph, include a list of references matching the inline citation numbers (e.g. Harvard_1, with institution names, authors, and DOIs if available.\n\n"
    )
    logger.debug("two-university comparison using llm: %s", Settings.llm)
    comparison_response = Settings.llm.complete(comparison_prompt).text

    return fix_output_citations(comparison_response)

# --- Citation enforcement -------------------------------------------------
# DOIs and author-year citations slip into the body despite the prompt; we
# detect them programmatically and either retry with the LLM or strip them.

_DOI_URL_RE = re.compile(r"https?://(?:dx\.)?doi\.org/\S+", re.IGNORECASE)
_BARE_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
# Matches "(Author, 2023)", "(Smith et al., 2023)", "(Smith and Jones, 2023)"
_AUTHOR_YEAR_RE = re.compile(
    r"\(\s*[A-Z][A-Za-zÀ-ſ'\-]+"
    r"(?:\s+et\s+al\.?|\s+and\s+[A-Z][A-Za-zÀ-ſ'\-]+)?"
    r"\s*,?\s+(?:19|20)\d{2}\s*\)"
)
_REFERENCES_HEADER_RE = re.compile(
    r"^\s*#{0,6}\s*references\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def _split_body_refs(text: str):
    """Cut at the last 'References' header. Returns (body, refs)."""
    matches = list(_REFERENCES_HEADER_RE.finditer(text))
    if not matches:
        return text, ""
    cut = matches[-1].start()
    return text[:cut], text[cut:]


def _citation_violations(body: str):
    issues = []
    for m in _DOI_URL_RE.finditer(body):
        issues.append(("doi_url_in_body", m.group(0)))
    for m in _BARE_DOI_RE.finditer(body):
        issues.append(("bare_doi_in_body", m.group(0)))
    for m in _AUTHOR_YEAR_RE.finditer(body):
        issues.append(("author_year_in_body", m.group(0)))
    return issues


def _strip_dois_in_body(text: str) -> str:
    """Hard final-pass: remove any DOI URLs/bare DOIs left in the body."""
    body, refs = _split_body_refs(text)
    if not refs:
        # No references section to lean on — leaving DOIs in place would be
        # less damaging than stripping them and leaving floating citations.
        return text
    body = _DOI_URL_RE.sub("", body)
    body = _BARE_DOI_RE.sub("", body)
    # Clean up artefacts left behind. Crucially, do NOT collapse newlines —
    # only horizontal whitespace — or markdown paragraph breaks die.
    body = re.sub(r"\(\s*\)", "", body)
    body = re.sub(r"[ \t]{2,}", " ", body)
    body = re.sub(r"[ \t]+\n", "\n", body)
    return body + refs


def fix_output_citations(input):
    """Clean up citations to match strict format rules.

    Rules enforced:
      - All inline citations in the body use [N] markers — never raw DOIs,
        never (Author, year) style.
      - DOI URLs appear ONLY in the references list at the end.
      - Continuous numbering starting at [1]; unused references removed;
        duplicates deduplicated.
      - References list in APA style with one entry per line.

    First pass: LLM cleanup with explicit good/bad examples.
    If regex still finds violations: one LLM retry with the concrete snippets.
    Final pass: hard regex strip of any DOI URLs left in the body.
    """
    fixing_prompt = (
        f"{input}\n\n"
        "Reformat the answer above so it follows STRICT rules. The output is "
        "rendered as markdown by the frontend so blank lines between blocks "
        "matter.\n\n"
        "CITATIONS:\n"
        "- All inline citations in the body use [N] markers (e.g. [1], [2]).\n"
        "- DOI/OpenAlex URLs appear ONLY in the references list — NEVER in the body.\n"
        "- No author-year style anywhere in the body.\n"
        "- Numbering is continuous starting at [1]; remove unused references; "
          "deduplicate identical references and use the single number.\n"
        "- If a reference belongs to multiple institutions, rewrite the sentence "
          "to make the collaboration explicit.\n\n"
        "REFERENCES list format (this is the template — follow exactly):\n"
        "### References\n"
        "\n"
        "[1] First, A., Second, B., & Third, C. (2020). Title of the paper. "
          "*Journal*, 12(3), 45-67. https://doi.org/10.x/y\n"
        "\n"
        "[2] Fourth, D., Fifth, E., & Sixth, F. (2022). Another title. "
          "*Venue*, 7(1), 1-15. https://doi.org/10.x/y\n"
        "\n"
        "Note the BLANK line between every entry — markdown needs that to "
        "render each reference on its own line. Use APA style, replace authors "
        "after the third with 'et al.' EVERY entry must end with a working URL "
        "(use the OpenAlex Work URL e.g. https://openalex.org/W4400024729 when "
        "no DOI is available). Do not include references without a link.\n\n"
        "BODY structure:\n"
        "- Open with a SINGLE bolded one-line takeaway (a `**...**` line) "
          "before any heading. Example:\n"
        "  '**TU Wien's Data Science unit produced 47 IR papers in 2024, with "
          "growing focus on retrieval-augmented generation and human-AI "
          "alignment.**'\n"
        "- Then level-3 headings: ### Overview / ### Key contributions / "
          "### Comparative analysis / ### Trends / ### References (use the "
          "headings that fit; not every section is mandatory).\n"
        "- Within a section, be CONSISTENT: either all bullet items "
          "(each starting with `*` or `-`) OR all prose paragraphs. Do NOT "
          "interleave a bullet and an unbulleted paragraph at the same level.\n"
        "- Each fact, statistic, or paper mention appears ONCE. Don't restate "
          "the same number or finding across sections.\n\n"
        "GOOD inline:  'BERT [3] introduced bidirectional pre-training.'\n"
        "BAD inline:   'BERT (Devlin et al., 2019) introduced...'\n"
        "BAD inline:   'BERT (https://doi.org/10.18653/v1/N19-1423) introduced...'\n"
        "BAD inline:   'BERT [Devlin2019] introduced...'\n\n"
        "Do not change the substance of the answer beyond renumbering, "
        "moving DOIs into the references, rewording author-year inline forms, "
        "and applying the markdown structure above. The whole output must be "
        "in English. Reply with ONLY the cleaned text — no preamble, no "
        "explanation."
    )
    fixed = Settings.llm.complete(fixing_prompt).text

    body, _refs = _split_body_refs(fixed)
    issues = _citation_violations(body)
    if issues:
        # Show the LLM concrete snippets to repair (cap at 8 to keep the
        # prompt short).
        snippet_list = "\n".join(
            f"- {kind}: {snippet!r}" for kind, snippet in issues[:8]
        )
        retry_prompt = (
            f"{fixed}\n\n"
            "The body of this answer still contains forbidden citation forms:\n"
            f"{snippet_list}\n\n"
            "Replace each one with the [N] inline marker that points at the "
            "matching entry in the references list. Add the entry to the "
            "references list if it is missing there. Do not change anything else. "
            "Reply with ONLY the corrected text."
        )
        logger.debug("citation enforcer retrying: %d violations", len(issues))
        fixed = Settings.llm.complete(retry_prompt).text

    # Hard final pass — anything still raw in the body gets stripped.
    fixed = _strip_dois_in_body(fixed)
    return fixed


def use_abstracts_universities(structured_institution):
    for institution, data in structured_institution.items():
        top_docs = data["top_documents"]
        summaries = use_abstracts_as_summaries(top_docs)
        data["summary"] = summaries
    return structured_institution

def use_abstracts_university(docs):
    return use_abstracts_as_summaries(docs)

def use_abstracts_as_summaries(documents):
    summaries = []
    for i, node in enumerate(documents, start=1):
        meta = getattr(node, "metadata", {})
        title = meta.get("title", f"Document {i}")
        abstract = meta.get("abstract", "")
        authors = meta.get("authors", "Unknown authors")
        doi = meta.get("doi", meta.get("DOI", ""))
        url = meta.get("url") or doi or meta.get("openalex_url") or ""
        publication_year = meta.get("publication_year", "")

        summaries.append({
            "index": i,
            "title": title,
            "authors": authors,
            "year": publication_year,
            "doi": doi,
            "url": url,
            "summary": abstract  # Use abstract directly
        })
    return summaries

#static_index = build_index("data/works.json")
#retriever = static_index.as_retriever(similarity_top_k=5)
#citation_engine = CitationQueryEngine(retriever=retriever, llm=Settings.llm)