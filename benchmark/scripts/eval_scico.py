"""Offline-data evaluation of the SciCo / local ACL index.

`Offline-data`, not `offline`: the LLM APIs are used (synthesis + critic +
LLM-as-judge), but the only retrieval source is the local SciCo /
local-ACL VectorStoreIndex. OpenAlex / CORE / TU Wien Repositum / Neo4j
are all locked out.

Modes
-----
- `--mode retrieval` (cheap, deterministic): call search_local_acl_papers
  directly with each row's query and grade the top-k retrieval. No LLM.

- `--mode agent` (expensive, the real test): run the full agent
  (process_input_agent_sync) with ONLY the SciCo tool exposed and a
  system prompt that scopes it to the corpus. The agent's evidence
  buffer drives the same retrieval-style graders; the agent's prose
  answer is fed to the NLG metrics (ROUGE-L, BERTScore) and — if
  --judge is set — an LLM-as-judge.

Question types and graders
--------------------------
- paper_retrieval     : recall@k, precision@k, F1@k, MRR
- institution_list    : recall, precision, F1 (case-insensitive substring)
                        over the affiliation union of retrieved papers
- country_precision   : precision = retrieved papers w/ expected country
                        / total retrieved (passes if >= min_precision)
- free_form           : retrieval recall + precision + MRR over
                        gold_paper_ids; if gold_summary present:
                        ROUGE-L, BERTScore, and (when --judge MODEL set in
                        agent mode) LLM-as-judge on coverage / faithfulness
                        / specificity.

Always reported in agent mode
-----------------------------
- Phantom citation rate    — references that aren't in the agent's evidence
- Affiliation correctness  — F1 of indexed affiliations vs ground_truth.csv
- Wall-clock seconds, prompt/completion/total tokens, est. USD cost

Examples
--------
    # cheap retrieval-only (no LLM)
    python scripts/eval_scico.py --mode retrieval

    # full agent run, gpt-4o-mini, full metric suite
    set -a && . ./.env && set +a
    python scripts/eval_scico.py --mode agent --model gpt-4o-mini

    # with LLM-as-judge for free_form
    python scripts/eval_scico.py --mode agent --model claude-sonnet-4-6 \\
        --judge gpt-4o-mini

    # disable BERTScore (saves ~5 min on first row downloading roberta-large)
    python scripts/eval_scico.py --mode agent --no-bertscore
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.eval.classifier import grade_classifier_row  # noqa: E402
from backend.eval.metrics import (  # noqa: E402
    affiliation_correctness,
    bertscore_f1,
    institution_list_metrics,
    inter_judge_agreement,
    load_ground_truth_affiliations,
    phantom_citation_rate,
    precision_recall_mrr,
    rouge_l_f1,
    rouge_l_vs_evidence,
)
from backend.eval.gold import (  # noqa: E402
    compute_gold_paper_ids,
    load_affiliations_corpus,
    summarize_gold_sizes,
)
from backend.rag.agent_tools import (  # noqa: E402
    build_acl_only_tools,
    collected_evidence,
    reset_evidence_buffer,
    search_local_acl_papers,
)


# ---------------------------------------------------------------------------
# SciCo-scoped agent prompt
# ---------------------------------------------------------------------------

SCICO_SYSTEM_PROMPT = """\
You are a research-comparison assistant constrained to one offline corpus:
the local SciCo index over the ACL Anthology, with ROR-normalized
affiliations. You have TWO retrieval tools — pick the right one per query:

- search_local_acl_papers     — fast retrieval over paper ABSTRACTS.
  Use for institution / topic / overview / "what has X published" questions.
- search_scico_fulltext_papers — retrieval over paper FULL TEXTS.
  Use for paper-vs-paper methodology comparisons, evaluation-protocol
  questions, and anything that needs detail beyond the abstract.

Strict rules:
- You MUST call at least one retrieval tool before answering.
- You may NOT use your training data to invent papers, authors, or DOIs.
  If the corpus has nothing on the question, say so plainly.
- Every cited paper must have come back from a tool call this turn.
- Answer with the markdown structure: bolded one-line takeaway, then
  level-3 headings (### Overview / ### Findings / ### References).
- Inline citations use [N] markers; the references list at the end uses
  APA style with one entry per line and a blank line between entries.
  Every entry ends with the paper's URL (DOI when present, otherwise the
  ACL Anthology URL the tool returned).
- Each fact, statistic, or paper mention appears once.
"""


# Closed-book variant: same task framing, no mention of tools so the
# LLM doesn't try to cosplay tool calls in the output. The model is
# expected to answer purely from its parametric knowledge of ACL/NLP
# literature. Citations are still requested so the output schema stays
# comparable to the agentic / pipeline modes — phantom_rate then
# measures how much the model made up.
SCICO_CLOSED_BOOK_PROMPT = """\
You are a research-comparison assistant. The question concerns the ACL
Anthology corpus of computational-linguistics / NLP papers. Answer
purely from your own knowledge of the field — you have no retrieval
tool available in this turn.

Strict rules:
- Do not invent paper IDs, URLs, or DOIs. If you don't know a specific
  paper that supports a claim, leave the citation out rather than fake it.
- If you genuinely don't know enough to answer, say so plainly instead
  of fabricating evidence.
- Answer with the markdown structure: bolded one-line takeaway, then
  level-3 headings (### Overview / ### Findings / ### References).
- Inline citations use [N] markers; the references list at the end uses
  APA style with one entry per line and a blank line between entries,
  ending with the paper's URL. Only include papers you are confident
  about.
- Each fact, statistic, or paper mention appears once.
"""


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_testset(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
    return rows


def _retrieved_ids(records: List[Dict[str, Any]]) -> List[str]:
    out = []
    for r in records:
        pid = r.get("paper_id") or r.get("title")
        if pid:
            out.append(pid)
    return out


def _affiliation_union(records: List[Dict[str, Any]]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for r in records:
        for aff in r.get("affiliations") or []:
            if not aff:
                continue
            key = aff.strip()
            if key and key.lower() not in seen:
                seen.add(key.lower())
                out.append(key)
    return out


# ---------------------------------------------------------------------------
# Per-row execution
# ---------------------------------------------------------------------------

def _retrieve_only(row: Dict[str, Any], k_default: int) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    reset_evidence_buffer()
    records = search_local_acl_papers(
        query=row.get("query", ""),
        year_start=row.get("year_start"),
        year_end=row.get("year_end"),
        institution=row.get("institution"),
        samples=int(row.get("k", k_default)),
    )
    if records and isinstance(records, list) and records[0].get("error"):
        return [], records[0]["error"]
    return records, None


def _build_agent_prompt(row: Dict[str, Any]) -> str:
    """For most rows the query is the prompt. For paper_vs_field rows the
    target paper's body is essential context — the agent can't compare a
    paper to the field without seeing what's in the paper. Prepend a
    concise card with title + abstract + fulltext excerpt."""
    query = row.get("query", "")
    paper = row.get("paper") or {}
    if not paper or row.get("category") != "paper_vs_field":
        return query
    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    fulltext = (paper.get("fulltext") or "").strip()
    # Cap the fulltext fed into the prompt — 30KB is enough context for any
    # frontier model while keeping prompt cost manageable.
    if len(fulltext) > 30000:
        fulltext = fulltext[:30000] + " ... [truncated]"
    parts = [
        "TARGET PAPER (to compare against the field):",
        f"  Title: {title}" if title else "",
        f"  Year: {paper.get('year')}" if paper.get("year") else "",
        "",
        f"  Abstract:\n{abstract}" if abstract else "",
        "",
        f"  Body:\n{fulltext}" if fulltext else "",
        "",
        "QUESTION:",
        query,
    ]
    return "\n".join(p for p in parts if p is not None).strip()


def _run_agent(
    row: Dict[str, Any], model: str
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any], Optional[str]]:
    """Returns (evidence_records, summary, telemetry, error)."""
    from backend.eval.runs import run_agent_with_telemetry

    prompt = _build_agent_prompt(row)
    summary, tele = run_agent_with_telemetry(
        prompt=prompt,
        model=model,
        tools=build_acl_only_tools(),
        system_prompt=SCICO_SYSTEM_PROMPT,
    )
    return collected_evidence(), summary, tele, tele.get("error")


_PIPELINE_CLASSIFIER_PROMPT = """\
You are routing a research question to one of five processing pipelines
over the ACL Anthology. Decide which pipeline category fits best AND
extract the relevant entities. Output ONE JSON object, no preamble, no
markdown fence.

Categories:

- overview_university            : "what does <university> publish on
                                    <topic>"; one university, one topic.
- u2u_comparison                 : "compare <uni1> and <uni2> on <topic>";
                                    two universities, one topic.
- cross_country_method           : "compare methods between <countryA>
                                    and <countryB> for <topic>"; two
                                    countries (ISO codes), one topic.
- university_country_comparison  : "<university> in <country> on <topic>";
                                    one university × one country.
- field_evolution                : "how has <topic> evolved over time";
                                    one topic, year window.

Required output JSON schema:
{{
  "category": "<one of the five categories above>",
  "topic": "<short topic phrase>",
  "uni1": "<institution display name or null>",
  "uni2": "<second institution name or null>",
  "country": "<ISO 3166-1 alpha-2 code or null>",
  "country_b": "<second ISO code for cross_country_method, else null>",
  "year_start": <integer or null>,
  "year_end": <integer or null>,
  "queries": ["<query1>", "<query2>", "<query3>"]
}}

Rules:
- `queries` are 1-2 short noun-phrase retrieval queries (NOT questions)
  suitable for vector-index embedding search.
- Use ISO 3166-1 alpha-2 codes for countries (e.g., "US", "CN", "DE").
- If a field doesn't apply to the chosen category, set it to null.
- Output ONLY the JSON object. No commentary.

Question:
{question}
"""


def _classify_and_extract_scico(question: str, model: str) -> Dict[str, Any]:
    """LLM-driven classifier + entity extractor for the SciCo pipeline.

    Returns a dict with keys: category, topic, uni1, uni2, country,
    country_b, year_start, year_end, queries. Falls back to a safe
    default (field_evolution + raw question as the only query) on any
    parse failure — the caller should still produce a valid answer,
    just one routed through the generic overview handler.
    """
    from llama_index.core import Settings
    from llama_index.core.llms import ChatMessage
    from backend.rag.rag_utils import configure_llm, extract_first_json_object
    fallback = {
        "category": "field_evolution",
        "topic": question,
        "uni1": None, "uni2": None, "country": None, "country_b": None,
        "year_start": None, "year_end": None,
        "queries": [question],
    }
    try:
        configure_llm(model)
        msgs = [ChatMessage(
            role="user",
            content=_PIPELINE_CLASSIFIER_PROMPT.format(question=question),
        )]
        resp = Settings.llm.chat(msgs)
        raw = str(resp.message.content) if resp and resp.message else ""
        parsed = extract_first_json_object(raw) or {}
    except Exception:
        return fallback
    out = dict(fallback)
    for k in ("category", "topic", "uni1", "uni2", "country", "country_b",
              "year_start", "year_end", "queries"):
        if parsed.get(k) is not None:
            out[k] = parsed[k]
    valid_cats = {
        "overview_university", "u2u_comparison", "cross_country_method",
        "university_country_comparison", "field_evolution",
    }
    if out["category"] not in valid_cats:
        out["category"] = "field_evolution"
    if not isinstance(out["queries"], list) or not out["queries"]:
        out["queries"] = [question]
    return out


_PIPELINE_QUERY_REWRITE_PROMPT = """\
You are preparing retrieval queries for a vector index over the ACL
Anthology of computational-linguistics papers. The user asked the
question below. Produce 2-3 focused keyword queries (one per line, no
numbering, no extra punctuation) that, together, will cover the
question's intent and retrieve relevant papers. Each query should be a
short noun phrase or topic string suitable for embedding search — not
a full sentence, not a question. Output ONLY the queries, one per line.

Question:
{question}
"""


def _generate_retrieval_queries(question: str, model: str) -> List[str]:
    """LLM call: rewrite a question into 2-3 retrieval queries.
    Falls back to the raw question if the LLM call fails or returns
    nothing usable. Returns a non-empty list of stripped strings."""
    from llama_index.core import Settings
    from llama_index.core.llms import ChatMessage
    from backend.rag.rag_utils import configure_llm
    try:
        configure_llm(model)
        msgs = [ChatMessage(
            role="user",
            content=_PIPELINE_QUERY_REWRITE_PROMPT.format(question=question),
        )]
        resp = Settings.llm.chat(msgs)
        raw = str(resp.message.content) if resp and resp.message else ""
        queries = [ln.strip(" -•*\t") for ln in raw.splitlines() if ln.strip()]
        # Drop quote chars / numeric prefixes the model sometimes emits
        cleaned = []
        for q in queries:
            q = q.lstrip("0123456789. )-").strip().strip('"').strip("'")
            if q:
                cleaned.append(q)
        if cleaned:
            return cleaned[:3]
    except Exception:
        pass
    return [question]


def _consume_handler_to_summary(handler, data, prompt, model):
    """Run a streaming pipeline handler, collect SSE chunks, return
    (summary_text, retrieved_papers). The handler's final SSE payload
    is a JSON dict with keys `summary` (the text) and `retrieved_papers`
    (flat list of per-paper records). Profiles is consumed if present
    but treated as supplementary."""
    summary = ""
    retrieved_papers: List[Dict[str, Any]] = []
    for chunk in handler(data, prompt, online=False, model=model, stream_fn=None):
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
                if isinstance(parsed.get("retrieved_papers"), list):
                    retrieved_papers = parsed["retrieved_papers"]
                # Fall back: if the handler emitted institutional
                # profiles with per-profile papers (online mode), flatten
                # those into the same record shape.
                if not retrieved_papers:
                    for prof in (parsed.get("profiles") or []):
                        for p in (prof.get("papers") or []):
                            retrieved_papers.append({
                                "title": p.get("title"),
                                "authors": p.get("authors") or [],
                                "publication_year": p.get("publication_year") or p.get("year"),
                                "DOI": p.get("doi") or p.get("DOI"),
                                "url": p.get("url") or p.get("doi"),
                                "affiliations": p.get("affiliations") or [],
                                "paper_id": p.get("paper_id") or p.get("doi") or p.get("title"),
                            })
    return summary, retrieved_papers


def _run_pipeline(
    row: Dict[str, Any], model: str
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any], Optional[str]]:
    """Non-agentic RAG pipeline, dispatched per SciCo question category.

    Flow:
      1) LLM rewrites the question into 2-3 retrieval queries (1 call).
      2) Dispatch to a category-specific handler in pipeline.py:
         - overview_university              → handle_university
         - u2u_comparison                   → handle_university_university
         - cross_country_method             → handle_cross_country_method
         - university_country_comparison    → handle_university_country_comparison
         - field_evolution                  → handle_overview
      3) Handler retrieves from the local ACL index + does its
         category-specific summarisation/comparison (2-3 LLM calls).
      4) Returns (records, summary, telemetry, error).

    Dispatch uses the row's `classifier_gold.question_category` —
    no LLM classifier is invoked, so misclassification noise is zero.
    Token usage across the full pipeline (rewrite + handler) is
    captured via a TokenCountingHandler.
    """
    import time
    from llama_index.core import Settings
    from llama_index.core.callbacks import (
        CallbackManager, CBEventType, EventPayload, TokenCountingHandler,
    )
    # BaseCallbackHandler moved between minor versions: try the modern
    # location first, fall back to the legacy submodule, then a final
    # fallback to the top-level callbacks package for very old versions.
    try:
        from llama_index.core.callbacks.base_handler import BaseCallbackHandler
    except ImportError:
        try:
            from llama_index.core.callbacks.base import BaseCallbackHandler
        except ImportError:
            from llama_index.core.callbacks import BaseCallbackHandler  # type: ignore
    from backend.rag.rag_utils import configure_llm
    import backend.rag.rag_utils as _rag_utils
    from backend.rag.local_acl_indexing import (
        DEFAULT_ACL_PERSIST_DIR, load_acl_index,
    )
    from backend.rag.pipeline import (
        handle_university, handle_university_university, handle_overview,
        handle_cross_country_method, handle_university_country_comparison,
    )
    from backend.eval.metrics import estimate_cost_usd  # type: ignore

    if _rag_utils.static_index is None:
        import time as _t_idx
        print(f"# loading ACL index from {DEFAULT_ACL_PERSIST_DIR} "
              f"(silent step — ~5-10s from /dev/shm, ~9min from NFS)",
              flush=True)
        _t_idx0 = _t_idx.perf_counter()
        _rag_utils.static_index = load_acl_index(DEFAULT_ACL_PERSIST_DIR)
        print(f"# ACL index loaded in {_t_idx.perf_counter() - _t_idx0:.1f}s",
              flush=True)

    # Captures `reasoning_content` from every LLM call this row issues
    # (classifier + handler calls). Qwen-3 reasoning models populate it
    # when enable_thinking is True; non-reasoning models leave it empty
    # and traces stays []. Stored at telemetry["reasoning_traces"] so
    # downstream judging keeps working unchanged.
    class _ReasoningCaptureHandler(BaseCallbackHandler):
        def __init__(self):
            super().__init__(event_starts_to_ignore=[], event_ends_to_ignore=[])
            self.traces: List[str] = []
        def start_trace(self, trace_id=None): pass
        def end_trace(self, trace_id=None, trace_map=None): pass
        def on_event_start(self, event_type, payload=None, event_id="",
                           parent_id="", **kwargs):
            return event_id
        def on_event_end(self, event_type, payload=None, event_id="", **kwargs):
            if event_type != CBEventType.LLM or not payload:
                return
            resp = payload.get(EventPayload.RESPONSE)
            if resp is None:
                return
            rc = None
            msg = getattr(resp, "message", None)
            if msg is not None:
                ak = getattr(msg, "additional_kwargs", None) or {}
                rc = ak.get("reasoning_content")
            if not rc:
                raw = getattr(resp, "raw", None)
                try:
                    if isinstance(raw, dict):
                        rc = ((raw.get("choices") or [{}])[0]
                              .get("message", {}).get("reasoning_content"))
                    elif raw is not None:
                        choices = getattr(raw, "choices", None) or []
                        if choices:
                            m = getattr(choices[0], "message", None)
                            rc = getattr(m, "reasoning_content", None)
                except Exception:
                    pass
            if rc:
                self.traces.append(str(rc))

    counter = TokenCountingHandler()
    reasoning_capture = _ReasoningCaptureHandler()
    saved_cb_manager = Settings.callback_manager
    Settings.callback_manager = CallbackManager([counter, reasoning_capture])

    t0 = time.perf_counter()
    err: Optional[str] = None
    summary = ""
    retrieved_records: List[Dict[str, Any]] = []
    classified: Dict[str, Any] = {}
    question = row.get("query", "")
    gold_cg = row.get("classifier_gold") or {}
    gold_category = gold_cg.get("question_category")

    try:
        configure_llm(model)
        # Step 1: LLM classifier — routes the question to one of the
        # five SciCo pipeline categories AND extracts the entities
        # (uni1/uni2/country/year window) + 2-3 retrieval queries. One
        # LLM call replaces both classification and query rewriting.
        classified = _classify_and_extract_scico(question, model)
        category = classified["category"]
        queries = classified.get("queries") or [question]

        # Step 2: build a `data` dict the handler expects + dispatch
        base = {
            "queries": queries,
            "topic": classified.get("topic"),
            "year_start": classified.get("year_start"),
            "year_end": classified.get("year_end"),
        }

        if category == "overview_university":
            base["university_name"] = classified.get("uni1")
            base["country"] = classified.get("country") or ""
            handler = handle_university
        elif category == "u2u_comparison":
            base["university_name"] = classified.get("uni1")
            base["comparison_university_name"] = classified.get("uni2")
            handler = handle_university_university
        elif category == "cross_country_method":
            base["country_a"] = (classified.get("country") or "").upper()
            base["country_b"] = (classified.get("country_b") or "").upper()
            handler = handle_cross_country_method
        elif category == "university_country_comparison":
            base["university_name"] = classified.get("uni1")
            base["country"] = (classified.get("country") or "").upper()
            handler = handle_university_country_comparison
        elif category == "field_evolution":
            handler = handle_overview
        else:
            raise ValueError(f"no pipeline handler for category {category!r}")

        # Step 3: run the handler, collect summary + retrieved papers
        summary, retrieved_records = _consume_handler_to_summary(handler, base, question, model)
        summary = summary or ""
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    finally:
        wall = time.perf_counter() - t0
        Settings.callback_manager = saved_cb_manager

    records: List[Dict[str, Any]] = retrieved_records or []

    prompt_tokens = int(counter.prompt_llm_token_count)
    completion_tokens = int(counter.completion_llm_token_count)
    try:
        cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
    except Exception:
        cost = None
    tele = {
        "wall_seconds": wall,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost,
        "error": err,
        # Classifier output — keep both predicted and gold for offline
        # classifier-accuracy analysis later.
        "predicted_category": classified.get("category"),
        "predicted_queries": classified.get("queries"),
        "predicted_uni1": classified.get("uni1"),
        "predicted_uni2": classified.get("uni2"),
        "predicted_country": classified.get("country"),
        "predicted_country_b": classified.get("country_b"),
        "predicted_year_start": classified.get("year_start"),
        "predicted_year_end": classified.get("year_end"),
        "gold_category": gold_category,
        # Reasoning trace from every LLM call this row issued. Empty
        # list for non-reasoning models (gemini/claude/gpt/gemma).
        # Populated for qwen-3.5-397b / qwen-3.6-35b when enable_thinking
        # is True in configure_llm (rag_utils.py).
        "reasoning_traces": reasoning_capture.traces,
    }
    return records, summary, tele, err


def _run_closed_book(
    row: Dict[str, Any], model: str
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any], Optional[str]]:
    """Closed-book baseline: the LLM answers the question with NO
    retrieval tool at all. Useful as a "parametric knowledge" control
    against the retrieval-augmented agent run.

    Returns the same 4-tuple as _run_agent so the downstream code is
    identical. `records` is empty (since nothing was retrieved), which
    makes the retrieval metrics (P@k, R@k, MRR) all zero by definition.
    The generation/judge metrics still apply — that's the point.
    """
    import time
    from llama_index.core import Settings
    from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
    from llama_index.core.llms import ChatMessage
    from backend.rag.rag_utils import configure_llm
    from backend.eval.metrics import estimate_cost_usd  # type: ignore

    # Attach a token counter for the duration of this row (same pattern
    # as run_agent_with_telemetry). Without it, every closed-book row
    # records 0/0 tokens and we lose cost-per-question info.
    counter = TokenCountingHandler()
    saved_cb_manager = Settings.callback_manager
    Settings.callback_manager = CallbackManager([counter])

    t0 = time.perf_counter()
    err: Optional[str] = None
    summary = ""
    reasoning_trace: str = ""
    try:
        configure_llm(model)
        prompt = _build_agent_prompt(row)
        # Closed-book uses SCICO_CLOSED_BOOK_PROMPT — no mention of
        # retrieval tools, so the LLM doesn't cosplay tool calls in its
        # output. Still asks for inline [N] citations + reference list so
        # phantom_rate is comparable to the agentic / pipeline modes.
        chat_msgs = [
            ChatMessage(role="system", content=SCICO_CLOSED_BOOK_PROMPT),
            ChatMessage(role="user", content=prompt),
        ]
        resp = Settings.llm.chat(chat_msgs)
        summary = str(resp.message.content) if resp and resp.message else ""
        # Pull reasoning_content (Qwen-3 thinking models). Empty for
        # non-reasoning models. Stored verbatim under telemetry.
        if resp and resp.message:
            ak = getattr(resp.message, "additional_kwargs", None) or {}
            reasoning_trace = str(ak.get("reasoning_content") or "")
            if not reasoning_trace:
                raw = getattr(resp, "raw", None)
                try:
                    if isinstance(raw, dict):
                        reasoning_trace = str(
                            (raw.get("choices") or [{}])[0]
                            .get("message", {}).get("reasoning_content") or "")
                    elif raw is not None:
                        choices = getattr(raw, "choices", None) or []
                        if choices:
                            m = getattr(choices[0], "message", None)
                            reasoning_trace = str(getattr(m, "reasoning_content", "") or "")
                except Exception:
                    pass
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    finally:
        wall = time.perf_counter() - t0
        Settings.callback_manager = saved_cb_manager

    prompt_tokens = int(counter.prompt_llm_token_count)
    completion_tokens = int(counter.completion_llm_token_count)
    try:
        cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
    except Exception:
        cost = None

    tele = {
        "wall_seconds": wall,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cost_usd": cost,
        "error": err,
        # Single-call closed-book mode: keep this as a list-of-one for
        # schema-symmetry with _run_pipeline's reasoning_traces.
        "reasoning_traces": [reasoning_trace] if reasoning_trace else [],
    }
    return [], summary, tele, err


# ---------------------------------------------------------------------------
# Country precision (still its own grader since it's threshold-based)
# ---------------------------------------------------------------------------

def _grade_country_precision(
    retrieved_records: List[Dict[str, Any]],
    expected_country: str,
    min_precision: float,
) -> Dict[str, Any]:
    if not retrieved_records:
        return {"precision": float("nan"), "matched": 0, "total": 0, "passed": False}
    cc = expected_country.upper()
    matched = sum(
        1 for r in retrieved_records
        if cc in [c.upper() for c in (r.get("countries") or []) if c]
    )
    precision = matched / len(retrieved_records)
    return {
        "precision": precision,
        "matched": matched,
        "total": len(retrieved_records),
        "passed": precision >= min_precision,
        "min_precision": min_precision,
    }


# ---------------------------------------------------------------------------
# LLM-as-judge for free_form
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are scoring a research-summary answer against a reference (gold) summary.

Question:
{question}

Reference (gold) summary:
{gold}

System answer:
{prediction}

Score the system answer on a 1-5 integer scale (5 = matches the reference's
substance, 4 = solid, 3 = adequate, <3 = needs work):
- coverage: addresses the same key points as the reference?
- faithfulness: does it stick to facts that could come from the same corpus,
  with no obvious inventions?
- specificity: cites concrete papers / authors / years rather than vague claims?

Reply with a single JSON object only, no prose:
{{"coverage": <int>, "faithfulness": <int>, "specificity": <int>, "feedback": "<one short sentence>"}}
"""


JUDGE_PROMPT_GROUNDED = """\
You are scoring a research-summary answer for a scientometric question
over the ACL Anthology. There is no reference summary; judge the answer
against the question and the retrieved evidence only.

Question:
{question}

Retrieved evidence (paper titles + abstracts the system had access to):
{evidence}

System answer:
{prediction}

Score on a 1-5 integer scale (5 = excellent, 3 = adequate, 1 = poor) on
each axis independently:
- faithfulness: every factual claim (institution name, year, paper title,
  research contribution) is supported by something in the retrieved
  evidence. No fabricated entities. Hallucinated facts → 1.
- coverage: the answer addresses the question's scope (e.g., for
  institutional comparison, it actually compares; for an overview, it
  names key contributions). Half the question unanswered → ≤2.
- attribution_accuracy: citations or named papers in the answer correctly
  correspond to real papers in the evidence, and the claims attributed
  to them are accurate. Swapped/wrong attribution → 1.
- relevance: the answer stays on the question's topic; does not drift to
  unrelated subareas. Off-topic content → ≤2.

Reply with a single JSON object only, no prose:
{{"faithfulness": <int>, "coverage": <int>, "attribution_accuracy": <int>, "relevance": <int>, "feedback": "<one short sentence>"}}
"""


GROUNDED_AXES = ("faithfulness", "coverage", "attribution_accuracy", "relevance")


def _judge_free_form(question: str, gold: str, prediction: str, judge_model: str) -> Dict[str, Any]:
    from llama_index.core import Settings
    from backend.rag.rag_utils import configure_llm, extract_first_json_object

    saved_llm = Settings.llm
    try:
        configure_llm(judge_model)
        prompt = JUDGE_PROMPT.format(question=question, gold=gold, prediction=prediction)
        raw = Settings.llm.complete(prompt).text
    finally:
        Settings.llm = saved_llm

    try:
        parsed = extract_first_json_object(raw)
    except (ValueError, json.JSONDecodeError):
        return {
            "coverage": None, "faithfulness": None, "specificity": None,
            "feedback": f"unparseable judge output: {raw[:200]}",
        }
    return {
        "coverage": parsed.get("coverage"),
        "faithfulness": parsed.get("faithfulness"),
        "specificity": parsed.get("specificity"),
        "feedback": parsed.get("feedback"),
    }


def _format_evidence_for_judge(evidence: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """Concise textual rendering of the retrieved evidence for the judge prompt."""
    chunks: List[str] = []
    used = 0
    for i, rec in enumerate(evidence or [], start=1):
        title = (rec.get("title") or "").strip()
        body = (rec.get("abstract") or rec.get("text") or "").strip()
        pid = rec.get("paper_id") or ""
        block = f"[{i}] {pid} — {title}\n{body[:800]}"
        if used + len(block) > max_chars:
            break
        chunks.append(block)
        used += len(block)
    return "\n\n".join(chunks) if chunks else "(no evidence retrieved)"


def _judge_free_form_grounded(
    question: str,
    prediction: str,
    evidence: List[Dict[str, Any]],
    judge_model: str,
) -> Dict[str, Any]:
    """Reference-free rubric scoring against retrieved evidence.
    Returns dict with GROUNDED_AXES (int|None) + feedback string."""
    from llama_index.core import Settings
    from backend.rag.rag_utils import configure_llm, extract_first_json_object

    saved_llm = Settings.llm
    raw = ""
    try:
        configure_llm(judge_model)
        prompt = JUDGE_PROMPT_GROUNDED.format(
            question=question,
            evidence=_format_evidence_for_judge(evidence),
            prediction=prediction,
        )
        # Transient upstream errors (e.g., Aqueduct 400 wrapping a
        # connection error, OpenAI rate-limit, model overload) shouldn't
        # abort the entire eval. Emit None scores for the affected row
        # and let the run continue — the aggregate will skip Nones.
        try:
            raw = Settings.llm.complete(prompt).text
        except Exception as e:
            return {
                **{axis: None for axis in GROUNDED_AXES},
                "feedback": f"judge call failed [{judge_model}]: "
                            f"{type(e).__name__}: {str(e)[:200]}",
            }
    finally:
        Settings.llm = saved_llm

    try:
        parsed = extract_first_json_object(raw)
    except (ValueError, json.JSONDecodeError):
        return {
            **{axis: None for axis in GROUNDED_AXES},
            "feedback": f"unparseable judge output: {raw[:200]}",
        }
    out: Dict[str, Any] = {axis: parsed.get(axis) for axis in GROUNDED_AXES}
    out["feedback"] = parsed.get("feedback")
    return out


def _judge_multi_grounded(
    question: str,
    prediction: str,
    evidence: List[Dict[str, Any]],
    judge_models: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Run grounded judging across N judge models. Returns
    {judge_model: {axis: score, ..., 'feedback': ...}}."""
    out: Dict[str, Dict[str, Any]] = {}
    for jm in judge_models:
        out[jm] = _judge_free_form_grounded(question, prediction, evidence, jm)
    return out


# ---------------------------------------------------------------------------
# Aggregate helpers
# ---------------------------------------------------------------------------

def _add(agg: Dict[str, List[float]], key: str, value: Optional[float]) -> None:
    if value is None:
        return
    if isinstance(value, float) and math.isnan(value):
        return
    agg.setdefault(key, []).append(float(value))


def _format_metric(name: str, vals: List[float]) -> str:
    if not vals:
        return ""
    mean = sum(vals) / len(vals)
    return f"# {name}: {mean:.3f}  (n={len(vals)})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--testset", default="data/eval/scico_testset.jsonl",
                   help="JSONL test cases (default: data/eval/scico_testset.jsonl).")
    p.add_argument("--mode", choices=("retrieval", "agent", "closed_book", "pipeline"), default="retrieval",
                   help="retrieval = call the tool directly (free, fast). "
                        "agent = run the SciCo-scoped agent for each row.")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="LLM for the agent (only --mode agent). Default: gpt-4o-mini.")
    p.add_argument("--judge", default=None,
                   help="Single LLM-as-judge model (legacy gold-summary rubric). "
                        "Only meaningful in --mode agent.")
    p.add_argument("--judges", nargs="+", default=None,
                   metavar="MODEL",
                   help="Multiple judge models for the grounded reference-free "
                        "rubric (faithfulness, coverage, attribution_accuracy, "
                        "relevance). Reports per-axis means AND inter-judge "
                        "Spearman ρ. Example: --judges gpt-4o-mini "
                        "claude-sonnet-4-6 gemini-2.5-flash")
    p.add_argument("--ground-truth-csv", default="data/datastructure/ground_truth.csv",
                   help="Path to SciCo ground_truth.csv (default: "
                        "data/datastructure/ground_truth.csv). Used for "
                        "affiliation_correctness.")
    p.add_argument("--affiliations-dir", type=Path, default=None,
                   help="Directory of enriched per-paper JSONs (with "
                        "topics.cso.union populated). Required to compute "
                        "gold paper-id sets per question at runtime. If "
                        "unset, retrieval metrics are skipped. SLOW over "
                        "NFS — prefer --gold-paper-ids when possible.")
    p.add_argument("--gold-paper-ids", type=Path, default=None,
                   help="Precomputed {row_id: [paper_id, ...]} JSON "
                        "(output of scripts/precompute_gold_paper_ids.py). "
                        "When set, skips loading the affiliations corpus "
                        "entirely. ~5 MB file, loads in <1 sec.")
    p.add_argument("--generate-only", action="store_true",
                   help="Pure answer-generation mode. Skips corpus load, "
                        "gold-set computation, retrieval metrics, judges, "
                        "BERTScore — runs the agent and dumps records + "
                        "summary_text + telemetry to --answers-out. All "
                        "scoring is done in a later pass via --answers-in.")
    p.add_argument("--classifier-accuracy", nargs="+", default=None,
                   metavar="MODEL",
                   help="One or more model names to grade on classifier "
                        "accuracy (reproduces CIKM Table 1). For each row "
                        "with classifier_gold, calls generate_questions and "
                        "exact-matches each predicted field against gold. "
                        "Example: --classifier-accuracy gpt-4o-mini "
                        "claude-sonnet-4-6 gemini-2.5-pro")
    p.add_argument("--persona", choices=("all", "head", "postdoc", "phd"),
                   default="all",
                   help="Restrict to rows tagged with this persona. Default: all. "
                        "Use 'phd' to only grade methodology-tier questions.")
    p.add_argument("--k", type=int, default=15,
                   help="Default top-k retrieved (per-row 'k' overrides). Default 15.")
    p.add_argument("--no-bertscore", action="store_true",
                   help="Skip BERTScore (avoids ~5 min model download on first run).")
    p.add_argument("--answers-out", type=Path, default=None,
                   help="If set, save per-row agent output (records + summary + "
                        "telemetry) to this JSONL. Lets a later judging-only "
                        "pass re-score the same answers without re-paying for "
                        "the agent. Pair with no --judges/--judge to get a "
                        "pure answer-only phase.")
    p.add_argument("--answers-in", type=Path, default=None,
                   help="Load per-row agent output from a JSONL written by a "
                        "previous --answers-out run. Skips agent calls entirely; "
                        "judging + grading proceed against the loaded answers. "
                        "Speeds up re-judging matrix runs by 10-50x and avoids "
                        "re-spending agent budget when adding judges later.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print every retrieved paper id per row.")
    args = p.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    testset_path = Path(args.testset)
    if not testset_path.exists():
        print(f"Testset not found: {testset_path}", file=sys.stderr)
        return 2

    rows = _load_testset(testset_path)
    if not rows:
        print(f"Empty testset: {testset_path}", file=sys.stderr)
        return 2

    if args.persona != "all":
        rows = [r for r in rows if r.get("persona") == args.persona]
        if not rows:
            print(f"No rows match persona={args.persona!r}", file=sys.stderr)
            return 2

    judge_active = bool(args.judge) and args.mode == "agent"
    judges_grounded = list(args.judges or []) if args.mode == "agent" else []
    bertscore_active = args.mode == "agent" and not args.no_bertscore
    classifier_models = args.classifier_accuracy or []

    # Per-judge × axis × row score tracking, populated as we iterate.
    # judge_scores[judge][axis] = [s1, s2, ...] (one entry per row that ran)
    judge_scores: Dict[str, Dict[str, List[Optional[float]]]] = {
        j: {axis: [] for axis in GROUNDED_AXES} for j in judges_grounded
    }

    # Defaults that every downstream branch references at least once.
    # Hoisting these out of the if/else below so an UnboundLocalError
    # can't happen when the affiliations corpus branch runs in non-
    # generate-only mode.
    gold_aff: Dict[str, Any] = {}
    corpus = None
    gold_by_rid: Dict[str, set] = {}

    # In --generate-only mode the agent runs and we save the raw output;
    # every scoring step (gold load, corpus load, retrieval/NLG metrics,
    # judges, classifier accuracy) is skipped. The dense vector index used
    # by `search_local_acl_papers` is the only thing that loads.
    if args.generate_only:
        if args.answers_out is None:
            print("--generate-only requires --answers-out", file=sys.stderr)
            return 2
        # All three modes (agent / closed_book / retrieval) are valid
        # output sources — retrieval saves records but no summary, the
        # other two save both. The downstream judge phase keys off `mode`
        # in the JSONL to decide which metrics to compute.
        judges_grounded = []
        classifier_models = []
        bertscore_active = False
        judge_active = False
    else:
        gold_aff = load_ground_truth_affiliations(args.ground_truth_csv) if args.mode == "agent" else {}

    # Load the affiliations corpus once for runtime gold computation.
    # OR — much faster — load a precomputed {rid: [paper_id]} JSON.
    if args.generate_only:
        pass  # already initialised above
    elif args.gold_paper_ids:
        if not args.gold_paper_ids.is_file():
            print(f"WARN: --gold-paper-ids {args.gold_paper_ids} not found; "
                  f"retrieval gold will be empty.", file=sys.stderr)
        else:
            print(f"# loading precomputed gold paper_ids from {args.gold_paper_ids} ...")
            with args.gold_paper_ids.open() as fh:
                raw = json.load(fh)
            for rid, pids in raw.items():
                gold_by_rid[rid] = set(pids)
            sizes = [len(s) for s in gold_by_rid.values()]
            print(f"# gold-set size stats: min={min(sizes) if sizes else 0} "
                  f"median={sorted(sizes)[len(sizes)//2] if sizes else 0} "
                  f"max={max(sizes) if sizes else 0} "
                  f"zero_gold={sum(1 for s in sizes if s == 0)}/{len(sizes)}")
    elif args.affiliations_dir:
        if not args.affiliations_dir.is_dir():
            print(f"WARN: --affiliations-dir {args.affiliations_dir} not found; "
                  f"retrieval gold will be empty.", file=sys.stderr)
        else:
            print(f"# loading affiliations corpus from {args.affiliations_dir} ...")
            corpus = load_affiliations_corpus(args.affiliations_dir)
            for r in rows:
                gold_by_rid[r["id"]] = compute_gold_paper_ids(r, corpus)
            stats = summarize_gold_sizes([(r, gold_by_rid[r["id"]]) for r in rows])
            print(f"# gold-set size stats: min={stats['min']} "
                  f"median={stats['median']} max={stats['max']} "
                  f"mean={stats['mean']:.1f} zero_gold={stats['zero_gold']}/{stats['n']}")
    else:
        print("# NOTE: neither --gold-paper-ids nor --affiliations-dir set; "
              "gold computed per-row falls back to row['gold_paper_ids'] "
              "(empty for most of the new question schema). "
              "Retrieval metrics will be NaN.", file=sys.stderr)

    # (model, field) -> [bool, ...]
    classifier_results: Dict[tuple, list] = {}

    header = (
        f"# SciCo eval — {len(rows)} cases, mode={args.mode}, k={args.k}"
        f"{f', persona={args.persona}' if args.persona != 'all' else ''}"
        f"{f', model={args.model}' if args.mode == 'agent' else ''}"
        f"{f', judge={args.judge}' if judge_active else ''}"
        f"{f', judges={judges_grounded}' if judges_grounded else ''}"
        f"{', +bertscore' if bertscore_active else ''}"
        f"{f', classifier_accuracy={classifier_models}' if classifier_models else ''}"
    )
    print(header)
    print()

    agg: Dict[str, List[float]] = {}
    # persona -> metric -> list[float]
    agg_per_persona: Dict[str, Dict[str, List[float]]] = {}

    def _add_pp(persona: str, name: str, value: Optional[float]) -> None:
        if value is None:
            return
        if isinstance(value, float) and math.isnan(value):
            return
        agg_per_persona.setdefault(persona, {}).setdefault(name, []).append(float(value))
    skipped = 0
    errors = 0

    # --- answer-then-judge split plumbing ---
    # answers_in: load saved agent output, skip _run_agent
    # answers_out: persist agent output per row for later re-judging
    answers_in_map: Dict[str, Dict[str, Any]] = {}
    if args.answers_in is not None:
        if not args.answers_in.is_file():
            print(f"--answers-in {args.answers_in} not found", file=sys.stderr)
            return 2
        with args.answers_in.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("id"):
                    answers_in_map[rec["id"]] = rec
        print(f"# loaded {len(answers_in_map)} saved answers from {args.answers_in}")
    # Resume support: if --answers-out points at an existing non-empty
    # file, build a set of already-completed row IDs and open in append
    # mode so a re-submitted job picks up where the previous one stopped.
    # Mid-row writes never happen because we json.dump() the full record
    # per row and flush — so any complete line in the file is trustworthy.
    answers_out_fh = None
    completed_ids: set = set()
    if args.answers_out is not None:
        args.answers_out.parent.mkdir(parents=True, exist_ok=True)
        if args.answers_out.is_file() and args.answers_out.stat().st_size > 0:
            with args.answers_out.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if rec.get("id"):
                        completed_ids.add(rec["id"])
            answers_out_fh = args.answers_out.open("a", encoding="utf-8")
            print(f"# RESUMING — found {len(completed_ids)} completed rows in "
                  f"{args.answers_out}; appending the rest")
        else:
            answers_out_fh = args.answers_out.open("w", encoding="utf-8")
            print(f"# writing per-row answers to {args.answers_out}")

    for row in rows:
        rid = row.get("id", "?")
        rtype = row.get("type", "?")
        persona = row.get("persona", "unknown")
        k = int(row.get("k", args.k))

        # Skip rows already completed in a prior run of this same --answers-out
        # file. Only applies when --answers-in is NOT set (which would
        # short-circuit generation anyway).
        if args.answers_in is None and rid in completed_ids:
            print(f"[{rid}] already in {args.answers_out.name}; skipping")
            skipped += 1
            continue

        # Run retrieval / agent — or load from a prior --answers-out file.
        summary_text = ""
        tele: Dict[str, Any] = {}
        err: Optional[str] = None
        if args.answers_in is not None:
            saved = answers_in_map.get(rid)
            if saved is None:
                print(f"[{rid}] no saved answer in --answers-in; skipping")
                skipped += 1
                continue
            records = saved.get("records") or []
            summary_text = saved.get("summary_text") or ""
            tele = saved.get("telemetry") or {}
            err = saved.get("error") or None
        elif args.mode == "retrieval":
            import time as _t
            _t0 = _t.perf_counter()
            records, err = _retrieve_only(row, args.k)
            tele = {
                "wall_seconds": _t.perf_counter() - _t0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "error": err,
            }
            summary_text = ""  # no generation in pure-retrieval mode
        elif args.mode == "closed_book":
            records, summary_text, tele, err = _run_closed_book(row, args.model)
        elif args.mode == "pipeline":
            records, summary_text, tele, err = _run_pipeline(row, args.model)
        else:
            records, summary_text, tele, err = _run_agent(row, args.model)

        # Persist this row's agent output BEFORE grading (so judging-only
        # reruns get the same records even when grading fails). Records
        # are written as-is; downstream rerun must use the same testset.
        if answers_out_fh is not None:
            json.dump({
                "id": rid,
                "type": rtype,
                "persona": persona,
                "model": args.model,
                "mode": args.mode,
                "records": records,
                "summary_text": summary_text,
                "telemetry": tele,
                "error": err,
            }, answers_out_fh, ensure_ascii=False)
            answers_out_fh.write("\n")
            answers_out_fh.flush()

        if err:
            print(f"[{rid}] ({rtype}) ERROR: {err}")
            errors += 1
            continue

        # --generate-only short-circuit: skip every grading step. The
        # full eval (retrieval metrics, judges, BERTScore, classifier
        # accuracy, etc.) runs in a separate pass via --answers-in.
        if args.generate_only:
            telemetry_secs = (tele or {}).get("wall_seconds") or 0
            n_records = len(records or [])
            print(f"[{rid}] generated  records={n_records}  "
                  f"wall={telemetry_secs:.1f}s  "
                  f"prompt_tok={(tele or {}).get('prompt_tokens', 0)}  "
                  f"completion_tok={(tele or {}).get('completion_tokens', 0)}")
            continue

        retrieved_ids = _retrieved_ids(records)

        # Type-specific graders.
        if rtype in {"paper_retrieval", "free_form"}:
            if rtype == "paper_retrieval":
                gold_ids = row.get("expected_paper_ids") or []
            elif corpus is not None:
                # Runtime gold computed against the current affiliations corpus.
                gold_ids = sorted(gold_by_rid.get(rid, set()))
            else:
                gold_ids = row.get("gold_paper_ids") or []
            ret = precision_recall_mrr(retrieved_ids, gold_ids, k=k)
            if not gold_ids:
                print(f"[{rid}] ({rtype}) no gold paper ids — retrieved {len(retrieved_ids)}")
                skipped += 1
            else:
                _add(agg, f"{rtype}.precision@k", ret["precision"])
                _add(agg, f"{rtype}.recall@k", ret["recall"])
                _add(agg, f"{rtype}.f1@k", ret["f1"])
                _add(agg, f"{rtype}.mrr", ret["mrr"])
                _add_pp(persona, f"{rtype}.precision@k", ret["precision"])
                _add_pp(persona, f"{rtype}.recall@k", ret["recall"])
                _add_pp(persona, f"{rtype}.f1@k", ret["f1"])
                _add_pp(persona, f"{rtype}.mrr", ret["mrr"])
                print(
                    f"[{rid}] {rtype}  "
                    f"P@k={ret['precision']:.2f}  R@k={ret['recall']:.2f}  "
                    f"F1@k={ret['f1']:.2f}  MRR={ret['mrr']:.2f}"
                )

        elif rtype == "institution_list":
            sys_affs = _affiliation_union(records)
            res = institution_list_metrics(sys_affs, row.get("expected_institutions") or [])
            if math.isnan(res["recall"]):
                print(f"[{rid}] (institution_list) no gold — found {len(sys_affs)} sys affs")
                skipped += 1
            else:
                _add(agg, "institution_list.precision", res["precision"])
                _add(agg, "institution_list.recall", res["recall"])
                _add(agg, "institution_list.f1", res["f1"])
                print(
                    f"[{rid}] institution_list  "
                    f"P={res['precision']:.2f}  R={res['recall']:.2f}  "
                    f"F1={res['f1']:.2f}  ({len(sys_affs)} sys affs)"
                )

        elif rtype == "country_precision":
            res = _grade_country_precision(
                records,
                row.get("expected_country", ""),
                float(row.get("min_precision", 0.5)),
            )
            _add(agg, "country_precision", res["precision"])
            mark = "PASS" if res.get("passed") else "FAIL"
            print(
                f"[{rid}] country_precision precision={res['precision']:.2f}  "
                f"({res['matched']}/{res['total']} for "
                f"{row.get('expected_country','?')})  [{mark} >= {res['min_precision']}]"
            )

        else:
            print(f"[{rid}] unknown type {rtype!r} — skipping")
            skipped += 1
            continue

        # Free-form NLG metrics + LLM-judge.
        if rtype == "free_form" and summary_text:
            # ROUGE-L against retrieved evidence — always available in agent
            # mode, no gold summary needed. Approximates "did the system
            # paraphrase retrieved content vs invent things".
            if args.mode == "agent" and records:
                rl_ev = rouge_l_vs_evidence(summary_text, records)
                _add(agg, "free_form.rougeL_vs_evidence", rl_ev)
                print(f"        rougeL_vs_evidence={rl_ev:.3f}")

            # Gold-summary metrics (only when gold provided).
            if row.get("gold_summary"):
                gold = row["gold_summary"]
                rl = rouge_l_f1(summary_text, gold)
                _add(agg, "free_form.rougeL", rl)
                line = f"        rougeL={rl:.3f}"
                if bertscore_active:
                    bs = bertscore_f1(summary_text, gold)
                    _add(agg, "free_form.bertscore", bs)
                    line += f"  bertscore={bs:.3f}"
                print(line)

                if judge_active:
                    judge = _judge_free_form(row.get("query", ""), gold, summary_text, args.judge)
                    for axis in ("coverage", "faithfulness", "specificity"):
                        val = judge.get(axis)
                        if isinstance(val, (int, float)):
                            _add(agg, f"judge.{axis}", float(val))
                    print(
                        f"        judge: coverage={judge.get('coverage')}, "
                        f"faithfulness={judge.get('faithfulness')}, "
                        f"specificity={judge.get('specificity')}  "
                        f"— {judge.get('feedback')}"
                    )

            # Multi-judge grounded rubric (reference-free; runs whenever
            # --judges is set and we have evidence to ground in).
            if judges_grounded and args.mode == "agent":
                multi = _judge_multi_grounded(
                    row.get("query", ""), summary_text, records, judges_grounded,
                )
                for jm in judges_grounded:
                    jres = multi.get(jm, {})
                    line_axes = []
                    for axis in GROUNDED_AXES:
                        val = jres.get(axis)
                        score = float(val) if isinstance(val, (int, float)) else None
                        judge_scores[jm][axis].append(score)
                        if score is not None:
                            _add(agg, f"judge_grounded.{jm}.{axis}", score)
                            _add(agg, f"judge_grounded.mean.{axis}", score)
                            line_axes.append(f"{axis}={int(score) if score == int(score) else score}")
                        else:
                            line_axes.append(f"{axis}=?")
                    print(f"        judge[{jm}]: {', '.join(line_axes)}  — {jres.get('feedback')}")

        # Per-row faithfulness + telemetry (agent mode only).
        if args.mode == "agent":
            ph = phantom_citation_rate(summary_text, records)
            if not math.isnan(ph["phantom_rate"]):
                _add(agg, "phantom_rate", ph["phantom_rate"])
                print(
                    f"        phantom={ph['phantom_rate']:.2f}  "
                    f"({ph['phantom_count']}/{ph['total_refs']} refs not in evidence; "
                    f"{ph['inline_marker_count']} inline markers)"
                )

            if gold_aff:
                aff_res = affiliation_correctness(records, gold_aff)
                if not math.isnan(aff_res["mean_f1"]):
                    _add(agg, "affiliation_correctness.f1", aff_res["mean_f1"])
                    print(
                        f"        aff_correctness mean_F1={aff_res['mean_f1']:.2f}  "
                        f"({aff_res['graded']} graded, {aff_res['skipped']} skipped)"
                    )

            if tele:
                _add(agg, "wall_seconds", tele["wall_seconds"])
                _add(agg, "prompt_tokens", float(tele["prompt_tokens"]))
                _add(agg, "completion_tokens", float(tele["completion_tokens"]))
                _add(agg, "total_tokens", float(tele["total_tokens"]))
                cost_str = (f"${tele['cost_usd']:.4f}"
                            if tele.get("cost_usd") is not None else "n/a")
                print(
                    f"        telemetry: {tele['wall_seconds']:.1f}s  "
                    f"prompt_tok={tele['prompt_tokens']}  "
                    f"completion_tok={tele['completion_tokens']}  "
                    f"cost={cost_str}"
                )
                if tele.get("cost_usd") is not None:
                    _add(agg, "cost_usd", tele["cost_usd"])

        if args.verbose:
            for pid in retrieved_ids:
                print(f"    {pid}")

        # Classifier-accuracy: orthogonal to retrieval/agent, runs per
        # model on every row that carries classifier_gold.
        gold = row.get("classifier_gold")
        if classifier_models and gold:
            for cmodel in classifier_models:
                res = grade_classifier_row(
                    row.get("query", ""), gold, cmodel,
                    paper=row.get("paper"),
                )
                if res.get("error"):
                    print(f"        classifier[{cmodel}] ERROR: {res['error']}")
                    continue
                for field, ok in res["matches"].items():
                    classifier_results.setdefault((cmodel, field), []).append(bool(ok))
                summary_bits = [
                    f"{field}={'✓' if ok else '✗'}"
                    for field, ok in sorted(res["matches"].items())
                ]
                print(f"        classifier[{cmodel}]: {' '.join(summary_bits)}")

    print()
    for name, vals in sorted(agg.items()):
        line = _format_metric(name, vals)
        if line:
            print(line)

    # Inter-judge agreement (Spearman ρ across LLM judges, per axis).
    if judges_grounded and len(judges_grounded) >= 2:
        print()
        print(f"# Inter-judge agreement (Spearman ρ across {len(judges_grounded)} judges)")
        for axis in GROUNDED_AXES:
            per_judge = {j: judge_scores[j][axis] for j in judges_grounded}
            agreement = inter_judge_agreement(per_judge)
            mean_rho = agreement["mean_rho"]
            if isinstance(mean_rho, float) and mean_rho == mean_rho:
                print(f"#   {axis:<22} mean ρ = {mean_rho:.3f}  ({agreement['pairs']} pairs)")
                for pair_key, rho in sorted(agreement["per_pair"].items()):
                    if isinstance(rho, float) and rho == rho:
                        print(f"#     {pair_key:<50} ρ = {rho:.3f}")

    # Per-persona aggregates (only if any persona has data).
    if agg_per_persona:
        print()
        print("# Per-persona aggregates")
        for persona in sorted(agg_per_persona):
            for name, vals in sorted(agg_per_persona[persona].items()):
                if not vals:
                    continue
                mean = sum(vals) / len(vals)
                print(f"#   [{persona}] {name}: {mean:.3f}  (n={len(vals)})")

    if classifier_results:
        print()
        print("# Classifier accuracy (per model × field)")
        # Per-model totals so we can include an average row.
        per_model: Dict[str, list] = {}
        for (cmodel, field), vals in sorted(classifier_results.items()):
            if not vals:
                continue
            acc = sum(vals) / len(vals)
            print(f"#   {cmodel:30s}  {field:30s}  {acc:.3f}  (n={len(vals)})")
            per_model.setdefault(cmodel, []).extend(vals)
        for cmodel in sorted(per_model):
            vals = per_model[cmodel]
            print(f"#   {cmodel:30s}  {'OVERALL':30s}  "
                  f"{sum(vals) / len(vals):.3f}  (n={len(vals)})")

    if skipped:
        print(f"# {skipped} row(s) skipped (no gold or unknown type)")
    if errors:
        print(f"# {errors} row(s) errored")
    if answers_out_fh is not None:
        answers_out_fh.close()
        print(f"# wrote {args.answers_out}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
