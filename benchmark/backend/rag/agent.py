"""Agentic alternative to the deterministic process_input pipeline.

A FunctionAgent picks tools (OpenAlex / CORE / TU Wien / Neo4j KG) based on
the user's prompt instead of relying on upfront classification. After each
draft answer a critic LLM scores coverage / specificity / faithfulness on a
1-5 scale; if any score < 4 or gaps are reported, the agent runs again with
the gaps as feedback. Bounded by ``MAX_REFINE_ITERATIONS``.

Output format matches process_input_streaming: progress messages as
``data: {"update": "..."}\\n\\n`` and a final
``data: {"summary": ..., "profiles": [...]}\\n\\n``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from llama_index.core import Settings
from llama_index.core.agent.workflow import FunctionAgent

from backend.rag.agent_tools import (
    build_default_tools,
    collected_evidence,
    collected_profiles,
    reset_buffers,
)
from backend.rag.rag_utils import (
    configure_llm,
    extract_first_json_object,
    fix_output_citations,
)

logger = logging.getLogger(__name__)

MAX_REFINE_ITERATIONS = 3
ACCEPT_MEAN_SCORE = 4.0  # mean of quality axes must reach this
MIN_REFERENCES = 6       # min distinct papers cited inline
# Faithfulness is grounded — every tool call records its returned papers
# into an evidence buffer that we hand to the critic, so it can actually
# cross-check citations against retrieved metadata instead of speculating.
SCORE_AXES = ("coverage", "specificity", "structure", "depth", "faithfulness")

# Persistent per-worker event loop. asyncio.run() creates and tears down a
# new loop for every call, but Gemini's gRPC client and Anthropic's httpx
# transport cache resources on the loop where they were first used — running
# the second agent.run() on a fresh loop trips "Event loop is closed".
# Keeping one loop alive for the worker's lifetime sidesteps that entirely.
_AGENT_LOOP = None


def _agent_loop():
    global _AGENT_LOOP
    if _AGENT_LOOP is None or _AGENT_LOOP.is_closed():
        _AGENT_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_AGENT_LOOP)
    return _AGENT_LOOP


def _run_on_loop(coro):
    return _agent_loop().run_until_complete(coro)


SYSTEM_PROMPT = """\
You are a research-comparison assistant for academic publications. You answer
questions about scientific output of universities, research units, papers, and
research domains by calling tools to retrieve evidence, then writing a
structured answer with inline citations.

HARD RULE: You MUST call at least one retrieval tool before producing the
final answer. Never answer from your training data alone. If you write text
that cites specific papers, those papers MUST come from a tool result you
just called this turn — not from memory.

Available tools (call them by name; pick the best fit):
- list_university_publications(university_name, topic?, year_start?, year_end?)
    — papers affiliated with a specific university; the right tool for
    "what has <university> published on X?". Returns rows you can cite.
- get_university_publications_profile(university_name, topic?, year_start?, year_end?)
    — aggregated stats for a university (publications-per-year, top authors,
    collaborators, top-cited papers); call this for institution questions
    so the frontend can render its charts.
- search_openalex_by_topic(query, year_start?, year_end?, institution_name?)
    — cross-institution topic search across ~250M papers. Use when no
    specific institution is given, or to find third-party comparators.
- search_core_fulltext(query)
    — full-text papers, only when methodology-level detail is needed.
- search_tu_wien_papers(query, year_start?, org_units?)
    — TU Wien Repositum (preferred for TU Wien-specific queries).
- search_internal_kg_research_unit(unit_name, year?, year_start?, year_end?)
    — internal Neo4j KG of TU Wien research units (Data Science, Knowledge
    Graphs, Visual Computing, etc.).

Strategy:
1. Identify the kind of question: a specific paper, a single institution, two
   institutions, a domain overview, a research unit, or a TU Wien query.
2. Call the relevant tool(s):
   - "What has <university> contributed/published" → call
     list_university_publications AND get_university_publications_profile.
   - Comparing two universities → call both for each institution.
   - TU Wien queries → prefer the TU Wien / KG tools first.
   - Domain overview without institutions → search_openalex_by_topic.
3. If initial results are sparse, broaden the query, raise samples, or try
   a sibling tool. Aim for AT LEAST 6-10 distinct citations.
4. Write the final answer with [1], [2], ... inline citations and an
   APA-style numbered references section listing only papers you actually
   cite, with DOIs. Cite at least 6 distinct papers when the question
   warrants it.

Rules:
- Only cite papers actually returned by tool calls. Never invent DOIs,
  authors, or years.
- If the user provides their own paper (look for a "User-provided paper:"
  block in the message), include it explicitly in any comparison.
- Be specific: name authors, years, and methods. Avoid vague claims.
- Stop calling tools once you have enough; don't loop indefinitely.

Output structure (markdown — the frontend renders this as markdown, so
blank lines between blocks matter):
- Open with ONE bolded single-line takeaway BEFORE any heading, like:
  **TU Wien's Data Science unit produced 47 IR papers in 2024, with
  growing focus on retrieval-augmented generation.**
- Then level-3 headings: ### Overview / ### Key contributions /
  ### Comparative analysis / ### Trends / ### References (use the ones
  that fit the question — not every section is mandatory).
- Within a section, be CONSISTENT: either all bullet items (each starting
  with `*` or `-`) OR all prose paragraphs. Don't interleave a bullet and
  a free-standing paragraph at the same level.
- Each fact, statistic, or paper mention appears ONCE. Don't restate the
  same number or finding across sections.
- References section: each entry on its own line WITH A BLANK LINE between
  entries (markdown needs the blank line). Every entry must end with a
  working URL — tool results include a `url` field (DOI when available,
  OpenAlex Work URL otherwise); use it. Don't list a paper without a link.

Citation format (STRICT):
- All inline citations use [N] markers (e.g., [1], [2]). DOI URLs appear
  ONLY in the references list at the end, never in the body.
- No author-year style anywhere in the body.
- Numbering is continuous starting at [1]; deduplicate identical refs; drop
  any reference you don't actually cite inline.
- References list at the end uses APA, one entry per line; replace authors
  after the third with "et al.".

GOOD body: "BERT [3] introduced bidirectional pre-training."
GOOD refs: "[3] Devlin, J., Chang, M.-W., & Lee, K. (2019). BERT: Pre-training
of deep bidirectional transformers. https://doi.org/10.18653/v1/N19-1423"

BAD (do NOT produce):
- "BERT (Devlin et al., 2019) introduced..."
- "BERT (https://doi.org/10.18653/v1/N19-1423) introduced..."
- "BERT [Devlin2019] introduced..."
"""


def _build_user_message(prompt: str, paper: Optional[dict]) -> str:
    if not paper:
        return prompt
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    return (
        f"{prompt}\n\n"
        f"User-provided paper:\n"
        f"Title: {title}\n"
        f"Abstract: {abstract}\n"
    )


async def _run_agent_async(agent: FunctionAgent, user_msg: str) -> str:
    # max_iterations bounds wall time per row. Cap at 5: the agent
    # should pick a tool, run a couple of refinements, then synthesise.
    # Looping past that wastes API budget for marginal gains.
    try:
        response = await agent.run(user_msg=user_msg, max_iterations=5)
    except Exception as e:
        if "Max iterations" in str(e) or "WorkflowRuntimeError" in type(e).__name__:
            return ("[Agent did not converge within 5 tool-call turns. "
                    "No final answer generated.]")
        raise
    return str(response)


def _format_evidence_sheet(evidence: list, max_records: int = 60) -> str:
    """Render the agent's retrieved evidence as a numbered sheet for the critic."""
    lines = []
    for i, rec in enumerate(evidence[:max_records], start=1):
        title = (rec.get("title") or "").strip().replace("\n", " ")
        authors = rec.get("authors") or []
        if isinstance(authors, list):
            authors_str = ", ".join(a for a in authors if a) or "Unknown authors"
        else:
            authors_str = str(authors)
        year = rec.get("publication_year") or "n.d."
        doi = rec.get("DOI") or rec.get("url") or "no link"
        affs = rec.get("affiliations") or []
        affs_str = "; ".join(a for a in affs if a) if affs else "no affiliations recorded"
        lines.append(
            f"E{i}. {title} — {authors_str} ({year}). {doi}. Affiliations: {affs_str}"
        )
    return "\n".join(lines) if lines else "(no evidence recorded — the agent made no tool calls)"


def _critic(prompt: str, draft: str, evidence: list) -> dict:
    """Score a draft using the SAME evidence the agent retrieved.

    The critic gets a slim metadata sheet — title, authors, year, DOI,
    affiliations — for every paper any tool returned during this run. That
    grounds the faithfulness axis: instead of speculating about whether a
    citation's institution is correct, the critic can check the affiliation
    field on the matching evidence record. Earlier versions had no
    grounding and the critic would hallucinate "this paper isn't really
    from TU Wien"-style accusations, which then misled the next iteration.
    """
    evidence_sheet = _format_evidence_sheet(evidence)
    critic_prompt = (
        "You are a strict reviewer. You can fact-check the draft against "
        "the agent's actual retrieval evidence (listed below).\n\n"
        f"User's question:\n{prompt}\n\n"
        f"Evidence retrieved by the agent (papers available for citation):\n"
        f"{evidence_sheet}\n\n"
        f"Draft answer:\n{draft}\n\n"
        "Score on a 1-5 integer scale (5 = truly excellent, 4 = solid, "
        "3 = adequate, <3 = needs work):\n"
        "- coverage: addresses every aspect of the question?\n"
        "- specificity: cites concrete papers, authors, years (not vague claims)?\n"
        "- structure: well-formed markdown — single bolded takeaway up top, "
          "level-3 headings, consistent bullets-or-prose within each section, "
          "references with a blank line between entries?\n"
        "- depth: substantive analysis (methods, contributions, trends, "
          "comparisons) rather than a flat list?\n"
        "- faithfulness: every claim in the draft is supported by the "
          "evidence above. Specifically check that each [N] citation maps "
          "to a paper in the evidence list, that author names match, that "
          "institutional attributions are consistent with the affiliations "
          "field. Mark down for inventions or mis-attributions YOU CAN PROVE "
          "from the evidence. If a claim isn't refuted by any evidence "
          "record, treat it as supported.\n\n"
        "Also report:\n"
        "- reference_count: integer, distinct [N] markers cited inline (deduplicated).\n\n"
        "List up to 3 SPECIFIC gaps. GOOD examples:\n"
        "- 'only 4 distinct references — agent should call more tools to reach >=6'\n"
        "- 'no comparative analysis section'\n"
        "- 'bullets and free paragraphs mixed inside Key Contributions'\n"
        "- 'citation [4] attributes paper X to TU Wien but evidence E12 lists "
          "affiliations: Milano-Bicocca only'\n\n"
        "If a gap claim is FACTUAL, you must cite the evidence record (E#) "
        "that supports it. Do NOT invent factual gaps that the evidence "
        "doesn't clearly contradict.\n\n"
        "Reply with a single JSON object, no prose:\n"
        '{"coverage": <int>, "specificity": <int>, "structure": <int>, '
        '"depth": <int>, "faithfulness": <int>, "reference_count": <int>, '
        '"gaps": [<str>, ...]}\n'
    )
    raw = Settings.llm.complete(critic_prompt).text
    try:
        return extract_first_json_object(raw)
    except (ValueError, json.JSONDecodeError):
        logger.warning("critic returned unparseable output: %s", raw[:200])
        return {
            "coverage": 3, "specificity": 3, "structure": 3, "depth": 3,
            "faithfulness": 3, "reference_count": 0, "gaps": [],
        }


def _is_acceptable(critique: dict) -> bool:
    """Accept when (mean of quality axes) >= ACCEPT_MEAN_SCORE AND
    reference_count >= MIN_REFERENCES. Gaps are advisory — used to steer
    the next iteration when scores are below the bar — but a single
    minor gap doesn't force another full agent pass when the work is
    already at quality. (Earlier "no gaps" gating made the loop run all
    3 iterations even on solid drafts because critics always nitpick.)
    """
    if critique.get("reference_count", 0) < MIN_REFERENCES:
        return False
    scores = [float(critique.get(k, 0) or 0) for k in SCORE_AXES]
    if not scores:
        return False
    return sum(scores) / len(scores) >= ACCEPT_MEAN_SCORE


def process_input_agent(
    prompt: str,
    paper: Optional[dict] = None,
    online: bool = True,  # kept for parity with the pipeline signature
    model: str = "gpt-4o-mini",
    stream_fn=None,
    tools=None,
    system_prompt: Optional[str] = None,
):
    """Sync generator running the agent + critic-refine loop.

    Yields SSE chunks compatible with the existing /api/process_input_stream
    consumer: progress messages via ``stream_fn`` and a final
    ``data: {"summary": ..., "profiles": [], "mode": "agent",
              "iterations": N}\\n\\n`` payload.

    Override ``tools`` and ``system_prompt`` to constrain the agent — used
    by the SciCo eval to run a SciCo-only agent (just the local ACL tool,
    with a system prompt that forbids speculating beyond the corpus).
    """

    def emit(msg: str):
        if stream_fn:
            yield from stream_fn(msg)

    yield from emit(f"Configuring agent ({model})")
    configure_llm(model)
    reset_buffers()
    agent = FunctionAgent(
        tools=tools if tools is not None else build_default_tools(),
        llm=Settings.llm,
        system_prompt=system_prompt or SYSTEM_PROMPT,
    )

    user_msg = _build_user_message(prompt, paper)
    yield from emit("Agent reasoning over tools...")
    draft = _run_on_loop(_run_agent_async(agent, user_msg))

    iterations = 0
    for iterations in range(1, MAX_REFINE_ITERATIONS + 1):
        yield from emit(f"Critic evaluating draft (iter {iterations})")
        evidence = collected_evidence()
        critique = _critic(prompt, draft, evidence)
        logger.debug("critique: %s", critique)

        if _is_acceptable(critique):
            yield from emit(f"Critic accepted after {iterations} iteration(s)")
            break

        gaps = critique.get("gaps") or []
        scores = {k: critique.get(k) for k in SCORE_AXES}
        score_values = [float(v or 0) for v in scores.values()]
        mean_score = (sum(score_values) / len(score_values)) if score_values else 0.0
        ref_count = critique.get("reference_count", 0)
        yield from emit(
            f"Refining (scores={scores}, mean={mean_score:.2f}, "
            f"references={ref_count}, gaps={gaps})"
        )

        # If references are below the floor, surface that as a gap even when
        # the critic forgot to call it out.
        if ref_count < MIN_REFERENCES and not any("reference" in g.lower() for g in gaps):
            gaps.append(
                f"only {ref_count} distinct citations — call more tools to "
                f"reach at least {MIN_REFERENCES} cited papers"
            )

        followup = (
            f"Original question:\n{prompt}\n\n"
            f"Your previous draft:\n{draft}\n\n"
            f"A reviewer flagged these gaps (factual gaps reference an "
            f"evidence record E#; structural gaps don't):\n"
            + "\n".join(f"- {g}" for g in gaps)
            + "\n\nProduce an improved answer addressing the gaps. For each "
              "factual gap that cites an E# record, verify against your own "
              "tool results — if the reviewer is right, fix the citation or "
              "drop the claim. Call additional tools if you need more "
              "evidence. Keep [N] inline citations and the references "
              "section."
        )
        draft = _run_on_loop(_run_agent_async(agent, followup))
    else:
        yield from emit(f"Max refinement iterations ({MAX_REFINE_ITERATIONS}) reached")

    yield from emit("Fixing citations")
    final = fix_output_citations(draft)

    payload = {
        "summary": final,
        "profiles": collected_profiles(),
        "mode": "agent",
        "iterations": iterations,
    }
    yield f"data: {json.dumps(payload)}\n\n".encode("utf-8")


def process_input_agent_sync(
    prompt: str,
    paper: Optional[dict] = None,
    online: bool = True,
    model: str = "gpt-4o-mini",
    tools=None,
    system_prompt: Optional[str] = None,
):
    """Non-streaming variant returning (summary, profiles).

    Mirrors process_input's contract so /api/process_input can dispatch
    cleanly to either mode. ``tools`` / ``system_prompt`` are passed through
    to ``process_input_agent`` for SciCo-style scoped runs.
    """
    summary = None
    profiles: list = []
    for chunk in process_input_agent(
        prompt, paper=paper, online=online, model=model, stream_fn=None,
        tools=tools, system_prompt=system_prompt,
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
