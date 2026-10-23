import os

import pytest

from backend.openalex.utils import get_ror_id, get_university_name_from_ror, search_paper
from backend.rag.rag_utils import generate_questions

pytestmark = pytest.mark.live

LIVE_ENABLED = os.getenv("RUN_LIVE_TESTS") == "1"


@pytest.mark.skipif(not LIVE_ENABLED, reason="Set RUN_LIVE_TESTS=1 to hit real APIs")
def test_classifier_returns_expected_structure_for_single_university_query():
    """Smoke-test: gpt-4o-mini classifies a prompt and returns the expected schema."""
    result = generate_questions(
        "What has the University of Aberdeen contributed to topic modeling in 2024?",
        paper=None,
        llm_model="gpt-4o-mini",
    )

    assert isinstance(result, dict)
    assert result.get("type") in {
        "comparison",
        "university",
        "paper",
        "overview",
        "university_university",
        "single_university",
        "internal_kg",
        "internal_kg_research_unit",
        "tuwien",
    }
    assert "queries" in result and isinstance(result["queries"], list) and result["queries"]
    assert (result.get("university_name") or "").lower().startswith("university of aberdeen") or \
           (result.get("university_name") or "").lower().startswith("aberdeen")
    assert result.get("year_start") in {2024, None}
    assert result.get("year_end") in {2024, None}


@pytest.mark.skipif(not LIVE_ENABLED, reason="Set RUN_LIVE_TESTS=1 to hit real APIs")
def test_classifier_detects_paper_question():
    result = generate_questions(
        "What is the main contribution of this paper?",
        paper={"title": "BERT", "abstract": "We introduce BERT, a transformer model."},
        llm_model="gpt-4o-mini",
    )
    assert result.get("type") == "paper"


@pytest.mark.skipif(not LIVE_ENABLED, reason="Set RUN_LIVE_TESTS=1 to hit real APIs")
def test_openalex_ror_lookup_for_aberdeen():
    """OpenAlex ROR-ID lookup is the entry point for every institutional query."""
    ror_id = get_ror_id("University of Aberdeen")
    assert ror_id, "expected a ROR id"
    assert ror_id.startswith("https://ror.org/"), f"unexpected ROR format: {ror_id}"

    name = get_university_name_from_ror(ror_id.replace("https://ror.org/", ""))
    assert name and "aberdeen" in name.lower()


@pytest.mark.skipif(not LIVE_ENABLED, reason="Set RUN_LIVE_TESTS=1 to hit real APIs")
def test_openalex_paper_search_returns_structured_results():
    """search_paper feeds the index-building stage; verify it returns the expected schema."""
    papers = search_paper(
        "topic modeling",
        year_start=2024,
        year_end=2024,
        samples=3,
    )
    assert isinstance(papers, list)
    assert len(papers) >= 1, "expected at least one 2024 paper on topic modeling"

    paper = papers[0]
    for key in ("title", "abstract", "DOI", "publication_year", "authors", "affiliations"):
        assert key in paper, f"missing key {key!r} in structured paper"
    assert paper["publication_year"] == 2024


def _parse_pipeline_result(result):
    """process_input returns inconsistently across branches — normalize here."""
    import json as _json

    if isinstance(result, tuple):
        summary, profiles = result
        if isinstance(summary, (bytes, bytearray)):
            payload = summary.decode("utf-8")
        else:
            payload = summary
    elif isinstance(result, (bytes, bytearray)):
        payload = result.decode("utf-8")
        profiles = None
    else:
        chunks = bytearray()
        for chunk in result:
            chunks += chunk if isinstance(chunk, (bytes, bytearray)) else chunk.encode("utf-8")
        payload = bytes(chunks).decode("utf-8")
        profiles = None

    if payload.lstrip().startswith("data:"):
        last = [p for p in payload.split("\n\n") if p.strip().startswith("data:")][-1]
        parsed = _json.loads(last.replace("data:", "", 1).strip())
        summary = parsed.get("summary")
        profiles = profiles or parsed.get("profiles")
    else:
        summary = payload

    return summary, profiles or []


@pytest.mark.skipif(not LIVE_ENABLED, reason="Set RUN_LIVE_TESTS=1 to hit real APIs")
@pytest.mark.skipif(not os.getenv("RUN_FULL_PIPELINE"), reason="Set RUN_FULL_PIPELINE=1 to run the full lookup pipeline (slower, more API spend)")
def test_full_single_university_pipeline():
    """End-to-end: classify, lookup, search, index, retrieve, summarize."""
    from backend.rag.pipeline import process_input

    result = process_input(
        "What has the University of Aberdeen contributed to topic modeling in 2024?",
        paper=None,
        online=True,
        model="gpt-4o-mini",
    )

    summary, profiles = _parse_pipeline_result(result)

    assert isinstance(summary, str) and len(summary) > 100, "expected a non-trivial summary"
    assert profiles, "expected at least one institution profile"
    assert profiles[0].get("Institution")
    assert profiles[0].get("Publications", 0) >= 1
