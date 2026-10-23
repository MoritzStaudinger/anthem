"""Live tests for the agent mode. Gated by RUN_LIVE_TESTS=1.

The full pipeline test costs ~$0.02; the agent typically costs $0.05-0.20 per
run because it makes more LLM calls. Gated separately by RUN_AGENT_TESTS=1
on top of RUN_LIVE_TESTS=1 so it doesn't fire on every live run.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.live

LIVE_ENABLED = os.getenv("RUN_LIVE_TESTS") == "1"
AGENT_ENABLED = os.getenv("RUN_AGENT_TESTS") == "1"


@pytest.mark.skipif(not LIVE_ENABLED, reason="Set RUN_LIVE_TESTS=1 to hit real APIs")
@pytest.mark.skipif(not AGENT_ENABLED, reason="Set RUN_AGENT_TESTS=1 to run the agent (slow, more API spend)")
def test_agent_returns_answer_with_citations():
    """End-to-end agent run: prompt → tool calls → critic → final answer."""
    from backend.rag.agent import process_input_agent_sync

    summary, profiles = process_input_agent_sync(
        "What has the University of Aberdeen contributed to topic modeling in 2024?",
        paper=None,
        online=True,
        model="gpt-4o-mini",
    )

    assert isinstance(summary, str) and len(summary) > 200, (
        f"expected a non-trivial answer, got {summary!r}"
    )
    # Should include at least one inline [N] citation
    assert "[1]" in summary or "[2]" in summary, (
        f"expected inline citations, got: {summary[:300]}"
    )
    # The agent shouldn't be inventing institution profiles (those are pipeline-only).
    # profiles=[] is the expected shape from agent mode.
    assert profiles == []
