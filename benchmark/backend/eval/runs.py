"""Wrap a SciCo agent run to capture wall-clock and token counts.

We attach a TokenCountingHandler to llama-index's global callback manager
for the duration of one agent call, then read the tallies back out. Each
call resets the counter so per-row stats are clean.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler

from backend.eval.metrics import estimate_cost_usd


def run_agent_with_telemetry(
    prompt: str,
    model: str,
    tools=None,
    system_prompt: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run process_input_agent_sync once and return (summary, telemetry).

    telemetry keys:
        wall_seconds, prompt_tokens, completion_tokens, total_tokens,
        cost_usd (None if pricing unknown), error (None on success).
    """
    from backend.rag.agent import process_input_agent_sync

    counter = TokenCountingHandler()
    saved_cb_manager = Settings.callback_manager
    Settings.callback_manager = CallbackManager([counter])

    error = None
    summary = ""
    t0 = time.perf_counter()
    try:
        summary, _profiles = process_input_agent_sync(
            prompt=prompt,
            paper=None,
            online=True,
            model=model,
            tools=tools,
            system_prompt=system_prompt,
        )
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    finally:
        wall = time.perf_counter() - t0
        Settings.callback_manager = saved_cb_manager

    prompt_tokens = int(counter.prompt_llm_token_count)
    completion_tokens = int(counter.completion_llm_token_count)
    total_tokens = prompt_tokens + completion_tokens
    cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)

    return (summary or ""), {
        "wall_seconds": wall,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost,
        "error": error,
    }
