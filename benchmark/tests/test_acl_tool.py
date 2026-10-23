"""Deterministic tests for the ACL retrieval helpers and tool.

These don't load the actual MiniLM model or hit any persisted index — they
verify the no-index error path and the recall arithmetic only. Live tests
that exercise the full retrieval pipeline live under scripts/eval_acl_retrieval.py.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_search_local_acl_papers_returns_error_when_index_missing(tmp_path, monkeypatch):
    """No index on disk → tool must return a single-element error record,
    not raise. Lets the agent surface a useful message instead of crashing."""
    # Re-import to drop any cached index state from earlier tests.
    import backend.rag.local_acl_indexing as acl_idx
    importlib.reload(acl_idx)
    import backend.rag.agent_tools as agent_tools
    importlib.reload(agent_tools)

    nonexistent = tmp_path / "no_index_here"
    monkeypatch.setattr(agent_tools, "DEFAULT_ACL_PERSIST_DIR", str(nonexistent))

    out = agent_tools.search_local_acl_papers("any query", samples=5)
    assert isinstance(out, list) and len(out) == 1
    assert "error" in out[0]
    assert "no_index_here" in out[0]["error"] or "ACL index" in out[0]["error"]


def test_eval_recall_helper():
    """The recall helper from the eval CLI."""
    import importlib.util
    eval_path = ROOT / "scripts" / "eval_acl_retrieval.py"
    spec = importlib.util.spec_from_file_location("_eval_acl_retrieval", eval_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert mod._recall(["a", "b", "c"], ["a", "b"]) == 1.0
    assert mod._recall(["a", "x"], ["a", "b"]) == 0.5
    assert mod._recall([], ["a", "b"]) == 0.0
    # No gold → NaN convention
    import math
    assert math.isnan(mod._recall(["a"], []))
