"""Classifier-accuracy grader.

Calls backend.rag.rag_utils.generate_questions(question, llm_model=model)
and exact-matches its output dict against the row's `classifier_gold`
fields. Per-field accuracy aggregated across rows reproduces CIKM
Table 1: how well does each model parse a question into the structured
metadata the pipeline depends on?

This grader is *pipeline-mode-specific*: the agent doesn't use
generate_questions (it picks tools and arguments itself), so this metric
is only meaningful when judging the pipeline path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Fields the classifier emits → fields on the gold side.
# `country_b` is the second country for cross_country_method rows; gold-side
# is None for all other categories so the grader auto-skips the comparison
# (see _norm() — None gold means the field isn't graded).
FIELD_PAIRS = [
    ("type", "question_type"),
    ("university_name", "uni1"),
    ("comparison_university_name", "uni2"),
    ("country", "country"),
    ("country_b", "country_b"),
    ("topic", "topic"),
    ("year_start", "year_start"),
    ("year_end", "year_end"),
]


def _norm(v: Any) -> Any:
    """Empty strings and the literal "null" both count as None."""
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if not s or s.lower() == "null":
            return None
        return s
    return v


def _norm_year(v: Any) -> Optional[int]:
    v = _norm(v)
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def grade_classifier_row(
    question: str,
    gold: Dict[str, Any],
    model: str,
    paper: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Grade one classifier call against gold metadata.

    Returns:
        {
          "pred": {classifier output dict},
          "matches": {field: bool, ...},      # one entry per FIELD_PAIRS
          "graded_fields": {field: bool, ...} # whether each field is graded
                                              # (skipped when gold is None and
                                              # the field is uni2 / country)
        }

    The gold side is the row's `classifier_gold` dict. Fields where the
    gold is missing AND the pipeline isn't expected to populate them
    (uni2, country, topic-when-overview) are excluded from the count so
    we don't penalize the classifier for correctly leaving them empty.
    """
    from backend.rag.rag_utils import generate_questions

    try:
        pred = generate_questions(question, paper=paper, llm_model=model) or {}
    except Exception as e:
        logger.exception("classifier call failed for model %s", model)
        return {
            "pred": {},
            "matches": {},
            "graded_fields": {},
            "error": f"{type(e).__name__}: {e}",
        }

    matches: Dict[str, bool] = {}
    graded: Dict[str, bool] = {}

    for pred_field, gold_field in FIELD_PAIRS:
        if pred_field in ("year_start", "year_end"):
            p = _norm_year(pred.get(pred_field))
            g = _norm_year(gold.get(gold_field))
        else:
            p = _norm(pred.get(pred_field))
            g = _norm(gold.get(gold_field))

        # Skip fields where the row genuinely has no gold and the
        # classifier wouldn't be expected to invent one. Mirrors the
        # SKIP_IF_NULL set in legacy/question_evaluation.py.
        gold_missing = g is None
        skippable = pred_field in {"comparison_university_name", "country"}
        if gold_missing and skippable:
            graded[pred_field] = False
            continue

        graded[pred_field] = True
        matches[pred_field] = (p == g)

    return {"pred": pred, "matches": matches, "graded_fields": graded, "error": None}
