"""Convert legacy/data/questions.json (97 CIKM-style questions) into our
JSONL test-set format consumed by scripts/eval_scico.py.

Each legacy row becomes one JSONL row with:
  - id, type=free_form (so the existing graders run unchanged)
  - query, optional gold_summary / gold_paper_ids (TODO until you add them)
  - classifier_gold: the metadata fields used by the classifier-accuracy
    grader to reproduce the CIKM Table 1 (question_type, uni1, uni2,
    country, topic, year_start, year_end, years)
  - paper: when the row is `paper_vs_field`, the paper1_* fields packed
    into the {title, abstract, fulltext, doi, year, ...} structure that
    process_input_agent / process_input expects

The legacy `question_type` is normalized to OUR classifier output scheme
via TYPE_MAP (e.g. overview_university -> single_university) so the
classifier-accuracy grader can exact-match against the pipeline's output.

Usage
-----
    python scripts/convert_legacy_questions.py
    python scripts/convert_legacy_questions.py --out data/eval/scico_testset.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Legacy question_type -> our classifier output type. Mirrors the table in
# legacy/question_evaluation.py::TYPE_MAP, adjusted to the labels emitted
# by backend.rag.rag_utils.generate_questions.
TYPE_MAP: Dict[str, str] = {
    "overview_university": "single_university",
    "university_comparison": "university",
    "university_country_comparison": "university",
    "university_to_university_comparison": "university_university",
    "paper_vs_field": "comparison",
    "field_evolution": "overview",
}

# Legacy question_type -> evaluation persona. head = breadth/overview;
# postdoc = topic/field-level synthesis; phd = methodology-level detail
# (generated separately by scripts/generate_phd_questions.py).
PERSONA_MAP: Dict[str, str] = {
    "overview_university": "head",
    "university_comparison": "head",
    "university_country_comparison": "head",
    "university_to_university_comparison": "head",
    "paper_vs_field": "postdoc",
    "field_evolution": "postdoc",
}

# Default per-category caps. Total = 14*4 (head) + 10*2 (postdoc) = 76,
# leaving 24 slots for the PhD generator within the 100-question budget.
DEFAULT_HEAD_CAP = 14
DEFAULT_POSTDOC_CAP = 10
DEFAULT_RANDOM_SEED = 42


def _coerce_int(x: Any) -> Optional[int]:
    if x in (None, "", "null"):
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _clean(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        return s or None
    return v


def _build_paper_dict(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Pack paper1_* fields into a {title, abstract, fulltext, ...} dict.
    Returns None when no paper info is present (most non-paper_vs_field rows)."""
    title = _clean(row.get("paper1_title"))
    abstract = _clean(row.get("paper1_abstract"))
    if not (title or abstract):
        return None
    return {
        "title": title or "",
        "abstract": abstract or "",
        "fulltext": "",  # not in the legacy schema; eval can ignore
        "doi": _clean(row.get("paper1_doi")),
        "year": _coerce_int(row.get("paper1_year")),
        "authors": _clean(row.get("paper1_authors")),
        "affiliations": _clean(row.get("paper1_affiliations")),
        "countries": _clean(row.get("paper1_countries")),
        "keywords": _clean(row.get("paper1_keywords")),
        "topic": _clean(row.get("paper1_topic")),
    }


def _convert_row(idx: int, row: Dict[str, Any]) -> Dict[str, Any]:
    legacy_type = row.get("question_type")
    normalized_type = TYPE_MAP.get(legacy_type, legacy_type)
    persona = PERSONA_MAP.get(legacy_type, "head")

    classifier_gold = {
        "question_type": normalized_type,        # our classifier scheme
        "question_category": legacy_type,        # granular generator category
        "uni1": _clean(row.get("uni1")),
        "uni2": _clean(row.get("uni2")),
        "country": _clean(row.get("country1")),
        "topic": _clean(row.get("topic")),
        "year_start": _coerce_int(row.get("year_start")),
        "year_end": _coerce_int(row.get("year_end")),
        "years": _coerce_int(row.get("years")),
    }

    out: Dict[str, Any] = {
        "id": f"scico-{legacy_type}-{idx:03d}",
        "type": "free_form",
        "persona": persona,
        "depth": "overview" if persona == "head" else "topical",
        "query": row.get("question", ""),
        # Empty placeholders — fill these in when you have a reference
        # answer or known-good paper_ids for retrieval grading.
        "gold_summary": "",
        "gold_paper_ids": [],
        "classifier_gold": classifier_gold,
    }
    paper = _build_paper_dict(row)
    if paper is not None:
        out["paper"] = paper
    return out


def _maybe_backup(path: Path) -> None:
    if not path.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(f".{stamp}.bak.jsonl")
    shutil.copy2(path, backup)
    logging.info("backed up existing %s -> %s", path.name, backup.name)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--source",
        default=str(ROOT / "legacy" / "data" / "questions.json"),
        help="Legacy CIKM-style questions.json (default: legacy/data/questions.json).",
    )
    p.add_argument(
        "--out",
        default=str(ROOT / "data" / "eval" / "scico_testset.jsonl"),
        help="Output JSONL path (default: data/eval/scico_testset.jsonl).",
    )
    p.add_argument(
        "--head-cap", type=int, default=DEFAULT_HEAD_CAP,
        help=(f"Max rows per head-tier question_type (default: "
              f"{DEFAULT_HEAD_CAP}). Default leaves room for the PhD "
              f"generator within a 100-question total."),
    )
    p.add_argument(
        "--postdoc-cap", type=int, default=DEFAULT_POSTDOC_CAP,
        help=f"Max rows per postdoc-tier question_type (default: {DEFAULT_POSTDOC_CAP}).",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_RANDOM_SEED,
        help=f"Random seed for subsampling (default: {DEFAULT_RANDOM_SEED}).",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Skip subsampling — emit every legacy row (97 in total).",
    )
    p.add_argument(
        "--no-backup", action="store_true",
        help="Skip backing up the existing --out file before overwrite.",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    src = Path(args.source)
    if not src.exists():
        print(f"ERROR: source not found: {src}", file=sys.stderr)
        return 2

    rows = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        print(f"ERROR: expected a JSON list in {src}", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_backup:
        _maybe_backup(out_path)

    # Group by legacy question_type and subsample within each.
    import random
    rng = random.Random(args.seed)
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(row.get("question_type") or "unknown", []).append(row)

    selected_per_type: Dict[str, List[Dict[str, Any]]] = {}
    for legacy_type, group in by_type.items():
        persona = PERSONA_MAP.get(legacy_type, "head")
        if args.all:
            cap = len(group)
        elif persona == "head":
            cap = args.head_cap
        else:
            cap = args.postdoc_cap
        if cap >= len(group):
            selected_per_type[legacy_type] = list(group)
        else:
            # Deterministic shuffle so re-runs with the same seed pick
            # the same rows; first cap items after shuffle.
            shuffled = list(group)
            rng.shuffle(shuffled)
            selected_per_type[legacy_type] = shuffled[:cap]

    # Re-numbering by category to keep ids stable across reruns.
    written: List[Dict[str, Any]] = []
    type_counts: Dict[str, int] = {}
    persona_counts: Dict[str, int] = {}
    for legacy_type in sorted(selected_per_type):
        for j, row in enumerate(selected_per_type[legacy_type], start=1):
            converted = _convert_row(j, row)
            written.append(converted)
            type_counts[legacy_type] = type_counts.get(legacy_type, 0) + 1
            persona_counts[converted["persona"]] = persona_counts.get(converted["persona"], 0) + 1

    with out_path.open("w", encoding="utf-8") as fh:
        for r in written:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"# wrote {len(written)} questions to {out_path}")
    for k in sorted(type_counts):
        print(f"  {k}: {type_counts[k]}")
    print(f"# persona breakdown: {persona_counts}")
    print(f"# (PhD-tier rows are added separately by scripts/generate_phd_questions.py)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
