"""LLM-paraphrase the templated head/postdoc questions to remove the robotic
'What did X contribute to T...' phrasing while preserving every entity that
the classifier_gold / runtime-gold logic depends on (institution names,
country codes, topic strings, years).

The PhD-methodology rows are already LLM-generated (Qwen) and naturally
phrased — they're skipped. The paper_vs_field template is also paraphrased
since its question text is rigid even though the row carries a full paper.

Approach:
    1. For each eligible row, derive the set of constraint strings that
       MUST appear verbatim in any paraphrase (institutions, countries,
       topic, both year values).
    2. Ask the LLM for N variants in JSON.
    3. Keep only variants that contain every constraint substring
       (case-insensitive). The first surviving variant replaces `query`.
    4. If no variant survives, the original template stays — degrades to
       a no-op for that row rather than silently breaking gold.

Usage:
    python scripts/paraphrase_scico_questions.py \\
        --in  data/eval/scico_candidates.jsonl \\
        --out data/eval/scico_candidates.jsonl

Env:
    AQUEDUCT_API_KEY — required (free key from TU Wien DataLab).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("paraphrase_scico")

# Categories whose query text comes from a fixed template and benefits
# from rewriting. PhD-methodology already varies naturally.
PARAPHRASEABLE = {
    "overview_university",
    "university_comparison",
    "university_country_comparison",
    "university_to_university_comparison",
    "cross_country_method",
    "field_evolution",
    "paper_vs_field",
}

AQUEDUCT_URL = "https://aqueduct.ai.datalab.tuwien.ac.at/v1/chat/completions"
DEFAULT_MODEL = "qwen-3.5-397b"


# -------------------------- Aqueduct client --------------------------

def _call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    timeout: float = 120.0,
    retries: int = 3,
) -> Optional[str]:
    api_key = os.environ.get("AQUEDUCT_API_KEY")
    if not api_key:
        raise SystemExit("AQUEDUCT_API_KEY not set. `set -a && . ./.env && set +a`")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.7,  # variety, but not chaotic
        "max_tokens": 600,
    }
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(AQUEDUCT_URL, headers=headers, json=body, timeout=timeout)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            logger.warning("Aqueduct returned %d on attempt %d", r.status_code, attempt)
        except requests.exceptions.RequestException as e:
            logger.warning("Aqueduct request failed (attempt %d): %s", attempt, e)
        time.sleep(2 * attempt)
    return None


# -------------------------- constraint extraction --------------------------

def _constraints_for(row: Dict[str, Any]) -> List[str]:
    """Return strings that MUST appear in any paraphrase (case-insensitive).
    Built from classifier_gold + paper info; drives the verbatim check."""
    cg = row.get("classifier_gold") or {}
    out: List[str] = []
    for k in ("uni1", "uni2", "country", "country_b", "topic"):
        v = cg.get(k)
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    for k in ("year_start", "year_end"):
        v = cg.get(k)
        if isinstance(v, int):
            out.append(str(v))
    # Also keep the paper's title for paper_vs_field rows so the rewrite
    # doesn't fabricate a different reference.
    paper = row.get("paper") or {}
    if paper.get("title") and row.get("classifier_gold", {}).get("question_category") == "paper_vs_field":
        # Don't require the full title in the question (often it's
        # implicit via 'this paper'). The constraint check below
        # still ensures topic stays intact.
        pass
    return out


def _all_present(text: str, constraints: List[str]) -> bool:
    tl = text.lower()
    return all(c.lower() in tl for c in constraints)


# -------------------------- the paraphrase call --------------------------

SYSTEM_PROMPT = (
    "You rewrite scientometric questions to sound natural and human, like "
    "a researcher actually asking a colleague. You preserve every named "
    "entity, country code, topic phrase, and year EXACTLY as given. You "
    "produce strictly valid JSON output, no prose."
)


def _build_user_prompt(query: str, constraints: List[str]) -> str:
    constraint_lines = "\n".join(f"  - {c!r}" for c in constraints)
    return (
        f"Original question:\n{query}\n\n"
        f"Constraints — every one of these strings MUST appear verbatim "
        f"(case-insensitive) in each rewrite:\n{constraint_lines}\n\n"
        f"Rewrite the question in 3 different natural phrasings. Vary "
        f"sentence structure, opening words, and verb choice. Do not add "
        f"any new entities, methods, or dates. Do not invent things. "
        f"Output strictly JSON of the form:\n"
        f'{{"variants": ["...", "...", "..."]}}'
    )


_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    # Strip codefence wrappers
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fallback: pull the first JSON object out of the string
    m = _JSON_BLOCK_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return None


def paraphrase_one(
    row: Dict[str, Any],
    *,
    model: str,
) -> Optional[str]:
    """Return a new query string for `row`, or None if no valid variant."""
    query = row.get("query", "")
    constraints = _constraints_for(row)
    user = _build_user_prompt(query, constraints)
    raw = _call_llm(SYSTEM_PROMPT, user, model=model)
    if not raw:
        return None
    obj = _extract_json(raw)
    if not obj:
        logger.debug("could not parse JSON from LLM: %s", raw[:120])
        return None
    variants = obj.get("variants") or []
    if not isinstance(variants, list):
        return None
    for v in variants:
        if not isinstance(v, str):
            continue
        if _all_present(v, constraints):
            return v.strip()
    return None


# -------------------------- main --------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", dest="out_path", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=None,
                   help="Only paraphrase the first N eligible rows (smoke test).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print before/after but don't write the output.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    with args.in_path.open() as f:
        rows = [json.loads(l) for l in f if l.strip()]
    logger.info("loaded %d rows from %s", len(rows), args.in_path)

    paraphrased = 0
    skipped_phd = 0
    failed = 0
    eligible_count = 0
    for i, row in enumerate(rows):
        cat = (row.get("classifier_gold") or {}).get("question_category")
        if cat not in PARAPHRASEABLE:
            skipped_phd += 1
            continue
        eligible_count += 1
        if args.limit is not None and eligible_count > args.limit:
            continue
        new_query = paraphrase_one(row, model=args.model)
        if new_query is None:
            failed += 1
            logger.info("[%d/%d] %s — no valid variant, keeping original",
                        i + 1, len(rows), cat)
            continue
        original = row["query"]
        row["query"] = new_query
        # Stash the original under a private field so users can revert/audit
        # in the candidates CSV without having to re-run generation.
        row["_original_query"] = original
        paraphrased += 1
        logger.info("[%d/%d] %s\n  ORIG: %s\n  NEW : %s",
                    i + 1, len(rows), cat, original, new_query)

    print(f"\nparaphrased {paraphrased} / failed {failed} / phd-skipped {skipped_phd}")

    if args.dry_run:
        print("--dry-run: no output written.")
        return 0

    # Atomic-ish write: backup → write to temp → rename
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.in_path.resolve() == args.out_path.resolve():
        backup = args.out_path.with_suffix(args.out_path.suffix + ".prerewrite.bak")
        shutil.copy2(args.in_path, backup)
        print(f"# backed up original to {backup}")

    tmp = args.out_path.with_suffix(args.out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(args.out_path)
    print(f"# wrote {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
