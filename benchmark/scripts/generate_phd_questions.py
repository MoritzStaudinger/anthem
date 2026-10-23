"""Generate PhD-tier methodology questions against the SciCo corpus.

The classifier-driven and template-driven questions in the legacy 97 cover
head-of-unit (overview) and postdoc (topic-level) personas. The third
tier — PhD-level methodology questions whose answers require reading the
paper body, not just the abstract — has to come from the full text itself.

This script scans <scico-root>/pymu_llm/*.md for papers with substantial
content, prompts an LLM (gpt-4o-mini by default — ~$0.0005 per paper) to
write 1-3 questions per paper plus the source excerpt that contains the
answer. Output rows have:

  persona     = "phd"
  depth       = "methodology"
  type        = "free_form"               (existing graders run unchanged)
  gold_paper_ids = ["<paper_id>"]
  gold_context = "<source paragraph from the paper>"
  expected_tools = ["drill_into_papers_fulltext"]
  classifier_gold = same shape as legacy rows, derived from affiliations JSON

Rows are appended to data/eval/scico_testset.jsonl. Re-runs replace any
prior phd-tier rows in that file (matched by id prefix).

Usage
-----
    set -a && . ./.env && set +a
    python scripts/generate_phd_questions.py
    # or override caps:
    python scripts/generate_phd_questions.py --max-questions 24 --per-paper 2
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


PHD_PROMPT = """\
You are writing exam-style questions for a PhD student who needs to
understand the METHODOLOGY of the paper below. Generate {n} questions
whose answers REQUIRE reading the paper body — abstract-level info is
not enough. Avoid factual lookup questions (year, author, venue);
focus on:
- algorithmic / architectural detail
- experimental setup / evaluation protocol
- hyperparameters or design choices
- comparisons made within the paper to prior work or baselines
- ablations or limitations the paper itself discusses

CONSTRAINTS:
- Each question MUST be a single sentence, 30 words or fewer. Be concise.
- No multi-part questions ("X, and also Y"); pick the most important
  angle and ask only that.

For each question, also extract a SHORT excerpt from the paper (1-3
sentences, verbatim) that contains the answer. This becomes the
gold_context for graders.

Paper id: {paper_id}
Paper title: {title}
Paper body (truncated):
---
{body}
---

Reply with a single JSON array, no prose, no markdown fences. Each
element:
{{
  "question": "<question text, ≤30 words>",
  "gold_context": "<excerpt from the paper, verbatim>"
}}
"""

# Hard cap enforced after the LLM responds — drops or truncates any
# question that's over budget regardless of what the prompt said.
QUESTION_MAX_WORDS = 30


def _safe_read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


import re as _re


def _year_from_paper_id(paper_id: str) -> Optional[int]:
    """Cheap year derivation from the paper_id alone (avoids needing the
    affiliations JSON for filtering). Modern: 2021.acl-long.1 → 2021.
    Legacy: P80-1001 → 1980 (yy>=60 → 19yy, else 20yy)."""
    m = _re.match(r"(\d{4})", paper_id)
    if m:
        return int(m.group(1))
    m = _re.match(r"^[A-Z](\d{2})", paper_id)
    if m:
        yy = int(m.group(1))
        return 1900 + yy if yy >= 60 else 2000 + yy
    return None


# *ACL main conference venues + Findings.
# Findings papers go through the same peer review as main, so we treat
# them as substantive research for question generation. We still exclude
# demos, tutorials, student research workshops, SemEval, WMT, and other
# workshop tracks.
_MAIN_FINDINGS_MODERN_RE = _re.compile(
    r"^\d{4}\.("
    r"acl-(long|short)|"
    r"emnlp-main|"
    r"naacl-(long|short)|"
    r"eacl-(long|short)|"
    r"aacl-main|"
    r"tacl|"
    r"cl-\d|"
    r"findings-(acl|emnlp|naacl|eacl|aacl)"
    r")\.",
    _re.IGNORECASE,
)
# Legacy IDs (pre-2020) — Findings didn't exist as a separate track yet.
# P=ACL, D=EMNLP, N=NAACL, E=EACL, J=CL journal, Q=TACL, H=HLT (≈ NAACL).
# Excludes: W=Workshops, S=SemEval, L=LREC, C=COLING, A=ANLP, K=CoNLL,
# X=TIPSTER, Y=PACLIC, R=RANLP, M=MUC.
_MAIN_FINDINGS_LEGACY_RE = _re.compile(r"^[PDNEJQH]\d{2}-", _re.IGNORECASE)


def _is_main_conference(paper_id: str) -> bool:
    """True iff paper_id is from an *ACL main conference or Findings track.

    Includes: ACL/EMNLP/NAACL/EACL/AACL main + short tracks, TACL, CL
    journal, Findings-*. Excludes demos, tutorials, workshops, SemEval,
    WMT, COLING, LREC.
    """
    return bool(
        _MAIN_FINDINGS_MODERN_RE.match(paper_id)
        or _MAIN_FINDINGS_LEGACY_RE.match(paper_id)
    )


def _enumerate_fulltext_papers(
    scico_root: Path,
    min_text_chars: int = 5000,
    md_dir: Optional[Path] = None,
    aff_dir: Optional[Path] = None,
    min_year: Optional[int] = None,
    max_year: Optional[int] = None,
    shuffle_seed: Optional[int] = None,
    main_only: bool = False,
) -> List[Dict[str, Any]]:
    """Return [{paper_id, text, affiliation_meta}] for every paper with a
    pymu_llm/*.md file whose body is substantial enough to author
    methodology questions against.

    Filters:
      - min_year / max_year       : inclusive bounds on publication year
                                     derived from paper_id (cheap, no I/O)
      - min_text_chars            : skip papers with shorter MD body
      - main_only                 : restrict to *ACL main conferences
                                     (ACL/EMNLP/NAACL/EACL/AACL/TACL/CL).
                                     Excludes findings, demos, workshops,
                                     SemEval, WMT, COLING, LREC, etc.
    Selection order is shuffled (seeded) when shuffle_seed is given;
    otherwise alphabetical by paper_id, which skews to legacy IDs first.
    """
    fulltext_dir = md_dir if md_dir is not None else (scico_root / "pymu_llm")
    if aff_dir is None:
        legacy = scico_root / "affiliations"
        aff_dir = legacy if legacy.is_dir() else scico_root
    if not fulltext_dir.is_dir():
        return []

    md_files = list(fulltext_dir.glob("*.md"))
    # Cheap paper_id filters BEFORE the expensive read+JSON-parse.
    if min_year is not None or max_year is not None or main_only:
        keep = []
        for md in md_files:
            pid = md.stem
            if main_only and not _is_main_conference(pid):
                continue
            y = _year_from_paper_id(pid)
            if y is None:
                # papers without a parseable year are dropped when any
                # year filter is set; otherwise kept
                if min_year is not None or max_year is not None:
                    continue
            else:
                if min_year is not None and y < min_year:
                    continue
                if max_year is not None and y > max_year:
                    continue
            keep.append(md)
        md_files = keep

    if shuffle_seed is not None:
        import random
        random.Random(shuffle_seed).shuffle(md_files)
    else:
        md_files.sort()

    out = []
    for md in md_files:
        text = _safe_read(md).strip()
        if len(text) < min_text_chars:
            continue
        paper_id = md.stem
        aff_path = aff_dir / f"{paper_id}.json"
        aff: Dict[str, Any] = {}
        if aff_path.exists():
            try:
                aff = json.loads(_safe_read(aff_path))
            except json.JSONDecodeError:
                aff = {}
        out.append({"paper_id": paper_id, "text": text, "aff": aff})
    return out


def _affiliations_from_aff(aff: Dict[str, Any]) -> List[str]:
    seen: set = set()
    out: List[str] = []
    for a in aff.get("authors") or []:
        for x in a.get("affiliations") or []:
            name = x.get("clean_institution_name") or x.get("raw_affiliation")
            if name and name not in seen:
                seen.add(name)
                out.append(name)
    return out


def _build_classifier_gold(paper_id: str, aff: Dict[str, Any]) -> Dict[str, Any]:
    """PhD-level questions classify as 'paper' in our scheme — answering
    a specific paper. uni1 is the first listed affiliation, year_start
    and year_end both equal to the paper year."""
    year = aff.get("year")
    affs = _affiliations_from_aff(aff)
    return {
        "question_type": "paper",
        "question_category": "phd_methodology",
        "uni1": affs[0] if affs else None,
        "uni2": None,
        "country": None,
        "topic": None,  # PhD methodology questions aren't topic-bound
        "year_start": int(year) if year is not None else None,
        "year_end": int(year) if year is not None else None,
        "years": None,
    }


def _call_llm_via_aqueduct(
    model: str,
    prompt: str,
    timeout: int = 300,
    retries: int = 3,
) -> str:
    """Direct HTTP call to TU Wien Aqueduct (OpenAI-compatible) using
    `requests` only — avoids dragging in llama_index. Includes retries
    on read-timeouts / connection errors so a single slow generation
    doesn't kill the whole PhD batch."""
    import os, time
    import requests
    api_key = os.environ.get("AQUEDUCT_API_KEY")
    if not api_key:
        raise RuntimeError("AQUEDUCT_API_KEY not set in environment")
    base_url = "https://aqueduct.ai.datalab.tuwien.ac.at"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            r = requests.post(
                f"{base_url}/v1/chat/completions",
                headers=headers, json=payload, timeout=timeout,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except (requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError) as e:
            last_err = e
            wait = min(60, 5 * (2 ** (attempt - 1)))   # 5s, 10s, 20s
            logging.warning(
                "Aqueduct timeout/conn error (attempt %d/%d), retrying in %ds: %s",
                attempt, retries, wait, type(e).__name__,
            )
            time.sleep(wait)
        except requests.HTTPError as e:
            # 5xx may be transient; 4xx is our fault — don't retry.
            if e.response is not None and 500 <= e.response.status_code < 600:
                last_err = e
                wait = 5 * attempt
                logging.warning("Aqueduct %d (attempt %d/%d), retry in %ds",
                                e.response.status_code, attempt, retries, wait)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Aqueduct exhausted retries: {last_err}")


def _call_llm_via_llamaindex(model: str, prompt: str) -> str:
    """Fallback path that uses the Comparer_flask `configure_llm` helper
    (covers gpt-4o-mini, claude, gemini, etc.). Only available when the
    full Comparer_flask env is around."""
    from llama_index.core import Settings   # type: ignore
    from backend.rag.rag_utils import configure_llm   # type: ignore
    configure_llm(model)
    return Settings.llm.complete(prompt).text


def _extract_first_json_object_local(text: str):
    """Fallback bracket-balanced JSON extraction so we don't depend on
    backend.rag.rag_utils when running in the minimal env."""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return json.loads(text[start : i + 1])
    raise ValueError("no balanced JSON object found")


def _call_llm_for_questions(
    paper_id: str,
    title: str,
    body: str,
    n_per_paper: int,
    model: str,
    body_chars: int,
) -> List[Dict[str, str]]:
    """Run the question-generation prompt; return [{question, gold_context}].

    Routes to Aqueduct for Qwen models (direct HTTP, no llama_index
    dependency); otherwise falls back to the Comparer_flask LLM router."""
    prompt = PHD_PROMPT.format(
        n=n_per_paper,
        paper_id=paper_id,
        title=title or paper_id,
        body=body[:body_chars],
    )
    if model.startswith("qwen-"):
        raw = _call_llm_via_aqueduct(model, prompt)
    else:
        raw = _call_llm_via_llamaindex(model, prompt)

    # Pick whichever JSON parser is available
    try:
        from backend.rag.rag_utils import extract_first_json_object   # type: ignore
    except Exception:
        extract_first_json_object = _extract_first_json_object_local

    raw = raw.strip()
    # The model occasionally wraps in ```json fences; strip them.
    if raw.startswith("```"):
        raw = raw.strip("`")
        # remove possible leading 'json'
        first_nl = raw.find("\n")
        if first_nl != -1:
            raw = raw[first_nl + 1 :]
    # Try direct JSON, then bracket-balanced fallback.
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = extract_first_json_object(raw)
        except Exception:
            logging.warning("unparseable PhD question batch for %s: %s", paper_id, raw[:200])
            return []
    if isinstance(parsed, dict):
        # If wrapped in {"questions": [...]} unwrap.
        parsed = parsed.get("questions", [parsed])
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        q = (item.get("question") or "").strip()
        gc = (item.get("gold_context") or "").strip()
        if q:
            out.append({"question": q, "gold_context": gc})
    return out


def _maybe_backup(path: Path) -> Optional[Path]:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = path.with_suffix(f".{stamp}.bak.jsonl")
    shutil.copy2(path, backup)
    return backup


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scico-root", default="data/datastructure",
                   help="SciCo dataset root (default: data/datastructure).")
    p.add_argument("--testset", default=str(ROOT / "data" / "eval" / "scico_testset.jsonl"),
                   help="JSONL to append PhD rows to (default: scico_testset.jsonl).")
    p.add_argument("--max-questions", type=int, default=24,
                   help="Cap total PhD-tier questions written (default: 24).")
    p.add_argument("--per-paper", type=int, default=2,
                   help="Questions per paper (default: 2).")
    p.add_argument("--md-dir", type=Path, default=None,
                   help="Directory of PyMu_LLM Markdown files. If unset, "
                        "defaults to <scico-root>/pymu_llm/ (legacy layout).")
    p.add_argument("--aff-dir", type=Path, default=None,
                   help="Directory of per-paper affiliation JSONs. If unset, "
                        "tries <scico-root>/affiliations/ then falls back to "
                        "<scico-root> itself (flat enriched layout).")
    p.add_argument("--min-text-chars", type=int, default=5000,
                   help="Skip papers whose Markdown body is shorter than this (default: 5000).")
    p.add_argument("--min-year", type=int, default=2020,
                   help="Only consider papers published in this year or later "
                        "(default: 2020 — focus on recent research; set 1950 "
                        "to include the full corpus).")
    p.add_argument("--max-year", type=int, default=None,
                   help="Only consider papers published in this year or earlier.")
    p.add_argument("--shuffle-seed", type=int, default=42,
                   help="Random seed for shuffling candidate papers "
                        "(default: 42; set to 0 for alphabetical order).")
    p.add_argument("--main-only", action="store_true",
                   help="Restrict to *ACL main conferences (ACL/EMNLP/NAACL/"
                        "EACL/AACL/TACL/CL). Excludes findings, workshops, "
                        "demos, SemEval, WMT, COLING, LREC. Recommended for "
                        "PhD-tier methodology questions where workshop/demo "
                        "papers are often too thin to support deep questions.")
    p.add_argument("--body-chars", type=int, default=12000,
                   help="Truncate each paper body to this many chars in the prompt "
                        "(default: 12000; keeps prompt under ~6k tokens).")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="LLM to use for question generation (default: gpt-4o-mini).")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip backing up scico_testset.jsonl before rewriting.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    scico_root = Path(args.scico_root)
    if not scico_root.is_dir():
        print(f"ERROR: scico_root not found: {scico_root}", file=sys.stderr)
        return 2

    candidates = _enumerate_fulltext_papers(
        scico_root, args.min_text_chars,
        md_dir=args.md_dir, aff_dir=args.aff_dir,
        min_year=args.min_year, max_year=args.max_year,
        shuffle_seed=args.shuffle_seed if args.shuffle_seed else None,
        main_only=args.main_only,
    )
    effective_md_dir = args.md_dir or (scico_root / "pymu_llm")
    if not candidates:
        print(
            f"WARNING: no papers in {effective_md_dir} with >={args.min_text_chars} chars of body. "
            "Nothing to generate.",
            file=sys.stderr,
        )
        return 0

    print(f"# {len(candidates)} candidate paper(s) with substantial full text")
    print(f"# generating up to {args.max_questions} PhD questions "
          f"({args.per_paper} per paper, model={args.model})")

    # Pre-load the existing testset + drop any prior PhD rows. We'll
    # write back after EACH successful paper so a crash mid-generation
    # doesn't lose work.
    testset_path = Path(args.testset)
    if not args.no_backup:
        backup = _maybe_backup(testset_path)
        if backup is not None:
            print(f"# backed up existing {testset_path.name} -> {backup.name}")
    existing = _load_jsonl(testset_path)
    existing_kept = [
        r for r in existing
        if not str(r.get("id", "")).startswith("scico-phd_methodology-")
    ]

    def _question_too_long(q: str) -> bool:
        return len(q.split()) > QUESTION_MAX_WORDS

    new_rows: List[Dict[str, Any]] = []
    dropped_too_long = 0
    for paper in candidates:
        if len(new_rows) >= args.max_questions:
            break
        paper_id = paper["paper_id"]
        aff = paper["aff"]
        title = aff.get("title") or paper_id
        remaining = args.max_questions - len(new_rows)
        n_to_ask = min(args.per_paper, remaining)
        try:
            items = _call_llm_for_questions(
                paper_id=paper_id,
                title=title,
                body=paper["text"],
                n_per_paper=n_to_ask,
                model=args.model,
                body_chars=args.body_chars,
            )
        except Exception as e:
            print(f"  {paper_id}: ERROR ({type(e).__name__}: {e}); skipping")
            continue

        kept_for_paper = 0
        for j, item in enumerate(items, start=1):
            if len(new_rows) >= args.max_questions:
                break
            qtext = (item.get("question") or "").strip()
            if not qtext or _question_too_long(qtext):
                dropped_too_long += 1
                continue
            new_rows.append({
                "id": f"scico-phd_methodology-{paper_id}-{j:02d}",
                "type": "free_form",
                "persona": "phd",
                "depth": "methodology",
                "query": qtext,
                "gold_summary": "",
                "gold_paper_ids": [paper_id],
                "gold_context": item.get("gold_context", ""),
                "expected_tools": ["drill_into_papers_fulltext"],
                "classifier_gold": _build_classifier_gold(paper_id, aff),
                "paper": {
                    "title": title,
                    "abstract": aff.get("abstract") or "",
                    "fulltext": "",
                    "doi": aff.get("doi"),
                    "year": aff.get("year"),
                    "authors": [a.get("name") for a in aff.get("authors") or [] if a.get("name")],
                },
            })
            kept_for_paper += 1
        print(f"  {paper_id}: +{kept_for_paper} (total {len(new_rows)})")

        # Incremental save: flush the testset to disk after each paper
        # so a later crash doesn't lose what's been generated so far.
        _write_jsonl(testset_path, existing_kept + new_rows)

    if not new_rows:
        print("# generator produced 0 rows — nothing to write.")
        return 0

    combined = existing_kept + new_rows
    if dropped_too_long:
        print(f"# dropped {dropped_too_long} question(s) over {QUESTION_MAX_WORDS} words")

    print(f"# wrote {len(new_rows)} new PhD rows; total in {testset_path}: {len(combined)}")
    persona_breakdown: Dict[str, int] = {}
    for r in combined:
        persona = r.get("persona", "?")
        persona_breakdown[persona] = persona_breakdown.get(persona, 0) + 1
    print(f"# persona breakdown: {persona_breakdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
