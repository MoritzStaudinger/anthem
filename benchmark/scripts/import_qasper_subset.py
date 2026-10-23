"""Pull a curated subset of QASPER (Allen AI, EMNLP 2021) questions and
align them with the SciCo per-paper corpus, producing JSONL rows in the
SciCo schema.

Why we do this
--------------
QASPER provides human-written PhD-level methodology questions over NLP
papers WITH human-annotated answer spans + evidence paragraphs — exactly
the ground truth our LLM-generated phd_methodology rows lack. By
importing a 20-25 question subset we get:

    1. Independent ground truth (questions and answers both human-written)
       so the eval isn't critiqued for circularity (Qwen generates,
       GPT-4 judges Qwen's questions).
    2. Calibration against published QASPER metrics — reviewers can
       compare our agent's token-F1 to literature numbers.
    3. A clean story: 'we extend QASPER (single-paper QA) with cross-
       document scientometric queries — the latter is our contribution'.

What this script does
---------------------
1. Loads `allenai/qasper` from HuggingFace.
2. Joins QASPER papers to the SciCo affiliations corpus by:
     a) arxiv_id (primary) — both QASPER and ACL Anthology often have it
     b) normalized-title fallback (lowercased, stopword-stripped)
3. For each matched paper, keeps up to N questions (default 2), preferring
   extractive answers over yes/no and unanswerable.
4. Samples up to --target-count questions overall, stratified by paper-
   year so we don't draw 20 questions from the same 3 papers.
5. Maps each (question, answer, evidence) triple to the SciCo row schema
   with `category = "phd_qasper"`. Adds `gold_answer` and `gold_evidence`
   fields (NEW — the SciCo eval needs the small `eval_scico.py` extension
   to read these for token-F1 scoring).
6. Writes to JSONL that can be concatenated onto the main candidates file.

License
-------
QASPER is CC-BY-4.0; redistribute with citation:
  Dasigi et al. 'A Dataset of Information-Seeking Questions and Answers
  Anchored in Research Papers', NAACL 2021.

Usage
-----
    python scripts/import_qasper_subset.py \\
        --affiliations-dir /share/.../ror_parsed_affiliations_90_with_topics \\
        --target-count 25 \\
        --out data/eval/qasper_subset.jsonl

Then merge with existing candidates:
    cat data/eval/scico_candidates.jsonl data/eval/qasper_subset.jsonl \\
        > data/eval/scico_candidates_with_qasper.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("import_qasper")


# -------------------------- title normalization --------------------------

_STOP = {"a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "at",
         "by", "with", "via", "using", "from"}


def _normalize_title(t: str) -> str:
    """Lowercase + strip non-alphanumeric + drop very short / stopword tokens.
    Used for fuzzy title matching between QASPER and SciCo corpora — exact
    string match would miss case differences and punctuation drift."""
    if not t:
        return ""
    t = re.sub(r"[^a-zA-Z0-9 ]", " ", t.lower())
    toks = [tok for tok in t.split() if len(tok) > 2 and tok not in _STOP]
    return " ".join(toks)


# -------------------------- SciCo corpus loading --------------------------

def _load_scico_index(aff_dir: Path) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, dict]]:
    """Build two indexes over the SciCo corpus + a meta lookup:
       arxiv_id  → paper_id
       norm_title → paper_id
       paper_id  → {title, year, abstract}
    """
    arxiv_idx: Dict[str, str] = {}
    title_idx: Dict[str, str] = {}
    meta: Dict[str, dict] = {}
    t0 = time.time()
    n = 0
    for fp in aff_dir.glob("*.json"):
        try:
            d = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            continue
        pid = d.get("paper_id") or fp.stem
        title = d.get("title") or ""
        year = d.get("year")
        meta[pid] = {"title": title, "year": year, "abstract": d.get("abstract") or ""}
        # ACL records carry arxiv ids only sometimes — most under `arxiv_id`
        # or inside an `external_ids` block. We check both.
        arxiv = None
        for src in (d.get("arxiv_id"), (d.get("external_ids") or {}).get("arxiv"),
                    (d.get("external_ids") or {}).get("ArXiv")):
            if src:
                arxiv = str(src).strip().lower()
                break
        if arxiv:
            arxiv_idx[arxiv] = pid
        nt = _normalize_title(title)
        if nt:
            title_idx[nt] = pid
        n += 1
    logger.info("loaded %d SciCo papers in %.1fs (arxiv-indexed: %d, title-indexed: %d)",
                n, time.time() - t0, len(arxiv_idx), len(title_idx))
    return arxiv_idx, title_idx, meta


def _resolve_qasper_paper_to_scico(q_paper: dict, arxiv_idx: Dict[str, str],
                                    title_idx: Dict[str, str]) -> Optional[str]:
    """Try multiple strategies to map a QASPER record to a SciCo paper_id."""
    # 1. arxiv id (preferred)
    qa_arxiv = (q_paper.get("arxiv_id") or "").strip().lower()
    if qa_arxiv and qa_arxiv in arxiv_idx:
        return arxiv_idx[qa_arxiv]
    # 2. normalized title (fallback — fuzzy by lowercase + stopword stripping)
    qa_title = q_paper.get("title") or ""
    nt = _normalize_title(qa_title)
    if nt and nt in title_idx:
        return title_idx[nt]
    return None


# -------------------------- QASPER answer extraction --------------------------

def _extract_answer(answers_struct: list) -> Optional[Tuple[str, List[str], str]]:
    """Pull a usable (answer_string, evidence_paragraphs, answer_kind) from
    QASPER's annotated answers. QASPER often has multiple annotators per
    question — we take the first answer that is extractive or free-form.

    Returns None when the question is unanswerable or has only yes/no
    answers (we drop those — they don't fit the retrieval framing).
    """
    if not answers_struct:
        return None
    for entry in answers_struct:
        ans = entry.get("answer") or entry  # depending on the HF schema variant
        unanswerable = ans.get("unanswerable")
        if unanswerable:
            continue
        # Extractive spans win
        extractive = ans.get("extractive_spans") or []
        if extractive:
            answer_str = " | ".join(s.strip() for s in extractive if s and s.strip())
            evidence = ans.get("evidence") or []
            return answer_str, [e for e in evidence if e], "extractive"
        # Free-form (abstractive) is second choice
        free_form = (ans.get("free_form_answer") or "").strip()
        if free_form:
            evidence = ans.get("evidence") or []
            return free_form, [e for e in evidence if e], "free_form"
        # yes_no answers: dropped (not retrieval-shaped)
    return None


# -------------------------- main --------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--affiliations-dir", type=Path, required=True,
                   help="SciCo per-paper affiliations dir, used to match "
                        "QASPER papers into the SciCo corpus.")
    p.add_argument("--target-count", type=int, default=25,
                   help="How many QASPER questions to keep after stratified "
                        "sampling. Default 25.")
    p.add_argument("--questions-per-paper", type=int, default=2,
                   help="Max questions kept per QASPER paper, before "
                        "stratified sampling. Default 2.")
    p.add_argument("--qasper-split", default="train",
                   choices=("train", "validation", "test"),
                   help="QASPER split to draw from. Default train (largest). "
                        "Switch to test if you want strict held-out eval, "
                        "but its smaller paper count reduces matching hits.")
    p.add_argument("--out", type=Path, required=True,
                   help="Output JSONL path (SciCo schema). Concat onto your "
                        "main candidates file when ready.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("`datasets` not installed. `pip install datasets`")

    logger.info("loading QASPER split=%s ...", args.qasper_split)
    qasper = load_dataset("allenai/qasper", split=args.qasper_split)
    logger.info("QASPER: %d papers in split", len(qasper))

    arxiv_idx, title_idx, scico_meta = _load_scico_index(args.affiliations_dir)

    # Match QASPER papers → SciCo paper_ids
    matched: List[dict] = []
    for q_paper in qasper:
        scico_pid = _resolve_qasper_paper_to_scico(q_paper, arxiv_idx, title_idx)
        if not scico_pid:
            continue
        matched.append({"scico_pid": scico_pid, "q_paper": q_paper})
    logger.info("QASPER papers matched to SciCo: %d / %d (%.1f%%)",
                len(matched), len(qasper), 100 * len(matched) / max(1, len(qasper)))
    if not matched:
        raise SystemExit(
            "No QASPER papers matched the SciCo corpus. Check arxiv_id / "
            "title fields on the SciCo side — try `--verbose` to see what "
            "shape the indexes have."
        )

    # Extract questions per matched paper
    rows: List[dict] = []
    rng = random.Random(args.seed)
    rng.shuffle(matched)

    for m in matched:
        scico_pid = m["scico_pid"]
        qa_block = m["q_paper"].get("qas") or {}
        # The HF QASPER schema is column-oriented; reshape to a list of question dicts
        questions = qa_block.get("question") or []
        answers_struct_list = qa_block.get("answers") or []
        if not questions:
            continue
        # Build per-question tuples
        per_q: List[Tuple[str, list]] = []
        for i, q in enumerate(questions):
            answers_for_q = []
            if i < len(answers_struct_list):
                # HF stores as {"answer": [list of dicts]}
                entry = answers_struct_list[i]
                if isinstance(entry, dict) and "answer" in entry:
                    answers_for_q = entry["answer"] or []
                elif isinstance(entry, list):
                    answers_for_q = entry
            per_q.append((q, answers_for_q))
        rng.shuffle(per_q)
        kept = 0
        for q_text, answers_struct in per_q:
            if kept >= args.questions_per_paper:
                break
            extracted = _extract_answer(answers_struct)
            if not extracted:
                continue
            answer_str, evidence_paras, kind = extracted
            rows.append({
                "scico_pid": scico_pid,
                "question": q_text,
                "answer": answer_str,
                "evidence": evidence_paras,
                "answer_kind": kind,
                "qasper_paper_id": m["q_paper"].get("id"),
                "year": scico_meta.get(scico_pid, {}).get("year"),
                "title": scico_meta.get(scico_pid, {}).get("title"),
            })
            kept += 1

    logger.info("extracted %d question candidates from %d matched papers",
                len(rows), len({r['scico_pid'] for r in rows}))

    # Stratified sample to target count: by paper-year bucket so we don't
    # draw all questions from one era.
    by_decade: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        y = r.get("year") or 0
        decade = f"{(y // 10) * 10}s" if y else "unknown"
        by_decade[decade].append(r)
    logger.info("rows by decade: %s", {k: len(v) for k, v in by_decade.items()})

    if len(rows) <= args.target_count:
        sampled = rows
    else:
        # Round-robin draw across decades
        sampled: List[dict] = []
        decade_lists = {k: list(v) for k, v in by_decade.items()}
        for v in decade_lists.values():
            rng.shuffle(v)
        decades = list(decade_lists.keys())
        rng.shuffle(decades)
        while len(sampled) < args.target_count and any(decade_lists.values()):
            for d in decades:
                if not decade_lists[d]:
                    continue
                sampled.append(decade_lists[d].pop())
                if len(sampled) >= args.target_count:
                    break

    logger.info("after stratified sample: %d rows", len(sampled))

    # Convert to SciCo schema
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(sampled, start=1):
            row = {
                "id": f"scico-phd_qasper-{i:03d}",
                "type": "free_form",
                "persona": "phd",
                "depth": "deep",
                "query": r["question"],
                "gold_summary": "",
                "gold_paper_ids": [r["scico_pid"]],
                "gold_answer": r["answer"],
                "gold_evidence": r["evidence"],
                "gold_answer_kind": r["answer_kind"],  # extractive | free_form
                "classifier_gold": {
                    "question_type": "paper",
                    "question_category": "phd_qasper",
                    "uni1": None, "uni1_ror": None,
                    "uni2": None, "uni2_ror": None,
                    "country": None, "country_b": None,
                    "topic": None,
                    "year_start": None, "year_end": None,
                    "years": None,
                },
                "paper": {
                    "paper_id": r["scico_pid"],
                    "title": r.get("title", ""),
                    "year": r.get("year"),
                },
                "_provenance": {
                    "source": "qasper",
                    "qasper_paper_id": r["qasper_paper_id"],
                    "license": "CC-BY-4.0",
                    "citation": "Dasigi et al. NAACL 2021",
                },
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("wrote %d rows to %s", len(sampled), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
