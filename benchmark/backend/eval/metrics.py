"""Metric helpers for the SciCo eval harness.

Stateless, deterministic functions only. Heavy NLG metrics (BERTScore)
load their model lazily on first use. The eval runner wires these into
the per-row dispatch.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def precision_recall_mrr(
    retrieved: List[str], expected: Iterable[str], k: Optional[int] = None
) -> Dict[str, float]:
    """Standard rank-aware metrics for an ordered retrieval list.

    - precision@k: fraction of the top-k retrieved that are expected.
    - recall@k:    fraction of expected items that appear in the top-k.
    - f1@k:        harmonic mean.
    - mrr:         1 / rank-of-first-expected (1-indexed); 0 if none.

    NaN is returned when expected is empty (no gold to grade against).
    """
    expected_set = set(expected or [])
    if not expected_set:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "mrr": float("nan"),
        }
    top = retrieved[:k] if k else retrieved
    hits = [pid for pid in top if pid in expected_set]
    precision = len(hits) / len(top) if top else 0.0
    recall = len(hits) / len(expected_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    mrr = 0.0
    for i, pid in enumerate(top, start=1):
        if pid in expected_set:
            mrr = 1 / i
            break
    return {"precision": precision, "recall": recall, "f1": f1, "mrr": mrr}


def institution_list_metrics(
    sys_affs: List[str], expected: List[str]
) -> Dict[str, float]:
    """Case-insensitive substring P/R/F1 between two institution lists.

    A retrieved institution counts as a hit if it overlaps any expected
    institution (substring either direction). Same logic for expected
    being matched. Asymmetric but matches how analysts read messy
    affiliation strings ("Alibaba Group (China)" ↔ "Alibaba Group").
    """
    if not expected:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "system_size": len(sys_affs),
        }
    sys_lower = [a.lower() for a in sys_affs]
    exp_lower = [e.lower() for e in expected]

    def overlaps(a: str, b: str) -> bool:
        return a in b or b in a

    matched_exp = sum(1 for e in exp_lower if any(overlaps(e, a) for a in sys_lower))
    matched_sys = sum(1 for a in sys_lower if any(overlaps(a, e) for e in exp_lower))
    recall = matched_exp / len(exp_lower)
    precision = (matched_sys / len(sys_lower)) if sys_lower else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "system_size": len(sys_affs),
    }


# ---------------------------------------------------------------------------
# Faithfulness metrics
# ---------------------------------------------------------------------------

_REFERENCES_HEADER_RE = re.compile(
    r"^\s*#{0,6}\s*references\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_DOI_URL_RE = re.compile(r"https?://(?:dx\.)?doi\.org/\S+", re.IGNORECASE)
_OPENALEX_URL_RE = re.compile(r"https?://openalex\.org/\S+", re.IGNORECASE)
_ACL_URL_RE = re.compile(r"https?://aclanthology\.org/\S+", re.IGNORECASE)
_ACL_PAPER_ID_RE = re.compile(r"\b(20\d{2}\.[a-z\-]+\.\d+)\b", re.IGNORECASE)
_INLINE_MARKER_RE = re.compile(r"\[(\d+)\]")
_REF_LINE_RE = re.compile(r"^\s*\[(\d+)\]\s+(.+?)\s*$")


def _normalize_url(url: str) -> str:
    return url.strip().rstrip(").,;").lower()


def parse_references(answer: str) -> List[Dict[str, Any]]:
    """Extract structured reference entries from the answer's references
    section. Returns [{number, text, url, paper_id}], where url is any
    DOI / OpenAlex / ACL anthology URL found in the entry, and paper_id
    any ACL Anthology id (e.g. 2021.acl-long.236) detected in the entry.
    """
    matches = list(_REFERENCES_HEADER_RE.finditer(answer))
    if not matches:
        return []
    refs_text = answer[matches[-1].end():]
    out: List[Dict[str, Any]] = []
    for line in refs_text.split("\n"):
        m = _REF_LINE_RE.match(line)
        if not m:
            continue
        number, body = m.group(1), m.group(2)
        url = None
        for r in (_DOI_URL_RE, _ACL_URL_RE, _OPENALEX_URL_RE):
            um = r.search(body)
            if um:
                url = _normalize_url(um.group(0))
                break
        pm = _ACL_PAPER_ID_RE.search(body)
        out.append({
            "number": int(number),
            "text": body,
            "url": url,
            "paper_id": pm.group(1) if pm else None,
        })
    return out


# ACL Anthology DOI prefixes — stripping these from a DOI URL yields
# the anthology paper_id, which is what compute_gold_paper_ids() and
# parse_references() both speak. Used by _evidence_keys when records
# carry only a DOI (e.g. agent evidence buffer — see
# backend/rag/agent_tools.py:_record_evidence, no paper_id field).
_ACL_DOI_PREFIXES = (
    "10.18653/v1/",
    "10.3115/",
    "https://doi.org/10.18653/v1/",
    "https://doi.org/10.3115/",
    "http://doi.org/10.18653/v1/",
    "http://doi.org/10.3115/",
    "doi.org/10.18653/v1/",
    "doi.org/10.3115/",
)


def _paper_id_from_doi(s: str) -> Optional[str]:
    """Strip ACL DOI prefix → anthology paper_id. Also handles
    aclanthology.org URLs. Returns None if nothing recognisable."""
    if not s:
        return None
    sl = s.strip().lower()
    for pre in _ACL_DOI_PREFIXES:
        if sl.startswith(pre):
            return s.strip()[len(pre):].rstrip("/")
    if "aclanthology.org/" in sl:
        tail = s.strip().rstrip("/").rsplit("aclanthology.org/", 1)[-1]
        return tail or None
    pm = _ACL_PAPER_ID_RE.search(s)
    return pm.group(1) if pm else None


def _evidence_keys(evidence: List[Dict[str, Any]]) -> Tuple[set, set]:
    urls: set = set()
    paper_ids: set = set()
    for rec in evidence or []:
        for fld in ("url", "DOI", "doi"):
            v = rec.get(fld)
            if v:
                urls.add(_normalize_url(str(v)))
        pid = rec.get("paper_id")
        if not pid:
            # Agent evidence records carry DOI but no paper_id. Derive
            # paper_id from the DOI / url so phantom matching connects.
            for fld in ("DOI", "doi", "url"):
                pid = _paper_id_from_doi(str(rec.get(fld) or ""))
                if pid:
                    break
            if not pid:
                pm = _ACL_PAPER_ID_RE.search(str(rec.get("title") or ""))
                if pm:
                    pid = pm.group(1)
        if pid:
            paper_ids.add(str(pid))
    return urls, paper_ids


def phantom_citation_rate(
    answer: str, evidence: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Fraction of the answer's references that don't match any paper in
    the agent's evidence buffer. Match by URL or ACL paper_id. Lower is
    better (zero phantoms = every cited paper actually came from a tool
    call this turn)."""
    refs = parse_references(answer)
    if not refs:
        return {
            "phantom_rate": float("nan"),
            "phantom_count": 0,
            "total_refs": 0,
            "inline_marker_count": len(_INLINE_MARKER_RE.findall(answer)),
        }
    urls, paper_ids = _evidence_keys(evidence)
    phantoms = 0
    for r in refs:
        url_hit = r["url"] in urls if r.get("url") else False
        pid_hit = r["paper_id"] in paper_ids if r.get("paper_id") else False
        if not (url_hit or pid_hit):
            phantoms += 1
    return {
        "phantom_rate": phantoms / len(refs),
        "phantom_count": phantoms,
        "total_refs": len(refs),
        "inline_marker_count": len(_INLINE_MARKER_RE.findall(answer)),
    }


# ---------------------------------------------------------------------------
# Affiliation correctness against SciCo ground_truth.csv
# ---------------------------------------------------------------------------

def load_ground_truth_affiliations(
    csv_path: str | Path,
) -> Dict[str, set]:
    """Build {paper_id -> set of clean institution names} from the SciCo
    ground_truth.csv. Each row's authors_normalized_json is a JSON list of
    authors with affiliations[].clean_institution_name.
    """
    out: Dict[str, set] = {}
    p = Path(csv_path)
    if not p.exists():
        logger.warning("ground_truth.csv not found at %s", p)
        return out
    with p.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            paper_id = (row.get("anthology_id") or "").strip()
            if not paper_id:
                continue
            try:
                authors = json.loads(row.get("authors_normalized_json") or "[]")
            except json.JSONDecodeError:
                continue
            insts: set = set()
            for author in authors or []:
                for aff in author.get("affiliations") or []:
                    name = aff.get("clean_institution_name") or aff.get("raw_affiliation")
                    if name:
                        insts.add(str(name).strip().lower())
            if insts:
                out[paper_id] = insts
    logger.info("loaded ground-truth affiliations for %d papers", len(out))
    return out


def affiliation_correctness(
    retrieved_records: List[Dict[str, Any]],
    gold_lookup: Dict[str, set],
) -> Dict[str, Any]:
    """For each retrieved paper present in the gold lookup, compute F1
    of indexed clean institution names vs gold clean institution names
    (case-insensitive set match). Mean across papers.

    Tells you whether the agent's evidence pipeline is faithful to the
    SciCo gold standard — i.e. is the system citing the *right*
    institutions per paper?
    """
    if not gold_lookup:
        return {"mean_f1": float("nan"), "graded": 0, "skipped": len(retrieved_records)}
    f1s: List[float] = []
    skipped = 0
    for rec in retrieved_records:
        pid = rec.get("paper_id")
        if not pid or pid not in gold_lookup:
            skipped += 1
            continue
        sys = {str(a).strip().lower() for a in (rec.get("affiliations") or []) if a}
        gold = gold_lookup[pid]
        if not sys and not gold:
            f1s.append(1.0)
            continue
        if not sys or not gold:
            f1s.append(0.0)
            continue
        tp = len(sys & gold)
        precision = tp / len(sys)
        recall = tp / len(gold)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        f1s.append(f1)
    return {
        "mean_f1": (sum(f1s) / len(f1s)) if f1s else float("nan"),
        "graded": len(f1s),
        "skipped": skipped,
    }


# ---------------------------------------------------------------------------
# NLG metrics
# ---------------------------------------------------------------------------

_ROUGE_SCORER = None


def rouge_l_f1(prediction: str, reference: str) -> float:
    """ROUGE-L F1 between system prediction and a single reference.

    Uses LCS over tokens. Cheap deterministic surface-overlap metric;
    pessimistic on long-form rewordings, but the field expects it.
    """
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        from rouge_score import rouge_scorer  # type: ignore[import-untyped]
        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    if not prediction or not reference:
        return 0.0
    return _ROUGE_SCORER.score(reference, prediction)["rougeL"].fmeasure


def rouge_l_vs_evidence(prediction: str, evidence: List[Dict[str, Any]]) -> float:
    """ROUGE-L F1 between the prediction and the concatenated text of
    retrieved evidence (titles + abstracts). Approximates "did the system
    paraphrase retrieved content, or did it invent things". No human
    reference needed; cheap to compute on every row.

    Returns 0.0 when prediction or evidence text is empty.
    """
    if not prediction or not evidence:
        return 0.0
    parts: List[str] = []
    for rec in evidence:
        title = rec.get("title") or ""
        body = rec.get("abstract") or rec.get("text") or ""
        if title or body:
            parts.append(f"{title}\n{body}".strip())
    if not parts:
        return 0.0
    return rouge_l_f1(prediction, " ".join(parts))


def _rank(values: List[float]) -> List[float]:
    """Average ranks for Spearman (ties get the mean of their positions)."""
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation; returns nan if degenerate."""
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    rx, ry = _rank(xs), _rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(len(rx)))
    dx = (sum((rx[i] - mx) ** 2 for i in range(len(rx)))) ** 0.5
    dy = (sum((ry[i] - my) ** 2 for i in range(len(ry)))) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def inter_judge_agreement(
    per_judge_scores: Dict[str, List[Optional[float]]],
) -> Dict[str, Any]:
    """Pairwise Spearman ρ across LLM judges for a single rubric axis.

    per_judge_scores: judge_model_name -> list of scores (one per question).
        None / NaN entries are skipped pairwise.

    Returns {pairs: N, mean_rho: X, per_pair: {(a,b): rho, ...}}.
    Use Spearman because the 1-5 Likert scores are ordinal, not interval.
    """
    import itertools
    judges = sorted(per_judge_scores.keys())
    if len(judges) < 2:
        return {"pairs": 0, "mean_rho": float("nan"), "per_pair": {}}
    pairs: Dict[Tuple[str, str], float] = {}
    rhos: List[float] = []
    for a, b in itertools.combinations(judges, 2):
        sa = per_judge_scores[a]
        sb = per_judge_scores[b]
        paired = [
            (float(x), float(y))
            for x, y in zip(sa, sb)
            if isinstance(x, (int, float)) and isinstance(y, (int, float))
        ]
        if len(paired) < 2:
            continue
        rho = _spearman([p[0] for p in paired], [p[1] for p in paired])
        pairs[(a, b)] = rho
        if not (isinstance(rho, float) and rho != rho):  # not NaN
            rhos.append(rho)
    return {
        "pairs": len(pairs),
        "mean_rho": (sum(rhos) / len(rhos)) if rhos else float("nan"),
        "per_pair": {f"{a}|{b}": v for (a, b), v in pairs.items()},
    }


_BERT_SCORE_LOADED = False


def bertscore_f1(
    prediction: str, reference: str, model_type: str = "roberta-large"
) -> float:
    """BERTScore F1 against a single reference, using the requested
    underlying transformer. Loads the model once per process (multi-MB).

    Returns 0.0 when either side is empty. Reads the average F1 across
    tokens (bert_score returns a tensor per item).
    """
    if not prediction or not reference:
        return 0.0
    from bert_score import score  # type: ignore[import-untyped]

    global _BERT_SCORE_LOADED
    _BERT_SCORE_LOADED = True
    _, _, F1 = score(
        [prediction], [reference], model_type=model_type, lang="en", verbose=False,
    )
    return float(F1[0].item())


# ---------------------------------------------------------------------------
# Pricing — rough USD per 1K tokens (input, output). Update as needed.
# ---------------------------------------------------------------------------

MODEL_PRICING = {
    # OpenAI
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o": (0.0025, 0.01),
    # Anthropic
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-opus-4-7": (0.015, 0.075),
    # Google
    "gemini-2.5-pro": (0.00125, 0.01),
    "gemini-2.5-flash": (0.000075, 0.0003),
    # Aqueduct (TU Wien gateway) — pricing not public, leave None to
    # report tokens but no $-amount.
    "qwen-3.5-397b": None,
    "qwen-3.6-35b": None,
}


def estimate_cost_usd(
    model: str, prompt_tokens: int, completion_tokens: int
) -> Optional[float]:
    rates = MODEL_PRICING.get(model)
    if rates is None:
        return None
    in_rate, out_rate = rates
    return (prompt_tokens / 1000.0) * in_rate + (completion_tokens / 1000.0) * out_rate
