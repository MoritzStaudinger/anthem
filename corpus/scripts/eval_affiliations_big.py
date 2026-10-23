#!/usr/bin/env python3
"""
Evaluate author-affiliation extraction vs ground truth across MANY run folders.

Supports two scoring tracks:

1) ROR-based scoring
   - Prediction sources provide per-paper author/affiliation objects with affiliations[].ror_id
   - Current implementation supports: JSON dir (<paper_id>.json), JSONL dir (*.jsonl with records containing authors),
     Markdown dir (*.md containing embedded JSON), and (optionally) TEI-ROR as JSON dir.

2) Raw affiliation scoring
   - Prediction sources provide raw affiliation strings per author (affiliations[].raw_affiliation or affiliations[].affiliation)
   - Supports:
       * JSON dir (<paper_id>.json) (same schema as your model JSON)
       * TEI XML dir (<paper_id>.tei.xml or <paper_id>.xml) (Grobid TEI)
       * JSONL dir (*.jsonl)
       * Markdown dir (*.md with embedded JSON)

Author matching rule:
- match by last name: last whitespace-split token of author name (or <surname> in TEI) must match after normalization.

Outputs:
- Precision, Recall, Macro F1, Micro F1 for:
    * single_author (author–affiliation pair level)
    * all_authors_exact (paper-level exact match: all authors+affiliations correct; for RAW this uses fuzzy matching)

- Coverage:
    * ROR coverage rate: (#affiliations with ror_id) / (#affiliations total seen)
    * RAW coverage rate: (#affiliations with raw string) / (#affiliations total seen)

Ground truth expects columns:
- anthology_id
- authors_normalized_json (for ROR scoring; GT affiliations with ror_id)
- authors_raw_json        (for raw affiliation scoring; GT affiliations with affiliation string)

Run folders expected under --runs_root:
RAW sources:
  gemini_jsonl, gemini_md, gemini_tei, gpt_jsonl, gpt_md, gpt_tei, qwen_tei
ROR sources:
  gemini_jsonl_ror, gemini_md_ror, gemini_tei_ror, gpt_jsonl_ror, gpt_md_ror, gpt_tei_ror, qwen_tei_ror

Usage:
  python eval_affiliations_multi.py \
    --gt_file /path/to/ground_truth.tsv \
    --runs_root /path/to/experiments_root \
    --sep $'\\t' \
    --raw_threshold 0.80 \
    --out_json /path/to/report.json

Notes / assumptions:
- For *_tei (RAW) folders: expects TEI XML; parsed via Grobid-ish TEI heuristics (teiHeader/biblStruct).
- For *_ror folders: assumed to be JSON dir with <paper_id>.json and schema: {"authors":[...]}.
- For JSONL and MD: this script uses heuristics to recover a JSON structure containing "authors".
  If your JSONL/MD schema differs, adjust load_jsonl_dir() / load_md_dir() accordingly.
"""

import argparse
import json
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


# ----------------------------
# Multi-source evaluation config
# ----------------------------

RAW_SOURCES = [
    ("../data/acl_affiliations_json/gemini_jsonl", "json_dir"),
    ("../data/acl_affiliations_json/gemini_md",    "json_dir"),
    ("../data/acl_affiliations_json/gemini_tei",   "json_dir"),
    ("../data/gemini_affiliations",   "json_dir"),        
    ("../data/acl_affiliations_json/gpt_jsonl",    "json_dir"),
    ("../data/acl_affiliations_json/gpt_all",    "json_dir"),
    ("../data/acl_affiliations_json/gpt_md",       "json_dir"),
    ("../data/acl_affiliations_json/gpt_tei",      "json_dir"),
    ("../data/acl_affiliations_json/qwen_tei",     "json_dir"),
        ("../data/acl_affiliations_json/qwen_all",     "json_dir"),
]

ROR_SOURCES = [
    ("../data/acl_affiliations_json/gemini_jsonl_ror", "json_dir"),
    ("../data/acl_affiliations_json/gemini_md_ror",    "json_dir"),
    ("../data/acl_affiliations_json/gemini_tei_ror",   "json_dir"),
    ("../data/ror_parsed_affiliations_90",   "json_dir"),
    ("../data/acl_affiliations_json/gpt_jsonl_ror",    "json_dir"),
    ("../data/acl_affiliations_json/gpt_all_ror",    "json_dir"),
    ("../data/acl_affiliations_json/gpt_md_ror",       "json_dir"),
    ("../data/acl_affiliations_json/gpt_tei_ror",      "json_dir"),
    ("../data/acl_affiliations_json/qwen_tei_ror",     "json_dir"),
            ("../data/acl_affiliations_json/qwen_all_ror",     "json_dir"),
]


# ----------------------------
# Normalization utilities
# ----------------------------

def ascii_fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return s.encode("ascii", "ignore").decode("ascii")


def norm_lastname(name: str) -> str:
    if not name:
        return ""
    name = ascii_fold(name).strip().lower()
    name = re.sub(r"\s+", " ", name)
    parts = name.split(" ")
    return parts[-1] if parts else ""


def norm_affil_text(s: str) -> str:
    """
    Canonical raw affiliation string (for exact comparisons and tokenization).
    - ASCII fold, lowercase
    - punctuation -> space
    - collapse whitespace
    """
    if not s:
        return ""
    s = ascii_fold(s).lower()
    s = s.replace("\n", " ")
    s = re.sub(r"[^\w\s]", " ", s)     # punctuation -> space
    s = re.sub(r"\s+", " ", s).strip()
    return s


def affil_tokens(s: str) -> Set[str]:
    s = norm_affil_text(s)
    if not s:
        return set()
    toks = set(s.split(" "))
    # optional light cleanup: remove ultra-common tokens that add noise
    stop = {"of", "the", "and", "for", "in", "at", "on", "a", "an", "dept", "department", "school", "university"}
    return {t for t in toks if t and t not in stop}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def safe_json_loads(x: str):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, (dict, list)):
        return x
    x = str(x).strip()
    if not x:
        return None
    return json.loads(x)


# ----------------------------
# Metrics helpers
# ----------------------------

@dataclass(frozen=True)
class PairCounts:
    tp: int
    fp: int
    fn: int

    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return (2 * p * r / (p + r)) if (p + r) else 0.0


def union_counts(counts: List[PairCounts]) -> PairCounts:
    return PairCounts(
        tp=sum(c.tp for c in counts),
        fp=sum(c.fp for c in counts),
        fn=sum(c.fn for c in counts),
    )


# ----------------------------
# Ground truth loading
# ----------------------------

def load_ground_truth(
    gt_path: str,
    sep: str,
    id_col: str = "anthology_id",
    authors_norm_col: str = "authors_normalized_json",
    authors_raw_col: str = "authors_raw_json",
) -> Tuple[Dict[str, Dict[str, Set[str]]], Dict[str, Dict[str, List[str]]]]:
    """
    Returns:
      gt_ror[paper_id][lastname] = set(ror_ids)
      gt_raw[paper_id][lastname] = list(raw_affiliation_strings)
    """
    df = pd.read_csv(gt_path, sep=sep, dtype=str, keep_default_na=False)

    missing = [c for c in [id_col, authors_norm_col, authors_raw_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in GT file: {missing}. Found: {list(df.columns)}")

    gt_ror: Dict[str, Dict[str, Set[str]]] = {}
    gt_raw: Dict[str, Dict[str, List[str]]] = {}

    for _, row in df.iterrows():
        pid = str(row[id_col]).strip()
        if not pid:
            continue

        # ROR GT
        authors_norm = safe_json_loads(row[authors_norm_col])
        per_paper_ror: Dict[str, Set[str]] = {}
        if authors_norm:
            for a in authors_norm:
                ln = norm_lastname(a.get("name", ""))
                if not ln:
                    continue
                for aff in (a.get("affiliations", []) or []):
                    ror = aff.get("ror_id")
                    if ror:
                        per_paper_ror.setdefault(ln, set()).add(str(ror).strip())
        gt_ror[pid] = per_paper_ror

        # Raw GT
        authors_raw = safe_json_loads(row[authors_raw_col])
        per_paper_raw: Dict[str, List[str]] = {}
        if authors_raw:
            for a in authors_raw:
                ln = norm_lastname(a.get("name", ""))
                if not ln:
                    continue
                for aff in (a.get("affiliations", []) or []):
                    s = aff.get("affiliation") or ""
                    s = str(s).strip()
                    if s:
                        per_paper_raw.setdefault(ln, []).append(s)
        gt_raw[pid] = per_paper_raw

    return gt_ror, gt_raw


def load_gt_paper_ids(gt_path: str, sep: str, id_col: str) -> Set[str]:
    """
    Load only the set of paper IDs from the GT file.
    Used to restrict evaluation and file reading to labeled papers.
    """
    df = pd.read_csv(gt_path, sep=sep, dtype=str, usecols=[id_col], keep_default_na=False)
    ids = set(df[id_col].astype(str).str.strip())
    ids.discard("")
    return ids


# ----------------------------
# JSON dir loaders (per-paper JSON)
# ----------------------------

def load_ror_from_json_dir(
    json_dir: str,
    allowed_ids: Set[str],
) -> Tuple[Dict[str, Dict[str, Set[str]]], int, int]:
    """
    Load ROR predictions from a directory of JSON files (<paper_id>.json) with schema:
      {"authors":[{"name":"...", "affiliations":[{"ror_id":"..."}]}]}

    Restricts to paper IDs in allowed_ids.

    Hardened against:
      - authors being None / non-list
      - author entries being None / non-dict
      - affiliations being None / non-list
      - affiliation entries being None / non-dict
      - ror_id being non-string / whitespace
    """
    ror_map: Dict[str, Dict[str, Set[str]]] = {}
    n_aff_total = 0
    n_aff_with_ror = 0

    for fn in os.listdir(json_dir):
        if not fn.endswith(".json"):
            continue

        pid = fn[:-5]
        if pid not in allowed_ids:
            continue

        path = os.path.join(json_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            # bad/unreadable JSON -> treat as missing predictions
            continue

        authors = obj.get("authors") if isinstance(obj, dict) else None
        if not isinstance(authors, list):
            authors = []

        per_paper_ror: Dict[str, Set[str]] = {}

        for a in authors:
            if not isinstance(a, dict):
                continue

            ln = norm_lastname(a.get("name", ""))
            if not ln:
                continue

            affs = a.get("affiliations") or []
            if not isinstance(affs, list):
                continue

            for aff in affs:
                if not isinstance(aff, dict):
                    # handles None, strings, etc.
                    continue

                n_aff_total += 1
                ror = aff.get("ror_id")
                if ror is None:
                    continue

                ror = str(ror).strip()
                if ror:
                    n_aff_with_ror += 1
                    per_paper_ror.setdefault(ln, set()).add(ror)

        ror_map[pid] = per_paper_ror

    return ror_map, n_aff_total, n_aff_with_ror


def load_ror_and_raw_from_json_dir(
    json_dir: str,
    allowed_ids: Set[str],
) -> Tuple[
    Dict[str, Dict[str, Set[str]]],
    Dict[str, Dict[str, List[str]]],
    int, int, int, int
]:
    """
    Load JSON files (<paper_id>.json) with schema:
      {
        "authors": [
          {"name": "...",
           "affiliations": [{"ror_id": "...", "raw_affiliation": "..."} ...]
          }, ...
        ]
      }

    Restricts to paper IDs in allowed_ids.

    Hardened against:
      - authors being None / non-list
      - author entries being None / non-dict
      - affiliations being None / non-list
      - affiliation entries being None / non-dict (e.g., null)
      - raw_affiliation missing; falls back to aff["affiliation"] if present

    Returns:
      ror_map[paper_id][lastname] = set(ror_ids)
      raw_map[paper_id][lastname] = list(raw_affiliation_strings)
      n_aff_total, n_aff_with_ror, n_aff_total_raw, n_aff_with_raw
    """
    ror_map: Dict[str, Dict[str, Set[str]]] = {}
    raw_map: Dict[str, Dict[str, List[str]]] = {}

    n_aff_total = 0
    n_aff_with_ror = 0
    n_aff_total_raw = 0
    n_aff_with_raw = 0

    for fn in os.listdir(json_dir):
        if not fn.endswith(".json"):
            continue

        pid = fn[:-5]
        if pid not in allowed_ids:
            continue

        path = os.path.join(json_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            # Bad/unreadable JSON -> treat as missing predictions
            continue

        authors = obj.get("authors") if isinstance(obj, dict) else None
        if not isinstance(authors, list):
            authors = []

        per_ror: Dict[str, Set[str]] = {}
        per_raw: Dict[str, List[str]] = {}

        for a in authors:
            if not isinstance(a, dict):
                continue

            ln = norm_lastname(a.get("name", ""))
            if not ln:
                continue

            affs = a.get("affiliations") or []
            if not isinstance(affs, list):
                continue

            for aff in affs:
                if not isinstance(aff, dict):
                    # handles None entries and other malformed data
                    continue

                # ROR coverage
                n_aff_total += 1
                ror = aff.get("ror_id")
                if ror is not None:
                    ror = str(ror).strip()
                    if ror:
                        n_aff_with_ror += 1
                        per_ror.setdefault(ln, set()).add(ror)

                # RAW coverage
                n_aff_total_raw += 1
                raw = aff.get("raw_affiliation")
                if raw is None or (isinstance(raw, str) and not raw.strip()):
                    # fallback key some pipelines use
                    raw = aff.get("affiliation")

                raw = "" if raw is None else str(raw).strip()
                if raw:
                    n_aff_with_raw += 1
                    per_raw.setdefault(ln, []).append(raw)

        ror_map[pid] = per_ror
        raw_map[pid] = per_raw

    return ror_map, raw_map, n_aff_total, n_aff_with_ror, n_aff_total_raw, n_aff_with_raw



# ----------------------------
# JSONL / MD loaders (heuristics)
# ----------------------------

def load_jsonl_dir(
    jsonl_dir: str,
    allowed_ids: Set[str],
) -> Tuple[
    Dict[str, Dict[str, List[str]]],  # raw_map
    Dict[str, Dict[str, Set[str]]],   # ror_map
    int, int, int, int                # raw_total, raw_with_value, ror_total, ror_with_value
]:
    """
    Reads *.jsonl files in a directory.

    Supports common JSONL per-line shapes:
      - Each line is a per-paper object containing "authors"
      - Paper id in one of: anthology_id | paper_id | id
      - Or paper id inferred from filename stem (<paper_id>.jsonl)

    Authors expected under: obj["authors"] (list of dicts)
    Affiliations under: author["affiliations"] (list of dicts)
      - RAW string under: raw_affiliation OR affiliation
      - ROR under: ror_id
    """
    raw_map: Dict[str, Dict[str, List[str]]] = {}
    ror_map: Dict[str, Dict[str, Set[str]]] = {}

    raw_total = raw_with = 0
    ror_total = ror_with = 0

    def get_pid(obj: dict, fallback_pid: str) -> str:
        for k in ("anthology_id", "paper_id", "id"):
            v = obj.get(k)
            if v and str(v).strip():
                return str(v).strip()
        return fallback_pid

    for fn in os.listdir(jsonl_dir):
        if not fn.endswith(".jsonl"):
            continue
        fallback_pid = fn[:-6]
        path = os.path.join(jsonl_dir, fn)

        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue

                    if not isinstance(obj, dict):
                        continue

                    pid = get_pid(obj, fallback_pid)
                    if pid not in allowed_ids:
                        continue

                    authors = obj.get("authors") or []
                    if not isinstance(authors, list):
                        continue

                    per_raw = raw_map.setdefault(pid, {})
                    per_ror = ror_map.setdefault(pid, {})

                    for a in authors:
                        if not isinstance(a, dict):
                            continue
                        ln = norm_lastname(a.get("name", ""))
                        if not ln:
                            continue

                        affs = a.get("affiliations") or []
                        if not isinstance(affs, list):
                            continue

                        for aff in affs:
                            if not isinstance(aff, dict):
                                continue

                            # RAW
                            raw_total += 1
                            raw = aff.get("raw_affiliation") or aff.get("affiliation") or ""
                            raw = str(raw).strip()
                            if raw:
                                raw_with += 1
                                per_raw.setdefault(ln, []).append(raw)

                            # ROR
                            ror_total += 1
                            ror = aff.get("ror_id")
                            if ror:
                                ror_with += 1
                                per_ror.setdefault(ln, set()).add(str(ror).strip())
        except Exception:
            continue

    return raw_map, ror_map, raw_total, raw_with, ror_total, ror_with


def _extract_first_json_blob(text: str) -> Optional[object]:
    """
    Heuristic: find the first JSON object/array in text.
    Works well if markdown contains a fenced ```json ... ``` block or inline JSON.
    """
    if not text:
        return None

    m = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    m = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not m:
        return None

    blob = m.group(1).strip()
    try:
        return json.loads(blob)
    except Exception:
        return None


def load_md_dir(
    md_dir: str,
    allowed_ids: Set[str],
) -> Tuple[
    Dict[str, Dict[str, List[str]]],  # raw_map
    Dict[str, Dict[str, Set[str]]],   # ror_map
    int, int, int, int
]:
    """
    Reads *.md files in a directory. Paper id inferred from filename stem.
    Expects markdown to contain a JSON blob describing authors/affiliations.
    """
    raw_map: Dict[str, Dict[str, List[str]]] = {}
    ror_map: Dict[str, Dict[str, Set[str]]] = {}
    raw_total = raw_with = 0
    ror_total = ror_with = 0

    for fn in os.listdir(md_dir):
        if not fn.endswith(".md"):
            continue

        pid = fn[:-3]
        if pid not in allowed_ids:
            continue

        path = os.path.join(md_dir, fn)
        try:
            text = open(path, "r", encoding="utf-8").read()
        except Exception:
            continue

        obj = _extract_first_json_blob(text)
        if obj is None:
            continue

        # Normalize to dict with "authors"
        if isinstance(obj, list):
            obj = {"authors": obj}
        if not isinstance(obj, dict):
            continue

        authors = obj.get("authors") or []
        if not isinstance(authors, list):
            continue

        per_raw = raw_map.setdefault(pid, {})
        per_ror = ror_map.setdefault(pid, {})

        for a in authors:
            if not isinstance(a, dict):
                continue
            ln = norm_lastname(a.get("name", ""))
            if not ln:
                continue

            affs = a.get("affiliations") or []
            if not isinstance(affs, list):
                continue

            for aff in affs:
                if not isinstance(aff, dict):
                    continue

                # RAW
                raw_total += 1
                raw = aff.get("raw_affiliation") or aff.get("affiliation") or ""
                raw = str(raw).strip()
                if raw:
                    raw_with += 1
                    per_raw.setdefault(ln, []).append(raw)

                # ROR
                ror_total += 1
                ror = aff.get("ror_id")
                if ror:
                    ror_with += 1
                    per_ror.setdefault(ln, set()).add(str(ror).strip())

    return raw_map, ror_map, raw_total, raw_with, ror_total, ror_with


# ----------------------------
# TEI loading (Grobid-ish) for RAW
# ----------------------------

def extract_tei_author_affils(tei_path: str) -> Dict[str, List[str]]:
    """
    Extract tei_raw[lastname] = list(raw_affiliation_strings) from Grobid TEI.

    - Robust to default TEI namespace.
    - Treats parse errors / empty files as missing predictions (returns {}).
    - Extracts affiliation primarily from orgName nodes under affiliation.
      Falls back to full affiliation text if orgName is absent.
    """
    try:
        if os.path.getsize(tei_path) == 0:
            return {}
        tree = ET.parse(tei_path)
        root = tree.getroot()
    except (ET.ParseError, OSError):
        return {}

    NS = {"tei": "http://www.tei-c.org/ns/1.0"}
    per_paper_raw: Dict[str, List[str]] = {}

    author_nodes = root.findall(
        ".//tei:teiHeader/tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author",
        namespaces=NS,
    )
    if not author_nodes:
        author_nodes = root.findall(".//tei:author", namespaces=NS)

    for author in author_nodes:
        surname_el = author.find(".//tei:persName/tei:surname", namespaces=NS)
        if surname_el is not None and (surname_el.text or "").strip():
            ln = norm_lastname(surname_el.text)
        else:
            pers_el = author.find(".//tei:persName", namespaces=NS)
            raw_name = ""
            if pers_el is not None:
                raw_name = "".join(pers_el.itertext()).strip()
            ln = norm_lastname(raw_name)

        if not ln:
            continue

        aff_texts: List[str] = []
        aff_nodes = author.findall("./tei:affiliation", namespaces=NS)

        for aff in aff_nodes:
            orgs = [
                " ".join("".join(o.itertext()).split()).strip()
                for o in aff.findall(".//tei:orgName", namespaces=NS)
            ]
            orgs = [o for o in orgs if o]

            if orgs:
                aff_texts.append(" ; ".join(orgs))
            else:
                txt = " ".join("".join(aff.itertext()).split()).strip()
                if txt:
                    aff_texts.append(txt)

        if aff_texts:
            per_paper_raw.setdefault(ln, []).extend(aff_texts)

    return per_paper_raw


def load_tei_dir(
    tei_dir: str,
    allowed_ids: Set[str],
) -> Tuple[Dict[str, Dict[str, List[str]]], int, int]:
    """
    Load Grobid TEI files, restricted to allowed_ids.

    Returns:
      tei_raw[paper_id][lastname] = list(raw_affiliation_strings)
      n_aff_total_raw, n_aff_with_raw
    """
    tei_raw: Dict[str, Dict[str, List[str]]] = {}
    n_aff_total_raw = 0
    n_aff_with_raw = 0

    for fn in os.listdir(tei_dir):
        if not (fn.endswith(".xml") or fn.endswith(".tei.xml")):
            continue

        pid = fn
        if pid.endswith(".tei.xml"):
            pid = pid[:-8]
        elif pid.endswith(".xml"):
            pid = pid[:-4]

        if pid not in allowed_ids:
            continue

        path = os.path.join(tei_dir, fn)
        per_paper = extract_tei_author_affils(path)

        for _, affs in per_paper.items():
            for s in affs:
                n_aff_total_raw += 1
                if s and str(s).strip():
                    n_aff_with_raw += 1

        tei_raw[pid] = per_paper

    return tei_raw, n_aff_total_raw, n_aff_with_raw


# ----------------------------
# Scoring: ROR exact set match
# ----------------------------

def ror_pair_counts_for_paper(gt_map: Dict[str, Set[str]], pred_map: Dict[str, Set[str]], pid: str) -> PairCounts:
    gt_pairs: Set[Tuple[str, str, str]] = set()
    pred_pairs: Set[Tuple[str, str, str]] = set()

    for ln, rors in (gt_map or {}).items():
        for r in rors:
            gt_pairs.add((pid, ln, r))

    for ln, rors in (pred_map or {}).items():
        for r in rors:
            pred_pairs.add((pid, ln, r))

    tp = len(gt_pairs & pred_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)
    return PairCounts(tp=tp, fp=fp, fn=fn)


def ror_exact_match_paper(gt_map: Dict[str, Set[str]], pred_map: Dict[str, Set[str]]) -> bool:
    gt_map = gt_map or {}
    pred_map = pred_map or {}
    if set(gt_map.keys()) != set(pred_map.keys()):
        return False
    for ln in gt_map.keys():
        if set(gt_map.get(ln, set())) != set(pred_map.get(ln, set())):
            return False
    return True


# ----------------------------
# Scoring: Raw affiliation fuzzy matching
# ----------------------------

def greedy_match_counts(gt_affs: List[str], pred_affs: List[str], threshold: float) -> PairCounts:
    """
    Match predicted raw affiliations to GT raw affiliations using greedy maximum Jaccard token similarity.
    Returns counts for a single author (lastname) within one paper.
    """
    gt = [a for a in gt_affs if a and a.strip()]
    pr = [a for a in pred_affs if a and a.strip()]

    gt_tokens = [affil_tokens(a) for a in gt]
    pr_tokens = [affil_tokens(a) for a in pr]

    used_gt = set()
    used_pr = set()
    matches = 0

    sims: List[Tuple[float, int, int]] = []
    for i, pt in enumerate(pr_tokens):
        for j, gt_t in enumerate(gt_tokens):
            sim = jaccard(pt, gt_t)
            sims.append((sim, i, j))

    sims.sort(reverse=True, key=lambda x: x[0])

    for sim, i, j in sims:
        if sim < threshold:
            break
        if i in used_pr or j in used_gt:
            continue
        used_pr.add(i)
        used_gt.add(j)
        matches += 1

    tp = matches
    fp = len(pr) - matches
    fn = len(gt) - matches
    return PairCounts(tp=tp, fp=fp, fn=fn)


def raw_pair_counts_for_paper(
    gt_map: Dict[str, List[str]],
    pred_map: Dict[str, List[str]],
    threshold: float
) -> PairCounts:
    lnames = set((gt_map or {}).keys()) | set((pred_map or {}).keys())
    counts = []
    for ln in lnames:
        c = greedy_match_counts((gt_map or {}).get(ln, []), (pred_map or {}).get(ln, []), threshold)
        counts.append(c)
    return union_counts(counts)


def raw_exact_match_paper(
    gt_map: Dict[str, List[str]],
    pred_map: Dict[str, List[str]],
    threshold: float
) -> bool:
    gt_map = gt_map or {}
    pred_map = pred_map or {}
    if set(gt_map.keys()) != set(pred_map.keys()):
        return False
    for ln in gt_map.keys():
        c = greedy_match_counts(gt_map.get(ln, []), pred_map.get(ln, []), threshold)
        if c.fp != 0 or c.fn != 0:
            return False
    return True


# ----------------------------
# Evaluation driver
# ----------------------------

def evaluate_ror(gt_ror, pred_ror, paper_ids: List[str]):
    per_paper_counts = []
    per_paper_f1s = []

    for pid in paper_ids:
        c = ror_pair_counts_for_paper(gt_ror.get(pid, {}), pred_ror.get(pid, {}), pid)
        per_paper_counts.append(c)
        if (c.tp + c.fp + c.fn) > 0:
            per_paper_f1s.append(c.f1())

    micro = union_counts(per_paper_counts)
    macro_f1 = sum(per_paper_f1s) / len(per_paper_f1s) if per_paper_f1s else 0.0

    # paper-level exact
    gt_papers = set(gt_ror.keys())
    pred_papers = set(pred_ror.keys())

    tp = fp = fn = 0
    for pid in pred_papers:
        if ror_exact_match_paper(gt_ror.get(pid, {}), pred_ror.get(pid, {})):
            tp += 1
        else:
            fp += 1
    for pid in gt_papers:
        if not ror_exact_match_paper(gt_ror.get(pid, {}), pred_ror.get(pid, {})):
            fn += 1

    paper_counts = PairCounts(tp=tp, fp=fp, fn=fn)
    paper_f1s = [1.0 if ror_exact_match_paper(gt_ror.get(pid, {}), pred_ror.get(pid, {})) else 0.0 for pid in paper_ids]
    paper_macro_f1 = sum(paper_f1s) / len(paper_f1s) if paper_f1s else 0.0

    return {
        "single_author": {
            "precision": micro.precision(),
            "recall": micro.recall(),
            "macro_f1": macro_f1,
            "micro_f1": micro.f1(),
        },
        "all_authors_exact": {
            "precision": paper_counts.precision(),
            "recall": paper_counts.recall(),
            "macro_f1": paper_macro_f1,
            "micro_f1": paper_counts.f1(),
        },
    }


def evaluate_raw(gt_raw, pred_raw, paper_ids: List[str], threshold: float):
    per_paper_counts = []
    per_paper_f1s = []

    for pid in paper_ids:
        c = raw_pair_counts_for_paper(gt_raw.get(pid, {}), pred_raw.get(pid, {}), threshold)
        per_paper_counts.append(c)
        if (c.tp + c.fp + c.fn) > 0:
            per_paper_f1s.append(c.f1())

    micro = union_counts(per_paper_counts)
    macro_f1 = sum(per_paper_f1s) / len(per_paper_f1s) if per_paper_f1s else 0.0

    # paper-level exact under fuzzy matching
    gt_papers = set(gt_raw.keys())
    pred_papers = set(pred_raw.keys())

    tp = fp = fn = 0
    for pid in pred_papers:
        if raw_exact_match_paper(gt_raw.get(pid, {}), pred_raw.get(pid, {}), threshold):
            tp += 1
        else:
            fp += 1
    for pid in gt_papers:
        if not raw_exact_match_paper(gt_raw.get(pid, {}), pred_raw.get(pid, {}), threshold):
            fn += 1

    paper_counts = PairCounts(tp=tp, fp=fp, fn=fn)
    paper_f1s = [1.0 if raw_exact_match_paper(gt_raw.get(pid, {}), pred_raw.get(pid, {}), threshold) else 0.0 for pid in paper_ids]
    paper_macro_f1 = sum(paper_f1s) / len(paper_f1s) if paper_f1s else 0.0

    return {
        "single_author": {
            "precision": micro.precision(),
            "recall": micro.recall(),
            "macro_f1": macro_f1,
            "micro_f1": micro.f1(),
        },
        "all_authors_exact": {
            "precision": paper_counts.precision(),
            "recall": paper_counts.recall(),
            "macro_f1": paper_macro_f1,
            "micro_f1": paper_counts.f1(),
        },
    }


# ----------------------------
# Source dispatchers
# ----------------------------

def load_source_raw(
    source_kind: str,
    source_path: str,
    allowed_ids: Set[str],
) -> Tuple[Dict[str, Dict[str, List[str]]], int, int]:
    """
    Returns: raw_map, raw_total, raw_with_value
    """
    if source_kind == "tei_dir":
        raw_map, raw_total, raw_with = load_tei_dir(source_path, allowed_ids=allowed_ids)
        return raw_map, raw_total, raw_with

    if source_kind == "json_dir":
        _, raw_map, _, _, raw_total, raw_with = load_ror_and_raw_from_json_dir(source_path, allowed_ids=allowed_ids)
        return raw_map, raw_total, raw_with

    if source_kind == "jsonl_dir":
        raw_map, _, raw_total, raw_with, _, _ = load_jsonl_dir(source_path, allowed_ids=allowed_ids)
        return raw_map, raw_total, raw_with

    if source_kind == "md_dir":
        raw_map, _, raw_total, raw_with, _, _ = load_md_dir(source_path, allowed_ids=allowed_ids)
        return raw_map, raw_total, raw_with

    raise ValueError(f"Unknown RAW source_kind: {source_kind}")


def load_source_ror(
    source_kind: str,
    source_path: str,
    allowed_ids: Set[str],
) -> Tuple[Dict[str, Dict[str, Set[str]]], int, int]:
    """
    Returns: ror_map, ror_total, ror_with_value
    """
    if source_kind == "json_dir":
        ror_map, ror_total, ror_with = load_ror_from_json_dir(source_path, allowed_ids=allowed_ids)
        return ror_map, ror_total, ror_with

    if source_kind == "jsonl_dir":
        _, ror_map, _, _, ror_total, ror_with = load_jsonl_dir(source_path, allowed_ids=allowed_ids)
        return ror_map, ror_total, ror_with

    if source_kind == "md_dir":
        _, ror_map, _, _, ror_total, ror_with = load_md_dir(source_path, allowed_ids=allowed_ids)
        return ror_map, ror_total, ror_with

    raise ValueError(f"Unknown ROR source_kind: {source_kind}")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_file", required=True, help="Path to GT CSV/TSV.")
    ap.add_argument("--runs_root", required=True, help="Root folder containing all run subfolders.")
    ap.add_argument("--sep", default="\t", help="Delimiter for GT file (default: tab).")
    ap.add_argument("--id_col", default="anthology_id", help="Paper ID column in GT file.")
    ap.add_argument("--authors_norm_col", default="authors_normalized_json", help="Normalized authors JSON column (ROR).")
    ap.add_argument("--authors_raw_col", default="authors_raw_json", help="Raw authors JSON column (raw affiliation).")
    ap.add_argument("--raw_threshold", type=float, default=0.80, help="Token-Jaccard threshold for raw matching.")
    ap.add_argument("--out_json", default="eval_report.json", help="Write JSON report here.")
    args = ap.parse_args()

    runs_root = os.path.abspath(args.runs_root)

    # Restrict to GT papers
    allowed_ids = load_gt_paper_ids(args.gt_file, sep=args.sep, id_col=args.id_col)
    paper_ids = sorted(allowed_ids)

    # Load GT
    gt_ror, gt_raw = load_ground_truth(
        args.gt_file,
        sep=args.sep,
        id_col=args.id_col,
        authors_norm_col=args.authors_norm_col,
        authors_raw_col=args.authors_raw_col,
    )

    # --- Evaluate all RAW sources ---
    raw_results: Dict[str, dict] = {}
    raw_coverage: Dict[str, dict] = {}
    raw_loaded_counts: Dict[str, dict] = {}

    for source_name, kind in RAW_SOURCES:
        source_path = os.path.join(runs_root, source_name)
        if not os.path.isdir(source_path):
            raw_results[source_name] = {"error": f"Missing directory: {source_path}"}
            continue

        pred_raw_map, raw_total, raw_with = load_source_raw(kind, source_path, allowed_ids=allowed_ids)
        raw_loaded_counts[source_name] = {"n_papers_loaded": len(pred_raw_map)}
        raw_coverage[source_name] = {
            "raw_affiliation_coverage_rate": (raw_with / raw_total) if raw_total else 0.0,
            "counts": {"raw_total": raw_total, "raw_with_value": raw_with},
        }
        raw_results[source_name] = evaluate_raw(gt_raw, pred_raw_map, paper_ids, threshold=args.raw_threshold)

    # --- Evaluate all ROR sources ---
    ror_results: Dict[str, dict] = {}
    ror_coverage: Dict[str, dict] = {}
    ror_loaded_counts: Dict[str, dict] = {}

    for source_name, kind in ROR_SOURCES:
        source_path = os.path.join(runs_root, source_name)
        if not os.path.isdir(source_path):
            ror_results[source_name] = {"error": f"Missing directory: {source_path}"}
            continue

        pred_ror_map, ror_total, ror_with = load_source_ror(kind, source_path, allowed_ids=allowed_ids)
        ror_loaded_counts[source_name] = {"n_papers_loaded": len(pred_ror_map)}
        ror_coverage[source_name] = {
            "ror_coverage_rate": (ror_with / ror_total) if ror_total else 0.0,
            "counts": {"aff_total": ror_total, "aff_with_ror": ror_with},
        }
        ror_results[source_name] = evaluate_ror(gt_ror, pred_ror_map, paper_ids)

    report = {
        "inputs": {
            "gt_file": os.path.abspath(args.gt_file),
            "runs_root": runs_root,
            "sep": args.sep,
            "id_col": args.id_col,
            "authors_norm_col": args.authors_norm_col,
            "authors_raw_col": args.authors_raw_col,
            "raw_threshold": args.raw_threshold,
            "raw_sources": RAW_SOURCES,
            "ror_sources": ROR_SOURCES,
        },
        "evaluation_scope": {
            "restricted_to_gt": True,
            "n_papers_gt": len(paper_ids),
        },
        "coverage": {
            "raw": raw_coverage,
            "ror": ror_coverage,
        },
        "metrics": {
            "raw": raw_results,
            "ror": ror_results,
        },
        "counts": {
            "n_papers_gt_ror_loaded": len(gt_ror),
            "n_papers_gt_raw_loaded": len(gt_raw),
            "raw_sources": raw_loaded_counts,
            "ror_sources": ror_loaded_counts,
        },
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    def pct(x: float) -> str:
        return f"{x*100:.2f}%"

    def print_eval_group(title: str, group: dict):
        print(title)
        for name, metrics in group.items():
            if isinstance(metrics, dict) and "error" in metrics:
                print(f"  {name}: ERROR: {metrics['error']}")
                continue
            sa = metrics["single_author"]
            ae = metrics["all_authors_exact"]
            print(f"  {name}:")
            print(f"    single_author  P/R/MacroF1/MicroF1: {pct(sa['precision'])} / {pct(sa['recall'])} / {pct(sa['macro_f1'])} / {pct(sa['micro_f1'])}")
            print(f"    all_authors_exact P/R/MacroF1/MicroF1: {pct(ae['precision'])} / {pct(ae['recall'])} / {pct(ae['macro_f1'])} / {pct(ae['micro_f1'])}")
        print()

    print("=== Multi-Source Evaluation Report (GT-restricted) ===")
    print(f"Runs root: {runs_root}")
    print(f"Papers evaluated (GT): {len(paper_ids)}")
    print()

    print("Coverage (RAW):")
    for name, cov in raw_coverage.items():
        print(f"  {name}: {pct(cov['raw_affiliation_coverage_rate'])} ({cov['counts']['raw_with_value']}/{cov['counts']['raw_total']})")
    print()

    print("Coverage (ROR):")
    for name, cov in ror_coverage.items():
        print(f"  {name}: {pct(cov['ror_coverage_rate'])} ({cov['counts']['aff_with_ror']}/{cov['counts']['aff_total']})")
    print()

    print_eval_group(f"RAW scoring (all RAW sources vs GT raw), threshold={args.raw_threshold}:", raw_results)
    print_eval_group("ROR scoring (all *_ror sources vs GT normalized):", ror_results)

    print(f"Wrote: {os.path.abspath(args.out_json)}")


if __name__ == "__main__":
    main()
