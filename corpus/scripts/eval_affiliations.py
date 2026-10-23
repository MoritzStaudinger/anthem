#!/usr/bin/env python3
"""
Evaluate author-affiliation extraction vs ground truth, with BOTH:
1) ROR-based scoring (prediction JSON folder)
2) Raw affiliation string scoring:
   a) from prediction JSON folder (raw_affiliation fields)
   b) from Grobid TEI XML folder

Author matching rule (as requested):
- match by last name: the last whitespace-split token of the author name must match after normalization.

Outputs:
- Precision, Recall, Macro F1, Micro F1
- ROR coverage rate (pred JSON only)
- Raw affiliation coverage rate (pred JSON + TEI)
- Metrics for:
    * single_author (author–affiliation level)
    * all_authors_exact (paper-level exact match: all authors+affiliations correct)

Raw affiliation matching:
- Uses fuzzy token-Jaccard matching with a configurable threshold (default 0.80),
  with greedy max-sim assignment per (paper, lastname).
- This handles minor punctuation / ordering differences robustly.

Ground truth expects columns:
- anthology_id
- authors_normalized_json (for ROR scoring)
- authors_raw_json        (for raw affiliation scoring)

Prediction JSON files:
- One per paper: <paper_id>.json
- Schema as in your example; raw strings from affiliation["raw_affiliation"], ROR from affiliation["ror_id"].

TEI XML files (Grobid):
- One per paper: <paper_id>.tei.xml  OR  <paper_id>.xml  (both supported)
- Extracts authors and affiliation text from <teiHeader> bibliographic structures.
  It is reasonably robust but TEI variants exist; if your TEI differs, we can adjust the XPath.

Usage:
  python eval_affiliations.py \
    --gt_file /path/to/ground_truth.tsv \
    --pred_dir /path/to/pred_json_folder \
    --tei_dir /path/to/grobid_tei_folder \
    --sep $'\\t' \
    --out_json /path/to/report.json

"""
from typing import Set
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
      gt_raw[paper_id][lastname] = list(raw_affiliation_strings)   (list to preserve duplicates if any)
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


# ----------------------------
# Prediction JSON loading
# ----------------------------

def load_pred_json(
    pred_dir: str,
    allowed_ids: Set[str],
) -> Tuple[
    Dict[str, Dict[str, Set[str]]],
    Dict[str, Dict[str, List[str]]],
    int, int, int, int
]:
    """
    Load prediction JSON outputs, but ONLY for papers whose IDs are in allowed_ids.

    Returns:
      pred_ror[paper_id][lastname] = set(ror_ids)
      pred_raw[paper_id][lastname] = list(raw_affiliation_strings)
      n_aff_total, n_aff_with_ror, n_aff_total_raw, n_aff_with_raw
    """
    pred_ror: Dict[str, Dict[str, Set[str]]] = {}
    pred_raw: Dict[str, Dict[str, List[str]]] = {}

    n_aff_total = 0
    n_aff_with_ror = 0
    n_aff_total_raw = 0
    n_aff_with_raw = 0

    for fn in os.listdir(pred_dir):
        if not fn.endswith(".json"):
            continue

        pid = fn[:-5]  # strip ".json"
        if pid not in allowed_ids:
            continue

        path = os.path.join(pred_dir, fn)

        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            # unreadable/bad json -> treat as missing predictions (skip)
            continue

        authors = obj.get("authors", []) or []

        per_paper_ror: Dict[str, Set[str]] = {}
        per_paper_raw: Dict[str, List[str]] = {}

        for a in authors:
            ln = norm_lastname(a.get("name", ""))
            if not ln:
                continue
            affs = a.get("affiliations", []) or []

            for aff in affs:
                # ROR coverage
                n_aff_total += 1
                ror = aff.get("ror_id")
                if ror:
                    n_aff_with_ror += 1
                    per_paper_ror.setdefault(ln, set()).add(str(ror).strip())

                # RAW coverage
                n_aff_total_raw += 1
                raw = aff.get("raw_affiliation") or ""
                raw = str(raw).strip()
                if raw:
                    n_aff_with_raw += 1
                    per_paper_raw.setdefault(ln, []).append(raw)

        pred_ror[pid] = per_paper_ror
        pred_raw[pid] = per_paper_raw

    return pred_ror, pred_raw, n_aff_total, n_aff_with_ror, n_aff_total_raw, n_aff_with_raw

# ----------------------------
# TEI loading (Grobid)
# ----------------------------

def strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def all_text(el: ET.Element) -> str:
    # concatenate text from element subtree
    parts = []
    if el.text:
        parts.append(el.text)
    for c in list(el):
        parts.append(all_text(c))
        if c.tail:
            parts.append(c.tail)
    return "".join(parts)

def load_ror_from_json_dir(
    json_dir: str,
    allowed_ids: Set[str],
) -> Tuple[Dict[str, Dict[str, Set[str]]], int, int]:
    """
    Load ROR predictions from a directory of JSON files (<paper_id>.json) with schema:
      {"authors":[{"name":"...", "affiliations":[{"ror_id":"..."}]}]}

    Restricts to paper IDs in allowed_ids.

    Returns:
      ror_map[paper_id][lastname] = set(ror_ids)
      n_aff_total, n_aff_with_ror  (coverage counters)
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

        authors = obj.get("authors", []) or []
        per_paper_ror: Dict[str, Set[str]] = {}

        for a in authors:
            ln = norm_lastname(a.get("name", ""))
            if not ln:
                continue
            affs = a.get("affiliations", []) or []
            for aff in affs:
                n_aff_total += 1
                ror = aff.get("ror_id")
                if ror:
                    n_aff_with_ror += 1
                    per_paper_ror.setdefault(ln, set()).add(str(ror).strip())

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

        authors = obj.get("authors", []) or []

        per_ror: Dict[str, Set[str]] = {}
        per_raw: Dict[str, List[str]] = {}

        for a in authors:
            ln = norm_lastname(a.get("name", ""))
            if not ln:
                continue
            affs = a.get("affiliations", []) or []

            for aff in affs:
                # ROR coverage
                n_aff_total += 1
                ror = aff.get("ror_id")
                if ror:
                    n_aff_with_ror += 1
                    per_ror.setdefault(ln, set()).add(str(ror).strip())

                # RAW coverage
                n_aff_total_raw += 1
                raw = aff.get("raw_affiliation") or ""
                raw = str(raw).strip()
                if raw:
                    n_aff_with_raw += 1
                    per_raw.setdefault(ln, []).append(raw)

        ror_map[pid] = per_ror
        raw_map[pid] = per_raw

    return ror_map, raw_map, n_aff_total, n_aff_with_ror, n_aff_total_raw, n_aff_with_raw


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

    # Target the canonical Grobid location first
    author_nodes = root.findall(
        ".//tei:teiHeader/tei:fileDesc/tei:sourceDesc/tei:biblStruct/tei:analytic/tei:author",
        namespaces=NS,
    )

    # Fallback: any author nodes anywhere (in case structure differs)
    if not author_nodes:
        author_nodes = root.findall(".//tei:author", namespaces=NS)

    for author in author_nodes:
        # Extract surname (preferred) else fallback to full persName text
        surname_el = author.find(".//tei:persName/tei:surname", namespaces=NS)
        if surname_el is not None and (surname_el.text or "").strip():
            ln = norm_lastname(surname_el.text)
        else:
            # fallback: use full persName text
            pers_el = author.find(".//tei:persName", namespaces=NS)
            raw_name = ""
            if pers_el is not None:
                raw_name = "".join(pers_el.itertext()).strip()
            ln = norm_lastname(raw_name)

        if not ln:
            continue

        # Extract affiliations
        aff_texts: List[str] = []
        aff_nodes = author.findall("./tei:affiliation", namespaces=NS)

        for aff in aff_nodes:
            # Prefer orgName children (institution/department), joined by " ; "
            orgs = [
                " ".join("".join(o.itertext()).split()).strip()
                for o in aff.findall(".//tei:orgName", namespaces=NS)
            ]
            orgs = [o for o in orgs if o]

            if orgs:
                # Keep as a single affiliation string per <affiliation> node
                aff_texts.append(" ; ".join(orgs))
            else:
                # Fallback to full affiliation text
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
    Load Grobid TEI files, but ONLY for papers whose IDs are in allowed_ids.

    Returns:
      tei_raw[paper_id][lastname] = list(normed_affiliation_strings)
      n_aff_total_raw, n_aff_with_raw
    """
    tei_raw: Dict[str, Dict[str, List[str]]] = {}
    n_aff_total_raw = 0
    n_aff_with_raw = 0

    for fn in os.listdir(tei_dir):
        if not (fn.endswith(".xml") or fn.endswith(".tei.xml")):
            continue

        # normalize file naming to paper_id
        pid = fn
        if pid.endswith(".tei.xml"):
            pid = pid[:-8]
        elif pid.endswith(".xml"):
            pid = pid[:-4]

        if pid not in allowed_ids:
            continue

        path = os.path.join(tei_dir, fn)

        # extract_tei_author_affils already has error handling (malformed XML -> {})
        per_paper = extract_tei_author_affils(path)

        # Count coverage
        for _, affs in per_paper.items():
            for s in affs:
                n_aff_total_raw += 1
                if s and str(s).strip():
                    n_aff_with_raw += 1

        tei_raw[pid] = per_paper

    return tei_raw, n_aff_total_raw, n_aff_with_raw

def load_gt_paper_ids(gt_path: str, sep: str, id_col: str) -> Set[str]:
    """
    Load only the set of paper IDs from the GT file.
    This is used to restrict evaluation and file reading to labeled papers.
    """
    df = pd.read_csv(gt_path, sep=sep, dtype=str, usecols=[id_col], keep_default_na=False)
    ids = set(df[id_col].astype(str).str.strip())
    # remove empty IDs if any
    ids.discard("")
    return ids
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

    # Precompute similarities
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
    """
    Aggregate greedy matching per (lastname) within the paper.
    """
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
    """
    Paper-level "all authors exact" under fuzzy raw matching:
    - Same set of lastnames
    - For each lastname, greedy matching yields fp=0 and fn=0
    """
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


import random

def show_examples(gt_raw, pred_raw, tei_raw, n=5, seed=0):
    random.seed(seed)
    pids = list(gt_raw.keys())
    random.shuffle(pids)
    shown = 0
    for pid in pids:
        g = gt_raw.get(pid, {})
        t = tei_raw.get(pid, {})
        p = pred_raw.get(pid, {})
        if not g:
            continue
        # choose one lastname present in GT
        ln = next(iter(g.keys()))
        print("="*80)
        print("paper:", pid, "| lastname:", ln)
        print("\nGT RAW:")
        print(g.get(ln, [])[:3])
        print("\nPRED RAW:")
        print(p.get(ln, [])[:3])
        print("\nTEI RAW:")
        print(t.get(ln, [])[:3])
        shown += 1
        if shown >= n:
            break

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_file", required=True, help="Path to GT CSV/TSV.")

    # Existing inputs (keep)
    ap.add_argument("--pred_dir", required=True, help="Folder with model prediction JSON files (<paper_id>.json).")
    ap.add_argument("--tei_dir", required=True, help="Folder with Grobid TEI XML files (<paper_id>.tei.xml or .xml).")

    # New input (add)
    ap.add_argument("--tei_ror_dir", required=True, help="Folder with TEI-ROR JSON files (<paper_id>.json), same schema as pred_dir.")

    ap.add_argument("--sep", default="\t", help="Delimiter for GT file (default: tab).")
    ap.add_argument("--id_col", default="anthology_id", help="Paper ID column in GT file.")
    ap.add_argument("--authors_norm_col", default="authors_normalized_json", help="Normalized authors JSON column (ROR).")
    ap.add_argument("--authors_raw_col", default="authors_raw_json", help="Raw authors JSON column (raw affiliation).")
    ap.add_argument("--raw_threshold", type=float, default=0.80, help="Token-Jaccard threshold for raw matching.")
    ap.add_argument("--out_json", default="eval_report.json", help="Write JSON report here.")
    args = ap.parse_args()

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

    # Load model predictions (JSON): ROR + RAW
    pred_ror, pred_raw, n_aff_total_pred, n_aff_with_ror_pred, n_raw_total_pred, n_raw_with_pred = \
        load_pred_json(args.pred_dir, allowed_ids=allowed_ids)

    # Load TEI XML: RAW only
    tei_raw, n_raw_total_tei, n_raw_with_tei = load_tei_dir(args.tei_dir, allowed_ids=allowed_ids)

    # Load TEI-ROR JSON: ROR only (new)
    tei_ror, n_aff_total_tei_ror, n_aff_with_ror_tei_ror = load_ror_from_json_dir(args.tei_ror_dir, allowed_ids=allowed_ids)

    # --- Evaluate ---
    # ROR
    ror_pred_metrics = evaluate_ror(gt_ror, pred_ror, paper_ids)
    ror_tei_ror_metrics = evaluate_ror(gt_ror, tei_ror, paper_ids)

    # RAW
    raw_pred_metrics = evaluate_raw(gt_raw, pred_raw, paper_ids, threshold=args.raw_threshold)
    raw_tei_metrics = evaluate_raw(gt_raw, tei_raw, paper_ids, threshold=args.raw_threshold)

    # --- Coverage ---
    pred_ror_cov = (n_aff_with_ror_pred / n_aff_total_pred) if n_aff_total_pred else 0.0
    pred_raw_cov = (n_raw_with_pred / n_raw_total_pred) if n_raw_total_pred else 0.0

    tei_raw_cov = (n_raw_with_tei / n_raw_total_tei) if n_raw_total_tei else 0.0
    tei_ror_cov = (n_aff_with_ror_tei_ror / n_aff_total_tei_ror) if n_aff_total_tei_ror else 0.0

    report = {
        "inputs": {
            "gt_file": os.path.abspath(args.gt_file),
            "pred_dir": os.path.abspath(args.pred_dir),
            "tei_dir": os.path.abspath(args.tei_dir),
            "tei_ror_dir": os.path.abspath(args.tei_ror_dir),
            "sep": args.sep,
            "id_col": args.id_col,
            "authors_norm_col": args.authors_norm_col,
            "authors_raw_col": args.authors_raw_col,
            "raw_threshold": args.raw_threshold,
        },
        "evaluation_scope": {
            "restricted_to_gt": True,
            "n_papers_gt": len(paper_ids),
        },
        "coverage": {
            "pred_json": {
                "ror_coverage_rate": pred_ror_cov,
                "raw_affiliation_coverage_rate": pred_raw_cov,
                "counts": {
                    "aff_total": n_aff_total_pred,
                    "aff_with_ror": n_aff_with_ror_pred,
                    "raw_total": n_raw_total_pred,
                    "raw_with_value": n_raw_with_pred,
                },
            },
            "tei_xml": {
                "raw_affiliation_coverage_rate": tei_raw_cov,
                "counts": {
                    "raw_total": n_raw_total_tei,
                    "raw_with_value": n_raw_with_tei,
                },
            },
            "tei_ror_json": {
                "ror_coverage_rate": tei_ror_cov,
                "counts": {
                    "aff_total": n_aff_total_tei_ror,
                    "aff_with_ror": n_aff_with_ror_tei_ror,
                },
            },
        },
        "metrics": {
            "ror_pred_json_vs_gt_normalized": ror_pred_metrics,
            "ror_tei_ror_json_vs_gt_normalized": ror_tei_ror_metrics,
            "raw_pred_json_vs_gt_raw": raw_pred_metrics,
            "raw_tei_xml_vs_gt_raw": raw_tei_metrics,
        },
        "counts": {
            "n_papers_gt_ror_loaded": len(gt_ror),
            "n_papers_gt_raw_loaded": len(gt_raw),
            "n_papers_pred_json_loaded": len(pred_ror),
            "n_papers_tei_xml_loaded": len(tei_raw),
            "n_papers_tei_ror_json_loaded": len(tei_ror),
        },
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    def pct(x: float) -> str:
        return f"{x*100:.2f}%"

    def print_block(title: str, m: dict):
        print(title)
        sa = m["single_author"]
        ae = m["all_authors_exact"]
        print("  single_author:")
        print(f"    Precision: {pct(sa['precision'])}")
        print(f"    Recall:    {pct(sa['recall'])}")
        print(f"    Macro F1:  {pct(sa['macro_f1'])}")
        print(f"    Micro F1:  {pct(sa['micro_f1'])}")
        print("  all_authors_exact:")
        print(f"    Precision: {pct(ae['precision'])}")
        print(f"    Recall:    {pct(ae['recall'])}")
        print(f"    Macro F1:  {pct(ae['macro_f1'])}")
        print(f"    Micro F1:  {pct(ae['micro_f1'])}")
        print()

    print("=== Evaluation Report (GT-restricted) ===")
    print(f"Papers evaluated (GT): {len(paper_ids)}")
    print(f"Papers loaded - Pred JSON: {len(pred_ror)} | TEI XML: {len(tei_raw)} | TEI-ROR JSON: {len(tei_ror)}")
    print()

    print("Coverage:")
    print(f"  Pred JSON ROR coverage:     {pct(pred_ror_cov)} ({n_aff_with_ror_pred}/{n_aff_total_pred})")
    print(f"  TEI-ROR JSON ROR coverage:  {pct(tei_ror_cov)} ({n_aff_with_ror_tei_ror}/{n_aff_total_tei_ror})")
    print(f"  Pred JSON RAW coverage:     {pct(pred_raw_cov)} ({n_raw_with_pred}/{n_raw_total_pred})")
    print(f"  TEI XML RAW coverage:       {pct(tei_raw_cov)} ({n_raw_with_tei}/{n_raw_total_tei})")
    print()

    # --- 4 blocks (as requested) ---
    print_block("ROR scoring (Pred JSON vs GT normalized):", ror_pred_metrics)
    print_block("ROR scoring (TEI-ROR JSON vs GT normalized):", ror_tei_ror_metrics)
    print_block(f"RAW scoring (Pred JSON vs GT raw), threshold={args.raw_threshold}:", raw_pred_metrics)
    print_block(f"RAW scoring (TEI XML vs GT raw), threshold={args.raw_threshold}:", raw_tei_metrics)

    print(f"Wrote: {os.path.abspath(args.out_json)}")


if __name__ == "__main__":
    main()
