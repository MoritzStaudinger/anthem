"""Side-by-side comparison of two extraction pipelines against the gold set.

Designed for the EMNLP-revision reviewer concern: "weak comparison to S2ORC".
Lets you produce a single table answering:
  - On the 781 manually-annotated gold papers, how does our custom pipeline
    (GROBID/PyMu/Hybrid + Qwen affiliation extractor + ROR linking) compare
    against S2ORC's pre-parsed affiliations?
  - How does that comparison change across decades? (S2ORC's coverage of
    pre-2020 ACL is notoriously patchy — this is where the value of our
    custom pipeline should show up.)

Computes for each pipeline:
  - Paper coverage: % of gold papers the pipeline produced ANY affiliation for
  - Raw-string precision/recall vs gold (per Jaccard ≥ threshold matching)
  - Per-decade breakdown
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Self-contained string normalisation + Jaccard, mirroring the helpers in
# eval_affiliations.py exactly (same fold, same stoplist, same Jaccard).
# We duplicate rather than import to avoid pulling pandas as a runtime
# dependency for this script (eval_affiliations imports pandas at module
# load for its CSV/NaN handling — overkill when we just need string ops).

def _ascii_fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return s.encode("ascii", "ignore").decode("ascii")


def _norm_affil_text(s: str) -> str:
    if not s:
        return ""
    s = _ascii_fold(s).lower()
    s = s.replace("\n", " ")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


_AFFIL_STOP = {"of", "the", "and", "for", "in", "at", "on", "a", "an",
               "dept", "department", "school", "university"}


def affil_tokens(s: str) -> Set[str]:
    s = _norm_affil_text(s)
    if not s:
        return set()
    return {t for t in s.split(" ") if t and t not in _AFFIL_STOP}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def load_ground_truth(gold_csv: Path) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Load ground_truth_merged.csv → ({pid: ror_ids}, {pid: raw_affs}).

    Mirrors eval_affiliations.load_ground_truth in semantics; uses stdlib csv
    instead of pandas. Reads authors_raw_json + authors_normalized_json,
    flattens to one list of affiliations per paper.
    """
    gt_ror: Dict[str, List[str]] = {}
    gt_raw: Dict[str, List[str]] = {}
    with open(gold_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("anthology_id") or "").strip()
            if not pid:
                continue
            try:
                authors_raw = json.loads(row.get("authors_raw_json") or "[]")
            except json.JSONDecodeError:
                authors_raw = []
            try:
                authors_norm = json.loads(row.get("authors_normalized_json") or "[]")
            except json.JSONDecodeError:
                authors_norm = []

            raw_affs: List[str] = []
            for a in authors_raw:
                for af in (a.get("affiliations") or []):
                    if isinstance(af, dict):
                        s = af.get("affiliation") or af.get("name")
                        if s:
                            raw_affs.append(str(s))
                    elif isinstance(af, str):
                        raw_affs.append(af)

            ror_ids: List[str] = []
            for a in authors_norm:
                for af in (a.get("affiliations") or []):
                    if isinstance(af, dict) and af.get("ror_id"):
                        ror_ids.append(str(af["ror_id"]))

            gt_raw[pid] = raw_affs
            gt_ror[pid] = ror_ids
    return gt_ror, gt_raw

logger = logging.getLogger("compare_pipelines")


# -------------------------- pred loaders --------------------------

def _load_pred_dir(pred_dir: Path) -> Dict[str, List[str]]:
    """Generic per-paper JSON loader.

    Accepts our schema (used by both your custom pipeline and the S2ORC
    normalizer): {paper_id, affiliations: [str, ...], ...}.

    Returns {paper_id: [raw_affil_strings, ...]}.
    """
    out: Dict[str, List[str]] = {}
    aff_dir = pred_dir / "affiliations"
    if not aff_dir.is_dir():
        # Allow pointing directly at the affiliations/ dir
        aff_dir = pred_dir
    for jp in aff_dir.glob("*.json"):
        try:
            rec = json.loads(jp.read_text())
        except json.JSONDecodeError:
            continue
        pid = rec.get("paper_id") or jp.stem
        affs = rec.get("affiliations") or []
        # Some pipelines store affiliations under authors[].affiliations
        if not affs:
            for a in (rec.get("authors") or []):
                for af in (a.get("affiliations") or []):
                    if isinstance(af, str):
                        affs.append(af)
                    elif isinstance(af, dict):
                        s = af.get("affiliation") or af.get("name")
                        if s:
                            affs.append(s)
        out[pid] = [str(s) for s in affs if s]
    return out


# -------------------------- metric helpers --------------------------

def _greedy_jaccard_pairs(
    gold: List[str], pred: List[str], threshold: float
) -> Tuple[int, int, int]:
    """Greedy 1-1 matching between gold and pred strings on token Jaccard.
    Returns (tp, fp, fn).
    """
    gold_tok = [affil_tokens(g) for g in gold]
    pred_tok = [affil_tokens(p) for p in pred]
    used_p: Set[int] = set()
    tp = 0
    for gi, gt in enumerate(gold_tok):
        best_j, best_pi = -1.0, -1
        for pi, pt in enumerate(pred_tok):
            if pi in used_p:
                continue
            j = jaccard(gt, pt)
            if j > best_j:
                best_j, best_pi = j, pi
        if best_j >= threshold and best_pi >= 0:
            used_p.add(best_pi)
            tp += 1
    fp = len(pred) - tp
    fn = len(gold) - tp
    return tp, fp, fn


def _decade(year: int) -> str:
    return f"{(year // 10) * 10}s"


# -------------------------- main --------------------------

def _summarize(
    label: str,
    gt: Dict[str, List[str]],
    pred: Dict[str, List[str]],
    pid_year: Dict[str, int],
    threshold: float,
):
    n_gold_papers = len(gt)
    n_covered = sum(1 for pid in gt if pid in pred and pred[pid])
    n_uncovered = n_gold_papers - n_covered

    total_tp = total_fp = total_fn = 0
    per_decade = defaultdict(lambda: [0, 0, 0, 0, 0])  # tp fp fn covered total

    for pid, gold_affs in gt.items():
        year = pid_year.get(pid, 0)
        decade = _decade(year) if year else "unknown"
        per_decade[decade][4] += 1  # total
        pred_affs = pred.get(pid, [])
        if pred_affs:
            per_decade[decade][3] += 1  # covered
            tp, fp, fn = _greedy_jaccard_pairs(gold_affs, pred_affs, threshold)
        else:
            tp, fp, fn = 0, 0, len(gold_affs)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        per_decade[decade][0] += tp
        per_decade[decade][1] += fp
        per_decade[decade][2] += fn

    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    print(f"\n=== {label} ===")
    print(f"  papers covered    : {n_covered}/{n_gold_papers} ({100*n_covered/n_gold_papers:.1f}%)")
    print(f"  papers uncovered  : {n_uncovered}")
    print(f"  raw-string (jaccard≥{threshold}) — P={p:.3f}  R={r:.3f}  F1={f1:.3f}")
    print(f"  per-decade breakdown:")
    print(f"    decade  | covered/total |  P     R     F1")
    print(f"    --------+---------------+--------------------")
    for decade in sorted(per_decade):
        tp, fp, fn, cov, tot = per_decade[decade]
        pp = tp / (tp + fp) if (tp + fp) else 0.0
        rr = tp / (tp + fn) if (tp + fn) else 0.0
        ff = 2 * pp * rr / (pp + rr) if (pp + rr) else 0.0
        cov_pct = (100 * cov / tot) if tot else 0
        print(f"    {decade:<7}|  {cov:>3}/{tot:<3}  ({cov_pct:>4.0f}%) | "
              f"{pp:.3f} {rr:.3f} {ff:.3f}")

    return {
        "label": label,
        "papers_total": n_gold_papers,
        "papers_covered": n_covered,
        "precision": p, "recall": r, "f1": f1,
        "per_decade": {k: dict(zip(["tp","fp","fn","covered","total"], v))
                       for k, v in per_decade.items()},
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold-csv", required=True,
                   help="Path to the manually-annotated ground truth CSV "
                        "(ground_truth_merged.csv from acl_processor/reports/).")
    p.add_argument("--pipeline-a-dir", required=True,
                   help="Predictions dir for pipeline A (your custom extractor).")
    p.add_argument("--pipeline-b-dir", required=True,
                   help="Predictions dir for pipeline B (S2ORC normalized).")
    p.add_argument("--label-a", default="ours",
                   help="Display label for pipeline A.")
    p.add_argument("--label-b", default="s2orc",
                   help="Display label for pipeline B.")
    p.add_argument("--jaccard-threshold", type=float, default=0.7,
                   help="Token-Jaccard threshold to count two affiliation "
                        "strings as the same (default 0.7).")
    p.add_argument("--out-json", default=None,
                   help="Optional path to write the full per-decade results "
                        "as JSON (in addition to stdout).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # eval_affiliations.load_ground_truth returns ({pid: ror_ids}, {pid: raw_affs}).
    # We use only raw_affs here — ROR comparison is a separate experiment.
    _, gt_raw = load_ground_truth(args.gold_csv)
    logger.info("gold papers: %d", len(gt_raw))

    pred_a = _load_pred_dir(Path(args.pipeline_a_dir))
    pred_b = _load_pred_dir(Path(args.pipeline_b_dir))
    logger.info("%s predictions: %d papers", args.label_a, len(pred_a))
    logger.info("%s predictions: %d papers", args.label_b, len(pred_b))

    # paper_id -> year (parsed from the ACL ID prefix when possible)
    pid_year: Dict[str, int] = {}
    for pid in gt_raw:
        # 2024.acl-long.123 → 2024; P18-1001 → 2018
        if pid[:4].isdigit():
            pid_year[pid] = int(pid[:4])
        elif len(pid) >= 3 and pid[1:3].isdigit():
            yr_short = int(pid[1:3])
            pid_year[pid] = 2000 + yr_short if yr_short < 50 else 1900 + yr_short

    res_a = _summarize(args.label_a, gt_raw, pred_a, pid_year,
                       args.jaccard_threshold)
    res_b = _summarize(args.label_b, gt_raw, pred_b, pid_year,
                       args.jaccard_threshold)

    # Delta table
    print(f"\n=== delta ({args.label_a} − {args.label_b}) ===")
    print(f"  coverage     : {res_a['papers_covered']-res_b['papers_covered']:+d} papers")
    print(f"  precision    : {res_a['precision']-res_b['precision']:+.3f}")
    print(f"  recall       : {res_a['recall']-res_b['recall']:+.3f}")
    print(f"  f1           : {res_a['f1']-res_b['f1']:+.3f}")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(
            {"pipeline_a": res_a, "pipeline_b": res_b,
             "jaccard_threshold": args.jaccard_threshold},
            indent=2,
        ))
        logger.info("wrote results JSON: %s", args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
