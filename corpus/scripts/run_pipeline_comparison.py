"""Run a matrix of affiliation-extraction pipelines against the 781-paper
SciCo gold set and produce a side-by-side comparison table.

Each pipeline is configured in a small YAML/JSON file (or via CLI for
the simple cases). For every paper in the gold, we feed each pipeline
all available source views (TEI XML, PyMu MD, hybrid OCR, S2ORC record,
abstract) — each pipeline picks the views it needs and writes a
PaperPrediction. After all predictions are collected, we score
precision/recall/F1 on raw-affiliation Jaccard matching (mirrors
eval_affiliations.py's logic) and dump per-pipeline + per-decade tables.

Designed to make these experiments cheap to add:
  1. Different LLMs: add a new entry to the config YAML
  2. Different inputs: change `input_template` / `input_kwargs` per entry
  3. Different pipelines: drop a class in scripts/aff_pipelines/ and
     import + instantiate in this script

Usage:
    python scripts/run_pipeline_comparison.py \\
        --config configs/pipelines.yaml \\
        --gold-csv reports/ground_truth_merged.csv \\
        --sources-root /share/moritz.staudinger/acl_data \\
        --out-dir reports/pipeline_comparison/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

# Self-contained string-normalisation + jaccard, mirroring
# scripts/compare_extraction_pipelines.py exactly so we don't pull pandas.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from compare_extraction_pipelines import (  # noqa: E402
    affil_tokens, jaccard, load_ground_truth, _greedy_jaccard_pairs, _decade,
)


def load_gold_names(gold_csv: Path) -> Dict[str, List[str]]:
    """Parallel to compare_extraction_pipelines.load_ground_truth but
    extracts the *canonical institution names* (`name` field in the
    authors_normalized_json column) — used by the name-similarity
    scoring track."""
    import csv as _csv
    out: Dict[str, List[str]] = {}
    with open(gold_csv, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            pid = (row.get("anthology_id") or "").strip()
            if not pid:
                continue
            try:
                authors_norm = json.loads(row.get("authors_normalized_json") or "[]")
            except json.JSONDecodeError:
                authors_norm = []
            names = []
            for a in authors_norm:
                for af in (a.get("affiliations") or []):
                    if isinstance(af, dict):
                        n = af.get("name")
                        if n:
                            names.append(str(n))
            out[pid] = names
    return out

from aff_pipelines import PaperSources, PaperPrediction
from aff_pipelines.passthrough import OursPassthrough, S2ORCPassthrough
from aff_pipelines.grobid_only import GrobidOnly
from aff_pipelines.llm_extract import LLMPipeline
from aff_pipelines.ror import RorResolver

logger = logging.getLogger("pipeline_compare")


# -------------------------- source loader --------------------------

def _load_sources(paper_id: str, sources_root: Path,
                  s2orc_raw_dir: Optional[Path] = None) -> PaperSources:
    """Hydrate every available paper representation that we have on disk."""
    tei_path = sources_root / "acl_pdfs_parsed" / f"{paper_id}.tei.xml"
    pymu_path = sources_root / "acl_pdfs_pymu_llm" / f"{paper_id}.md"
    hybrid_path = sources_root / "acl_pdfs_hybrid" / f"{paper_id}.jsonl"

    def _read_safe(p: Path) -> Optional[str]:
        if not p.is_file():
            return None
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return None

    s2orc_record = None
    if s2orc_raw_dir:
        s2_fp = s2orc_raw_dir / "s2orc" / f"{paper_id}.json"
        if s2_fp.is_file():
            try:
                s2orc_record = json.loads(s2_fp.read_text(errors="replace"))
            except json.JSONDecodeError:
                pass

    # Title + abstract pulled from our affiliations dir (most reliable)
    title = abstract = None
    year = None
    aff_path = sources_root / "ror_parsed_affiliations_90_with_topics" / "affiliations" / f"{paper_id}.json"
    if aff_path.is_file():
        try:
            d = json.loads(aff_path.read_text(errors="replace"))
            title = d.get("title")
            abstract = d.get("abstract")
            year = d.get("year") if isinstance(d.get("year"), int) else None
        except json.JSONDecodeError:
            pass

    return PaperSources(
        paper_id=paper_id,
        pdf_path=None,  # not needed unless a pipeline does its own PDF parsing
        tei_xml=_read_safe(tei_path),
        pymu_md=_read_safe(pymu_path),
        hybrid_txt=_read_safe(hybrid_path),
        s2orc_record=s2orc_record,
        title=title, abstract=abstract, year=year,
    )


# -------------------------- pipeline construction --------------------------

def _construct_pipeline(cfg: Dict[str, Any], paths: Dict[str, Path],
                        ror_resolver: Optional[RorResolver] = None) -> Any:
    """Build a pipeline object from one config dict.

    Supported pipeline `type` values:
        ours            -> OursPassthrough (already ROR-linked at source)
        s2orc           -> S2ORCPassthrough (ROR-link at read time if resolver given)
        grobid_only     -> GrobidOnly (ROR-link extracted strings if resolver given)
        llm             -> LLMPipeline (ROR-link extracted strings if resolver given)

    `use_ror: false` in any config entry suppresses ROR linking for that
    pipeline even if a resolver exists (useful for the raw_only baseline).
    """
    t = cfg["type"]
    pid = cfg["id"]
    apply_ror = ror_resolver if cfg.get("use_ror", True) else None
    if t == "ours":
        p = OursPassthrough(paths["scico_root"])
    elif t == "s2orc":
        p = S2ORCPassthrough(paths["s2orc_norm_root"], ror_resolver=apply_ror)
    elif t == "grobid_only":
        p = GrobidOnly(tei_dir=paths.get("tei_dir"), ror_resolver=apply_ror)
    elif t == "llm":
        p = LLMPipeline(
            pipeline_id=pid,
            backend=cfg["backend"],
            model=cfg["model"],
            input_template=cfg["input_template"],
            input_kwargs=cfg.get("input_kwargs", {}),
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 1500),
            ror_resolver=apply_ror,
        )
    elif t == "passthrough_file":
        # Predictions JSONL is written by an external script (e.g.,
        # scripts/fetch_openalex_affiliations.py). At comparison time
        # we never call .predict() on this — score-from-dir reads the
        # file directly. Return a stub so the construction phase
        # doesn't crash.
        class _PassthroughFile:
            pipeline_id = pid
            def predict(self, sources):
                raise RuntimeError(
                    f"pipeline {pid} (type=passthrough_file) must be used "
                    f"via --score-from-dir; its predictions JSONL is built "
                    f"externally, not at comparison time."
                )
        p = _PassthroughFile()
    else:
        raise ValueError(f"unknown pipeline type: {t}")
    # Make sure the pipeline_id matches the config (so report rows line up)
    p.pipeline_id = pid
    return p


# -------------------------- main --------------------------

def _run_pipeline(pipeline, sources_list, max_workers=1):
    """Run a pipeline over a list of PaperSources, with optional parallelism.
    LLM pipelines benefit from threads (I/O bound); passthrough/grobid
    pipelines are CPU-bound and gain little from threading."""
    preds = []
    if max_workers <= 1:
        for s in sources_list:
            preds.append(pipeline.predict(s))
        return preds
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(pipeline.predict, s): s for s in sources_list}
        for f in as_completed(futs):
            preds.append(f.result())
    return preds


def _normalize_ror(rid: Optional[str]) -> Optional[str]:
    """Strip the https://ror.org/ prefix so 'https://ror.org/042nb2s44'
    and '042nb2s44' compare equal. Returns None for empty input."""
    if not rid:
        return None
    rid = rid.strip()
    if not rid:
        return None
    # ROR IDs can come as URL, bare ID, with trailing slash. Normalize.
    rid = rid.rstrip("/")
    rid = rid.split("/")[-1]
    return rid.lower()


def _score_ror_hier(gold_set: set, pred_set: set,
                    expand) -> tuple:
    """Greedy 1-1 matching with hierarchy-aware equivalence.

    `expand` is a callable ror_id -> set of equivalent ids (self + parents
    + children, per the ROR `relationships` field). A pred id counts as
    a TP for a gold id when their expanded sets share at least one id.
    """
    used = set()
    tp = 0
    for g in gold_set:
        g_set = expand(g)
        for p in pred_set:
            if p in used:
                continue
            if g_set & expand(p):
                used.add(p)
                tp += 1
                break
    fp = len(pred_set) - tp
    fn = len(gold_set) - tp
    return tp, fp, fn


def _score_pipeline(label: str,
                    gt_raw: Dict[str, List[str]],
                    gt_ror: Dict[str, List[str]],
                    gt_names: Dict[str, List[str]],
                    preds: List[PaperPrediction], pid_year: Dict[str, int],
                    threshold: float,
                    ror_resolver=None) -> Dict[str, Any]:
    """Four scoring tracks per pipeline:
       - raw       : greedy token-Jaccard on raw affiliation strings
       - ror       : exact set match on canonical ROR IDs
       - ror_hier  : ROR set match with parent/child equivalence
                     (a Stanford-CS-Department pred matches a Stanford-Univ gold)
       - name      : greedy token-Jaccard on canonical institution NAMES
                     (post-ROR, so this measures 'did the linker pick the right
                     org regardless of whether the ID exactly matched')

    Every track reports overall + per-decade P/R/F1.
    """
    pred_by_pid = {p.paper_id: p for p in preds}
    n_gold = len(gt_raw)
    n_covered = n_ror_covered = n_name_covered = 0
    raw_tp = raw_fp = raw_fn = 0
    ror_tp = ror_fp = ror_fn = 0
    hier_tp = hier_fp = hier_fn = 0
    name_tp = name_fp = name_fn = 0
    per_decade_raw  = defaultdict(lambda: [0, 0, 0, 0, 0])
    per_decade_ror  = defaultdict(lambda: [0, 0, 0, 0, 0])
    per_decade_hier = defaultdict(lambda: [0, 0, 0, 0, 0])
    per_decade_name = defaultdict(lambda: [0, 0, 0, 0, 0])
    err_count = 0
    sum_seconds = 0.0
    sum_in = sum_out = 0
    n_ror_lookups = 0

    # Expansion function for hierarchy-aware matching. When no resolver,
    # collapses to identity (so hier == ror in that case).
    def _expand(rid: str) -> set:
        if not ror_resolver or not rid:
            return {rid} if rid else set()
        return ror_resolver.get_related_set(rid)

    # Two simultaneous accumulators per track:
    #   *_inc: inclusive — missing predictions count as FN (penalises coverage)
    #   the regular accumulators below: covered-only — only papers where the
    #   pipeline actually has data contribute to P/R/F1. This is the fairer
    #   metric for pipelines like S2ORC that have legitimately partial
    #   corpus coverage; we report coverage separately as its own column.
    raw_inc_tp = raw_inc_fp = raw_inc_fn = 0
    ror_inc_tp = ror_inc_fp = ror_inc_fn = 0
    hier_inc_tp = hier_inc_fp = hier_inc_fn = 0
    name_inc_tp = name_inc_fp = name_inc_fn = 0

    for pid, gold_affs in gt_raw.items():
        year = pid_year.get(pid, 0)
        d = _decade(year) if year else "unknown"
        for pd in (per_decade_raw, per_decade_ror, per_decade_hier, per_decade_name):
            pd[d][4] += 1
        gold_ror_set = {r for r in (_normalize_ror(x) for x in gt_ror.get(pid, [])) if r}
        gold_name_list = gt_names.get(pid, [])
        pred = pred_by_pid.get(pid)
        if pred is None or pred.error:
            err_count += int(bool(pred and pred.error))
            # Inclusive track still penalises (FN += gold size); covered-only
            # track simply skips this paper.
            raw_inc_fn += len(gold_affs)
            ror_inc_fn += len(gold_ror_set)
            hier_inc_fn += len(gold_ror_set)
            name_inc_fn += len(gold_name_list)
            continue

        pred_strs = [af.raw for a in pred.authors for af in a.affiliations if af.raw]
        pred_ror_set = {r for r in (_normalize_ror(af.ror_id)
                                    for a in pred.authors for af in a.affiliations) if r}
        pred_name_list = [af.clean_institution_name
                          for a in pred.authors for af in a.affiliations
                          if af.clean_institution_name]
        if pred_strs:
            n_covered += 1; per_decade_raw[d][3] += 1
        if pred_ror_set:
            n_ror_covered += 1; per_decade_ror[d][3] += 1; per_decade_hier[d][3] += 1
        if pred_name_list:
            n_name_covered += 1; per_decade_name[d][3] += 1
        n_ror_lookups += sum(1 for a in pred.authors for af in a.affiliations if af.ror_id)
        sum_seconds += pred.extraction_seconds or 0.0
        sum_in += pred.tokens_in or 0
        sum_out += pred.tokens_out or 0

        # --- raw greedy Jaccard ---
        tp, fp, fn = _greedy_jaccard_pairs(gold_affs, pred_strs, threshold)
        raw_tp += tp; raw_fp += fp; raw_fn += fn
        raw_inc_tp += tp; raw_inc_fp += fp; raw_inc_fn += fn
        per_decade_raw[d][0] += tp; per_decade_raw[d][1] += fp; per_decade_raw[d][2] += fn

        # --- exact ROR set match ---
        tp = len(gold_ror_set & pred_ror_set)
        fp = len(pred_ror_set - gold_ror_set)
        fn = len(gold_ror_set - pred_ror_set)
        ror_tp += tp; ror_fp += fp; ror_fn += fn
        ror_inc_tp += tp; ror_inc_fp += fp; ror_inc_fn += fn
        per_decade_ror[d][0] += tp; per_decade_ror[d][1] += fp; per_decade_ror[d][2] += fn

        # --- hierarchy-aware ROR match ---
        tp, fp, fn = _score_ror_hier(gold_ror_set, pred_ror_set, _expand)
        hier_tp += tp; hier_fp += fp; hier_fn += fn
        hier_inc_tp += tp; hier_inc_fp += fp; hier_inc_fn += fn
        per_decade_hier[d][0] += tp; per_decade_hier[d][1] += fp; per_decade_hier[d][2] += fn

        # --- name Jaccard (greedy) ---
        tp, fp, fn = _greedy_jaccard_pairs(gold_name_list, pred_name_list, threshold)
        name_tp += tp; name_fp += fp; name_fn += fn
        name_inc_tp += tp; name_inc_fp += fp; name_inc_fn += fn
        per_decade_name[d][0] += tp; per_decade_name[d][1] += fp; per_decade_name[d][2] += fn

    def _prf(tp, fp, fn):
        P = tp / (tp + fp) if (tp + fp) else 0.0
        R = tp / (tp + fn) if (tp + fn) else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) else 0.0
        return P, R, F1

    raw_P,  raw_R,  raw_F1  = _prf(raw_tp,  raw_fp,  raw_fn)
    ror_P,  ror_R,  ror_F1  = _prf(ror_tp,  ror_fp,  ror_fn)
    hier_P, hier_R, hier_F1 = _prf(hier_tp, hier_fp, hier_fn)
    name_P, name_R, name_F1 = _prf(name_tp, name_fp, name_fn)
    # Inclusive — penalises missing predictions as FN. Useful as a
    # "what-if-you-used-this-as-your-only-source" metric.
    raw_iP,  raw_iR,  raw_iF1  = _prf(raw_inc_tp,  raw_inc_fp,  raw_inc_fn)
    ror_iP,  ror_iR,  ror_iF1  = _prf(ror_inc_tp,  ror_inc_fp,  ror_inc_fn)
    hier_iP, hier_iR, hier_iF1 = _prf(hier_inc_tp, hier_inc_fp, hier_inc_fn)
    name_iP, name_iR, name_iF1 = _prf(name_inc_tp, name_inc_fp, name_inc_fn)

    def _decade_dict(pd):
        return {k: dict(zip(["tp","fp","fn","covered","total"], v)) for k, v in pd.items()}

    def _track(P, R, F1, tp, fp, fn, iP, iR, iF1, itp, ifp, ifn, dec, extra=None):
        d = {"precision": P, "recall": R, "f1": F1,
             "tp": tp, "fp": fp, "fn": fn,
             # Inclusive counterparts — useful when you want the metric
             # that DOES penalise missing predictions (e.g., S2ORC's
             # pre-2020 coverage gap should hurt its inclusive recall).
             "inclusive": {"precision": iP, "recall": iR, "f1": iF1,
                           "tp": itp, "fp": ifp, "fn": ifn},
             "per_decade": _decade_dict(dec)}
        if extra:
            d.update(extra)
        return d

    return {
        "pipeline_id": label,
        "papers_total": n_gold,
        "papers_covered_raw":  n_covered,
        "papers_covered_ror":  n_ror_covered,
        "papers_covered_name": n_name_covered,
        "error_rows": err_count,
        "raw":      _track(raw_P, raw_R, raw_F1, raw_tp, raw_fp, raw_fn,
                           raw_iP, raw_iR, raw_iF1, raw_inc_tp, raw_inc_fp, raw_inc_fn,
                           per_decade_raw),
        "ror":      _track(ror_P, ror_R, ror_F1, ror_tp, ror_fp, ror_fn,
                           ror_iP, ror_iR, ror_iF1, ror_inc_tp, ror_inc_fp, ror_inc_fn,
                           per_decade_ror, extra={"n_predicted_ror_ids": n_ror_lookups}),
        "ror_hier": _track(hier_P, hier_R, hier_F1, hier_tp, hier_fp, hier_fn,
                           hier_iP, hier_iR, hier_iF1, hier_inc_tp, hier_inc_fp, hier_inc_fn,
                           per_decade_hier),
        "name":     _track(name_P, name_R, name_F1, name_tp, name_fp, name_fn,
                           name_iP, name_iR, name_iF1, name_inc_tp, name_inc_fp, name_inc_fn,
                           per_decade_name),
        "extraction_seconds_total": sum_seconds,
        "tokens_in_total": sum_in, "tokens_out_total": sum_out,
    }


def _print_summary(scores: List[Dict[str, Any]]):
    """Two tables:
       1. COVERED-ONLY F1 (the fair metric — only scored on papers the
          pipeline actually has data for; doesn't penalise coverage gaps)
       2. INCLUSIVE F1 (missing predictions count as FN; this is what you
          get if you'd use ONLY this pipeline for all papers)
    Coverage % is shown alongside in table 1.
    """
    print()
    print("=== Track F1 — COVERED-ONLY (only papers with data; fair extraction quality) ===")
    header = (f"{'pipeline_id':<28}  "
              f"{'F1_raw':>7} {'F1_ror':>7} {'F1_hier':>8} {'F1_name':>8}  "
              f"{'cov%':>5}  "
              f"{'time(s)':>7} {'tok_in':>9} {'tok_out':>9}")
    print(header)
    print("-" * len(header))
    for s in scores:
        cov_pct = (s["papers_covered_raw"] / s["papers_total"]) * 100 if s["papers_total"] else 0
        print(f"{s['pipeline_id']:<28}  "
              f"{s['raw']['f1']:7.3f} {s['ror']['f1']:7.3f} "
              f"{s['ror_hier']['f1']:8.3f} {s['name']['f1']:8.3f}  "
              f"{cov_pct:>4.0f}%  "
              f"{s['extraction_seconds_total']:>7.0f} "
              f"{s['tokens_in_total']:>9} {s['tokens_out_total']:>9}")

    print()
    print("=== Track F1 — INCLUSIVE (missing predictions count as FN; combined coverage+quality) ===")
    print(f"{'pipeline_id':<28}  "
          f"{'F1_raw':>7} {'F1_ror':>7} {'F1_hier':>8} {'F1_name':>8}  "
          f"{'cov%':>5}")
    print("-" * 78)
    for s in scores:
        cov_pct = (s["papers_covered_raw"] / s["papers_total"]) * 100 if s["papers_total"] else 0
        print(f"{s['pipeline_id']:<28}  "
              f"{s['raw']['inclusive']['f1']:7.3f} "
              f"{s['ror']['inclusive']['f1']:7.3f} "
              f"{s['ror_hier']['inclusive']['f1']:8.3f} "
              f"{s['name']['inclusive']['f1']:8.3f}  "
              f"{cov_pct:>4.0f}%")

    print()
    print("Tracks: raw=token-Jaccard on raw strings, ror=exact ROR ID set, "
          "hier=ROR set with parent/child equivalence, name=token-Jaccard on "
          "canonical institution names.")
    print("cov% = papers the pipeline produced any prediction for. "
          "Covered-only metrics evaluate quality where the pipeline operates; "
          "inclusive metrics penalise both quality AND coverage gaps.")


def _f1_from_decade(d: Dict[str, int]) -> float:
    tp, fp, fn = d.get("tp", 0), d.get("fp", 0), d.get("fn", 0)
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * P * R / (P + R) if (P + R) else 0.0


def _print_per_decade(scores: List[Dict[str, Any]]) -> None:
    """One table per track. Rows = pipelines, columns = decades + overall."""
    if not scores:
        return
    # Collect every decade that appears in any track of any pipeline,
    # sorted chronologically with 'unknown' last.
    all_decades = set()
    for s in scores:
        for track in ("raw", "ror", "ror_hier", "name"):
            all_decades.update((s[track].get("per_decade") or {}).keys())
    chronological = sorted(d for d in all_decades if d != "unknown" and d.endswith("s"))
    if "unknown" in all_decades:
        chronological.append("unknown")

    for track in ("raw", "ror", "ror_hier", "name"):
        print(f"\n=== F1 by decade — {track} track ===")
        # Header
        hdr = f"{'pipeline_id':<28}  " + " ".join(f"{d:>7}" for d in chronological) + "  " + f"{'overall':>7}"
        print(hdr)
        print("-" * len(hdr))
        for s in scores:
            per_dec = s[track].get("per_decade") or {}
            cells = []
            for d in chronological:
                if d in per_dec and (per_dec[d]["tp"] + per_dec[d]["fp"] + per_dec[d]["fn"]) > 0:
                    cells.append(f"{_f1_from_decade(per_dec[d]):7.3f}")
                else:
                    cells.append(f"{'  -':>7}")
            row = f"{s['pipeline_id']:<28}  " + " ".join(cells) + f"  {s[track]['f1']:7.3f}"
            print(row)


def _write_csv_summary(scores: List[Dict[str, Any]], out_path: Path) -> None:
    """Wide CSV — one row per (pipeline, track), one column per decade + overall.
    Spreadsheet-friendly: paste into Excel, pivot however you want."""
    import csv as _csv
    # Build decade union (same logic as the printout)
    all_decades = set()
    for s in scores:
        for track in ("raw", "ror", "ror_hier", "name"):
            all_decades.update((s[track].get("per_decade") or {}).keys())
    chronological = sorted(d for d in all_decades if d != "unknown" and d.endswith("s"))
    if "unknown" in all_decades:
        chronological.append("unknown")

    fieldnames = (["pipeline_id", "track",
                   "coverage_pct",
                   "covered_precision", "covered_recall", "covered_f1",
                   "covered_tp", "covered_fp", "covered_fn",
                   "inclusive_precision", "inclusive_recall", "inclusive_f1",
                   "inclusive_tp", "inclusive_fp", "inclusive_fn"]
                  + [f"{d}_f1" for d in chronological]
                  + [f"{d}_n" for d in chronological])

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in scores:
            cov_pct = (s["papers_covered_raw"] / s["papers_total"]) * 100 if s["papers_total"] else 0
            for track in ("raw", "ror", "ror_hier", "name"):
                t = s[track]
                inc = t.get("inclusive") or {}
                row = {
                    "pipeline_id": s["pipeline_id"],
                    "track": track,
                    "coverage_pct": f"{cov_pct:.1f}",
                    "covered_precision": f"{t['precision']:.4f}",
                    "covered_recall":    f"{t['recall']:.4f}",
                    "covered_f1":        f"{t['f1']:.4f}",
                    "covered_tp":        t["tp"],
                    "covered_fp":        t["fp"],
                    "covered_fn":        t["fn"],
                    "inclusive_precision": f"{inc.get('precision', 0):.4f}",
                    "inclusive_recall":    f"{inc.get('recall', 0):.4f}",
                    "inclusive_f1":        f"{inc.get('f1', 0):.4f}",
                    "inclusive_tp":        inc.get("tp", 0),
                    "inclusive_fp":        inc.get("fp", 0),
                    "inclusive_fn":        inc.get("fn", 0),
                }
                per_dec = t.get("per_decade") or {}
                for d in chronological:
                    if d in per_dec and (per_dec[d]["tp"] + per_dec[d]["fp"] + per_dec[d]["fn"]) > 0:
                        row[f"{d}_f1"] = f"{_f1_from_decade(per_dec[d]):.4f}"
                        row[f"{d}_n"]  = per_dec[d]["total"]
                    else:
                        row[f"{d}_f1"] = ""
                        row[f"{d}_n"]  = per_dec.get(d, {}).get("total", 0)
                w.writerow(row)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True,
                    help="YAML or JSON listing pipeline configs (see "
                         "configs/pipelines.example.yaml).")
    ap.add_argument("--gold-csv", type=Path, required=True,
                    help="Merged ground-truth CSV (reports/ground_truth_merged.csv).")
    ap.add_argument("--sources-root", type=Path, required=True,
                    help="Root holding acl_pdfs_parsed/, acl_pdfs_pymu_llm/, "
                         "acl_pdfs_hybrid/, ror_parsed_affiliations_90_with_topics/.")
    ap.add_argument("--s2orc-norm-root", type=Path, default=None,
                    help="Optional: dir holding S2ORC-normalized per-paper "
                         "JSONs (output of scripts/s2orc_normalize.py).")
    ap.add_argument("--s2orc-raw-root", type=Path, default=None,
                    help="Optional: dir holding S2ORC raw records (output of "
                         "scripts/s2orc_download.py s2orc).")
    ap.add_argument("--tei-dir", type=Path, default=None,
                    help="Optional explicit TEI dir for grobid_only pipeline.")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--llm-workers", type=int, default=4,
                    help="Concurrent LLM requests per pipeline (default 4).")
    ap.add_argument("--limit-papers", type=int, default=None,
                    help="Cap to N gold papers — useful for smoke tests. "
                         "Picks the first N in sorted order; use --paper-ids "
                         "to target specific papers instead.")
    ap.add_argument("--paper-ids", nargs="+", default=None, metavar="ID",
                    help="Restrict the run to these specific paper IDs only. "
                         "Overrides --limit-papers when both are given. "
                         "Useful for smoke-testing a known-good paper rather "
                         "than whatever the gold CSV happens to sort first.")
    ap.add_argument("--jaccard-threshold", type=float, default=0.7)
    ap.add_argument("--ror-cache", type=Path,
                    default=Path("cache/ror_lookup_cache.json"),
                    help="Persistent JSON cache for ROR API matcher lookups.")
    ap.add_argument("--ror-hierarchy-cache", type=Path,
                    default=Path("cache/ror_hierarchy_cache.json"),
                    help="Persistent JSON cache for ROR org-by-id (relationships) "
                         "lookups. Drives the hierarchy-aware scoring track.")
    ap.add_argument("--ror-min-confidence", type=float, default=0.7,
                    help="Drop ROR matches under this confidence (default 0.7).")
    ap.add_argument("--no-ror", action="store_true",
                    help="Disable ROR linking entirely. Only the raw-string "
                         "Jaccard track is meaningful in this mode; the "
                         "ror/ror_hier/name tracks collapse to 0.")
    ap.add_argument("--pipelines", nargs="+", default=None, metavar="ID",
                    help="Run only the named pipeline(s) from the config. "
                         "Used by the per-pipeline slurm submit pattern — "
                         "each slurm task targets one id, then a final "
                         "aggregation pass scores all of them together via "
                         "--score-from-dir.")
    ap.add_argument("--skip-score", action="store_true",
                    help="Write predictions JSONL but skip scoring + report. "
                         "Use in per-pipeline slurm jobs that only need to "
                         "emit predictions; a separate aggregation job scores.")
    ap.add_argument("--score-from-dir", type=Path, default=None,
                    help="Skip pipeline execution; read predictions_*.jsonl "
                         "from this dir and run scoring + report only. "
                         "Counterpart to --skip-score for the per-pipeline "
                         "slurm pattern.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    # Load gold across all three views: raw strings, ROR ids, canonical names
    gt_ror, gt_raw = load_ground_truth(args.gold_csv)
    gt_names = load_gold_names(args.gold_csv)
    paper_ids = sorted(gt_raw.keys())
    if args.paper_ids:
        wanted = set(args.paper_ids)
        missing = wanted - set(paper_ids)
        if missing:
            raise SystemExit(f"--paper-ids requested {sorted(missing)} which "
                             f"aren't in the gold CSV")
        paper_ids = [p for p in paper_ids if p in wanted]
    elif args.limit_papers:
        paper_ids = paper_ids[: args.limit_papers]
    gt_raw = {p: gt_raw[p] for p in paper_ids}
    gt_ror = {p: gt_ror.get(p, []) for p in paper_ids}
    gt_names = {p: gt_names.get(p, []) for p in paper_ids}
    logger.info("gold papers: %d (raw strings) %d (with ≥1 ROR id) %d (with ≥1 name)",
                len(gt_raw),
                sum(1 for v in gt_ror.values() if v),
                sum(1 for v in gt_names.values() if v))

    # Decade map for the per-decade table
    pid_year = {}
    for pid in gt_raw:
        if pid[:4].isdigit():
            pid_year[pid] = int(pid[:4])
        elif len(pid) >= 3 and pid[1:3].isdigit():
            y = int(pid[1:3])
            pid_year[pid] = 2000 + y if y < 50 else 1900 + y

    # Load config
    try:
        import yaml
        cfg_text = args.config.read_text()
        cfg = yaml.safe_load(cfg_text)
    except ImportError:
        cfg = json.loads(args.config.read_text())  # fall back to JSON
    if not isinstance(cfg, dict) or "pipelines" not in cfg:
        raise SystemExit("config must have a top-level `pipelines:` list")

    paths = {
        "scico_root": args.sources_root / "ror_parsed_affiliations_90_with_topics",
        "s2orc_norm_root": args.s2orc_norm_root,
        "tei_dir": args.tei_dir or (args.sources_root / "acl_pdfs_parsed"),
    }
    # Shared ROR resolver — same cache across every pipeline this run.
    ror_resolver = None
    if not args.no_ror:
        args.ror_cache.parent.mkdir(parents=True, exist_ok=True)
        ror_resolver = RorResolver(
            cache_path=args.ror_cache,
            hierarchy_cache_path=args.ror_hierarchy_cache,
            min_confidence=args.ror_min_confidence,
        )
        logger.info("ROR resolver initialised (cache=%s, hier_cache=%s, threshold=%.2f)",
                    args.ror_cache, args.ror_hierarchy_cache, args.ror_min_confidence)
        # Warm hierarchy for gold ROR ids up-front so the hier scoring track
        # doesn't interleave API calls with the scoring loop.
        gold_ror_ids = {r for v in gt_ror.values() for r in v}
        if gold_ror_ids:
            ror_resolver.warm_hierarchy(gold_ror_ids)
    # Optional --pipelines filter: only run a subset of configured entries.
    pipeline_cfgs = cfg["pipelines"]
    if args.pipelines:
        wanted = set(args.pipelines)
        filtered = [c for c in pipeline_cfgs if c["id"] in wanted]
        missing = wanted - {c["id"] for c in filtered}
        if missing:
            raise SystemExit(
                f"--pipelines requested {sorted(missing)} but they aren't in the config"
            )
        pipeline_cfgs = filtered
    pipelines = [_construct_pipeline(c, paths, ror_resolver=ror_resolver)
                 for c in pipeline_cfgs]
    logger.info("built %d pipelines: %s", len(pipelines), [p.pipeline_id for p in pipelines])

    # Hydrate sources once
    t0 = time.time()
    sources = [_load_sources(pid, args.sources_root, args.s2orc_raw_root) for pid in paper_ids]
    logger.info("hydrated sources for %d papers in %.1fs", len(sources), time.time() - t0)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --score-from-dir short-circuit: skip running pipelines, just read
    # the predictions written by previous per-pipeline jobs and score.
    if args.score_from_dir:
        from aff_pipelines import PaperPrediction, PredictedAuthor, PredictedAffiliation
        all_scores = []
        for cfg_entry in pipeline_cfgs:
            pid_id = cfg_entry["id"]
            pred_path = args.score_from_dir / f"predictions_{pid_id}.jsonl"
            if not pred_path.is_file():
                logger.warning("missing %s — pipeline %s skipped from aggregation", pred_path, pid_id)
                continue
            preds = []
            with pred_path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line: continue
                    rec = json.loads(line)
                    authors = []
                    for a in (rec.get("authors") or []):
                        affs = [PredictedAffiliation(
                            raw=af.get("raw_affiliation") or "",
                            clean_institution_name=af.get("clean_institution_name"),
                            ror_id=af.get("ror_id"),
                            country_code=af.get("country_code"),
                        ) for af in (a.get("affiliations") or [])]
                        authors.append(PredictedAuthor(name=a.get("name") or "", affiliations=affs))
                    pl_meta = rec.get("_pipeline") or {}
                    preds.append(PaperPrediction(
                        paper_id=rec.get("paper_id") or "",
                        pipeline_id=pid_id, authors=authors,
                        extraction_seconds=pl_meta.get("extraction_seconds"),
                        tokens_in=pl_meta.get("tokens_in"),
                        tokens_out=pl_meta.get("tokens_out"),
                        error=pl_meta.get("error"),
                    ))
            if ror_resolver is not None:
                pred_ror_ids = {af.ror_id for pr in preds for a in pr.authors
                                for af in a.affiliations if af.ror_id}
                if pred_ror_ids:
                    ror_resolver.warm_hierarchy(pred_ror_ids)
            scores = _score_pipeline(pid_id, gt_raw, gt_ror, gt_names,
                                     preds, pid_year, args.jaccard_threshold,
                                     ror_resolver=ror_resolver)
            all_scores.append(scores)
            logger.info("scored %s from %d cached predictions", pid_id, len(preds))
    else:
        # Run each pipeline + collect predictions + (optionally) score
        all_scores = []
        for p in pipelines:
            logger.info("running pipeline %s ...", p.pipeline_id)
            is_llm = isinstance(p, LLMPipeline)
            workers = args.llm_workers if is_llm else 1
            t = time.time()
            preds = _run_pipeline(p, sources, max_workers=workers)
            elapsed = time.time() - t
            logger.info("pipeline %s: %d predictions in %.1fs", p.pipeline_id, len(preds), elapsed)

            # Persist predictions
            pred_path = args.out_dir / f"predictions_{p.pipeline_id}.jsonl"
            with pred_path.open("w", encoding="utf-8") as fh:
                for pr in preds:
                    fh.write(json.dumps(pr.to_dict(), ensure_ascii=False) + "\n")
            logger.info("  wrote %s", pred_path)

            if args.skip_score:
                continue

            # Warm hierarchy for any pred ROR ids this pipeline produced that
            # aren't already cached (so the hier scoring track sees full info).
            if ror_resolver is not None:
                pred_ror_ids = {af.ror_id for pr in preds for a in pr.authors
                                for af in a.affiliations if af.ror_id}
                if pred_ror_ids:
                    ror_resolver.warm_hierarchy(pred_ror_ids)

            scores = _score_pipeline(p.pipeline_id, gt_raw, gt_ror, gt_names,
                                     preds, pid_year, args.jaccard_threshold,
                                     ror_resolver=ror_resolver)
            all_scores.append(scores)

        if args.skip_score:
            logger.info("predictions written; --skip-score so no scoring done. "
                        "Run a separate aggregation pass with --score-from-dir %s",
                        args.out_dir)
            if ror_resolver is not None:
                ror_resolver.save()
            return 0

    # Persist the ROR cache once at the end (resolver auto-saves every 25
    # new lookups, but a final flush guarantees no losses).
    if ror_resolver is not None:
        ror_resolver.save()

    # Write the combined report + print summary
    report_path = args.out_dir / "comparison_report.json"
    with report_path.open("w") as fh:
        json.dump({
            "gold_csv": str(args.gold_csv),
            "jaccard_threshold": args.jaccard_threshold,
            "pipelines": all_scores,
        }, fh, indent=2)
    logger.info("wrote combined report: %s", report_path)

    _print_summary(all_scores)
    _print_per_decade(all_scores)

    csv_path = args.out_dir / "comparison_summary.csv"
    _write_csv_summary(all_scores, csv_path)
    logger.info("wrote per-decade CSV summary: %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
