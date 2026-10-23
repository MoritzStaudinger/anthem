"""Why is affiliation extraction so hard? Quantitative + qualitative.

Produces four analyses from reports/merged_per_paper.relinked.jsonl
and the ROR hierarchy cache:

  1. Per-paper agreement: how many pipelines hit ror_hier F1 >= threshold
     on each paper. Histogram + lists of "hard" (no pipeline succeeded)
     and "easy" (everyone succeeded) papers.

  2. Disagreement zones: papers where ~half the pipelines succeed — these
     are the actionable cases (ROR disambiguation, normalization).

  3. Error type breakdown per pipeline:
       TP           — predicted ROR matches gold (with hierarchy)
       FP_wrong_org — emitted a ROR id, but not in gold's equivalence
       FN_link_fail — gold raw string IS in pipeline.raw_affiliations but
                      ROR linked it to the wrong/no id (or pipeline emitted
                      a near-synonym that linked elsewhere)
       FN_extract_fail — gold raw is nowhere in pipeline output (extraction missed)

  4. Decade-stratified version of (3).

Writes reports/affiliation_difficulty.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

logger = logging.getLogger("aff_diff")

# Pipelines with near-zero coverage just produce noise in the analysis.
LOW_COV_DROP = {
    "ablation_gemini25_titleabstract",
    "ablation_gemma4_e2b_titleabstract",
    "ablation_qwen36_35b_titleabstract",
    "ablation_gemma4_e2b_hybrid5k",  # cov 2%
    "s2orc",                          # not yet downloaded
}


def _bare(rid: str) -> str:
    if not rid:
        return ""
    return rid.strip().rstrip("/").split("/")[-1].lower()


def _eq(bare: str, hier: Dict[str, Dict[str, list]]) -> Set[str]:
    if not bare:
        return set()
    h = hier.get(bare) or {}
    return {bare, *(h.get("parent") or []), *(h.get("child") or [])}


def _norm(s: str | None) -> str:
    return " ".join((s or "").strip().lower().split())


def _f1(tp: int, fp: int, fn: int) -> float:
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * P * R / (P + R) if (P + R) else 0.0


_OLD_ID = re.compile(r"^[A-Z](\d\d)-\d+")


def _year_from_id(pid: str) -> int | None:
    if not pid:
        return None
    if pid[:4].isdigit():
        return int(pid[:4])
    m = _OLD_ID.match(pid)
    if m:
        nn = int(m.group(1))
        # Legacy ACL: NN >= 65 → 1965-1999, otherwise 20NN.
        return 1900 + nn if nn >= 65 else 2000 + nn
    return None


def _decade(year: int | None) -> str:
    return f"{(year // 10) * 10}s" if year else "unknown"


def _score_hier_with_buckets(
    pred_ror: List[str], gold_ror: List[str], hier: Dict[str, dict],
) -> Tuple[int, int, int, List[str]]:
    """Return (tp, fp_wrong_org, fn_total, fn_gold_ror_ids).
    The last is the list of gold ROR ids we failed to match, for
    later classification into link_fail vs extract_fail."""
    pred_b = [_bare(x) for x in pred_ror if x]
    gold_b = [_bare(x) for x in gold_ror if x]
    pred_eq = [(b, _eq(b, hier)) for b in pred_b]
    consumed = [False] * len(pred_eq)
    tp = 0
    missed: List[str] = []
    for g in gold_b:
        matched = False
        for i, (pb, eq) in enumerate(pred_eq):
            if consumed[i]:
                continue
            if g == pb or g in eq or pb in _eq(g, hier):
                consumed[i] = True
                tp += 1
                matched = True
                break
        if not matched:
            missed.append(g)
    fp = sum(1 for c in consumed if not c)
    return tp, fp, len(missed), missed


def _classify_fn(
    pred_raw: List[str], gold_raw: List[str], gold_missed_ror: List[str],
    gold_ror_full: List[str], gold_names: List[str],
) -> Tuple[int, int]:
    """For each missed gold ROR, decide: link_fail vs extract_fail.

    Heuristic: if the corresponding gold raw string (or canonical name)
    appears (case-insensitive substring) in any pred_raw, the pipeline
    DID extract the affiliation — the linker just didn't resolve it.
    Otherwise, extraction missed it.
    """
    pred_norm = [_norm(s) for s in pred_raw if s]
    pred_joined = " ".join(pred_norm)
    link_fail = 0
    extract_fail = 0
    # Build gold parallel arrays so we can pair missed ROR id ↔ gold raw/name
    by_ror: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for r, raw, nm in zip(gold_ror_full, gold_raw + [""] * len(gold_ror_full),
                          gold_names + [""] * len(gold_ror_full)):
        if not r:
            continue
        by_ror[_bare(r)].append((raw or "", nm or ""))
    for missed in gold_missed_ror:
        # The gold raw / canonical name corresponding to this ROR id
        candidates = by_ror.get(missed) or []
        found = False
        for raw, nm in candidates:
            for needle in (raw, nm):
                if not needle:
                    continue
                n = _norm(needle)
                if not n:
                    continue
                # Try the full string and the most distinctive 1-2 tokens
                tokens = [t for t in n.split() if len(t) >= 4]
                if n in pred_joined:
                    found = True
                    break
                # Fall back: any 4+ char token from the gold name appears
                # in any pred string. Cheap proxy for "the institution
                # is mentioned somewhere".
                if tokens and any(t in pred_joined for t in tokens):
                    found = True
                    break
            if found:
                break
        if found:
            link_fail += 1
        else:
            extract_fail += 1
    return link_fail, extract_fail


def _load_year_map(gt_csv: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not gt_csv.is_file():
        return out
    with gt_csv.open() as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("anthology_id") or "").strip()
            try:
                y = int(row.get("year") or "")
            except ValueError:
                continue
            if pid:
                out[pid] = y
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path,
                    default=Path("reports/merged_per_paper.relinked.jsonl"))
    ap.add_argument("--hierarchy", type=Path,
                    default=Path("reports/ror_hierarchy_cache.json"))
    ap.add_argument("--gold-csv", type=Path,
                    default=Path("reports/ground_truth_merged.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("reports/affiliation_difficulty.md"))
    ap.add_argument("--success-threshold", type=float, default=0.5,
                    help="F1 threshold to count a pipeline as 'succeeded' on a paper.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    hier = json.loads(args.hierarchy.read_text()) if args.hierarchy.is_file() else {}
    year_map = _load_year_map(args.gold_csv)
    logger.info("hierarchy: %d ids, year map: %d papers", len(hier), len(year_map))

    # Per (pipeline, paper_id) -> {tp, fp, fn, fn_link, fn_extract, f1}
    per_paper: Dict[Tuple[str, str], Dict[str, float]] = {}
    # Per pipeline aggregate counts
    pp_totals: Dict[str, Counter] = defaultdict(Counter)
    # Per (pipeline, decade) aggregate counts
    ppd_totals: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    # Per paper: dict pipeline_id -> f1
    paper_f1: Dict[str, Dict[str, float]] = defaultdict(dict)
    # Per paper: gold size (for hard-paper analysis)
    paper_gold_size: Dict[str, int] = {}

    pipelines_seen: Set[str] = set()
    papers_seen: List[str] = []

    with args.merged.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pid = d.get("paper_id") or ""
            dec = _decade(year_map.get(pid) or _year_from_id(pid))
            gold = d.get("gold") or {}
            gold_ror = gold.get("ror_ids") or []
            gold_raw = gold.get("affiliations_raw") or []
            gold_nm = gold.get("names") or []
            papers_seen.append(pid)
            paper_gold_size[pid] = len(gold_ror)
            for ppid, pred in (d.get("predictions") or {}).items():
                if ppid in LOW_COV_DROP:
                    continue
                pipelines_seen.add(ppid)
                pred_raw = pred.get("affiliations_raw") or []
                pred_ror = pred.get("ror_ids") or []
                tp, fp, fn, missed = _score_hier_with_buckets(pred_ror, gold_ror, hier)
                fn_link, fn_extract = _classify_fn(
                    pred_raw, gold_raw, missed, gold_ror, gold_nm,
                )
                f1 = _f1(tp, fp, fn)
                per_paper[(ppid, pid)] = {
                    "tp": tp, "fp": fp, "fn": fn,
                    "fn_link": fn_link, "fn_extract": fn_extract, "f1": f1,
                }
                pp_totals[ppid]["tp"] += tp
                pp_totals[ppid]["fp"] += fp
                pp_totals[ppid]["fn"] += fn
                pp_totals[ppid]["fn_link"] += fn_link
                pp_totals[ppid]["fn_extract"] += fn_extract
                k = (ppid, dec)
                ppd_totals[k]["tp"] += tp
                ppd_totals[k]["fp"] += fp
                ppd_totals[k]["fn"] += fn
                ppd_totals[k]["fn_link"] += fn_link
                ppd_totals[k]["fn_extract"] += fn_extract
                paper_f1[pid][ppid] = f1

    pipelines = sorted(pipelines_seen)
    decades_present = sorted({_decade(year_map.get(p) or _year_from_id(p))
                              for p in papers_seen})
    decades_chrono = sorted([d for d in decades_present
                             if d != "unknown" and d.endswith("s")],
                            key=lambda s: int(s[:-1]))
    if "unknown" in decades_present:
        decades_chrono.append("unknown")

    # Sort pipelines by overall ror_hier F1 (covered = inclusive here,
    # since we didn't drop empty preds)
    overall_f1 = {
        p: _f1(pp_totals[p]["tp"], pp_totals[p]["fp"], pp_totals[p]["fn"])
        for p in pipelines
    }
    order = sorted(pipelines, key=lambda p: -overall_f1[p])

    # Build the report
    lines: List[str] = []
    lines.append("# Why is affiliation extraction so hard?")
    lines.append("")
    lines.append(f"Built from `reports/merged_per_paper.relinked.jsonl` "
                 f"(n={len(papers_seen)} papers, "
                 f"{len(pipelines)} pipelines after dropping "
                 f"{len(LOW_COV_DROP)} near-zero-coverage ablations). "
                 f"All numbers are on the ROR-hier track.")
    lines.append("")

    # ---------------- (1) Per-paper agreement ----------------
    lines.append("## 1. Per-paper agreement")
    lines.append("")
    lines.append(f"For each paper, how many of the {len(pipelines)} pipelines "
                 f"reached F1 ≥ {args.success_threshold}? "
                 f"A paper where 0 succeeded is *universally hard*; "
                 f"one where all succeeded is *easy*.")
    lines.append("")
    success_counts: Counter = Counter()
    for pid in papers_seen:
        n = sum(1 for f in paper_f1[pid].values()
                if f >= args.success_threshold)
        success_counts[n] += 1
    lines.append("| # pipelines that succeeded | # papers | share |")
    lines.append("|---:|---:|---:|")
    total_papers = len(papers_seen)
    for n in range(len(pipelines) + 1):
        count = success_counts.get(n, 0)
        share = 100 * count / total_papers if total_papers else 0
        lines.append(f"| {n} | {count} | {share:.1f}% |")
    lines.append("")
    # Hardest papers (0 successes, non-empty gold)
    hard = [p for p in papers_seen
            if paper_gold_size.get(p, 0) > 0
            and not any(f >= args.success_threshold for f in paper_f1[p].values())]
    lines.append(f"**Universally hard ({len(hard)} papers — 0 pipelines hit "
                 f"F1≥{args.success_threshold}, gold non-empty):**")
    lines.append("")
    lines.append("| paper_id | year | gold size | best F1 | best pipeline |")
    lines.append("|---|---:|---:|---:|---|")
    sample = sorted(hard,
                    key=lambda p: -max((paper_f1[p].get(pp, 0)
                                        for pp in pipelines), default=0))[:15]
    for p in sample:
        year = year_map.get(p) or _year_from_id(p) or ""
        best_pp, best_f1 = max(paper_f1[p].items(),
                               key=lambda kv: kv[1], default=("", 0.0))
        lines.append(f"| `{p}` | {year} | {paper_gold_size.get(p, 0)} | "
                     f"{best_f1:.2f} | `{best_pp}` |")
    lines.append("")

    # ---------------- (2) Disagreement zones ----------------
    lines.append("## 2. Disagreement zones")
    lines.append("")
    n_pipes = len(pipelines)
    lo, hi = int(n_pipes * 0.3), int(n_pipes * 0.7)
    lines.append(f"Papers where between {lo} and {hi} of {n_pipes} pipelines "
                 f"hit F1≥{args.success_threshold} — these are the *ambiguous* "
                 f"cases. Likely failure modes: ROR ambiguity (Stanford U vs "
                 f"SLAC vs Stanford CS), name normalisation (CMU vs Carnegie "
                 f"Mellon University), or OCR'd characters that some parsers "
                 f"fix and others don't.")
    lines.append("")
    disagreement = [p for p in papers_seen
                    if lo <= sum(1 for f in paper_f1[p].values()
                                 if f >= args.success_threshold) <= hi
                    and paper_gold_size.get(p, 0) > 0]
    disagreement.sort(key=lambda p: -paper_gold_size.get(p, 0))
    lines.append(f"**{len(disagreement)} papers in the disagreement zone.** "
                 f"Top by gold size:")
    lines.append("")
    lines.append("| paper_id | year | gold size | # success | best F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for p in disagreement[:15]:
        year = year_map.get(p) or _year_from_id(p) or ""
        n_succ = sum(1 for f in paper_f1[p].values()
                     if f >= args.success_threshold)
        best_f1 = max((paper_f1[p].get(pp, 0) for pp in pipelines), default=0)
        lines.append(f"| `{p}` | {year} | {paper_gold_size.get(p, 0)} | "
                     f"{n_succ}/{n_pipes} | {best_f1:.2f} |")
    lines.append("")

    # ---------------- (3) Error type breakdown per pipeline ----------------
    lines.append("## 3. Error type breakdown per pipeline")
    lines.append("")
    lines.append("- **TP**: ROR id matched (self or parent/child of gold)")
    lines.append("- **FP_wrong_org**: pipeline emitted a ROR id not in gold's equivalence")
    lines.append("- **FN_link_fail**: gold institution IS in the predicted raw "
                 "strings (or a 4+-char token of its canonical name is), "
                 "but the linker landed elsewhere or returned None — "
                 "*the extraction worked, the linking didn't*")
    lines.append("- **FN_extract_fail**: gold institution is nowhere in the "
                 "predicted raw strings — *the extraction missed it*")
    lines.append("")
    lines.append("| pipeline | TP | FP_wrong_org | FN_link_fail | FN_extract_fail "
                 "| FP/TP | FN_link/(FN_link+FN_extract) | F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for ppid in order:
        c = pp_totals[ppid]
        tp, fp = c["tp"], c["fp"]
        fnl, fne = c["fn_link"], c["fn_extract"]
        fp_ratio = fp / tp if tp else 0.0
        link_share = fnl / (fnl + fne) if (fnl + fne) else 0.0
        lines.append(
            f"| `{ppid}` | {tp} | {fp} | {fnl} | {fne} | "
            f"{fp_ratio:.2f} | {link_share:.2f} | {overall_f1[ppid]:.3f} |"
        )
    lines.append("")
    lines.append("Reading this table:")
    lines.append("- High **FP/TP ratio** = noisy linker (ROR API picking the "
                 "wrong sibling org). Fixable with country-context lookup or "
                 "stricter confidence threshold.")
    lines.append("- High **FN_link/(FN_link+FN_extract) share** = extraction is "
                 "fine, linking is the bottleneck. This is where most of the "
                 "headroom lives.")
    lines.append("- High **FN_extract** = the LLM or parser genuinely missed "
                 "the institution from the source text.")
    lines.append("")

    # ---------------- (4) Decade-stratified failure mix ----------------
    lines.append("## 4. Decade-stratified failure modes")
    lines.append("")
    lines.append("Shifts in FN_link / FN_extract / FP_wrong_org by decade, "
                 "summed across the top-10 pipelines.")
    lines.append("")
    top_pipes = order[:10]
    lines.append("| decade | TP | FP_wrong_org | FN_link_fail | FN_extract_fail | "
                 "FN_link share | papers |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for dec in decades_chrono:
        tp = fp = fnl = fne = 0
        for ppid in top_pipes:
            c = ppd_totals.get((ppid, dec), Counter())
            tp += c.get("tp", 0)
            fp += c.get("fp", 0)
            fnl += c.get("fn_link", 0)
            fne += c.get("fn_extract", 0)
        share = fnl / (fnl + fne) if (fnl + fne) else 0.0
        papers = sum(1 for p in papers_seen
                     if _decade(year_map.get(p) or _year_from_id(p)) == dec)
        lines.append(f"| {dec} | {tp} | {fp} | {fnl} | {fne} | {share:.2f} | "
                     f"{papers} |")
    lines.append("")
    lines.append("Interpretation: a rising **FN_extract** share with older "
                 "decades means OCR/parser failures dominate at the front of "
                 "the corpus. A rising **FN_link** share means the institution "
                 "IS extracted but ROR can't resolve the (often abbreviated, "
                 "OCR-corrupted) historical name.")
    lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    logger.info("wrote %s (%d lines)", args.out, len(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
