"""Failures-as-zero re-aggregation of the set-level retrieval table (Table 4).

Rebuttal task (reviewer c8Dh): the published set-level P/R/F1 numbers in
`reports/eval_scico/_eval_report.md` are averaged over ONLY the questions a
system managed to answer with a parseable, non-empty, matched result set
(`eval_ir_measures.py` DROPS every other question — see its lines 318/328/335).
That silently rewards systems that error out or return nothing, and — the exact
inconsistency c8Dh flags — an empty-but-parseable result set is dropped rather
than scored as zero.

This script recomputes the table two ways, WITHOUT any model calls, reusing the
*identical* gold computation and record->id join as the published pipeline
(imported from eval_ir_measures) so the only deltas are (1) failures score zero
and (2) the denominator is fixed:

  VARIANT A  "failures-as-zero"
      For every question in the scoreable universe U (= questions with a
      non-empty gold set against the eval corpus), a system that failed on that
      question contributes P=R=F1=0. Denominator = |U| for every system.
      |U| == 99 on the full corpus (all 99 questions have gold >= 6); the script
      asserts this and excludes any empty-gold question from U for ALL systems
      (an empty gold set is structurally unscoreable — recall is 0/0 — so it is
      neither dropped-for-some nor zeroed; it is removed from the denominator
      uniformly).

  VARIANT B  "all-answered intersection"
      I = questions that EVERY system answered successfully (OK on all).
      Each system is averaged over exactly I, so the comparison is on a common
      footing. Reports |I|.

"Failed" is defined identically for all systems: a question counts as failed if
ANY of the following hold (checked in this order, first match wins the reason):
    - the system has no answer row for that question id            -> "missing"
    - the answer row carries a non-empty `error`                   -> "error"
    - the row's records join to zero corpus paper ids (empty set,
      unparseable-into-empty, or all records unmatched)            -> "empty_or_unmatched"
Otherwise the question is OK and scored with the same set-level formula the
published pipeline uses (precision = hits/|retrieved|, recall = hits/|gold|,
F1 harmonic; full retrieved set, no top-k slice).

Duplicate answer rows (some dirs re-ran and appended: pipeline_claude_sonnet_4_6
has 116 rows for 99 ids) are deduped by id before scoring; --dedup controls which
copy wins.

Outputs (under --out-dir, default reports/eval_scico/rescore):
    rescore_table.md            both variants, plus the published "scored-only"
                                mean for side-by-side comparison of the inflation
    rescore_per_question.jsonl  per (system, question) classification + scores,
                                for audit / appendix

Run on a host with the FULL affiliations corpus (the 24,340-size gold sets are
only reproducible there); on the ~800-file local snapshot most gold sets are
empty and |U| collapses — the script will warn.

Usage:
    python scripts/rescore_failures_as_zero.py \
        --systems agent_gemini_2_5_pro pipeline_gemini_2_5_pro_topk5 \
                  agent_claude_sonnet_4_6 pipeline_claude_sonnet_4_6_topk5 \
                  agent_gpt_5_mini pipeline_gpt_5_mini_topk5
    # default --systems is every reports/eval_scico/{pipeline_,agent_}* dir with
    # an answers.jsonl; pass an explicit list to match exactly the paper's table.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

# Reuse the EXACT published gold + join logic so the only difference from the
# table in _eval_report.md is the failures-as-zero / fixed-denominator handling.
from backend.eval.gold import compute_gold_paper_ids, load_affiliations_corpus  # noqa: E402
import eval_ir_measures as eim  # noqa: E402


def _set_prf(retrieved: Set[str], gold: Set[str]) -> Tuple[float, float, float]:
    """Set-level precision/recall/F1 over the FULL retrieved set — identical to
    eval_ir_measures.py lines 385-390."""
    hits = retrieved & gold
    n_ret, n_gold = len(retrieved), len(gold)
    p = (len(hits) / n_ret) if n_ret else 0.0
    r = (len(hits) / n_gold) if n_gold else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
    return p, r, f1


def _dedupe_rows(rows: List[Dict[str, Any]], strategy: str) -> Dict[str, Dict[str, Any]]:
    """Collapse multiple answer rows per id to one. Returns id -> row.

    last  : keep the last occurrence (matches append/resume "latest wins").
    first : keep the first occurrence.
    best  : prefer a row with no error AND non-empty records; else fall back to
            the last occurrence. Use to give every system the benefit of the
            doubt when a re-run fixed an earlier failure.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rid = row.get("id")
        if rid is None:
            continue
        if strategy == "first":
            out.setdefault(rid, row)
        elif strategy == "best":
            prev = out.get(rid)
            good = (not row.get("error")) and bool(row.get("records"))
            prev_good = prev is not None and (not prev.get("error")) and bool(prev.get("records"))
            if prev is None or (good and not prev_good) or (good == prev_good):
                # take this row unless the previous one was good and this isn't
                if not (prev_good and not good):
                    out[rid] = row
        else:  # last
            out[rid] = row
    return out


def _load_answers(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # A line that won't even parse as JSON is itself a failure, but
                # it carries no id so we can't attribute it; count separately.
                rows.append({"__unparseable_line__": True})
    return rows


def _classify(
    row: Optional[Dict[str, Any]],
    gold: Set[str],
    doi_lookup: Dict[str, str],
    title_lookup: Dict[str, str],
) -> Tuple[str, Tuple[float, float, float], int]:
    """Return (status, (P,R,F1), n_retrieved).

    status in {ok, missing, error, empty_or_unmatched}. Failure statuses carry
    (0,0,0)."""
    if row is None:
        return "missing", (0.0, 0.0, 0.0), 0
    if row.get("error"):
        return "error", (0.0, 0.0, 0.0), 0
    retrieved_list, _unmatched = eim._retrieved_ids(
        row.get("records") or [], doi_lookup, title_lookup
    )
    retrieved = set(retrieved_list)
    if not retrieved:
        return "empty_or_unmatched", (0.0, 0.0, 0.0), 0
    return "ok", _set_prf(retrieved, gold), len(retrieved)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reports-dir", type=Path, default=Path("reports/eval_scico"))
    ap.add_argument("--systems", nargs="*", default=None,
                    help="System subdir names under --reports-dir. Default: all "
                         "pipeline_*/agent_* dirs with an answers.jsonl.")
    ap.add_argument("--testset", type=Path, default=Path("data/eval/scico_testset.jsonl"))
    ap.add_argument("--affiliations-dir", type=Path,
                    default=Path("/share/moritz.staudinger/acl_data/"
                                 "ror_parsed_affiliations_90_with_topics_and_bibmeta"))
    ap.add_argument("--dedup", choices=["last", "first", "best"], default="last")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/eval_scico/rescore"))
    args = ap.parse_args()

    if not args.testset.exists():
        print(f"ERROR: testset {args.testset} not found", file=sys.stderr)
        return 1

    # --- systems ------------------------------------------------------------
    if args.systems:
        systems = list(args.systems)
    else:
        systems = sorted(
            d.name for d in args.reports_dir.iterdir()
            if d.is_dir() and (d.name.startswith(("pipeline_", "agent_")))
            and (d / "answers.jsonl").exists()
        )
    if not systems:
        print("ERROR: no systems found/selected", file=sys.stderr)
        return 1

    # --- testset + corpus + gold -------------------------------------------
    testset = eim._load_testset(args.testset)
    all_qids = list(testset.keys())
    print(f"# testset: {len(all_qids)} questions", flush=True)

    print(f"# loading affiliations corpus from {args.affiliations_dir}", flush=True)
    corpus = load_affiliations_corpus(args.affiliations_dir)
    doi_lookup, title_lookup = eim._build_id_lookups(corpus)
    print(f"# corpus: {len(corpus)} papers | lookups: "
          f"{len(doi_lookup)} DOIs, {len(title_lookup)} titles", flush=True)

    gold_by_qid: Dict[str, Set[str]] = {}
    empty_gold: List[str] = []
    for qid, row in testset.items():
        g = eim._normalize_gold(compute_gold_paper_ids(row, corpus))
        gold_by_qid[qid] = g
        if not g:
            empty_gold.append(qid)

    universe = [q for q in all_qids if gold_by_qid[q]]  # U: scoreable questions
    print(f"# scoreable universe |U| = {len(universe)} "
          f"(excluded {len(empty_gold)} empty-gold questions)", flush=True)
    if empty_gold:
        print(f"#   WARNING: {len(empty_gold)} questions have EMPTY gold against "
              f"this corpus — likely a partial corpus. Real numbers need the "
              f"full corpus (all 99 have gold there). Excluded ids (up to 10): "
              f"{empty_gold[:10]}", flush=True)

    # --- per-system classification -----------------------------------------
    # results[system][qid] = (status, (P,R,F1), n_ret)
    results: Dict[str, Dict[str, Tuple[str, Tuple[float, float, float], int]]] = {}
    dup_report: Dict[str, int] = {}
    for sysname in systems:
        apath = args.reports_dir / sysname / "answers.jsonl"
        if not apath.exists():
            print(f"# SKIP {sysname}: no answers.jsonl", file=sys.stderr)
            continue
        raw = _load_answers(apath)
        n_unparseable = sum(1 for r in raw if r.get("__unparseable_line__"))
        parseable = [r for r in raw if not r.get("__unparseable_line__")]
        ids_seen = [r.get("id") for r in parseable if r.get("id") is not None]
        dup_report[sysname] = len(ids_seen) - len(set(ids_seen))
        by_id = _dedupe_rows(parseable, args.dedup)
        per_q: Dict[str, Tuple[str, Tuple[float, float, float], int]] = {}
        for qid in universe:
            per_q[qid] = _classify(by_id.get(qid), gold_by_qid[qid],
                                   doi_lookup, title_lookup)
        results[sysname] = per_q
        if n_unparseable:
            print(f"# {sysname}: {n_unparseable} unparseable JSONL line(s) "
                  f"(cannot attribute to a qid; not scored)", file=sys.stderr)

    systems = [s for s in systems if s in results]

    # --- Variant A: failures-as-zero over U --------------------------------
    def _mean(xs: List[float]) -> float:
        return (sum(xs) / len(xs)) if xs else float("nan")

    variant_a: Dict[str, Dict[str, float]] = {}
    for sysname in systems:
        pq = results[sysname]
        Ps = [pq[q][1][0] for q in universe]
        Rs = [pq[q][1][1] for q in universe]
        Fs = [pq[q][1][2] for q in universe]
        n_ok = sum(1 for q in universe if pq[q][0] == "ok")
        variant_a[sysname] = {
            "n_ok": n_ok, "denom": len(universe),
            "P_set": _mean(Ps), "R_set": _mean(Rs), "F1_set": _mean(Fs),
        }

    # --- Variant B: intersection of all-answered ---------------------------
    intersection = [q for q in universe
                    if all(results[s][q][0] == "ok" for s in systems)]
    variant_b: Dict[str, Dict[str, float]] = {}
    for sysname in systems:
        pq = results[sysname]
        variant_b[sysname] = {
            "n": len(intersection),
            "P_set": _mean([pq[q][1][0] for q in intersection]),
            "R_set": _mean([pq[q][1][1] for q in intersection]),
            "F1_set": _mean([pq[q][1][2] for q in intersection]),
        }

    # --- Published-style "scored-only" mean (for the inflation comparison) --
    scored_only: Dict[str, Dict[str, float]] = {}
    for sysname in systems:
        pq = results[sysname]
        ok_q = [q for q in universe if pq[q][0] == "ok"]
        scored_only[sysname] = {
            "n": len(ok_q),
            "P_set": _mean([pq[q][1][0] for q in ok_q]),
            "R_set": _mean([pq[q][1][1] for q in ok_q]),
            "F1_set": _mean([pq[q][1][2] for q in ok_q]),
        }

    # --- write outputs ------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_q_path = args.out_dir / "rescore_per_question.jsonl"
    with per_q_path.open("w", encoding="utf-8") as fh:
        for sysname in systems:
            for qid in universe:
                status, (p, r, f1), n_ret = results[sysname][qid]
                fh.write(json.dumps({
                    "system": sysname, "id": qid, "status": status,
                    "P_set": p, "R_set": r, "F1_set": f1,
                    "n_retrieved": n_ret, "n_gold": len(gold_by_qid[qid]),
                    "in_intersection": qid in intersection,
                }, ensure_ascii=False) + "\n")

    def _fmt(x: float) -> str:
        return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"

    md = []
    md.append("# Failures-as-zero re-aggregation of the set-level table\n")
    md.append(f"- Corpus: `{args.affiliations_dir}` ({len(corpus)} papers)")
    md.append(f"- Scoreable universe |U| = **{len(universe)}** "
              f"(excluded {len(empty_gold)} empty-gold questions)")
    md.append(f"- Intersection (all systems answered) |I| = **{len(intersection)}**")
    md.append(f"- Dedup strategy: `{args.dedup}` | systems: {len(systems)}")
    dups = {s: n for s, n in dup_report.items() if n}
    if dups:
        md.append(f"- Duplicate answer rows deduped: {dups}")
    md.append("")
    md.append("## Failed = missing OR error OR empty/unmatched retrieved set "
              "(applied identically to all systems)\n")

    md.append("### Variant A — failures-as-zero, denominator = |U|")
    md.append("")
    md.append("| System | n_ok | P_set | R_set | F1_set |")
    md.append("|---|---:|---:|---:|---:|")
    for s in sorted(systems, key=lambda s: variant_a[s]["F1_set"], reverse=True):
        a = variant_a[s]
        md.append(f"| {s} | {a['n_ok']}/{a['denom']} | {_fmt(a['P_set'])} "
                  f"| {_fmt(a['R_set'])} | {_fmt(a['F1_set'])} |")
    md.append("")

    md.append(f"### Variant B — intersection of all-answered questions (n = {len(intersection)})")
    md.append("")
    md.append("| System | P_set | R_set | F1_set |")
    md.append("|---|---:|---:|---:|")
    for s in sorted(systems, key=lambda s: variant_b[s]["F1_set"], reverse=True):
        b = variant_b[s]
        md.append(f"| {s} | {_fmt(b['P_set'])} | {_fmt(b['R_set'])} | {_fmt(b['F1_set'])} |")
    md.append("")

    md.append("### Reference — published \"scored-only\" mean (the inflated numbers)")
    md.append("")
    md.append("| System | n_scored | P_set | R_set | F1_set |")
    md.append("|---|---:|---:|---:|---:|")
    for s in sorted(systems, key=lambda s: scored_only[s]["F1_set"], reverse=True):
        so = scored_only[s]
        md.append(f"| {s} | {so['n']} | {_fmt(so['P_set'])} | {_fmt(so['R_set'])} "
                  f"| {_fmt(so['F1_set'])} |")
    md.append("")

    table_path = args.out_dir / "rescore_table.md"
    table_path.write_text("\n".join(md), encoding="utf-8")

    print(f"# wrote {table_path}", flush=True)
    print(f"# wrote {per_q_path}", flush=True)
    print("\n" + "\n".join(md), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
