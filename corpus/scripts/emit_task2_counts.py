"""Emit per-paper micro-F1 counts for the 3-pipeline extraction comparison.

Reads reports/pipeline_comparison/merged_per_paper.jsonl (gold + all pipeline
predictions, 781 papers) and writes per-paper [tp, fp, fn] for each target
pipeline, for two ROR tracks:
    hier  : hierarchy-aware set match (parent/child equivalence) — F1_hier
    exact : plain ROR set intersection                          — F1_ROR

Output (per track): reports/bootstrap/task2_<track>.jsonl, rows
    {"paper_id": ..., "counts": {"<pipeline>": [tp, fp, fn], ...}}
feed straight into bootstrap_paired.py.

Scoring reuses the exact helpers from relink_strategy_compare (norm_ror,
score_ror_hier, hierarchy expand) which themselves mirror
run_pipeline_comparison._score_ror_hier — so Task 2 and the ROR strategy table
are scored identically. A pipeline whose record is missing or carries `error`
contributes an empty prediction set (all gold -> FN), i.e. the inclusive
convention.

Usage:
    python scripts/emit_task2_counts.py
    python scripts/emit_task2_counts.py --pipelines gemini25_combined claude_sonnet_combined gpt5nano_combined
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from relink_strategy_compare import norm_ror, make_expand, score_ror_hier  # noqa: E402

DEFAULT_PIPELINES = ["gemini25_combined", "claude_sonnet_combined", "gpt5nano_combined"]


def _ror_set(entry) -> Set[str]:
    if not isinstance(entry, dict) or entry.get("error"):
        return set()
    return {b for b in (norm_ror(x) for x in (entry.get("ror_ids") or [])) if b}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", type=Path,
                    default=REPO / "reports/pipeline_comparison/merged_per_paper.jsonl")
    ap.add_argument("--hierarchy-cache", type=Path,
                    default=REPO / "reports/ror_hierarchy_cache.json")
    ap.add_argument("--pipelines", nargs="+", default=DEFAULT_PIPELINES)
    ap.add_argument("--include-related", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports/bootstrap")
    args = ap.parse_args()

    hier = json.loads(args.hierarchy_cache.read_text()) if args.hierarchy_cache.exists() else {}
    expand = make_expand(hier, args.include_related)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hier_rows: List[dict] = []
    exact_rows: List[dict] = []
    n = 0
    miss = {p: 0 for p in args.pipelines}
    with args.merged.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            pid = d.get("paper_id")
            gold = {b for b in (norm_ror(x) for x in ((d.get("gold") or {}).get("ror_ids") or [])) if b}
            preds = d.get("predictions") or {}
            hcounts: Dict[str, List[int]] = {}
            ecounts: Dict[str, List[int]] = {}
            for p in args.pipelines:
                entry = preds.get(p)
                if entry is None or (isinstance(entry, dict) and entry.get("error")):
                    miss[p] += 1
                pset = _ror_set(entry)
                tp, fp, fn = score_ror_hier(gold, pset, expand)
                hcounts[p] = [tp, fp, fn]
                ecounts[p] = [len(gold & pset), len(pset - gold), len(gold - pset)]
            hier_rows.append({"paper_id": pid, "counts": hcounts})
            exact_rows.append({"paper_id": pid, "counts": ecounts})
            n += 1

    (args.out_dir / "task2_hier.jsonl").write_text(
        "\n".join(json.dumps(r) for r in hier_rows) + "\n")
    (args.out_dir / "task2_exact.jsonl").write_text(
        "\n".join(json.dumps(r) for r in exact_rows) + "\n")

    # Sanity: print aggregate point F1 per pipeline per track (should match
    # comparison_report.json if the same gold/preds).
    def agg_f1(rows):
        out = {}
        for p in args.pipelines:
            tp = sum(r["counts"][p][0] for r in rows)
            fp = sum(r["counts"][p][1] for r in rows)
            fn = sum(r["counts"][p][2] for r in rows)
            denom = 2 * tp + fp + fn
            out[p] = (2 * tp / denom) if denom else 0.0
        return out
    print(f"# {n} papers | missing/error per pipeline: {miss}", flush=True)
    print(f"# aggregate F1_hier (point): "
          f"{ {k: round(v,4) for k,v in agg_f1(hier_rows).items()} }", flush=True)
    print(f"# aggregate F1_ROR  (point): "
          f"{ {k: round(v,4) for k,v in agg_f1(exact_rows).items()} }", flush=True)
    print(f"# wrote {args.out_dir/'task2_hier.jsonl'} and task2_exact.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
