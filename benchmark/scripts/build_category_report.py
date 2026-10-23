"""Per-category breakdown report. For every dir, splits rows by
question_category and emits matrices (dir × category) for coverage,
faithfulness (Qwen judge), set-level F1, ROUGE-L, BERTScore.

This is the breakdown you need to disentangle "old vs new pipeline"
results from "old vs new question mix" results — since the
field_evolution rows that previously crashed (now answered) drag the
new pipeline averages down.

Categories (from testset.classifier_gold.question_category):
    overview_university, u2u_comparison, cross_country_method,
    university_country_comparison, field_evolution, paper_vs_field

Writes:
    reports/eval_scico/_category_report.md

Usage:
    python scripts/build_category_report.py
    python scripts/build_category_report.py --judge qwen_qwen3_5_397b_a17b
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


CATEGORIES = [
    "overview_university",
    "university_to_university_comparison",
    "cross_country_method",
    "university_country_comparison",
    "field_evolution",
]
CAT_SHORT = {
    "overview_university": "ov_uni",
    "university_to_university_comparison": "u2u",
    "cross_country_method": "x_country",
    "university_country_comparison": "uni_x_country",
    "field_evolution": "field_ev",
}


def _load_jsonl(p: Path):
    if not p.is_file():
        return []
    out = []
    for line in p.open():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _id_to_category(testset_path: Path) -> Dict[str, str]:
    out = {}
    for r in _load_jsonl(testset_path):
        rid = r.get("id")
        cg = r.get("classifier_gold") or {}
        cat = cg.get("question_category")
        if rid and cat:
            out[rid] = cat
    return out


def _nanmean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return (sum(xs) / len(xs)) if xs else float("nan")


def _fmt(v, prec=3):
    if v is None or (isinstance(v, float) and (v != v)):
        return "—"
    return f"{v:.{prec}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("reports/eval_scico"))
    ap.add_argument("--testset", type=Path,
                    default=Path("data/eval/scico_testset.jsonl"))
    ap.add_argument("--judge", default="qwen_qwen3_5_397b_a17b",
                    help="Judge slug for the faithfulness table (default: qwen-3.5-397b)")
    ap.add_argument("--out", type=Path,
                    default=Path("reports/eval_scico/_category_report.md"))
    ap.add_argument("--min-ok", type=int, default=50,
                    help="Drop dirs with fewer than this many healthy rows.")
    args = ap.parse_args()

    id2cat = _id_to_category(args.testset)
    if not id2cat:
        print(f"ERROR: no testset categories loaded from {args.testset}")
        return 1

    # Collect per-row data per dir
    dirs_data: Dict[str, Dict[str, Any]] = {}
    for d in sorted(args.root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        if not (name.startswith("agent_") or name.startswith("pipeline_")):
            continue
        if "smoke" in name or name.endswith("_test"):
            continue
        af = d / "answers.jsonl"
        if not af.is_file():
            continue

        ok_ids = set()
        for r in _load_jsonl(af):
            if r.get("error"):
                continue
            if not (r.get("summary_text") or "").strip():
                continue
            if not r.get("records"):
                continue
            if r.get("id"):
                ok_ids.add(r["id"])
        if len(ok_ids) < args.min_ok:
            continue

        # IR metrics per row
        ir_rows: Dict[str, Dict[str, float]] = {}
        for r in _load_jsonl(d / "ir_metrics.jsonl"):
            if r.get("type") == "summary":
                continue
            rid = r.get("id")
            if rid:
                ir_rows[rid] = r

        # Judge ragas per row
        judge_rows: Dict[str, Dict[str, float]] = {}
        jf = d / f"ragas_{args.judge}.jsonl"
        if jf.is_file():
            for r in _load_jsonl(jf):
                rid = r.get("id")
                if rid:
                    judge_rows[rid] = r

        dirs_data[name] = {
            "ok_ids": ok_ids,
            "ir": ir_rows,
            "judge": judge_rows,
        }

    if not dirs_data:
        print(f"No dirs survived the >= {args.min_ok}-ok filter.")
        return 0

    # Build per-category matrices
    md = []
    md.append("# Per-category breakdown\n")
    md.append(f"_Filter: dirs with ≥ {args.min_ok} healthy rows. "
              f"Judge for faithfulness column: `{args.judge}`._\n")
    md.append(f"_Categories: {', '.join(CAT_SHORT[c] for c in CATEGORIES)}._\n")

    # ---- Matrix 1: Coverage (ok_count per category) ----
    md.append("\n## 1. Coverage per category (rows with healthy answer)\n")
    md.append("| Dir | " + " | ".join(CAT_SHORT[c] for c in CATEGORIES) + " | total |")
    md.append("|---|" + "|".join(["---:"] * (len(CATEGORIES) + 1)) + "|")
    for name in sorted(dirs_data):
        ok = dirs_data[name]["ok_ids"]
        per_cat = [sum(1 for rid in ok if id2cat.get(rid) == c) for c in CATEGORIES]
        md.append(f"| {name} | " + " | ".join(str(x) for x in per_cat) + f" | {len(ok)} |")

    # ---- Matrix 2: Faithfulness (per category, single judge) ----
    md.append(f"\n## 2. Faithfulness per category (RAGAS, judge: {args.judge})\n")
    md.append("_(Each cell: mean / n. n = rows judged in that category for this dir.)_\n")
    md.append("| Dir | " + " | ".join(CAT_SHORT[c] for c in CATEGORIES) + " | overall |")
    md.append("|---|" + "|".join(["---:"] * (len(CATEGORIES) + 1)) + "|")
    for name in sorted(dirs_data):
        per_cat_cells = []
        all_vals = []
        for c in CATEGORIES:
            vals = []
            for rid, r in dirs_data[name]["judge"].items():
                if id2cat.get(rid) != c:
                    continue
                v = r.get("faithfulness")
                if isinstance(v, (int, float)):
                    vals.append(v)
            all_vals.extend(vals)
            if vals:
                per_cat_cells.append(f"{_nanmean(vals):.3f}/{len(vals)}")
            else:
                per_cat_cells.append("—")
        overall = (f"{_nanmean(all_vals):.3f}/{len(all_vals)}"
                   if all_vals else "—")
        md.append(f"| {name} | " + " | ".join(per_cat_cells) + f" | {overall} |")

    # ---- Matrix 3: F1_set (retrieval) per category ----
    md.append("\n## 3. F1_set (set-level retrieval F1) per category\n")
    md.append("_(Each cell: mean F1_set / n. n = rows with IR scoring in this category.)_\n")
    md.append("| Dir | " + " | ".join(CAT_SHORT[c] for c in CATEGORIES) + " | overall |")
    md.append("|---|" + "|".join(["---:"] * (len(CATEGORIES) + 1)) + "|")
    for name in sorted(dirs_data):
        per_cat_cells = []
        all_vals = []
        for c in CATEGORIES:
            vals = []
            for rid, r in dirs_data[name]["ir"].items():
                if id2cat.get(rid) != c:
                    continue
                v = r.get("F1_set")
                if isinstance(v, (int, float)):
                    vals.append(v)
            all_vals.extend(vals)
            if vals:
                per_cat_cells.append(f"{_nanmean(vals):.3f}/{len(vals)}")
            else:
                per_cat_cells.append("—")
        overall = (f"{_nanmean(all_vals):.3f}/{len(all_vals)}"
                   if all_vals else "—")
        md.append(f"| {name} | " + " | ".join(per_cat_cells) + f" | {overall} |")

    # ---- Matrix 4: ROUGE-L vs evidence ----
    md.append("\n## 4. ROUGE-L vs evidence per category\n")
    md.append("| Dir | " + " | ".join(CAT_SHORT[c] for c in CATEGORIES) + " | overall |")
    md.append("|---|" + "|".join(["---:"] * (len(CATEGORIES) + 1)) + "|")
    for name in sorted(dirs_data):
        per_cat_cells = []
        all_vals = []
        for c in CATEGORIES:
            vals = []
            for rid, r in dirs_data[name]["ir"].items():
                if id2cat.get(rid) != c:
                    continue
                v = r.get("rougeL_vs_evidence")
                if isinstance(v, (int, float)):
                    vals.append(v)
            all_vals.extend(vals)
            if vals:
                per_cat_cells.append(f"{_nanmean(vals):.3f}")
            else:
                per_cat_cells.append("—")
        overall = (f"{_nanmean(all_vals):.3f}" if all_vals else "—")
        md.append(f"| {name} | " + " | ".join(per_cat_cells) + f" | {overall} |")

    # ---- Matrix 5: BERTScore F1 vs evidence ----
    md.append("\n## 5. BERTScore F1 vs evidence per category\n")
    md.append("| Dir | " + " | ".join(CAT_SHORT[c] for c in CATEGORIES) + " | overall |")
    md.append("|---|" + "|".join(["---:"] * (len(CATEGORIES) + 1)) + "|")
    for name in sorted(dirs_data):
        per_cat_cells = []
        all_vals = []
        for c in CATEGORIES:
            vals = []
            for rid, r in dirs_data[name]["ir"].items():
                if id2cat.get(rid) != c:
                    continue
                v = r.get("bertscore_f1_vs_evidence")
                if isinstance(v, (int, float)):
                    vals.append(v)
            all_vals.extend(vals)
            if vals:
                per_cat_cells.append(f"{_nanmean(vals):.3f}")
            else:
                per_cat_cells.append("—")
        overall = (f"{_nanmean(all_vals):.3f}" if all_vals else "—")
        md.append(f"| {name} | " + " | ".join(per_cat_cells) + f" | {overall} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"# wrote {args.out}  ({len(dirs_data)} dirs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
