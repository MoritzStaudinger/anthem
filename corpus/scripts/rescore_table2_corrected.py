"""Re-score Table 2 pipelines against the HUMAN-corrected ROR gold.

The published Table 2 F1_hier is measured against linker-anchored gold
(authors_normalized_json ror_ids obtained by ROR lookup). The ROR annotation
showed that gold inflates multisearch-linked pipelines. This re-scores each
pipeline's predicted ror_ids against the hand-annotated per-string gold
(from adjudication_sheet.xlsx), reporting old vs corrected F1_hier + the delta,
and emitting per-paper counts for a bootstrap.

Corrected gold per paper = union over the paper's raw affiliation strings of the
human correct_ror (blank/none = no ROR). Predicted per paper = the pipeline's
ror_ids from merged_per_paper.jsonl (unchanged — we only fix the gold).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from relink_strategy_compare import load_gold, norm_ror, make_expand, score_ror_hier  # noqa
from score_relink_vs_human import load_human_gold  # noqa


def f1(tp, fp, fn):
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    return P, R, (2 * P * R / (P + R) if (P + R) else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet", type=Path,
                    default=REPO / "reports/relink_compare/adjudication_sheet.xlsx")
    ap.add_argument("--gold-csv", type=Path,
                    default=Path("/home/moritz/Downloads/11688_Singing_from_the_Same_AN_Data/"
                                 "sample_data/affiliation_ground_truth.csv"))
    ap.add_argument("--merged", type=Path,
                    default=REPO / "reports/pipeline_comparison/merged_per_paper.jsonl")
    ap.add_argument("--hierarchy-cache", type=Path,
                    default=REPO / "reports/ror_hierarchy_cache.warmed.json")
    ap.add_argument("--pipelines", nargs="+",
                    default=["ours", "claude_sonnet_combined", "gpt5nano_combined",
                             "openalex", "s2orc"])
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports/pipeline_comparison")
    args = ap.parse_args()

    human, _ = load_human_gold(args.sheet)
    papers, _ = load_gold(args.gold_csv)
    hier = json.loads(args.hierarchy_cache.read_text())
    expand = make_expand(hier, False)

    # corrected per-paper gold (by anthology id)
    corr = {}
    ann_strings = 0
    for pid, p in papers.items():
        g = set()
        for s in p["raw_strings"]:
            k = s.lower()
            if k in human:
                ann_strings += 1
                g |= human[k]
        corr[pid] = g

    merged = {}
    for line in args.merged.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            merged[d["paper_id"]] = d

    pipes = [p for p in args.pipelines
             if any(p in (m.get("predictions") or {}) for m in merged.values())]
    missing = [p for p in args.pipelines if p not in pipes]

    # accumulate old + corrected counts
    acc = {p: {"o": [0, 0, 0], "c": [0, 0, 0]} for p in pipes}
    counts = {pid: {"paper_id": pid, "counts": {}} for pid in merged}
    for pid, m in merged.items():
        gold_old = {b for b in (norm_ror(x) for x in (m.get("gold") or {}).get("ror_ids") or []) if b}
        gold_cor = corr.get(pid, set())
        for p in pipes:
            e = (m.get("predictions") or {}).get(p) or {}
            pred = set() if (not isinstance(e, dict) or e.get("error")) else \
                {b for b in (norm_ror(x) for x in e.get("ror_ids") or []) if b}
            to = score_ror_hier(gold_old, pred, expand)
            tc = score_ror_hier(gold_cor, pred, expand)
            for i in range(3):
                acc[p]["o"][i] += to[i]; acc[p]["c"][i] += tc[i]
            counts[pid]["counts"][p] = list(tc)

    (args.out_dir / "table2_corrected_counts.jsonl").write_text(
        "\n".join(json.dumps(r) for r in counts.values()) + "\n")

    md = []
    md.append("# Table 2 re-scored against HUMAN-corrected ROR gold\n")
    md.append(f"- Corrected gold: {ann_strings} paper-string links from "
              "hand annotation (blank/none = no ROR)")
    if missing:
        md.append(f"- (not in merged predictions: {missing})")
    md.append("")
    md.append("| Pipeline | F1_hier (published/old gold) | F1_hier (corrected gold) | Δ | "
              "P_corr | R_corr |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for p in pipes:
        _, _, fo = f1(*acc[p]["o"])
        Pc, Rc, fc = f1(*acc[p]["c"])
        md.append(f"| {p} | {fo:.3f} | **{fc:.3f}** | {fc-fo:+.3f} | {Pc:.3f} | {Rc:.3f} |")
    md.append("")
    md.append(f"Per-paper counts (corrected) for bootstrap: `table2_corrected_counts.jsonl`")
    out = args.out_dir / "table2_corrected.md"
    out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n# wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
