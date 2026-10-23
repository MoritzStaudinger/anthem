"""Transparent cost accounting for the ANTHEM extraction side (Table 2 + corpus).

Three cost surfaces the RAG-benchmark cost script does NOT cover:
  1. Extraction ablation (Table 2): each model over the 781-paper gold set.
  2. Corpus construction: the chosen main model over all 117,386 papers.
     The enriched corpus stores no per-paper tokens, so this is EXTRAPOLATED
     from the 781-paper token rate (flagged as such).
  3. Local-GPU baseline (Qwen 2.5 7B LoRA): a compute cost in A100-hours, not
     tokens — reported as hours (and euros if --a100-eur-hr is given).

Token totals are read from reports/pipeline_comparison/comparison_report.json.
Prices come from a rate card (--rates JSON, per 1M tokens) so every number is
auditable and you can drop in your actual negotiated rates.

Rate card format (per 1M tokens; keyed by pipeline_id):
  {
    "gemini25_combined": {"name": "Gemini 2.5 Flash", "in_per_1m": 0.10, "out_per_1m": 0.40},
    "lora_qwen25_7b_aff": {"name": "Qwen 2.5 7B (LoRA)", "gpu": true},
    ...
  }

Usage:
  python scripts/cost_extraction.py --rates rates_extraction.json --a100-eur-hr 2.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

REPO = Path(__file__).resolve().parent.parent


def _fmt_eur(x: Optional[float]) -> str:
    return "n/a" if x is None else f"€{x:,.2f}"


def _fmt_eur4(x: Optional[float]) -> str:
    return "n/a" if x is None else f"€{x:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path,
                    default=REPO / "reports/pipeline_comparison/comparison_report.json")
    ap.add_argument("--rates", type=Path, required=True,
                    help="Rate card JSON (per 1M tokens), keyed by pipeline_id.")
    ap.add_argument("--corpus-size", type=int, default=117386)
    ap.add_argument("--gold-size", type=int, default=781)
    ap.add_argument("--main-pipeline", default="gemini25_combined",
                    help="Pipeline used to build the released corpus.")
    ap.add_argument("--a100-eur-hr", type=float, default=None,
                    help="€ per A100-hour for the local-GPU baseline cost.")
    ap.add_argument("--construction-actual-eur", type=float, default=None,
                    help="Actual paid cost of the full-corpus construction (then-current "
                         "prices). Reported as the headline; the current-price "
                         "extrapolation is kept as a 'reproduce-today' comparison.")
    ap.add_argument("--gpu-training-hrs", type=float, default=None,
                    help="One-time LoRA training cost in A100-hours (reported separately "
                         "from the full-corpus inference extrapolation).")
    ap.add_argument("--eur-per-usd", type=float, default=1.0,
                    help="If your rate card is in USD, set the USD->EUR factor "
                         "(default 1.0 = rates already in the target currency).")
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports/cost")
    args = ap.parse_args()

    rep = json.loads(args.report.read_text())
    by = {pl.get("pipeline_id"): pl for pl in rep["pipelines"]}
    rates = json.loads(args.rates.read_text())
    f = args.eur_per_usd

    rows = []          # ablation rows
    gpu_rows = []      # gpu rows
    for pid, rate in rates.items():
        if pid.startswith("_") or not isinstance(rate, dict):
            continue
        pl = by.get(pid)
        if not pl:
            rows.append({"pid": pid, "name": rate.get("name", pid), "absent": True})
            continue
        ti = pl.get("tokens_in_total") or 0
        to = pl.get("tokens_out_total") or 0
        sec = pl.get("extraction_seconds_total") or 0.0
        name = rate.get("name", pid)
        if rate.get("gpu"):
            hrs_781 = sec / 3600.0
            hrs_corpus = (sec / args.gold_size) * args.corpus_size / 3600.0
            eur = (hrs_corpus * args.a100_eur_hr) if args.a100_eur_hr else None
            gpu_rows.append({"pid": pid, "name": name, "sec_per_paper": sec / args.gold_size,
                             "hrs_781": hrs_781, "hrs_corpus": hrs_corpus, "eur": eur})
            continue
        ir = rate.get("in_per_1m"); orr = rate.get("out_per_1m")
        if ir is None or orr is None:
            rows.append({"pid": pid, "name": name, "ti": ti, "to": to, "cost": None})
            continue
        cost = (ti / 1e6 * ir + to / 1e6 * orr) * f
        rows.append({"pid": pid, "name": name, "ti": ti, "to": to,
                     "cost": cost, "per_paper": cost / args.gold_size})

    # Corpus construction = main pipeline extrapolated to the full corpus.
    main = next((r for r in rows if r.get("pid") == args.main_pipeline), None)
    construction = None
    if main and main.get("cost") is not None:
        scale = args.corpus_size / args.gold_size
        construction = {
            "name": main["name"],
            "tok_in": int((by[args.main_pipeline]["tokens_in_total"] or 0) * scale),
            "tok_out": int((by[args.main_pipeline]["tokens_out_total"] or 0) * scale),
            "cost": main["cost"] * scale,
            "per_paper": main["per_paper"],
        }

    # ---- render ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    md = []
    md.append("# ANTHEM extraction — transparent cost accounting\n")
    md.append(f"- Rate card: `{args.rates.name}` (per 1M tokens"
              f"{'' if f == 1.0 else f', ×{f} USD→EUR'})")
    md.append(f"- Gold set: {args.gold_size} papers | full corpus: "
              f"{args.corpus_size:,} papers")
    md.append("- Token totals from `comparison_report.json` (measured on the 781-paper "
              "gold set). Corpus construction is EXTRAPOLATED "
              f"(×{args.corpus_size/args.gold_size:.1f}) — the enriched corpus stores no "
              "per-paper tokens.")
    md.append("")

    md.append("## 1. Extraction ablation cost (Table 2 models, 781-paper gold set)")
    md.append("")
    md.append("| Model | pipeline_id | tok in | tok out | cost (781) | € / paper |")
    md.append("|---|---|---:|---:|---:|---:|")
    for r in rows:
        if r.get("absent"):
            md.append(f"| {r['name']} | {r['pid']} | — | — | (not in report) | — |")
            continue
        pp = _fmt_eur4(r.get("per_paper")) if r.get("cost") is not None else "n/a"
        md.append(f"| {r['name']} | {r['pid']} | {r['ti']:,} | {r['to']:,} "
                  f"| {_fmt_eur(r.get('cost'))} | {pp} |")
    md.append("")

    if construction:
        md.append("## 2. Corpus construction cost (main pipeline over full corpus)")
        md.append("")
        md.append(f"- Model: **{construction['name']}** over **{args.corpus_size:,}** papers")
        md.append(f"- Extrapolated tokens: {construction['tok_in']:,} in / "
                  f"{construction['tok_out']:,} out")
        if args.construction_actual_eur is not None:
            md.append(f"- **Actual cost as paid (then-current prices): "
                      f"{_fmt_eur(args.construction_actual_eur)}**")
            md.append(f"- For reference, at today's OpenRouter prices the same run "
                      f"would cost ~{_fmt_eur(construction['cost'])} "
                      f"(API prices have dropped since the run)")
        else:
            md.append(f"- **Estimated construction cost: {_fmt_eur(construction['cost'])}** "
                      f"({_fmt_eur4(construction['per_paper'])}/paper × {args.corpus_size:,})")
        md.append("")

    if gpu_rows:
        md.append("## 3. Local-GPU baseline (compute cost, not tokens)")
        md.append("")
        if args.gpu_training_hrs is not None:
            tr_eur = (args.gpu_training_hrs * args.a100_eur_hr) if args.a100_eur_hr else None
            md.append(f"- One-time LoRA **training**: {args.gpu_training_hrs:.0f} A100-h "
                      f"({_fmt_eur(tr_eur)})")
        md.append("- Full-corpus **inference** (extrapolated from the 781-paper per-paper "
                  "latency; the paper's own estimate is ~500–1000 A100-h):")
        md.append("")
        md.append("| Model | s/paper | A100-h (781) | A100-h (full corpus, inference) | € (inference) |")
        md.append("|---|---:|---:|---:|---:|")
        for g in gpu_rows:
            md.append(f"| {g['name']} | {g['sec_per_paper']:.1f} | {g['hrs_781']:.2f} "
                      f"| {g['hrs_corpus']:.0f} | {_fmt_eur(g['eur'])} |")
        md.append("")
        if args.a100_eur_hr is None:
            md.append("_Pass `--a100-eur-hr` to monetise the GPU hours._")
        md.append("")

    md.append("> **Assumptions to verify:** all prices come from the rate card; "
              "corpus-construction tokens are extrapolated from the 781-paper rate; "
              "GPU hours assume the full corpus runs at the measured per-paper latency. "
              "ROR API linking is free (no per-call charge; ~348k affiliation matches).")

    out = args.out_dir / "cost_extraction.md"
    out.write_text("\n".join(md))

    gpu = gpu_rows[0] if gpu_rows else None
    summary = {
        "construction": ({
            "model": construction["name"],
            "actual_eur": args.construction_actual_eur,
            "extrapolated_eur": construction["cost"],
        } if construction else None),
        "ablation_eur": sum(r["cost"] for r in rows
                            if not r.get("absent") and r.get("cost") is not None),
        "ablation_rows": [{"name": r["name"], "eur": r.get("cost")}
                          for r in rows if not r.get("absent")],
        "gpu": ({
            "model": gpu["name"],
            "train_hrs": args.gpu_training_hrs,
            "infer_hrs": gpu["hrs_corpus"],
            "train_eur": (args.gpu_training_hrs * args.a100_eur_hr)
                         if (args.gpu_training_hrs and args.a100_eur_hr) else None,
            "infer_eur": gpu["eur"],
        } if gpu else None),
    }
    (args.out_dir / "cost_extraction.json").write_text(json.dumps(summary, indent=2))
    print("\n".join(md))
    print(f"\n# wrote {out} and cost_extraction.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
