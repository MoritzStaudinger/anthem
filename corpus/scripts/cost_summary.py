"""Merge all ANTHEM cost surfaces into one paper-ready table.

Consumes the JSON outputs of the two cost scripts plus the RAGAS judge cost:
  - cost_extraction.json   (corpus construction + extraction ablation + GPU baseline)
  - cost_per_system.jsonl  (RAG-benchmark generation, from cost_accounting.py)
  - ragas_*.jsonl          (per-answer judge_cost_usd, second-judge RAGAS run)

Emits reports/cost/cost_summary.md — the itemised euro table for the paper's
cost/transparency paragraph, with the commercial-API total vs. the local
open-baseline compute contrast.

USD figures (RAG + RAGAS) are converted at --usd-eur; extraction figures are
already in the rate-card currency. State the FX assumption in the paper.

Usage:
  python scripts/cost_summary.py            # uses the default paths below
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parent.parent
COMPARER = Path("/home/moritz/PycharmProjects/Comparer_flask")


def _eur(x) -> str:
    return "n/a" if x is None else f"€{x:,.2f}"


def _ragas_cost(dirs: List[Path], usd_eur: float) -> Tuple[float, int]:
    total, n = 0.0, 0
    for d in dirs:
        for fp in glob.glob(str(d / "ragas_*.jsonl")):
            for line in open(fp):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                c = r.get("judge_cost_usd")
                if isinstance(c, (int, float)):
                    total += c
                    n += 1
    return total * usd_eur, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extraction-json", type=Path,
                    default=REPO / "reports/cost/cost_extraction.json")
    ap.add_argument("--rag-costs", type=Path,
                    default=COMPARER / "reports/eval_scico/cost/cost_per_system.jsonl")
    ap.add_argument("--ragas-dirs", type=Path, nargs="*", default=[
        COMPARER / "reports/eval_scico/agent_gemini_2_5_pro",
        COMPARER / "reports/eval_scico/pipeline_gemini_2_5_pro_topk5"])
    ap.add_argument("--usd-eur", type=float, default=0.92)
    ap.add_argument("--out", type=Path, default=REPO / "reports/cost/cost_summary.md")
    args = ap.parse_args()

    ext = json.loads(args.extraction_json.read_text())
    con = ext.get("construction") or {}
    construction_eur = con.get("actual_eur") if con.get("actual_eur") is not None \
        else con.get("extrapolated_eur")
    ablation_eur = ext.get("ablation_eur")
    gpu = ext.get("gpu") or {}

    rag_rows = [json.loads(l) for l in args.rag_costs.read_text().splitlines() if l.strip()] \
        if args.rag_costs.exists() else []
    rag_eur = sum((r.get("cost_usd") or 0) for r in rag_rows) * args.usd_eur
    rag_top = sorted(rag_rows, key=lambda r: -(r.get("cost_usd") or 0))[:1]
    rag_top_str = (f"{rag_top[0]['system']} { _eur((rag_top[0].get('cost_usd') or 0)*args.usd_eur)}"
                   if rag_top else "n/a")

    ragas_eur, ragas_n = _ragas_cost([Path(d) for d in args.ragas_dirs], args.usd_eur)

    api_total = sum(x for x in [construction_eur, ablation_eur, rag_eur, ragas_eur]
                    if isinstance(x, (int, float)))
    gpu_total = sum(x for x in [gpu.get("train_eur"), gpu.get("infer_eur")]
                    if isinstance(x, (int, float))) or None

    md = []
    md.append("# ANTHEM — total cost (all surfaces)\n")
    md.append(f"- USD→EUR: {args.usd_eur} (applied to RAG + RAGAS; extraction already in "
              "rate-card currency)")
    md.append("")
    md.append("| Surface | Detail | Cost |")
    md.append("|---|---|---:|")
    extra = ""
    if con.get("actual_eur") is not None and con.get("extrapolated_eur") is not None:
        extra = f" (as paid; ~{_eur(con['extrapolated_eur'])} at today's prices)"
    md.append(f"| Corpus construction | {con.get('model','?')} × full corpus | "
              f"**{_eur(construction_eur)}**{extra} |")
    md.append(f"| Extraction ablation | LLMs × 781-paper gold (Table 2) | {_eur(ablation_eur)} |")
    md.append(f"| RAG benchmark | {len(rag_rows)} system-runs (Table 4); top: {rag_top_str} "
              f"| {_eur(rag_eur)} |")
    md.append(f"| RAGAS judging | second judge × {ragas_n} answers | {_eur(ragas_eur)} |")
    md.append(f"| ROR linking | ~348k affiliation matches | free |")
    md.append(f"| **Total commercial API** | construction + ablation + RAG + RAGAS "
              f"| **{_eur(api_total)}** |")
    md.append("")
    md.append("### Local open-baseline compute (the transparency contrast)")
    md.append("")
    md.append(f"- {gpu.get('model','Qwen LoRA')}: **{gpu.get('train_hrs')} A100-h training** "
              f"({_eur(gpu.get('train_eur'))}) + **~{gpu.get('infer_hrs'):.0f} A100-h inference** "
              f"({_eur(gpu.get('infer_eur'))}) = **{_eur(gpu_total)}**")
    if isinstance(gpu_total, (int, float)) and api_total:
        md.append(f"- i.e. the entire commercial-API pipeline ({_eur(api_total)}) costs "
                  f"**~{gpu_total/api_total:.1f}× less** than just running the local open "
                  f"baseline ({_eur(gpu_total)}).")
    md.append("")
    md.append("_Extraction/construction prices from OpenRouter; corpus construction is the "
              "actual paid amount (then-current prices). RAG/RAGAS from logged telemetry._")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\n# wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
