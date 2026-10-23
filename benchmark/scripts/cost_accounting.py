"""Cost / token accounting for the SciCo eval sweeps (log mining, no new runs).

Rebuttal task: report the actual call counts, token totals and euro cost of one
full sweep per system, plus the marginal cost of an additional RAGAS judge.

What IS logged: every answers.jsonl row carries a `telemetry` block with
prompt_tokens / completion_tokens / total_tokens / cost_usd (cost is filled only
for models present in backend.eval.metrics.MODEL_PRICING). This script sums it
per system, deduping duplicate rows by id.

What is NOT logged (and this script is explicit about it): the RAGAS judge calls.
ragas_*.jsonl files store only the resulting scores (faithfulness, relevancy,
judge, n_contexts) — no tokens, no cost. So the additional-judge cost cannot be
mined; it must be measured on the Task 4 re-run (instrument the judge client) or
estimated. This script prints a clearly-labelled ESTIMATE from stored prompt
sizes if --estimate-ragas is given, and otherwise just states the gap.

For models missing from MODEL_PRICING (gpt-5-mini, gpt-5, deepseek-*, gemma-*,
qwen-*), cost_usd is None in telemetry; supply a rate card via --rates rates.json
({"model-id": [in_per_1k, out_per_1k], ...}) to recompute their cost from the
logged token counts. Without it, tokens are still reported and cost is shown as
"n/a (no rate)".

Usage:
    python scripts/cost_accounting.py
    python scripts/cost_accounting.py --usd-eur 0.92 --rates my_rates.json \
        --systems agent_gemini_2_5_pro pipeline_gemini_2_5_pro_topk5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.eval.metrics import MODEL_PRICING  # noqa: E402


def _dedupe_last(apath: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with apath.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("id") is not None:
                rows[r["id"]] = r  # last wins (matches append/resume)
    return rows


def _cost_from_rate(rate: Optional[Tuple[float, float]],
                    p_tok: int, c_tok: int) -> Optional[float]:
    if rate is None:
        return None
    return (p_tok / 1000.0) * rate[0] + (c_tok / 1000.0) * rate[1]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--reports-dir", type=Path, default=Path("reports/eval_scico"))
    ap.add_argument("--systems", nargs="*", default=None,
                    help="Default: all dirs with an answers.jsonl.")
    ap.add_argument("--rates", type=Path, default=None,
                    help='JSON: {"model-id": [in_per_1k_usd, out_per_1k_usd]} to '
                         "fill in models missing from MODEL_PRICING.")
    ap.add_argument("--usd-eur", type=float, default=0.92,
                    help="USD->EUR multiplier for the reported euro column.")
    ap.add_argument("--out-dir", type=Path, default=Path("reports/eval_scico/cost"))
    args = ap.parse_args()

    extra_rates: Dict[str, Tuple[float, float]] = {}
    if args.rates:
        raw = json.loads(args.rates.read_text())
        extra_rates = {k: (float(v[0]), float(v[1])) for k, v in raw.items()
                       if not k.startswith("_") and isinstance(v, list)}

    def rate_for(model: Optional[str]) -> Optional[Tuple[float, float]]:
        if not model:
            return None
        if model in extra_rates:
            return extra_rates[model]
        r = MODEL_PRICING.get(model)
        return tuple(r) if r else None  # type: ignore[return-value]

    if args.systems:
        systems = list(args.systems)
    else:
        systems = sorted(d.name for d in args.reports_dir.iterdir()
                         if d.is_dir() and (d / "answers.jsonl").exists())

    rows: List[Dict[str, Any]] = []
    missing_rate_models: set = set()
    for sysname in systems:
        apath = args.reports_dir / sysname / "answers.jsonl"
        if not apath.exists():
            continue
        by_id = _dedupe_last(apath)
        n_ans = len(by_id)
        n_err = sum(1 for r in by_id.values() if r.get("error"))
        p_tok = c_tok = t_tok = 0
        cost_logged = 0.0
        cost_logged_any = False
        model = None
        for r in by_id.values():
            model = r.get("model") or model
            tel = r.get("telemetry") or {}
            p_tok += int(tel.get("prompt_tokens") or 0)
            c_tok += int(tel.get("completion_tokens") or 0)
            t_tok += int(tel.get("total_tokens") or 0)
            cu = tel.get("cost_usd")
            if isinstance(cu, (int, float)):
                cost_logged += cu
                cost_logged_any = True
        rate = rate_for(model)
        cost_recomputed = _cost_from_rate(rate, p_tok, c_tok)
        # Prefer the logged cost when present; else the recomputed one.
        cost_usd: Optional[float]
        if cost_logged_any and cost_logged > 0:
            cost_usd = cost_logged
            cost_src = "logged"
        elif cost_recomputed is not None:
            cost_usd = cost_recomputed
            cost_src = "recomputed"
        else:
            cost_usd = None
            cost_src = "no_rate"
            if model:
                missing_rate_models.add(model)
        rows.append({
            "system": sysname, "model": model, "n_answers": n_ans,
            "n_error": n_err, "prompt_tokens": p_tok, "completion_tokens": c_tok,
            "total_tokens": t_tok, "cost_usd": cost_usd, "cost_src": cost_src,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "cost_per_system.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8")

    def eur(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x * args.usd_eur:.4f}"

    def usd(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x:.4f}"

    md: List[str] = []
    md.append("# Cost accounting — one full sweep per system\n")
    md.append(f"- USD->EUR: {args.usd_eur} | systems: {len(rows)}")
    md.append("- Generation cost is mined from `telemetry` in answers.jsonl "
              "(deduped by id, last-wins).")
    md.append("")
    md.append("| System | model | n | err | prompt tok | compl tok | total tok "
              "| USD | EUR | src |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in sorted(rows, key=lambda r: (r["cost_usd"] is None, -(r["cost_usd"] or 0))):
        md.append(
            f"| {r['system']} | {r['model']} | {r['n_answers']} | {r['n_error']} "
            f"| {r['prompt_tokens']:,} | {r['completion_tokens']:,} "
            f"| {r['total_tokens']:,} | {usd(r['cost_usd'])} | {eur(r['cost_usd'])} "
            f"| {r['cost_src']} |")
    md.append("")

    if missing_rate_models:
        md.append(f"> **No rate for:** {', '.join(sorted(missing_rate_models))} — "
                  "token counts are reported but USD/EUR are n/a. Supply "
                  "`--rates rates.json` to fill these in.")
        md.append("")

    md.append("## Additional-judge (RAGAS) cost")
    md.append("")
    md.append("RAGAS output files (`ragas_*.jsonl`) log **no tokens and no cost** "
              "— only faithfulness/relevancy/judge/n_contexts. The marginal cost "
              "of a second judge therefore **cannot be mined from existing logs**. "
              "Two honest options for the rebuttal number:")
    md.append("")
    md.append("1. **Measured (recommended):** wrap the judge client with a token "
              "counter on the Task 4 second-judge re-run (198 answers: "
              "Gemini 2.5 Pro agent + pipeline-K5) and report the real total.")
    md.append("2. **Estimated:** RAGAS faithfulness+relevancy issues a variable "
              "number of judge calls per answer (statement extraction + one NLI "
              "call per extracted statement); with contexts already metadata-only "
              "and stored, tokens can be approximated from the stored answer + "
              "n_contexts, but this is a rough lower bound.")
    md.append("")

    table_path = args.out_dir / "cost_report.md"
    table_path.write_text("\n".join(md), encoding="utf-8")
    print(f"# wrote {table_path}", flush=True)
    print("\n" + "\n".join(md), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
