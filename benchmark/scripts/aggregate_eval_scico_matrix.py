"""Aggregate per-agent eval reports into a single cross-agent table.

Reads:
   <out_root>/agent_<id>/  ... one dir per agent, each containing whatever
                              eval_scico.py wrote (JSON + log + per-row .jsonl).
   Auto-discovers report files via filename heuristics; doesn't depend
   on exact eval_scico.py output naming.

Writes:
   <out_root>/cross_agent_comparison.csv      flat — agents × metrics
   <out_root>/cross_agent_comparison.md       human-readable markdown
   <out_root>/cross_agent_per_category.csv    per (agent, category) breakdown
                                              when per-category breakdowns exist in reports

Usage:
   python scripts/aggregate_eval_scico_matrix.py reports/eval_scico/matrix_20260522/
"""
from __future__ import annotations

import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agg_eval_matrix")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")


# eval_scico.py writes a few different report file conventions across
# versions — we sniff for any of them.
_REPORT_CANDIDATES = [
    "report.json", "eval_report.json", "summary.json",
    "scico_eval_report.json", "results.json",
]


def _find_report(agent_dir: Path) -> Optional[Path]:
    """Locate the agent's metrics report file. Prefers known names;
    falls back to any *.json that contains a 'metrics' or 'aggregate' key."""
    for name in _REPORT_CANDIDATES:
        p = agent_dir / name
        if p.is_file():
            return p
    # Fallback: scan for any JSON that looks like a metrics dump
    for p in agent_dir.glob("*.json"):
        try:
            d = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and any(k in d for k in ("metrics", "aggregate", "averages", "macro", "per_category")):
            return p
    return None


def _flatten_metrics(d: dict, prefix: str = "") -> Dict[str, Any]:
    """Walk nested metric dicts and produce flat 'a.b.c' -> value pairs
    for numeric leaves. Skips dicts deeper than 4 levels to stay sane."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, (int, float)):
            out[key] = v
        elif isinstance(v, dict) and len(prefix.split(".")) < 4:
            out.update(_flatten_metrics(v, prefix=key + "."))
    return out


def _extract_log_summary(agent_dir: Path) -> Dict[str, Any]:
    """If we can't find a JSON report, eval_scico.py's stdout log
    usually contains aggregate numbers we can regex out."""
    out: Dict[str, Any] = {}
    log = agent_dir / "eval.log"
    if not log.is_file():
        return out
    text = log.read_text(errors="replace")
    # Look for common metric prints — relaxed regex; numbers prefixed by name
    for label, key in [
        ("Recall@k", "recall_at_k"),
        ("Precision@k", "precision_at_k"),
        ("F1@k", "f1_at_k"),
        ("MRR", "mrr"),
        ("ROUGE-L", "rouge_l"),
        ("Faithfulness", "judge_faithfulness"),
        ("Coverage", "judge_coverage"),
        ("Attribution", "judge_attribution_accuracy"),
        ("Relevance", "judge_relevance"),
    ]:
        m = re.search(rf"{label}\s*[:=]\s*([\-\d.]+)", text, re.IGNORECASE)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    return out


def main():
    if len(sys.argv) < 2:
        print("usage: aggregate_eval_scico_matrix.py <out_root>", file=sys.stderr)
        sys.exit(2)

    out_root = Path(sys.argv[1])
    if not out_root.is_dir():
        print(f"not a directory: {out_root}", file=sys.stderr)
        sys.exit(2)

    agent_dirs = sorted(d for d in out_root.iterdir()
                        if d.is_dir() and d.name.startswith("agent_"))
    if not agent_dirs:
        print(f"no agent_*/ subdirs in {out_root}", file=sys.stderr)
        sys.exit(2)

    rows: List[Dict[str, Any]] = []
    all_keys: set = set()
    per_category: List[Dict[str, Any]] = []

    for d in agent_dirs:
        agent_id = d.name.removeprefix("agent_")
        report_path = _find_report(d)
        row: Dict[str, Any] = {"agent": agent_id, "report_path": str(report_path) if report_path else ""}

        if report_path:
            try:
                report = json.loads(report_path.read_text())
            except json.JSONDecodeError:
                report = {}
            # Pull top-level macro metrics
            agg = (report.get("aggregate") or report.get("metrics")
                   or report.get("averages") or report.get("macro") or {})
            if isinstance(agg, dict):
                row.update(_flatten_metrics(agg))
            # Optional: per-category breakdown
            pc = report.get("per_category") or report.get("per_question_category") or {}
            if isinstance(pc, dict):
                for cat, cat_metrics in pc.items():
                    if not isinstance(cat_metrics, dict):
                        continue
                    pc_row = {"agent": agent_id, "category": cat}
                    pc_row.update(_flatten_metrics(cat_metrics))
                    per_category.append(pc_row)
        else:
            # Fall back to scraping numbers from eval.log
            row.update(_extract_log_summary(d))

        all_keys.update(row.keys())
        rows.append(row)
        logger.info("loaded %s — %d metrics captured", agent_id,
                    len([k for k, v in row.items() if isinstance(v, (int, float))]))

    if not any(any(isinstance(v, (int, float)) for v in r.values()) for r in rows):
        logger.warning("no numeric metrics extracted — check that eval_scico.py "
                       "wrote a JSON report or that its stdout was captured to eval.log")

    # --- write flat per-agent CSV ---
    sorted_keys = ["agent"] + sorted(k for k in all_keys if k != "agent" and k != "report_path") + ["report_path"]
    csv_path = out_root / "cross_agent_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sorted_keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in sorted_keys})
    logger.info("wrote %s", csv_path)

    # --- per-category CSV (if available) ---
    if per_category:
        pc_keys = ["agent", "category"] + sorted({k for r in per_category for k in r.keys()
                                                   if k not in ("agent", "category")})
        pc_path = out_root / "cross_agent_per_category.csv"
        with pc_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=pc_keys)
            w.writeheader()
            for r in per_category:
                w.writerow({k: r.get(k, "") for k in pc_keys})
        logger.info("wrote %s (%d agent×category rows)", pc_path, len(per_category))

    # --- markdown summary ---
    md_path = out_root / "cross_agent_comparison.md"
    # Pick the ~10 most informative numeric keys to surface
    numeric_keys = [k for k in sorted_keys
                    if k not in ("agent", "report_path")
                    and any(isinstance(r.get(k), (int, float)) for r in rows)]
    headline_keys = numeric_keys[:10]
    with md_path.open("w") as f:
        f.write(f"# Cross-agent eval comparison\n\n")
        f.write(f"Run dir: `{out_root}`\n")
        f.write(f"Agents: {len(rows)}\n\n")
        f.write("## Headline metrics\n\n")
        # Markdown table
        f.write("| agent | " + " | ".join(headline_keys) + " |\n")
        f.write("|" + "|".join(["---"] * (len(headline_keys) + 1)) + "|\n")
        for r in rows:
            cells = [r["agent"]]
            for k in headline_keys:
                v = r.get(k)
                cells.append(f"{v:.3f}" if isinstance(v, float)
                             else (str(v) if v is not None else "—"))
            f.write("| " + " | ".join(cells) + " |\n")
        f.write(f"\n_Full per-metric numbers in `cross_agent_comparison.csv`._\n")
        if per_category:
            f.write(f"_Per-category breakdown in `cross_agent_per_category.csv`._\n")
    logger.info("wrote %s", md_path)

    print(f"\n=== AGGREGATION DONE ===")
    print(f"  agents:        {len(rows)}")
    print(f"  numeric metrics per agent: {len(numeric_keys)}")
    print(f"  outputs:")
    print(f"    {csv_path}")
    print(f"    {md_path}")
    if per_category:
        print(f"    {out_root}/cross_agent_per_category.csv")


if __name__ == "__main__":
    main()
