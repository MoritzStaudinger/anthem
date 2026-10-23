"""Triage every <mode>_<model>/answers.jsonl under reports/eval_scico/
and print which runs are short of the 99 rows. For each incomplete run,
suggest a resume command using the right sbatch.

The resume mechanism: eval_scico.py opens --answers-out in append mode
when the file is non-empty, builds a set of completed row IDs, skips
them on the next pass. So re-submitting the same sbatch with the same
out_dir continues from where the previous job stopped.

Usage:
    python scripts/list_incomplete_runs.py
    python scripts/list_incomplete_runs.py --root reports/eval_scico
    python scripts/list_incomplete_runs.py --expected 99 --print-empty

Adds a column for the wall-clock per-row average (from telemetry), so
you can sanity-check whether a re-submit at the current 23:59:00 wall
limit will finish.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Dir-name suffix -> CLI model id. The eval was launched with these
# canonical names; the dir names dropped dashes/dots so this maps back.
# Add new entries as you wire new models in.
MODEL_ID_MAP: Dict[str, str] = {
    "claude_sonnet_4_6": "claude-sonnet-4-6",
    "claude_haiku_4_5": "claude-haiku-4-5",
    "claude_sonnet":    "claude-sonnet-4-6",   # legacy bare name → newest sonnet
    "gemini_2_5_pro":   "gemini-2.5-pro",
    "gemini25_pro":     "gemini-2.5-pro",
    "gpt_5_mini":       "gpt-5-mini",
    "gpt5_mini":        "gpt-5-mini",
    "gpt5_nano":        "gpt-5.4-nano",
    "qwen_3_6_35b":     "qwen-3.6-35b",
    "qwen36_35b":       "qwen-3.6-35b",
    "gemma_4_e2b_it":   "gemma-4-e2b-it",
    "gemma4_e2b_it":    "gemma-4-e2b-it",
    "gemma4_e2b":       "gemma-4-e2b-it",
    "gemma4_e2b-it":    "gemma-4-e2b-it",
    "deepseek_v4_flash": "deepseek/deepseek-v4-flash",
}

SBATCH_BY_MODE: Dict[str, str] = {
    "closed_book": "slurm/eval_scico_closed_book.sbatch",
    "pipeline":    "slurm/eval_scico_pipeline.sbatch",
    # eval_scico_answer.sbatch is the phase-1 generation script for
    # agent mode; eval_scico.sbatch is the older one-shot variant.
    # Default to answer for the answer-then-judge split flow.
    "agent":       "slurm/eval_scico_answer.sbatch",
}


def _parse_dir_name(name: str) -> Optional[Tuple[str, str]]:
    """Return (mode, model_tag) or None to skip. Mirrors the parser
    used by build_first_eval_table.py."""
    if name in {"pipeline_smoke"} or name.startswith("test_"):
        return None
    if name.endswith("_smoke") or name.endswith("_test"):
        return None
    if name.startswith("closed_book_"):
        return "closed_book", name[len("closed_book_"):]
    if name.startswith("closed_"):
        return "closed_book", name[len("closed_"):]
    if name.startswith("pipeline_"):
        return "pipeline", name[len("pipeline_"):]
    if name.startswith("agent_"):
        return "agent", name[len("agent_"):]
    if name.startswith("answers_"):
        return "agent", name[len("answers_"):]
    return None


def _count_rows_and_wall(p: Path) -> Tuple[int, float]:
    """Returns (n_rows, mean_wall_seconds_per_row)."""
    n = 0
    walls: List[float] = []
    if not p.is_file():
        return 0, 0.0
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            w = ((rec.get("telemetry") or {}).get("wall_seconds"))
            if isinstance(w, (int, float)):
                walls.append(float(w))
    mean_wall = (sum(walls) / len(walls)) if walls else 0.0
    return n, mean_wall


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("reports/eval_scico"))
    ap.add_argument("--expected", type=int, default=99,
                    help="Target row count per run.")
    ap.add_argument("--print-empty", action="store_true",
                    help="Include 0-row runs in the resume suggestions "
                         "(default: print them in a separate 'empty' "
                         "section without a resume command, since they "
                         "may have failed at load time).")
    ap.add_argument("--wall-budget-hours", type=float, default=23.0,
                    help="Wall budget assumed for the resume sbatch. Used to "
                         "warn when remaining rows × mean_wall exceeds it.")
    args = ap.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"root not found: {args.root}")

    rows = []
    skipped_names = []
    empty_names = []
    for child in sorted(args.root.iterdir()):
        if not child.is_dir():
            continue
        parsed = _parse_dir_name(child.name)
        if parsed is None:
            skipped_names.append(child.name)
            continue
        mode, tag = parsed
        n, mean_wall = _count_rows_and_wall(child / "answers.jsonl")
        if n == 0:
            empty_names.append((child.name, mode, tag))
            if not args.print_empty:
                continue
        rows.append({
            "dir": child.name,
            "mode": mode,
            "tag": tag,
            "n": n,
            "mean_wall_s": mean_wall,
            "missing": max(0, args.expected - n),
        })

    incomplete = [r for r in rows if r["n"] < args.expected]
    complete = [r for r in rows if r["n"] >= args.expected]

    print(f"# Triage of {args.root}  ({len(complete)} complete, "
          f"{len(incomplete)} incomplete, {len(empty_names)} empty, "
          f"{len(skipped_names)} skipped non-run dirs)")
    print()

    if complete:
        print("## COMPLETE  (cov == {})".format(args.expected))
        print("| dir | mode | tag | rows | mean s/row |")
        print("|---|---|---|---:|---:|")
        for r in sorted(complete, key=lambda r: (r["mode"], r["tag"])):
            print(f"| {r['dir']} | {r['mode']} | {r['tag']} | {r['n']} | "
                  f"{r['mean_wall_s']:.1f} |")
        print()

    if incomplete:
        print("## INCOMPLETE  (cov < {}) — resume commands below".format(args.expected))
        print("| dir | mode | tag | rows | missing | mean s/row | "
              "est. resume wall (h) |")
        print("|---|---|---|---:|---:|---:|---:|")
        for r in sorted(incomplete, key=lambda r: (r["mode"], r["tag"])):
            eta_h = (r["missing"] * r["mean_wall_s"]) / 3600 if r["mean_wall_s"] else 0
            print(f"| {r['dir']} | {r['mode']} | {r['tag']} | {r['n']} | "
                  f"{r['missing']} | {r['mean_wall_s']:.1f} | {eta_h:.2f} |")
        print()
        print("## Resume commands")
        print("```bash")
        for r in sorted(incomplete, key=lambda r: (r["mode"], r["tag"])):
            sbatch = SBATCH_BY_MODE.get(r["mode"])
            model_id = MODEL_ID_MAP.get(r["tag"], f"<UNKNOWN:{r['tag']}>")
            out_dir = f"reports/eval_scico/{r['dir']}"
            note = ""
            if model_id.startswith("<UNKNOWN"):
                note = "  # FIXME: add MODEL_ID_MAP entry for this tag"
            if sbatch is None:
                note += f"  # FIXME: no sbatch wired for mode={r['mode']}"
            eta_h = (r["missing"] * r["mean_wall_s"]) / 3600 if r["mean_wall_s"] else 0
            if eta_h > args.wall_budget_hours:
                note += (f"  # WARN: estimated resume wall {eta_h:.1f}h > "
                         f"{args.wall_budget_hours}h sbatch limit; consider "
                         "bumping --time or splitting")
            print(f"sbatch {sbatch} {model_id} {out_dir}{note}")
        print("```")
        print()

    if empty_names and not args.print_empty:
        print("## EMPTY  (answers.jsonl is 0-byte or absent — likely failed at "
              "load time, not a row-count issue)")
        for name, mode, tag in empty_names:
            print(f"- {name}  (mode={mode}, tag={tag})")
        print("Re-run with --print-empty to include them in the resume commands "
              "above.")
        print()

    if skipped_names:
        print("## SKIPPED  (non-run dirs / smoke / test — not in any table)")
        for name in skipped_names:
            print(f"- {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
