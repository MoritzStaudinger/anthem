"""Cross-pipeline per-paper merge — turns the side-by-side `predictions_*.jsonl`
files into one row per paper, showing what every pipeline said + a tally of
candidate affiliations.

Use case: audit the gold annotations. If 6 out of 8 pipelines extract
'ETH Zurich' for a paper but gold says 'ETHZ' (or doesn't list it), the
tally surfaces the candidate so you can decide whether the gold needs
revision. Also useful for computing inter-pipeline agreement (a proxy
for inter-annotator agreement when treating pipelines as 'annotators').

Output schema (one JSON per paper, one paper per line):
{
  "paper_id": "...",
  "gold": {
    "affiliations_raw":  ["MIT", "Stanford University"],
    "ror_ids":           ["https://ror.org/042nb2s44", ...],
    "names":             ["Massachusetts Institute of Technology", ...]
  },
  "predictions": {
    "ours":              {"affiliations_raw": [...], "ror_ids": [...], "names": [...]},
    "s2orc":             {...},
    "gpt4o_pymu5k":      {...},
    ...
  },
  "candidate_tally": {
    "MIT":               {"n_pipelines": 7,  "pipelines": [...], "in_gold": true},
    "Stanford CS Dept":  {"n_pipelines": 1,  "pipelines": [...], "in_gold": false},
    ...
  },
  "ror_tally": {
    "https://ror.org/042nb2s44": {"n_pipelines": 6, "pipelines": [...], "in_gold": true},
    ...
  },
  "agreement_stats": {
    "n_pipelines_total":    8,
    "n_pipelines_with_data": 7,
    "majority_affs":        ["MIT", "Stanford"],   # >= ceil(n/2) pipelines agree
    "unanimous_affs":       ["MIT"],                # all pipelines emit this
    "gold_only_affs":       ["Toyota Tech Inst"],   # in gold but 0 pipelines
    "novel_affs":           ["Stanford CS Dept"],   # in pipelines but not gold
  }
}

Usage:
    python scripts/merge_pipeline_predictions_per_paper.py \\
        --predictions-dir reports/pipeline_comparison/ \\
        --gold-csv reports/ground_truth_merged.csv \\
        --out reports/pipeline_comparison/merged_per_paper.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("merge_predictions")


# -------------------------- name canonicalisation --------------------------

def _ascii_fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return s.encode("ascii", "ignore").decode("ascii")


def _norm_for_tally(s: str) -> str:
    """Canonicalise an affiliation string for tally bucketing. We collapse
    case + punctuation + whitespace but keep distinct enough that
    'MIT' and 'Massachusetts Institute of Technology' end up in separate
    buckets (intentional — the tally surfaces both as candidates, not
    a single canonicalised entry, so reviewers can see what each model
    actually emitted)."""
    if not s:
        return ""
    s = _ascii_fold(s).strip().lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_ror(rid: Optional[str]) -> Optional[str]:
    if not rid:
        return None
    rid = rid.strip().rstrip("/")
    if not rid:
        return None
    return rid.split("/")[-1].lower()


# -------------------------- gold loader --------------------------

def _load_gold(gold_csv: Path) -> Dict[str, dict]:
    gold: Dict[str, dict] = {}
    with open(gold_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("anthology_id") or "").strip()
            if not pid:
                continue
            try:
                a_raw = json.loads(row.get("authors_raw_json") or "[]")
            except json.JSONDecodeError:
                a_raw = []
            try:
                a_norm = json.loads(row.get("authors_normalized_json") or "[]")
            except json.JSONDecodeError:
                a_norm = []
            raw_affs = []
            for a in a_raw:
                for af in (a.get("affiliations") or []):
                    if isinstance(af, dict):
                        s = af.get("affiliation") or af.get("name")
                        if s:
                            raw_affs.append(str(s))
                    elif isinstance(af, str):
                        raw_affs.append(af)
            ror_ids, names = [], []
            for a in a_norm:
                for af in (a.get("affiliations") or []):
                    if isinstance(af, dict):
                        if af.get("ror_id"):
                            ror_ids.append(str(af["ror_id"]))
                        if af.get("name"):
                            names.append(str(af["name"]))
            gold[pid] = {
                "affiliations_raw": raw_affs,
                "ror_ids": ror_ids,
                "names": names,
            }
    return gold


# -------------------------- prediction loader --------------------------

def _load_predictions(predictions_dir: Path) -> Dict[str, Dict[str, dict]]:
    """Returns {pipeline_id: {paper_id: prediction_dict}}."""
    out: Dict[str, Dict[str, dict]] = {}
    for fp in sorted(predictions_dir.glob("predictions_*.jsonl")):
        # filename: predictions_<pipeline_id>.jsonl
        pid = fp.stem.removeprefix("predictions_")
        with fp.open() as f:
            per_paper: Dict[str, dict] = {}
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                paper_id = rec.get("paper_id")
                if not paper_id:
                    continue
                # Flatten predicted affiliations across authors
                affs_raw, ror_ids, names = [], [], []
                for a in (rec.get("authors") or []):
                    for af in (a.get("affiliations") or []):
                        if af.get("raw_affiliation"):
                            affs_raw.append(af["raw_affiliation"])
                        if af.get("ror_id"):
                            ror_ids.append(af["ror_id"])
                        if af.get("clean_institution_name"):
                            names.append(af["clean_institution_name"])
                per_paper[paper_id] = {
                    "affiliations_raw": affs_raw,
                    "ror_ids": ror_ids,
                    "names": names,
                    "error": (rec.get("_pipeline") or {}).get("error"),
                }
            out[pid] = per_paper
        logger.info("loaded %s: %d papers", fp.name, len(per_paper))
    return out


# -------------------------- merge logic --------------------------

def _build_tally(per_pipeline: Dict[str, dict], key: str,
                 normalize=_norm_for_tally) -> Dict[str, dict]:
    """For each unique normalized value, list the pipelines that emitted it
    (with the canonical/raw form they used)."""
    tally: Dict[str, dict] = defaultdict(lambda: {
        "n_pipelines": 0, "pipelines": [], "raw_forms": set(),
    })
    for pid, pred in per_pipeline.items():
        seen_in_this_pipeline = set()
        for v in pred.get(key, []):
            n = normalize(v) if normalize else (v or "").strip().lower()
            if not n or n in seen_in_this_pipeline:
                continue
            seen_in_this_pipeline.add(n)
            tally[n]["n_pipelines"] += 1
            tally[n]["pipelines"].append(pid)
            tally[n]["raw_forms"].add(v)
    # Convert sets to sorted lists for JSON friendliness
    return {k: {
        "n_pipelines": v["n_pipelines"],
        "pipelines": sorted(v["pipelines"]),
        "raw_forms": sorted(v["raw_forms"]),
    } for k, v in tally.items()}


def _mark_in_gold(tally: Dict[str, dict], gold_values: List[str],
                  normalize=_norm_for_tally) -> Dict[str, dict]:
    gold_set = {normalize(g) if normalize else (g or "").strip().lower()
                for g in gold_values}
    gold_set.discard("")
    for k, v in tally.items():
        v["in_gold"] = k in gold_set
    return tally


def _agreement_stats(per_pipeline_for_paper: Dict[str, dict],
                     gold: dict, n_pipelines_total: int) -> dict:
    """Compute majority / unanimous / gold-only / novel sets on normalized
    raw-affiliation strings. Gives a quick sense of where pipelines agree
    with each other but diverge from gold."""
    tally = _build_tally(per_pipeline_for_paper, "affiliations_raw")
    gold_norm = {_norm_for_tally(g) for g in gold["affiliations_raw"]}
    gold_norm.discard("")
    n_with_data = sum(1 for v in per_pipeline_for_paper.values()
                      if v.get("affiliations_raw"))
    majority_n = max(1, math.ceil(n_with_data / 2)) if n_with_data else 0
    majority   = [k for k, v in tally.items() if v["n_pipelines"] >= majority_n]
    unanimous  = [k for k, v in tally.items() if v["n_pipelines"] == n_with_data and n_with_data > 0]
    gold_only  = [g for g in gold_norm if g not in tally]
    novel      = [k for k in tally if k not in gold_norm]
    return {
        "n_pipelines_total": n_pipelines_total,
        "n_pipelines_with_data": n_with_data,
        "majority_affs": sorted(majority),
        "unanimous_affs": sorted(unanimous),
        "gold_only_affs": sorted(gold_only),
        "novel_affs": sorted(novel),
    }


# -------------------------- main --------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions-dir", type=Path, required=True,
                    help="Dir holding predictions_<pipeline_id>.jsonl files "
                         "(written by run_pipeline_comparison.py).")
    ap.add_argument("--gold-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL — one row per paper.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    gold = _load_gold(args.gold_csv)
    logger.info("gold: %d papers", len(gold))

    per_pipeline = _load_predictions(args.predictions_dir)
    pipeline_ids = sorted(per_pipeline.keys())
    if not pipeline_ids:
        raise SystemExit(f"no predictions_*.jsonl files found in {args.predictions_dir}")
    logger.info("pipelines: %s", pipeline_ids)

    # Union of paper ids covered by any pipeline + the gold
    all_papers = set(gold.keys())
    for pid_id, per_paper in per_pipeline.items():
        all_papers.update(per_paper.keys())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for pid in sorted(all_papers):
            per_paper_for_pipelines = {
                pp_id: per_pipeline[pp_id].get(pid, {"affiliations_raw": [], "ror_ids": [], "names": []})
                for pp_id in pipeline_ids
            }
            gold_entry = gold.get(pid, {"affiliations_raw": [], "ror_ids": [], "names": []})

            # Tallies
            aff_tally = _mark_in_gold(
                _build_tally(per_paper_for_pipelines, "affiliations_raw"),
                gold_entry["affiliations_raw"],
            )
            ror_tally = _mark_in_gold(
                _build_tally(per_paper_for_pipelines, "ror_ids",
                             normalize=_normalize_ror),
                gold_entry["ror_ids"],
                normalize=_normalize_ror,
            )
            name_tally = _mark_in_gold(
                _build_tally(per_paper_for_pipelines, "names"),
                gold_entry["names"],
            )

            row = {
                "paper_id": pid,
                "gold": gold_entry,
                "predictions": per_paper_for_pipelines,
                "candidate_tally": aff_tally,
                "ror_tally": ror_tally,
                "name_tally": name_tally,
                "agreement_stats": _agreement_stats(
                    per_paper_for_pipelines, gold_entry, len(pipeline_ids),
                ),
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    logger.info("wrote %d merged rows -> %s", n_written, args.out)

    # Quick summary stats on the merge
    print(f"\nMerged {n_written} papers across {len(pipeline_ids)} pipelines.")
    print(f"Output: {args.out}")
    print(f"\nFor gold-audit review, key fields per row:")
    print(f"  agreement_stats.majority_affs   — extracted by ≥ ceil(N/2) pipelines")
    print(f"  agreement_stats.unanimous_affs  — extracted by ALL pipelines")
    print(f"  agreement_stats.gold_only_affs  — in gold but 0 pipelines extracted "
          f"(likely gold-side noise OR everyone misses it)")
    print(f"  agreement_stats.novel_affs      — extracted by ≥1 pipeline but NOT in gold "
          f"(candidates to add to gold OR systematic false positives)")
    print(f"  candidate_tally[<aff>].n_pipelines — vote count per affiliation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
