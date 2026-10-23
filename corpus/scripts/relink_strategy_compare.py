"""Compare four ROR-linking strategies on the 781-paper gold set.

STANDALONE / READ-ONLY w.r.t. the pipeline: this script does NOT modify the
production linker (ror.py / ror_link_v2.py / ror_link_annotation.py). It only
(a) fetches the raw ROR API `items[]` per unique affiliation string, (b) applies
four decision rules offline, and (c) scores each against gold using a faithful
re-implementation of run_pipeline_comparison._score_ror_hier (set-based,
hierarchy-aware, micro TP/FP/FN over papers).

Background (ROR API change, verified live 2026-07):
  As of 2026-05-26 the `?affiliation=` matcher defaults to the new "single
  search" strategy. Appending `&multisearch` forces the legacy multisearch
  strategy (substring splitting; matching_type FUZZY / COMMON TERMS). Both
  return items[] with per-item {chosen: bool, score: float, substring, organization}.
  Endpoint is v2 (https://api.ror.org/v2/organizations); v1 sunset 2025-12.

Four strategies (strategy x decision-rule 2x2, matching the requested names):
  current            = multisearch items, production RorResolver rule:
                       candidates = chosen items, else items[:1]; take argmax
                       score, accept iff score >= --min-score (default 0.7).
                       Reproduces the paper's current numbers (<=1 ROR/string).
  chosen_legacy      = multisearch items, take ALL items with chosen==true
                       (0..n ROR/string — recovers compound "A & B" affiliations
                       where ROR marks one chosen match per substring).
  single_argmax      = single_search items, argmax score >= --min-score (<=1/string).
  single_chosen      = single_search items, ALL items with chosen==true (<=1/string;
                       single search marks at most one chosen).

Scoring (per variant, micro over the 781 papers):
  gold ROR set / paper = normalized non-null ror_ids from authors_normalized_json.
  pred ROR set / paper = union over the paper's raw affiliation strings of the
                         rule's ROR output.
  F1_hier   : greedy 1-1 match, g~p iff expand(g) & expand(p) (expand = self +
              parents + children from ror_hierarchy_cache.json; identity fallback
              for ids absent from the cache — count reported).
  ROR-exact : plain set intersection (no hierarchy).
  coverage  : papers with >=1 predicted ROR / 781; also unique-string coverage.

Caches raw items[] to --items-cache so re-runs (e.g. different --min-score) hit
no API. Resumable. Does not write into any existing cache file.

Usage:
  python scripts/relink_strategy_compare.py                 # full run (fetches ~2x unique strings)
  python scripts/relink_strategy_compare.py --limit 20      # smoke: first 20 unique strings
  python scripts/relink_strategy_compare.py --min-score 0.9 # re-score offline from cache
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

REPO = Path(__file__).resolve().parent.parent
ROR_V2 = "https://api.ror.org/v2/organizations"

DEFAULT_GOLD = Path("/home/moritz/Downloads/11688_Singing_from_the_Same_AN_Data/"
                    "sample_data/affiliation_ground_truth.csv")
DEFAULT_HIER = REPO / "reports" / "ror_hierarchy_cache.json"
csv.field_size_limit(10_000_000)


# --------------------------------------------------------------------------- #
# ROR id normalization (mirrors run_pipeline_comparison._normalize_ror /
# RorResolver._strip_ror_url exactly)
# --------------------------------------------------------------------------- #
def norm_ror(rid: Optional[str]) -> Optional[str]:
    if not rid:
        return None
    rid = rid.strip().rstrip("/")
    if not rid:
        return None
    return rid.split("/")[-1].lower()


# --------------------------------------------------------------------------- #
# Gold loading
# --------------------------------------------------------------------------- #
def load_gold(csv_path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Return (papers, unique_strings).

    papers[pid] = {"raw_strings": [str, ...], "gold_ror": set(bare ids)}
    unique_strings = globally-unique raw affiliation strings (original casing kept
    for the API; dedup is case-insensitive).
    """
    papers: Dict[str, Dict[str, Any]] = {}
    seen_lower: Dict[str, str] = {}  # lower -> canonical original
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("anthology_id") or "").strip()
            if not pid:
                continue
            try:
                raw = json.loads(row.get("authors_raw_json") or "[]")
                norm = json.loads(row.get("authors_normalized_json") or "[]")
            except json.JSONDecodeError:
                continue
            raw_strings: List[str] = []
            for a in raw:
                for aff in (a.get("affiliations") or []):
                    s = (aff.get("affiliation") or "").strip()
                    if s:
                        raw_strings.append(s)
                        seen_lower.setdefault(s.lower(), s)
            gold_ror: Set[str] = set()
            for a in norm:
                for aff in (a.get("affiliations") or []):
                    b = norm_ror(aff.get("ror_id"))
                    if b:
                        gold_ror.add(b)
            papers[pid] = {"raw_strings": raw_strings, "gold_ror": gold_ror}
    unique_strings = sorted(seen_lower.values())
    return papers, unique_strings


# --------------------------------------------------------------------------- #
# ROR API fetch (raw items[]) with disk cache
# --------------------------------------------------------------------------- #
def _fetch_items(affiliation: str, strategy: str, timeout: int) -> List[Dict[str, Any]]:
    """One ROR call. strategy in {'single','multi'}. Returns slimmed items."""
    url = ROR_V2 + "?" + urllib.parse.urlencode({"affiliation": affiliation})
    if strategy == "single":
        url += "&single_search"
    elif strategy == "multi":
        url += "&multisearch"
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "acl-relink-compare/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 or 500 <= e.code < 600:
                time.sleep(2 ** attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
            continue
    else:
        raise RuntimeError(f"ROR fetch failed after retries: {strategy} {affiliation!r}")
    out = []
    for it in (data.get("items") or []):
        org = it.get("organization") or {}
        out.append({
            "ror": norm_ror(org.get("id")),
            "score": it.get("score"),
            "chosen": bool(it.get("chosen")),
            "matching_type": it.get("matching_type"),
            "substring": it.get("substring"),
        })
    return out


def build_items_cache(unique_strings: List[str], cache_path: Path,
                      rate_hz: float, timeout: int, limit: Optional[int]) -> Dict[str, Any]:
    cache: Dict[str, Any] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    targets = unique_strings[:limit] if limit else unique_strings
    delay = 1.0 / rate_hz if rate_hz > 0 else 0.0
    n_fetch = 0
    for i, s in enumerate(targets):
        for strat in ("single", "multi"):
            key = f"{strat}||{s}"
            if key in cache:
                continue
            cache[key] = _fetch_items(s, strat, timeout)
            n_fetch += 1
            if delay:
                time.sleep(delay)
            if n_fetch % 50 == 0:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(cache, ensure_ascii=False))
                print(f"# fetched {n_fetch} ({i+1}/{len(targets)} strings) — cache saved",
                      flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"# items cache: {len(cache)} entries ({n_fetch} new fetches)", flush=True)
    return cache


# --------------------------------------------------------------------------- #
# Decision rules -> set of bare ROR ids for one string
# --------------------------------------------------------------------------- #
def rule_current(items: List[Dict[str, Any]], min_score: float) -> Set[str]:
    """RorResolver reproduction: pool = chosen else items[:1]; argmax score>=thr."""
    pool = [it for it in items if it["chosen"]] or items[:1]
    best, best_s = None, -1.0
    for it in pool:
        s = float(it.get("score") or 0)
        if it.get("ror") and s >= min_score and s > best_s:
            best, best_s = it["ror"], s
    return {best} if best else set()


def rule_chosen_all(items: List[Dict[str, Any]]) -> Set[str]:
    return {it["ror"] for it in items if it["chosen"] and it.get("ror")}


def rule_argmax(items: List[Dict[str, Any]], min_score: float) -> Set[str]:
    best, best_s = None, -1.0
    for it in items:
        s = float(it.get("score") or 0)
        if it.get("ror") and s >= min_score and s > best_s:
            best, best_s = it["ror"], s
    return {best} if best else set()


STRATEGIES = {
    "current":       ("multi",  "rule_current"),
    "chosen_legacy": ("multi",  "rule_chosen_all"),
    "single_argmax": ("single", "rule_argmax"),
    "single_chosen": ("single", "rule_chosen_all"),
}


def string_ror_set(variant: str, s: str, cache: Dict[str, Any], min_score: float) -> Set[str]:
    strat, rule = STRATEGIES[variant]
    items = cache.get(f"{strat}||{s}") or []
    if rule == "rule_current":
        return rule_current(items, min_score)
    if rule == "rule_argmax":
        return rule_argmax(items, min_score)
    return rule_chosen_all(items)


# --------------------------------------------------------------------------- #
# Hierarchy expand + scoring (faithful to run_pipeline_comparison)
# --------------------------------------------------------------------------- #
def make_expand(hier: Dict[str, Any], include_related: bool):
    def expand(rid: str) -> Set[str]:
        b = norm_ror(rid)
        if not b:
            return set()
        h = hier.get(b)
        if not h:
            return {b}
        out = {b} | set(h.get("parent") or []) | set(h.get("child") or [])
        if include_related:
            out |= set(h.get("related") or [])
        return out
    return expand


def score_ror_hier(gold_set: Set[str], pred_set: Set[str], expand) -> Tuple[int, int, int]:
    used: Set[str] = set()
    tp = 0
    for g in gold_set:
        g_exp = expand(g)
        for p in pred_set:
            if p in used:
                continue
            if g_exp & expand(p):
                used.add(p)
                tp += 1
                break
    return tp, len(pred_set) - tp, len(gold_set) - tp


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    P = tp / (tp + fp) if (tp + fp) else 0.0
    R = tp / (tp + fn) if (tp + fn) else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F1


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold-csv", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--hierarchy-cache", type=Path, default=DEFAULT_HIER)
    ap.add_argument("--items-cache", type=Path,
                    default=REPO / "reports" / "ror_items_raw_cache.json")
    ap.add_argument("--min-score", type=float, default=0.7,
                    help="Score threshold for the argmax-based rules (current, "
                         "single_argmax). RorResolver default is 0.7.")
    ap.add_argument("--include-related", action="store_true",
                    help="Include ROR 'related' links in hierarchy expansion "
                         "(default: only parent+child, matching the pipeline).")
    ap.add_argument("--rate-hz", type=float, default=5.0)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only fetch the first N unique strings (smoke test).")
    ap.add_argument("--out-dir", type=Path, default=REPO / "reports" / "relink_compare")
    args = ap.parse_args()

    if not args.gold_csv.exists():
        print(f"ERROR: gold csv {args.gold_csv} not found", file=sys.stderr)
        return 1

    papers, unique_strings = load_gold(args.gold_csv)
    print(f"# gold: {len(papers)} papers | {len(unique_strings)} unique raw strings",
          flush=True)

    cache = build_items_cache(unique_strings, args.items_cache,
                              args.rate_hz, args.timeout, args.limit)

    hier = json.loads(args.hierarchy_cache.read_text()) if args.hierarchy_cache.exists() else {}
    expand = make_expand(hier, args.include_related)

    # Which strings are actually available in the cache (respects --limit).
    have_single = {s for s in unique_strings if f"single||{s}" in cache}
    have_multi = {s for s in unique_strings if f"multi||{s}" in cache}

    # Report hierarchy coverage of the gold ids (faithfulness of F1_hier).
    all_gold = set().union(*(p["gold_ror"] for p in papers.values())) if papers else set()
    gold_missing_hier = sum(1 for g in all_gold if g not in hier)
    print(f"# gold ROR ids: {len(all_gold)} | absent from hierarchy cache "
          f"(identity fallback): {gold_missing_hier}", flush=True)

    variants = list(STRATEGIES.keys())
    results: Dict[str, Dict[str, Any]] = {}
    per_paper_rows: List[Dict[str, Any]] = []

    for v in variants:
        strat = STRATEGIES[v][0]
        have = have_single if strat == "single" else have_multi
        # Only score papers whose every raw string was fetched (fair coverage
        # denominator under --limit; with a full run this is all 781 papers).
        scored_pids = ([pid for pid, p in papers.items()
                        if p["raw_strings"] and all(s in have for s in p["raw_strings"])]
                       if args.limit else list(papers))
        h_tp = h_fp = h_fn = 0
        e_tp = e_fp = e_fn = 0
        papers_covered = 0
        strings_linked = 0
        strings_total = 0
        for pid in scored_pids:
            p = papers[pid]
            pred: Set[str] = set()
            for s in p["raw_strings"]:
                strings_total += 1
                rs = string_ror_set(v, s, cache, args.min_score)
                if rs:
                    strings_linked += 1
                pred |= rs
            gold = p["gold_ror"]
            if pred:
                papers_covered += 1
            tp, fp, fn = score_ror_hier(gold, pred, expand)
            h_tp += tp; h_fp += fp; h_fn += fn
            e_tp += len(gold & pred); e_fp += len(pred - gold); e_fn += len(gold - pred)
            if v == variants[0]:
                per_paper_rows.append({"paper_id": pid, "n_gold": len(gold)})
        hP, hR, hF = prf(h_tp, h_fp, h_fn)
        eP, eR, eF = prf(e_tp, e_fp, e_fn)
        results[v] = {
            "n_papers": len(scored_pids),
            "paper_coverage": papers_covered / len(scored_pids) if scored_pids else 0.0,
            "string_coverage": strings_linked / strings_total if strings_total else 0.0,
            "hier": {"tp": h_tp, "fp": h_fp, "fn": h_fn, "P": hP, "R": hR, "F1": hF},
            "exact": {"tp": e_tp, "fp": e_fp, "fn": e_fn, "P": eP, "R": eR, "F1": eF},
        }

    # ---- render ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "relink_compare.json").write_text(
        json.dumps({"min_score": args.min_score,
                    "include_related": args.include_related,
                    "gold_csv": str(args.gold_csv),
                    "results": results}, indent=2))

    md = []
    md.append("# ROR linking strategy comparison (781-paper gold set)\n")
    md.append(f"- Gold: `{args.gold_csv.name}` — {len(papers)} papers, "
              f"{len(unique_strings)} unique strings")
    md.append(f"- min_score (argmax rules): {args.min_score} | "
              f"hierarchy expansion: parent+child{'+related' if args.include_related else ''}")
    md.append(f"- gold ROR ids absent from hierarchy cache (identity fallback): "
              f"{gold_missing_hier}/{len(all_gold)}")
    if args.limit:
        md.append(f"- **SMOKE RUN**: --limit {args.limit} unique strings; "
                  "coverage/F1 over the fully-fetched paper subset only")
    md.append("")
    md.append("| Strategy | ROR source | rule | n_papers | paper cov | string cov "
              "| F1_hier | P_hier | R_hier | F1_exact |")
    md.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    rule_desc = {"current": "multi / chosen-or-top1 argmax≥thr",
                 "chosen_legacy": "multi / all chosen",
                 "single_argmax": "single / argmax≥thr",
                 "single_chosen": "single / all chosen"}
    for v in variants:
        r = results[v]
        src = STRATEGIES[v][0]
        md.append(f"| {v} | {src} | {rule_desc[v]} | {r['n_papers']} "
                  f"| {r['paper_coverage']:.3f} | {r['string_coverage']:.3f} "
                  f"| {r['hier']['F1']:.3f} | {r['hier']['P']:.3f} | {r['hier']['R']:.3f} "
                  f"| {r['exact']['F1']:.3f} |")
    md.append("")
    md.append("_F1_hier is micro (aggregate TP/FP/FN over papers), hierarchy-aware, "
              "matching run_pipeline_comparison._score_ror_hier. paper cov = papers "
              "with ≥1 predicted ROR; string cov = unique raw strings that linked._")
    out_md = args.out_dir / "relink_compare.md"
    out_md.write_text("\n".join(md))
    print("\n" + "\n".join(md), flush=True)
    print(f"\n# wrote {out_md}", flush=True)

    # Per-paper counts for the paired bootstrap (bootstrap_paired.py). Only
    # meaningful on a full run, where all four strategies cover all 781 papers.
    if not args.limit:
        hier_rows, exact_rows = [], []
        for pid, p in papers.items():
            hc, ec = {}, {}
            gold = p["gold_ror"]
            for v in variants:
                pred: Set[str] = set()
                for s in p["raw_strings"]:
                    pred |= string_ror_set(v, s, cache, args.min_score)
                tp, fp, fn = score_ror_hier(gold, pred, expand)
                hc[v] = [tp, fp, fn]
                ec[v] = [len(gold & pred), len(pred - gold), len(gold - pred)]
            hier_rows.append({"paper_id": pid, "counts": hc})
            exact_rows.append({"paper_id": pid, "counts": ec})
        (args.out_dir / "relink_counts_hier.jsonl").write_text(
            "\n".join(json.dumps(r) for r in hier_rows) + "\n")
        (args.out_dir / "relink_counts_exact.jsonl").write_text(
            "\n".join(json.dumps(r) for r in exact_rows) + "\n")
        print(f"# wrote per-paper counts -> {args.out_dir}/relink_counts_{{hier,exact}}.jsonl",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
