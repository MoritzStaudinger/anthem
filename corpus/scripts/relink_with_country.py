"""Re-link predictions using paper-level country context.

For each paper, derive the set of country codes from the gold's
authors_normalized_json (in reports/ground_truth_merged.csv). For
each predicted raw string in that paper, hit ROR with
?affiliation=<raw>&country.country_code=<cc> for each candidate
country and keep the highest-confidence match above threshold.
Fall back to bare-string lookup when no country produces a hit.

This measures the UPPER BOUND of the country-aware linker — in
production, the country would have to be extracted by the LLM, not
read from the gold. If even the upper bound doesn't lift F1, country
context isn't the answer.

Writes:
  reports/ror_cache_country.json    — {raw, cc} -> match dict | None
  reports/merged_per_paper.country.jsonl   — predictions with country-aware ror_ids
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("relink_cc")
ROR_API = "https://api.ror.org/v2/organizations"


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "acl-processor/relink_with_country"
    return s


def _lookup(s: str, cc: Optional[str], sess: requests.Session,
            min_conf: float = 0.7) -> Optional[dict]:
    """ROR matcher with optional country filter. Returns best match or None."""
    if not s:
        return None
    params = {"affiliation": s}
    if cc:
        params["filter"] = f"country.country_code:{cc}"
    try:
        r = sess.get(ROR_API, params=params, timeout=15)
    except requests.RequestException as e:
        logger.warning("ROR err %r/%s: %s", s[:60], cc, e)
        return None
    if r.status_code != 200:
        if r.status_code != 429:
            logger.warning("ROR HTTP %d for %r/%s", r.status_code, s[:60], cc)
        return None
    items = (r.json() or {}).get("items") or []
    chosen = [it for it in items if it.get("chosen")] or items[:1]
    best = None; best_score = -1.0
    for it in chosen:
        try:
            sc = float(it.get("score") or 0)
        except (TypeError, ValueError):
            sc = 0.0
        if sc >= min_conf and sc > best_score:
            best, best_score = it, sc
    if not best:
        return None
    org = best.get("organization") or best
    names = org.get("names") or []
    canon = next((n.get("value") for n in names
                  if "ror_display" in (n.get("types") or [])), None)
    if not canon and names:
        canon = names[0].get("value", "")
    return {
        "ror_id": org.get("id") or "",
        "name": canon or "",
        "confidence": best_score,
        "country_filter": cc,
    }


def _load_paper_countries(gt_csv: Path) -> Dict[str, List[str]]:
    """paper_id → ordered list of unique gold country codes."""
    out: Dict[str, List[str]] = {}
    with gt_csv.open() as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("anthology_id") or "").strip()
            try:
                authors = json.loads(row.get("authors_normalized_json") or "[]")
            except json.JSONDecodeError:
                continue
            seen: List[str] = []
            for a in authors:
                for af in (a.get("affiliations") or []):
                    cc = (af.get("country_code") or "").strip().upper()
                    if cc and cc not in seen:
                        seen.append(cc)
            if pid:
                out[pid] = seen
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged-in", type=Path,
                    default=Path("reports/merged_per_paper.relinked.jsonl"))
    ap.add_argument("--merged-out", type=Path,
                    default=Path("reports/merged_per_paper.country.jsonl"))
    ap.add_argument("--gold-csv", type=Path,
                    default=Path("reports/ground_truth_merged.csv"))
    ap.add_argument("--bare-cache", type=Path,
                    default=Path("reports/ror_cache_predictions.json"),
                    help="Warm-start: bare-string lookups already done.")
    ap.add_argument("--cache-out", type=Path,
                    default=Path("reports/ror_cache_country.json"))
    ap.add_argument("--rate-limit-hz", type=float, default=4.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    paper_cc = _load_paper_countries(args.gold_csv)
    logger.info("loaded country sets for %d papers", len(paper_cc))

    # Bare cache: {raw -> {ror_id,name,confidence,...} | None}
    bare_cache: Dict[str, Optional[dict]] = {}
    if args.bare_cache.is_file():
        bare_cache = json.loads(args.bare_cache.read_text())
        logger.info("warmed %d bare lookups", len(bare_cache))

    # Country-aware cache: {(raw, cc) -> match | None}, JSON-serialized as
    # {"raw||cc": match}.
    cc_cache: Dict[str, Optional[dict]] = {}
    if args.cache_out.is_file():
        try:
            cc_cache = json.loads(args.cache_out.read_text())
            logger.info("warmed %d country-aware lookups", len(cc_cache))
        except json.JSONDecodeError:
            pass

    # Pass 1: enumerate (raw, cc) pairs we need
    todo: Set[Tuple[str, str]] = set()
    with args.merged_in.open() as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            pid = d.get("paper_id") or ""
            ccs = paper_cc.get(pid) or []
            if not ccs:
                continue
            for pred in (d.get("predictions") or {}).values():
                for raw in (pred.get("affiliations_raw") or []):
                    raw = (raw or "").strip()
                    if not raw: continue
                    for cc in ccs:
                        key = f"{raw}||{cc}"
                        if key in cc_cache:
                            continue
                        todo.add((raw, cc))
    logger.info("country-aware lookups needed: %d", len(todo))

    # Pass 2: resolve
    sess = _session()
    delay = 1.0 / max(args.rate_limit_hz, 0.1)
    last_save = time.time()
    for i, (raw, cc) in enumerate(sorted(todo)):
        m = _lookup(raw, cc, sess)
        cc_cache[f"{raw}||{cc}"] = m
        if (i + 1) % 200 == 0:
            resolved = sum(1 for v in cc_cache.values() if v)
            logger.info("  resolved %d/%d  (linked so far: %d)",
                        i + 1, len(todo), resolved)
        if time.time() - last_save > 60:
            args.cache_out.parent.mkdir(parents=True, exist_ok=True)
            args.cache_out.write_text(json.dumps(cc_cache, ensure_ascii=False, indent=2))
            last_save = time.time()
        time.sleep(delay)
    args.cache_out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_out.write_text(json.dumps(cc_cache, ensure_ascii=False, indent=2))

    # Pass 3: rewrite the merged JSONL using best-of(country-filtered, bare)
    with args.merged_in.open() as fin, args.merged_out.open("w") as fout:
        n_papers = 0; n_changed = 0
        for line in fin:
            line = line.strip()
            if not line: continue
            d = json.loads(line)
            pid = d.get("paper_id") or ""
            ccs = paper_cc.get(pid) or []
            changed = False
            for pred in (d.get("predictions") or {}).values():
                raws = pred.get("affiliations_raw") or []
                if not raws: continue
                new_ror, new_names = [], []
                old_ror = list(pred.get("ror_ids") or [])
                for raw in raws:
                    key = (raw or "").strip()
                    best: Optional[dict] = None
                    best_score = -1.0
                    for cc in ccs:
                        cand = cc_cache.get(f"{key}||{cc}")
                        if cand and cand.get("confidence", 0) > best_score:
                            best, best_score = cand, cand.get("confidence", 0)
                    # Fall back to bare cache if no country lookup hit
                    if not best:
                        cand = bare_cache.get(key)
                        if cand:
                            best = cand
                    if best and best.get("ror_id"):
                        new_ror.append(best["ror_id"])
                        new_names.append(best.get("name") or "")
                pred["ror_ids"] = new_ror
                pred["names"] = new_names
                if new_ror != old_ror:
                    changed = True
            if changed:
                n_changed += 1
            n_papers += 1
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
    logger.info("wrote %s (papers=%d, changed=%d)",
                args.merged_out, n_papers, n_changed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
