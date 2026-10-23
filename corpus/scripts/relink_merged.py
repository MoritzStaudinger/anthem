"""Re-run ROR linking on reports/merged_per_paper.jsonl in place.

Why this exists: the cluster-side linker `scripts/ror_link_predictions.py`
ran against an empty cache (the gold cache keys are `"<string>||<country>"`,
the prediction linker keys on bare strings) and the cluster's own cache
never made it back to the laptop. Net effect: most prediction rows in
merged_per_paper.jsonl have `ror_ids: []` even when the string is
trivially linkable (e.g., "CNRS" -> 0.900 confidence).

This script:
  1. Collects every unique raw affiliation string across all pipelines
  2. Resolves each via ROR's `?affiliation=<s>` matcher, with a warm
     start from the existing gold cache
  3. Writes a new merged JSONL with `ror_ids` / `names` populated for
     every prediction (and the gold side untouched)
  4. Caches results to reports/ror_cache_predictions.json so re-runs
     are instant
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROR_API = "https://api.ror.org/v2/organizations"

logger = logging.getLogger("relink")


def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "acl-processor/relink_merged"
    return s


def _warm_cache_from_gold(gold_cache_path: Path) -> Dict[str, dict]:
    """The existing gold cache uses 'string||Country' keys with a flattened
    single-best match value. We translate to bare-string keys."""
    if not gold_cache_path.is_file():
        return {}
    raw = json.loads(gold_cache_path.read_text())
    out: Dict[str, dict] = {}
    for k, v in raw.items():
        s = k.split("||", 1)[0].strip()
        if not s or not v:
            continue
        # gold cache stores {ror_id, ror_name, ror_country_code, ror_confidence}
        # we want the same shape we'll get from fresh API calls
        try:
            conf = float(v.get("ror_confidence") or 0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf < 0.7:
            continue
        if v.get("ror_id"):
            out.setdefault(s, {
                "ror_id": v["ror_id"],
                "name": v.get("ror_name") or "",
                "country_code": v.get("ror_country_code") or "",
                "confidence": conf,
            })
    return out


def _ror_lookup(s: str, sess: requests.Session, min_conf: float = 0.7) -> Optional[dict]:
    try:
        r = sess.get(ROR_API, params={"affiliation": s}, timeout=15)
    except requests.RequestException as e:
        logger.warning("ROR err %r: %s", s[:60], e)
        return None
    if r.status_code != 200:
        if r.status_code != 429:
            logger.warning("ROR HTTP %d for %r", r.status_code, s[:60])
        return None
    items = (r.json() or {}).get("items") or []
    chosen = [it for it in items if it.get("chosen")]
    pool = chosen or items[:1]
    best, best_score = None, -1.0
    for it in pool:
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
    locations = org.get("locations") or []
    cc = ""
    if locations:
        cc = (locations[0].get("geonames_details") or {}).get("country_code", "")
    return {
        "ror_id": org.get("id") or "",
        "name": canon or "",
        "country_code": cc,
        "confidence": best_score,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", type=Path,
                    default=Path("reports/merged_per_paper.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("reports/merged_per_paper.relinked.jsonl"))
    ap.add_argument("--gold-cache", type=Path,
                    default=Path("reports/ror_cache.json"))
    ap.add_argument("--pred-cache", type=Path,
                    default=Path("reports/ror_cache_predictions.json"))
    ap.add_argument("--rate-limit-hz", type=float, default=4.0,
                    help="Max ROR API calls per second (polite default 4).")
    ap.add_argument("--min-conf", type=float, default=0.7)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    # ---- 1. collect unique strings ----
    unique: set[str] = set()
    paper_count = 0
    with args.merged.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            paper_count += 1
            for pred in d.get("predictions", {}).values():
                for s in (pred.get("affiliations_raw") or []):
                    s = (s or "").strip()
                    if s:
                        unique.add(s)
    logger.info("scanned %d papers, %d unique prediction strings",
                paper_count, len(unique))

    # ---- 2. warm cache from gold + pred caches ----
    resolved: Dict[str, dict] = {}
    # missing flag — explicit so we don't re-query strings that we know are unlinkable
    unlinkable: set[str] = set()

    warm = _warm_cache_from_gold(args.gold_cache)
    resolved.update(warm)
    logger.info("warmed %d entries from gold cache", len(warm))

    if args.pred_cache.is_file():
        prev = json.loads(args.pred_cache.read_text())
        for k, v in prev.items():
            if v is None:
                unlinkable.add(k)
            else:
                resolved[k] = v
        logger.info("warmed %d entries (+ %d known-unlinkable) from pred cache",
                    len(prev) - len(unlinkable), len(unlinkable))

    needed = [s for s in sorted(unique)
              if s not in resolved and s not in unlinkable]
    logger.info("need fresh ROR calls for %d strings", len(needed))

    # ---- 3. resolve ----
    sess = _build_session()
    delay = 1.0 / max(args.rate_limit_hz, 0.1)
    last_save = time.time()
    for i, s in enumerate(needed):
        r = _ror_lookup(s, sess, min_conf=args.min_conf)
        if r is None:
            unlinkable.add(s)
        else:
            resolved[s] = r
        if (i + 1) % 50 == 0:
            logger.info("  resolved %d/%d  (linked so far: %d, unlinkable: %d)",
                        i + 1, len(needed),
                        sum(1 for k in resolved if k in unique),
                        sum(1 for k in unlinkable if k in unique))
        if time.time() - last_save > 60:
            _save_pred_cache(args.pred_cache, resolved, unlinkable, unique)
            last_save = time.time()
        time.sleep(delay)
    _save_pred_cache(args.pred_cache, resolved, unlinkable, unique)

    # ---- 4. rewrite merged JSONL ----
    n_papers_updated = 0
    with args.merged.open() as fin, args.out.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            changed = False
            for pred in d.get("predictions", {}).values():
                raws = pred.get("affiliations_raw") or []
                if not raws:
                    continue
                ror_ids, names, country_codes = [], [], []
                for s in raws:
                    key = (s or "").strip()
                    r = resolved.get(key)
                    if r:
                        ror_ids.append(r["ror_id"])
                        names.append(r["name"])
                        country_codes.append(r["country_code"])
                    else:
                        ror_ids.append(None)
                        names.append(None)
                        country_codes.append(None)
                if any(ror_ids):
                    pred["ror_ids"] = [r for r in ror_ids if r]
                    pred["names"] = [n for n, r in zip(names, ror_ids) if r]
                    changed = True
            if changed:
                n_papers_updated += 1
            fout.write(json.dumps(d, ensure_ascii=False) + "\n")
    logger.info("wrote %s  (papers with newly-linked predictions: %d)",
                args.out, n_papers_updated)
    return 0


def _save_pred_cache(path: Path, resolved: dict, unlinkable: set, in_use: set) -> None:
    # Persist only strings we've actually seen in this corpus (avoid bloat)
    cache = {}
    for k, v in resolved.items():
        if k in in_use:
            cache[k] = v
    for k in unlinkable:
        if k in in_use:
            cache[k] = None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
