"""ROR-link the manually annotated affiliations.

Reads the cleaned annotation CSV, queries ROR's `?affiliation=...`
endpoint for each unique (raw_affiliation, raw_country) pair, and
augments the CSV with four new columns:

    ror_id          — URL of the matched ROR entity (or empty if no match)
    ror_name        — canonical institution name from ROR
    ror_country_code— ISO country code
    ror_confidence  — match score (0-1) returned by ROR's matcher

A disk cache (default `reports/ror_cache.json`) avoids re-querying the
same affiliation string on re-runs. Polite to the public ROR API: small
delay between requests + retry on 429s.

After this script runs, merge_gold.py picks up the ROR columns
automatically and uses ror_name as the canonical institution name in
the merged per-paper gold JSONs.

Usage:
    python scripts/ror_link_annotation.py \\
        --input  annotation_template.cleaned.csv \\
        --output annotation_template.ror_linked.csv \\
        --cache  reports/ror_cache.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

ROR_API = "https://api.ror.org/v2/organizations"   # v2 is the current stable endpoint
NULL_TOKENS = {"!NULL", "NULL", "NA", "N/A", "#NULL!"}


def _cell(row: Dict[str, str], key: str) -> str:
    return (row.get(key) or "").strip()


def _is_null(s: str) -> bool:
    return s.upper() in NULL_TOKENS


# ---------------------------------------------------------------------------
# ROR API
# ---------------------------------------------------------------------------

def ror_match(
    affiliation: str,
    session: requests.Session,
    retries: int = 3,
    timeout: int = 15,
) -> List[Dict[str, Any]]:
    """Query the ROR matcher and return ALL chosen matches.

    ROR returns a ranked list; items with `chosen=True` are the orgs ROR
    is confident the input string refers to. There can be multiple
    chosen items when the affiliation mentions multiple institutions
    (e.g. "LIMSI & CNRS" → 2 chosen items, one per institution).
    Returns an empty list if no chosen matches; falls back to the
    top-scored unchosen candidate so the caller can decide whether to
    accept it via the confidence threshold.
    """
    if not affiliation:
        return []
    params = {"affiliation": affiliation}
    for attempt in range(retries):
        try:
            r = session.get(ROR_API, params=params, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                items = data.get("items") or []
                if not items:
                    return []
                chosen = [it for it in items if it.get("chosen")]
                if chosen:
                    return chosen
                # No chosen — return top candidate so confidence filter decides
                return items[:1]
            if r.status_code == 429:
                wait = 2 ** attempt
                logger.warning("ROR 429; backing off %ds", wait)
                time.sleep(wait)
                continue
            logger.warning("ROR HTTP %d for %r", r.status_code, affiliation[:80])
            return []
        except requests.RequestException as e:
            logger.warning("ROR error (attempt %d) for %r: %s",
                           attempt + 1, affiliation[:80], e)
            time.sleep(1 + attempt)
    return []


def _flatten_one_match(match: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the four fields from a single ROR matcher result item."""
    if not match:
        return {"ror_id": "", "ror_name": "", "ror_country_code": "", "ror_confidence": ""}
    org = match.get("organization") or match
    ror_id = org.get("id") or ""
    names = org.get("names") or []
    canonical = next((n.get("value") for n in names
                      if "ror_display" in (n.get("types") or [])), None)
    if not canonical and names:
        canonical = names[0].get("value", "")
    locations = org.get("locations") or []
    country_code = ""
    if locations:
        country_code = (locations[0].get("geonames_details") or {}).get("country_code", "")
    score = match.get("score")
    return {
        "ror_id": ror_id,
        "ror_name": canonical or "",
        "ror_country_code": country_code,
        "ror_confidence": f"{score:.3f}" if isinstance(score, (int, float)) else "",
    }


def _flatten_matches(matches: List[Dict[str, Any]], min_confidence: float) -> Dict[str, Any]:
    """Collapse multiple ROR matches into one CSV row using pipe-separated
    values. Matches below `min_confidence` are dropped. Returns empty
    strings if nothing passes."""
    kept = []
    for m in matches:
        flat = _flatten_one_match(m)
        try:
            score = float(flat.get("ror_confidence") or 0)
        except ValueError:
            score = 0
        if score >= min_confidence and flat.get("ror_id"):
            kept.append(flat)
    if not kept:
        return {"ror_id": "", "ror_name": "", "ror_country_code": "",
                "ror_confidence": "", "ror_n_matches": "0"}
    return {
        "ror_id":            " | ".join(k["ror_id"] for k in kept),
        "ror_name":          " | ".join(k["ror_name"] for k in kept),
        "ror_country_code":  " | ".join(k["ror_country_code"] for k in kept),
        "ror_confidence":    " | ".join(k["ror_confidence"] for k in kept),
        "ror_n_matches":     str(len(kept)),
    }


# ---------------------------------------------------------------------------
# Cache + main loop
# ---------------------------------------------------------------------------

def load_cache(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("cache file %s unparseable; starting fresh", path)
        return {}


def save_cache(path: Path, cache: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", type=Path, required=True,
                   help="Cleaned annotation CSV (from cleanup_annotation.py).")
    p.add_argument("--output", type=Path, required=True,
                   help="Augmented CSV with 4 new ROR columns.")
    p.add_argument("--cache", type=Path, default=Path("reports/ror_cache.json"),
                   help="JSON cache: {raw_affiliation -> ror_match_record}. "
                        "Reused across runs; safe to delete to force re-query.")
    p.add_argument("--mailto", type=str, default=None,
                   help="Email for User-Agent (polite to the ROR API).")
    p.add_argument("--min-confidence", type=float, default=0.7,
                   help="If a ROR match's score is below this, drop it "
                        "(treat as un-linkable). Default 0.7. Set to 0 to "
                        "keep every match regardless of confidence.")
    p.add_argument("--delay", type=float, default=0.1,
                   help="Seconds to sleep between API calls (default 0.1; "
                        "ROR's public limit is ~2000/5min).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N rows (smoke test).")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cache = load_cache(args.cache)
    logger.info("loaded %d cached ROR matches from %s", len(cache), args.cache)

    with args.input.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if args.limit:
        rows = rows[: args.limit]
    logger.info("processing %d annotation rows", len(rows))

    # Add ROR columns to the header if missing
    for k in ("ror_id", "ror_name", "ror_country_code", "ror_confidence", "ror_n_matches"):
        if k not in fieldnames:
            fieldnames.append(k)

    session = requests.Session()
    session.headers.update({
        "User-Agent": f"acl_processor/ror_link (mailto:{args.mailto})" if args.mailto
                      else "acl_processor/ror_link"
    })

    stats = Counter()
    api_calls = 0
    for i, row in enumerate(rows):
        raw_aff = _cell(row, "raw affiliation")
        if not raw_aff or _is_null(raw_aff):
            stats["skipped_no_affiliation"] += 1
            for k in ("ror_id", "ror_name", "ror_country_code", "ror_confidence"):
                row[k] = ""
            continue

        # Cache key includes country to disambiguate "Cambridge" UK vs US
        country = _cell(row, "raw country")
        cache_key = f"{raw_aff}||{country}"

        if cache_key in cache:
            cached = cache[cache_key]
            stats["cache_hit"] += 1
        else:
            matches = ror_match(raw_aff, session)
            cached = _flatten_matches(matches, args.min_confidence)
            if int(cached.get("ror_n_matches") or 0) == 0:
                stats["low_confidence_or_no_match"] += 1
            cache[cache_key] = cached
            api_calls += 1
            stats["api_call"] += 1
            if api_calls % 50 == 0:
                save_cache(args.cache, cache)
                logger.info("checkpointed cache at %d API calls", api_calls)
            time.sleep(args.delay)

        for k in ("ror_id", "ror_name", "ror_country_code", "ror_confidence", "ror_n_matches"):
            row[k] = cached.get(k, "")

        if cached.get("ror_id"):
            stats["matched"] += 1
        else:
            stats["no_match"] += 1

    save_cache(args.cache, cache)

    # Write augmented CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            for k in fieldnames:
                row.setdefault(k, "")
            w.writerow(row)

    # Report
    print()
    print(f"=== ROR linking summary ===")
    for k in ("matched", "no_match", "low_confidence_drop",
              "cache_hit", "api_call", "skipped_no_affiliation"):
        print(f"  {k:<22} {stats[k]:>6}")
    print(f"  total rows:            {len(rows):>6}")
    print()
    print(f"  match rate: {100*stats['matched']/max(1,len(rows)-stats['skipped_no_affiliation']):.1f}%")
    print(f"  cache file: {args.cache}")
    print(f"  output:     {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
