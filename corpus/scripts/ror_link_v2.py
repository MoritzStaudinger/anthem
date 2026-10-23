"""ROR v2 affiliation linker for per-paper extraction JSONs.

Lifted from extractor.ipynb cell 25 — the same pass that produced the
existing ror_parsed_affiliations_90 corpus (MIN_AFF_SCORE = 0.90,
prefers ROR's `chosen` flag, falls back to score threshold).

Three phases:
  1. SCAN every <paper_id>.json in --in-dir for unique raw_affiliation
     strings (deduplicated across the whole batch).
  2. QUERY ROR v2 once per unique string, persisting results to a JSON
     cache so re-runs are instant.
  3. ENRICH each input JSON in parallel: every aff dict gets ror_id,
     clean_institution_name, country, country_code, ror_score added
     when the resolver had a match. Writes to --out-dir; never modifies
     --in-dir in place.

Compared to scripts/ror_link_predictions.py (which targets the
comparison harness's predictions_*.jsonl format), this script targets
the per-paper extraction JSON format the Gemini / Qwen extractors
emit.

Usage:
    python scripts/ror_link_v2.py \\
        --in-dir  /share/.../gemini_affiliations_NEW \\
        --out-dir /share/.../ror_parsed_affiliations_NEW_90 \\
        --mapping-file cache/ror_v2_mapping_cache.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger("ror_link_v2")

ROR_V2_URL = "https://api.ror.org/v2/organizations"


# ---------------------------------------------------------------------------
# ROR v2 response helpers (notebook cell 25)
# ---------------------------------------------------------------------------

def extract_ror_display_name(org: Dict[str, Any]) -> str:
    names = org.get("names") or []
    for n in names:
        if isinstance(n, dict) and "ror_display" in (n.get("types") or []):
            return str(n.get("value", "")).strip()
    return names[0].get("value", "") if names else ""


def extract_country_info(org: Dict[str, Any]) -> Tuple[str, str]:
    for loc in (org.get("locations") or []):
        gd = loc.get("geonames_details") or {}
        c_name = gd.get("country_name")
        c_code = gd.get("country_code")
        if c_name or c_code:
            return (c_name or "Unknown"), (c_code or "??")
    return "Unknown", "??"


def query_ror_v2(raw_affil: str, min_score: float, require_chosen: bool,
                 rate_limit_delay: float, timeout: int = 20
                 ) -> Optional[Dict[str, Any]]:
    if not raw_affil:
        return None
    try:
        time.sleep(rate_limit_delay)
        resp = requests.get(ROR_V2_URL,
                            params={"affiliation": raw_affil},
                            timeout=timeout)
        if resp.status_code != 200:
            return None
        items = resp.json().get("items") or []
        if not items:
            return None
        chosen_item = next((x for x in items if x.get("chosen") is True), None)
        if require_chosen and not chosen_item:
            return None
        best_item = chosen_item or items[0]
        score = best_item.get("score", 0)
        if not chosen_item and score < min_score:
            return None
        org = best_item.get("organization") or best_item
        c_name, c_code = extract_country_info(org)
        return {
            "ror_id": org.get("id"),
            "ror_name": extract_ror_display_name(org),
            "country": c_name,
            "country_code": c_code,
            "score": score,
            "method": "chosen" if chosen_item else "score_threshold",
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Phase 1: scan unique affiliation strings
# ---------------------------------------------------------------------------

def _extract_strings(filepath: Path) -> set:
    found: set = set()
    try:
        d = json.loads(filepath.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return found
    for author in d.get("authors") or []:
        for aff in author.get("affiliations") or []:
            raw = aff if isinstance(aff, str) else aff.get("raw_affiliation")
            if raw:
                found.add(raw)
    # Also scan the affiliation_index keys/values (notebook handles both
    # author-side and index-side strings).
    for v in (d.get("affiliation_index") or {}).values():
        if not isinstance(v, dict):
            continue
        raw = v.get("raw_affiliation")
        if raw:
            found.add(raw)
    return found


# ---------------------------------------------------------------------------
# Phase 3: enrich per-paper JSONs using the resolved mapping
# ---------------------------------------------------------------------------

def _enrich_one(in_path: Path, out_path: Path,
                mapping: Dict[str, Optional[Dict[str, Any]]]) -> None:
    try:
        data = json.loads(in_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("skip %s: %s", in_path.name, e)
        return
    for author in data.get("authors") or []:
        affs = author.get("affiliations") or []
        unique_affs: List[Dict[str, Any]] = []
        seen_ror: set = set()
        for aff in affs:
            raw_text = aff if isinstance(aff, str) else aff.get("raw_affiliation")
            match = mapping.get(raw_text) if raw_text else None
            new_aff: Dict[str, Any] = {"raw_affiliation": raw_text}
            if isinstance(aff, dict):
                new_aff.update(aff)
            if match:
                # Dedup repeated ROR ids within the same author block —
                # matches the notebook's seen_ror logic.
                if match["ror_id"] in seen_ror:
                    continue
                seen_ror.add(match["ror_id"])
                new_aff.update({
                    "ror_id": match["ror_id"],
                    "clean_institution_name": match["ror_name"],
                    "country": match["country"],
                    "country_code": match["country_code"],
                    "ror_score": match["score"],
                })
            unique_affs.append(new_aff)
        author["affiliations"] = unique_affs
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", type=Path, required=True,
                    help="Directory of per-paper extraction JSONs (e.g. Gemini "
                         "extractor output).")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Where ROR-linked JSONs land. NEVER the same as --in-dir.")
    ap.add_argument("--mapping-file", type=Path,
                    default=Path("cache/ror_v2_mapping_cache.json"),
                    help="Persistent {raw_affiliation: match_dict} cache. "
                         "Shared across runs so repeated strings hit cache "
                         "instead of the API.")
    ap.add_argument("--min-score", type=float, default=0.90,
                    help="Minimum ROR v2 score for non-chosen matches. "
                         "Default 0.90 matches the existing corpus convention "
                         "(the '_90' in ror_parsed_affiliations_90_...).")
    ap.add_argument("--require-chosen", action="store_true",
                    help="Only accept matches where ROR returns chosen=true. "
                         "Maximum precision but kills recall on harder strings.")
    ap.add_argument("--max-workers-api", type=int, default=10)
    ap.add_argument("--max-workers-io", type=int, default=16)
    ap.add_argument("--rate-limit-delay", type=float, default=0.2,
                    help="Sleep between ROR API calls (per-worker). 0.2s × 10 "
                         "workers = ~50 req/s, well under ROR's 2000/5min cap.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    if not args.in_dir.is_dir():
        raise SystemExit(f"--in-dir not found: {args.in_dir}")
    if args.in_dir.resolve() == args.out_dir.resolve():
        raise SystemExit("--in-dir and --out-dir must differ (in-place not allowed)")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.mapping_file.parent.mkdir(parents=True, exist_ok=True)

    # ----- Phase 1: scan -----
    files = sorted([fp for fp in args.in_dir.glob("*.json")
                    if not fp.name.startswith("_")])
    logger.info("phase 1 — scanning unique affiliations in %d files ...",
                len(files))
    unique_affs: set = set()
    with ThreadPoolExecutor(max_workers=args.max_workers_io) as ex:
        for result in ex.map(_extract_strings, files):
            unique_affs.update(result)
    logger.info("  %d unique affiliation strings", len(unique_affs))

    # ----- Phase 2: ROR queries (with cache) -----
    cache: Dict[str, Optional[Dict[str, Any]]] = {}
    if args.mapping_file.is_file():
        try:
            cache = json.loads(args.mapping_file.read_text())
            logger.info("loaded %d cached ROR matches from %s",
                        len(cache), args.mapping_file)
        except json.JSONDecodeError:
            logger.warning("cache file corrupt, starting fresh")

    to_query = [s for s in unique_affs if s not in cache]
    logger.info("phase 2 — querying ROR v2 for %d new strings "
                "(%d already cached) ...", len(to_query), len(cache))
    if to_query:
        with ThreadPoolExecutor(max_workers=args.max_workers_api) as ex:
            results = list(ex.map(
                lambda s: query_ror_v2(s, args.min_score,
                                       args.require_chosen,
                                       args.rate_limit_delay),
                to_query,
            ))
        for raw, res in zip(to_query, results):
            cache[raw] = res
        args.mapping_file.write_text(json.dumps(cache, indent=2,
                                                ensure_ascii=False),
                                     encoding="utf-8")
        n_hit = sum(1 for v in results if v)
        logger.info("  %d / %d new strings resolved (%.1f%%)",
                    n_hit, len(results), 100 * n_hit / max(len(results), 1))

    # ----- Phase 3: enrich files -----
    logger.info("phase 3 — enriching %d files into %s ...",
                len(files), args.out_dir)
    def _do(fp: Path) -> None:
        out_p = args.out_dir / fp.name
        _enrich_one(fp, out_p, cache)
    with ThreadPoolExecutor(max_workers=args.max_workers_io) as ex:
        list(ex.map(_do, files))
    logger.info("done — output: %s", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
