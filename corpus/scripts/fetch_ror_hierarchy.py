"""Build a local ROR hierarchy cache for the comparison harness.

For every unique ROR ID across the relinked merged file (gold +
predictions), hit ROR /organizations/<id> and record its parent/child/
related relationships. Result lets rescore_merged.py do real
hierarchy-aware scoring instead of treating ror_hier == ror.

Output cache schema (bare-id keyed):
    {"<bare_id>": {"parent": [...], "child": [...], "related": [...]}}

Bare id = "0199hds37" (no URL, no slashes). ROR IDs in the data are
usually URLs like "https://ror.org/0199hds37"; we strip them.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("ror_hier")
ROR_BASE = "https://api.ror.org/v2/organizations"


def _strip(rid: str) -> str:
    if not rid:
        return ""
    return rid.strip().rstrip("/").split("/")[-1].lower()


def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "acl-processor/fetch_ror_hierarchy"
    return s


def _fetch_one(sess: requests.Session, bare: str, timeout: int = 15
               ) -> Dict[str, List[str]]:
    try:
        r = sess.get(f"{ROR_BASE}/{bare}", timeout=timeout)
    except requests.RequestException as e:
        logger.warning("fetch %s failed: %s", bare, e)
        return {"parent": [], "child": [], "related": []}
    if r.status_code != 200:
        if r.status_code != 429:
            logger.warning("ROR /%s -> HTTP %d", bare, r.status_code)
        return {"parent": [], "child": [], "related": []}
    org = r.json() or {}
    rels = org.get("relationships") or []
    out: Dict[str, List[str]] = {"parent": [], "child": [], "related": []}
    for rel in rels:
        kind = (rel.get("type") or "").lower()
        rid = _strip(rel.get("id") or "")
        if rid and kind in out:
            out[kind].append(rid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", type=Path,
                    default=Path("reports/merged_per_paper.relinked.jsonl"))
    ap.add_argument("--out", type=Path,
                    default=Path("reports/ror_hierarchy_cache.json"))
    ap.add_argument("--rate-limit-hz", type=float, default=4.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    # Collect all unique ROR ids
    bare_ids: set[str] = set()
    with args.merged.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            for r in (d.get("gold", {}).get("ror_ids") or []):
                b = _strip(r)
                if b:
                    bare_ids.add(b)
            for pred in (d.get("predictions") or {}).values():
                for r in (pred.get("ror_ids") or []):
                    b = _strip(r)
                    if b:
                        bare_ids.add(b)
    logger.info("collected %d unique ROR ids", len(bare_ids))

    # Warm-start from existing cache
    cache: Dict[str, Dict[str, List[str]]] = {}
    if args.out.is_file():
        try:
            cache = json.loads(args.out.read_text())
            logger.info("warmed %d entries from existing cache", len(cache))
        except json.JSONDecodeError:
            logger.warning("could not parse existing cache, rebuilding")

    needed = [b for b in sorted(bare_ids) if b not in cache]
    logger.info("need fresh lookups for %d ids", len(needed))

    sess = _session()
    delay = 1.0 / max(args.rate_limit_hz, 0.1)
    last_save = time.time()
    for i, bare in enumerate(needed):
        cache[bare] = _fetch_one(sess, bare)
        if (i + 1) % 100 == 0:
            with_rels = sum(1 for v in cache.values()
                            if v.get("parent") or v.get("child") or v.get("related"))
            logger.info("  fetched %d/%d  (with_relationships=%d)",
                        i + 1, len(needed), with_rels)
        if time.time() - last_save > 60:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
            last_save = time.time()
        time.sleep(delay)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cache, ensure_ascii=False, indent=2))
    with_rels = sum(1 for v in cache.values()
                    if v.get("parent") or v.get("child") or v.get("related"))
    logger.info("wrote %s (%d ids, %d have at least one relationship)",
                args.out, len(cache), with_rels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
