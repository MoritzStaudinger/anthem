"""Build LoRA training data: paper header text → gold authors+affiliations.

Pulls (input, output) pairs by combining three local sources:

  - Header text  = first --max-header-chars of <pymu_llm_dir>/<paper_id>.md
                   (the user's ACL-Anthology pymu_llm parse). That window
                   covers title + author block + affiliations before the
                   abstract for almost every modern paper.

  - DOI bridge   = <enriched_affs_dir>/<paper_id>.json has `doi`
                   populated from the ACL Anthology bibmeta merge.

  - Gold output  = OpenAlex authorships (looked up by DOI) — per-author
                   raw_affiliation_strings + canonical institutions +
                   ROR IDs + ISO country codes.

The 781 papers in --gold-csv (the eval test set) are EXCLUDED from
training to avoid train/test contamination.

We previously tried using S2ORC fulltext as the header source but the
current S2ORC release ships content.annotations populated on only
~1.9% of records — the abstract start can't be located reliably. The
pymu_llm parse is dense ACL coverage with consistent text shape.

Output (newline-delimited JSON):
    <out>/train.jsonl       (90% by paper)
    <out>/val.jsonl         (10%)

Row schema:
    {"input": "<paper header text>",
     "output": [
        {"name": "Jane Doe",
         "affiliations": [{
            "raw": "...",
            "institution": "...",
            "ror_id": "https://ror.org/...",
            "country_code": "US"
         }]}, ...
     ],
     "source": {"paper_id": "...", "doi": "...", "work_id": "W..."}}
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("oa_fetch")

OPENALEX_BASE = "https://api.openalex.org/works"

# ---------------------------------------------------------------------------
# Local sources
# ---------------------------------------------------------------------------

def _read_pymu_header(md_path: Path, max_chars: int) -> Optional[str]:
    """First N characters of the pymu_llm markdown, with whitespace
    normalised. Returns None if the file is missing or empty."""
    if not md_path.is_file():
        return None
    try:
        text = md_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    text = text.strip()
    if not text:
        return None
    return text[:max_chars]


def _doi_from_enriched(enriched_path: Path) -> Optional[str]:
    """Read the `doi` field from the enriched affiliation JSON."""
    try:
        d = json.loads(enriched_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return None
    doi = d.get("doi") or ""
    doi = doi.strip().lower()
    return doi or None


def _gold_paper_ids(gold_csv: Path) -> Set[str]:
    """The 781-paper eval set to EXCLUDE from training."""
    out: Set[str] = set()
    if not gold_csv.is_file():
        logger.warning("gold csv %s not found — no papers will be excluded "
                       "from training (eval contamination risk!)", gold_csv)
        return out
    with gold_csv.open() as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("anthology_id") or "").strip()
            if pid:
                out.add(pid)
    logger.info("excluding %d gold papers from training pool", len(out))
    return out


# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------

def _oa_session(mailto: Optional[str]) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = (
        f"acl-processor/fetch_oa_training (mailto:{mailto})"
        if mailto else "acl-processor/fetch_oa_training"
    )
    return s


def _oa_work_by_doi(doi: str, sess: requests.Session,
                    mailto: Optional[str]) -> Optional[Dict[str, Any]]:
    """Look up an OpenAlex work by DOI. Returns None on miss / error."""
    url = f"{OPENALEX_BASE}/doi:{doi}"
    params = {"mailto": mailto} if mailto else {}
    try:
        r = sess.get(url, params=params, timeout=20)
    except requests.RequestException as e:
        logger.debug("OA req err %s: %s", doi, e)
        return None
    if r.status_code != 200:
        if r.status_code not in (404, 403):
            logger.debug("OA HTTP %d for %s", r.status_code, doi)
        return None
    return r.json()


def _gold_from_oa(oa: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Build per-author gold from an OpenAlex work. Returns None if no
    author has a ROR-linked institution (skip these — they're useless
    for ROR-aware training)."""
    rows = []
    n_with_ror = 0
    for a in (oa.get("authorships") or []):
        author = (a.get("author") or {}).get("display_name") or ""
        raw_affs = a.get("raw_affiliation_strings") or []
        institutions = a.get("institutions") or []
        affs = []
        for i, inst in enumerate(institutions):
            ror = inst.get("ror") or ""
            cc = inst.get("country_code") or ""
            inst_name = inst.get("display_name") or ""
            raw = raw_affs[i] if i < len(raw_affs) else (raw_affs[0] if raw_affs else inst_name)
            if not inst_name:
                continue
            if ror:
                n_with_ror += 1
            affs.append({
                "raw": raw,
                "institution": inst_name,
                "ror_id": ror,
                "country_code": cc,
            })
        if author and affs:
            rows.append({"name": author, "affiliations": affs})
    if not rows or n_with_ror == 0:
        return None
    return rows


# ---------------------------------------------------------------------------
# Per-paper pipeline
# ---------------------------------------------------------------------------

def _process_paper(
    paper_id: str,
    pymu_dir: Path,
    enriched_dir: Path,
    sess: requests.Session,
    mailto: Optional[str],
    min_header_chars: int,
    max_header_chars: int,
    oa_cache: Dict[str, Optional[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    md_path = pymu_dir / f"{paper_id}.md"
    header = _read_pymu_header(md_path, max_header_chars)
    if not header or len(header) < min_header_chars:
        return None
    enriched_path = enriched_dir / f"{paper_id}.json"
    doi = _doi_from_enriched(enriched_path)
    if not doi:
        return None
    if doi in oa_cache:
        oa = oa_cache[doi]
    else:
        oa = _oa_work_by_doi(doi, sess, mailto)
        oa_cache[doi] = oa  # None == cached miss
    if oa is None:
        return None
    gold = _gold_from_oa(oa)
    if not gold:
        return None
    return {
        "input": header,
        "output": gold,
        "source": {
            "paper_id": paper_id,
            "doi": doi,
            "work_id": (oa.get("id") or "").rsplit("/", 1)[-1] or None,
        },
    }


def _save_oa_cache(path: Path, cache: Dict[str, Optional[Dict[str, Any]]]) -> None:
    """Slim cache file — keep only the authorships field which is all
    we need on re-runs."""
    slim = {}
    for k, v in cache.items():
        if v is None:
            slim[k] = None
        else:
            slim[k] = {
                "id": v.get("id"),
                "doi": v.get("doi"),
                "authorships": v.get("authorships"),
            }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(slim, ensure_ascii=False))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pymu-dir", required=True, type=Path,
                    help="Directory of pymu_llm markdown files (one per paper).")
    ap.add_argument("--enriched-affs-dir", required=True, type=Path,
                    help="Directory of enriched affiliation JSONs (output of "
                         "enrich_affiliations_with_acl_metadata.py — needs "
                         "`doi` field populated).")
    ap.add_argument("--gold-csv", type=Path,
                    default=Path("reports/ground_truth_merged.csv"),
                    help="CSV of 781 gold paper IDs to EXCLUDE from training.")
    ap.add_argument("--out-dir", type=Path, default=Path("data/lora_train"))
    ap.add_argument("--cache-path", type=Path,
                    default=Path("data/lora_train/.oa_cache.json"),
                    help="OpenAlex lookup cache (resumable across runs).")
    ap.add_argument("--target-rows", type=int, default=20_000)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--mailto", default=None,
                    help="Your email for the OpenAlex polite pool (10 req/s).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-header-chars", type=int, default=3000)
    ap.add_argument("--min-header-chars", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    random.seed(args.seed)

    if not args.pymu_dir.is_dir():
        raise SystemExit(f"missing pymu_dir: {args.pymu_dir}")
    if not args.enriched_affs_dir.is_dir():
        raise SystemExit(f"missing enriched_affs_dir: {args.enriched_affs_dir}")

    mailto = args.mailto or os.environ.get("OPENALEX_MAILTO")
    if not mailto:
        logger.warning("no --mailto / OPENALEX_MAILTO — OpenAlex public pool "
                       "(slower). Set it for 10 req/sec.")

    # Warm-start OpenAlex lookup cache
    oa_cache: Dict[str, Optional[Dict[str, Any]]] = {}
    if args.cache_path.is_file():
        try:
            oa_cache = json.loads(args.cache_path.read_text())
            logger.info("warmed %d OpenAlex lookups from cache", len(oa_cache))
        except json.JSONDecodeError:
            oa_cache = {}

    # Enumerate candidate paper_ids: enriched JSONs that have a DOI,
    # minus the gold set, intersected with pymu_llm coverage.
    gold = _gold_paper_ids(args.gold_csv)
    enriched_files = sorted(args.enriched_affs_dir.glob("*.json"))
    candidates = [fp.stem for fp in enriched_files if fp.stem not in gold]
    random.shuffle(candidates)
    logger.info("candidate papers: %d (enriched JSONs %d, gold-set excl %d)",
                len(candidates), len(enriched_files), len(gold))

    sess = _oa_session(mailto)
    rows: List[Dict[str, Any]] = []
    examined = 0
    last_save = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for pid in candidates:
            if len(rows) >= args.target_rows:
                break
            futures[ex.submit(
                _process_paper, pid, args.pymu_dir, args.enriched_affs_dir,
                sess, mailto, args.min_header_chars, args.max_header_chars,
                oa_cache,
            )] = pid
            if len(futures) >= 200:
                for fut in as_completed(list(futures.keys())[:100]):
                    futures.pop(fut)
                    examined += 1
                    try:
                        r = fut.result()
                    except Exception as e:
                        logger.debug("process err: %s", e)
                        continue
                    if r:
                        rows.append(r)
                    if examined % 500 == 0:
                        logger.info("examined %d, kept %d / target %d  "
                                    "(oa cache size %d)",
                                    examined, len(rows), args.target_rows,
                                    len(oa_cache))
                    if time.time() - last_save > 60:
                        _save_oa_cache(args.cache_path, oa_cache)
                        last_save = time.time()
                    if len(rows) >= args.target_rows:
                        break
        for fut in as_completed(futures):
            examined += 1
            try:
                r = fut.result()
            except Exception:
                continue
            if r:
                rows.append(r)
            if len(rows) >= args.target_rows:
                break

    _save_oa_cache(args.cache_path, oa_cache)
    logger.info("examined %d candidates, kept %d usable rows",
                examined, len(rows))

    # 90/10 split, deterministic
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_fraction))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_p = args.out_dir / "train.jsonl"
    val_p = args.out_dir / "val.jsonl"
    with train_p.open("w", encoding="utf-8") as fh:
        for r in train_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with val_p.open("w", encoding="utf-8") as fh:
        for r in val_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info("wrote %d train + %d val rows", len(train_rows), len(val_rows))
    logger.info("  %s", train_p)
    logger.info("  %s", val_p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
