"""Fetch per-paper, per-author affiliations from the OpenAlex graph
API for every paper in the 781-paper gold set, and write a predictions
JSONL in the same schema as the other aff-comparison pipelines.

This evaluates OpenAlex itself as a pipeline:
  - raw_affiliation_strings (raw track)
  - institution display_name (name track)
  - ror (ror / ror_hier tracks)
  - country_code

OpenAlex is the data source for the LoRA training gold; this fetcher
lets us put a hard number on OpenAlex's own affiliation quality before
trusting it as supervision. If OpenAlex's predictions score badly
against our gold, the LoRA model can't be expected to do better.

Output:
    <out>/predictions_openalex.jsonl
    rows: {paper_id, authors: [{name, affiliations: [{raw_affiliation,
                                                        clean_institution_name,
                                                        ror_id,
                                                        country_code}]}],
           _pipeline: {id: "openalex", ...}}

Usage:
    python scripts/fetch_openalex_affiliations.py \\
        --gold-csv reports/ground_truth_merged.csv \\
        --enriched-affs-dir /share/.../<...>_and_bibmeta \\
        --out reports/pipeline_comparison/predictions_openalex.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("oa_pipeline")
OPENALEX_BASE = "https://api.openalex.org/works"


def _load_env(cwd: Path) -> None:
    env = cwd / ".env"
    if not env.is_file():
        return
    for raw in env.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _session(mailto: Optional[str]) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = (
        f"acl-processor/fetch_openalex_affiliations (mailto:{mailto})"
        if mailto else "acl-processor/fetch_openalex_affiliations"
    )
    return s


def _doi_for_paper(paper_id: str, enriched_dir: Optional[Path]) -> Optional[str]:
    """Find the DOI for a paper.

    Tries in order:
      1. enriched_dir/<paper_id>.json `doi` field (most reliable)
      2. ACL Anthology's predictable DOI pattern for modern papers:
         <year>.<venue>... → 10.18653/v1/<paper_id>
    Returns None for legacy papers (P85-1010 style) without a DOI.
    """
    if enriched_dir is not None:
        fp = enriched_dir / f"{paper_id}.json"
        if fp.is_file():
            try:
                d = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
                doi = (d.get("doi") or "").strip().lower()
                if doi:
                    return doi
            except (OSError, json.JSONDecodeError):
                pass
    # Fallback: derive from modern ACL pattern. The first 4 chars of
    # post-2020 paper_ids are digits. Most have DOI 10.18653/v1/<id>.
    if paper_id and paper_id[:4].isdigit() and int(paper_id[:4]) >= 2020:
        return f"10.18653/v1/{paper_id}".lower()
    return None


def _lookup(doi: str, sess: requests.Session,
            mailto: Optional[str]) -> Optional[Dict[str, Any]]:
    url = f"{OPENALEX_BASE}/doi:{doi}"
    params = {"mailto": mailto} if mailto else {}
    try:
        r = sess.get(url, params=params, timeout=20)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    return r.json()


def _to_prediction_row(paper_id: str, oa: Optional[Dict[str, Any]],
                       wall: float, error: Optional[str]) -> Dict[str, Any]:
    """Convert OpenAlex authorships into the pipeline schema. Schema
    matches what S2ORCPassthrough and the LLM pipelines emit:
        authors: [{name, affiliations: [{raw_affiliation,
                                           clean_institution_name,
                                           ror_id,
                                           country_code}]}]
    """
    authors_out: List[Dict[str, Any]] = []
    if oa:
        for a in (oa.get("authorships") or []):
            name = (a.get("author") or {}).get("display_name") or ""
            raw_strs = a.get("raw_affiliation_strings") or []
            institutions = a.get("institutions") or []
            affs: List[Dict[str, Any]] = []
            # Pair each institution with a raw string when possible;
            # OpenAlex doesn't guarantee 1:1 alignment so we just emit
            # one entry per institution and reuse the first raw string
            # when the lists are uneven.
            for i, inst in enumerate(institutions):
                raw = raw_strs[i] if i < len(raw_strs) else (raw_strs[0] if raw_strs else "")
                affs.append({
                    "raw_affiliation": raw or (inst.get("display_name") or ""),
                    "clean_institution_name": inst.get("display_name") or "",
                    "ror_id": inst.get("ror") or "",
                    "country_code": inst.get("country_code") or "",
                })
            # If OA returned raw strings but no institutions, still emit
            # them so the `raw` scoring track has something to match on.
            if not affs and raw_strs:
                for r in raw_strs:
                    affs.append({
                        "raw_affiliation": r,
                        "clean_institution_name": "",
                        "ror_id": "",
                        "country_code": "",
                    })
            authors_out.append({"name": name, "affiliations": affs})
    return {
        "paper_id": paper_id,
        "authors": authors_out,
        "_pipeline": {
            "id": "openalex",
            "extraction_seconds": wall,
            "tokens_in": None,
            "tokens_out": None,
            "error": error,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold-csv", type=Path,
                   default=Path("reports/ground_truth_merged.csv"))
    p.add_argument("--enriched-affs-dir", type=Path,
                   default=Path("/share/moritz.staudinger/acl_data/"
                                "ror_parsed_affiliations_90_with_topics_and_bibmeta"),
                   help="Source of paper_id → DOI mapping.")
    p.add_argument("--out", type=Path,
                   default=Path("reports/pipeline_comparison/predictions_openalex.jsonl"))
    p.add_argument("--rate-limit-hz", type=float, default=10.0)
    p.add_argument("--mailto", default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    _load_env(Path.cwd())
    mailto = args.mailto or os.environ.get("OPENALEX_MAILTO")
    if not mailto:
        logger.warning("no --mailto / OPENALEX_MAILTO — using OA public "
                       "pool (~1 RPS). Set it for 10 RPS.")
        args.rate_limit_hz = min(args.rate_limit_hz, 1.0)

    if not args.gold_csv.is_file():
        raise SystemExit(f"missing {args.gold_csv}")
    enriched_dir: Optional[Path] = args.enriched_affs_dir
    if enriched_dir is None or not enriched_dir.is_dir():
        logger.warning("enriched affs dir not found (%s) — will derive DOIs "
                       "from anthology_id pattern for modern papers only. "
                       "Legacy papers without DOI in OA will be misses.",
                       enriched_dir)
        enriched_dir = None

    paper_ids: List[str] = []
    with args.gold_csv.open() as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("anthology_id") or "").strip()
            if pid:
                paper_ids.append(pid)
    logger.info("loaded %d gold paper_ids", len(paper_ids))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sess = _session(mailto)
    delay = 1.0 / max(args.rate_limit_hz, 0.1)
    n_hit = 0; n_no_doi = 0; n_oa_miss = 0; n_err = 0
    t_start = time.time()
    with args.out.open("w", encoding="utf-8") as fh:
        for i, pid in enumerate(paper_ids):
            t0 = time.perf_counter()
            err: Optional[str] = None
            oa = None
            doi = _doi_for_paper(pid, enriched_dir)
            if not doi:
                n_no_doi += 1
                err = "no DOI in enriched affs"
            else:
                try:
                    oa = _lookup(doi, sess, mailto)
                    if oa is None:
                        n_oa_miss += 1
                        err = "OpenAlex DOI lookup miss"
                    else:
                        n_hit += 1
                except Exception as e:
                    n_err += 1
                    err = f"{type(e).__name__}: {e}"
            wall = time.perf_counter() - t0
            row = _to_prediction_row(pid, oa, wall, err)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 50 == 0:
                rate = (i + 1) / (time.time() - t_start)
                logger.info("  fetched %d/%d  hits=%d  no_doi=%d  miss=%d  "
                            "err=%d  rate=%.1f req/s",
                            i + 1, len(paper_ids), n_hit, n_no_doi,
                            n_oa_miss, n_err, rate)
            if doi:
                time.sleep(delay)  # only sleep when we actually hit the API
    elapsed = time.time() - t_start
    logger.info("done in %.1fs: hits=%d, no_doi=%d, miss=%d, err=%d",
                elapsed, n_hit, n_no_doi, n_oa_miss, n_err)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
