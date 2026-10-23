"""Fetch per-paper, per-author affiliations from the S2 graph API for
every paper in the gold set, and write a predictions JSONL in the same
schema as the other aff-comparison pipelines.

Why this exists:
  - The bulk `papers/` dataset has no per-author affiliations.
  - The bulk `s2orc/` dataset has affiliations as character-offset
    annotation spans, but only ~1.9% of records in the current release
    have those populated.
  - The S2 graph API endpoint
    `/paper/<id>?fields=authors.name,authors.affiliations` returns the
    canonical per-paper affiliations S2 actually serves on their site.

This is what an Allen AI / S2 reviewer would expect us to evaluate
against. The other s2orc baseline (fulltext-span extraction) stays as
a separate row labelled "s2orc_spans" — both are S2-derived, but the
graph-API row captures S2's actual answer for "who works where on this
paper".

Output:
    <out>/predictions_s2_graph.jsonl
    each line:
        {"paper_id": "...",
         "authors": [
            {"name": "...",
             "affiliations": [{"raw_affiliation": "..."}, ...]}, ...],
         "_pipeline": {"id": "s2_graph", ...}}

Usage:
    python scripts/fetch_s2_graph_affiliations.py \\
        --gold-csv reports/ground_truth_merged.csv \\
        --out reports/pipeline_comparison/predictions_s2_graph.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("s2_graph")

S2_GRAPH = "https://api.semanticscholar.org/graph/v1/paper"


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


def _session(api_key: str | None) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "acl-processor/fetch_s2_graph_affiliations"
    if api_key:
        s.headers["x-api-key"] = api_key
    return s


def _fetch_paper(sess: requests.Session, paper_id: str,
                 timeout: int = 30) -> Dict[str, Any] | None:
    """Look up a paper by ACL Anthology ID via the S2 graph API.
    Returns {authors: [...]} or None on miss/error."""
    url = f"{S2_GRAPH}/ACL:{paper_id}"
    params = {"fields": "title,authors.name,authors.affiliations,externalIds"}
    try:
        r = sess.get(url, params=params, timeout=timeout)
    except requests.RequestException as e:
        logger.debug("net err %s: %s", paper_id, e)
        return None
    if r.status_code != 200:
        if r.status_code not in (404, 403):
            logger.debug("HTTP %d for %s", r.status_code, paper_id)
        return None
    return r.json()


def _to_prediction_row(paper_id: str, s2_record: Dict[str, Any] | None,
                       wall: float, error: str | None) -> Dict[str, Any]:
    """Convert an S2 graph API record into the predictions JSONL row
    schema used by run_pipeline_comparison.py."""
    authors_out: List[Dict[str, Any]] = []
    if s2_record:
        for a in (s2_record.get("authors") or []):
            name = a.get("name") or ""
            affs_raw = a.get("affiliations") or []
            # S2 graph API returns affiliations as list[str]
            affs = []
            for s in affs_raw:
                if isinstance(s, str) and s.strip():
                    affs.append({"raw_affiliation": s.strip()})
            authors_out.append({"name": name, "affiliations": affs})
    return {
        "paper_id": paper_id,
        "authors": authors_out,
        "_pipeline": {
            "id": "s2_graph",
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
                   default=Path("reports/ground_truth_merged.csv"),
                   help="CSV with `anthology_id` column listing gold paper IDs.")
    p.add_argument("--out", type=Path,
                   default=Path("reports/pipeline_comparison/predictions_s2_graph.jsonl"))
    p.add_argument("--rate-limit-hz", type=float, default=10.0,
                   help="Max requests/sec. Without S2_API_KEY ~1/s "
                        "is realistic; with a key the per-user quota "
                        "depends on tier (typically 100/s).")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    _load_env(Path.cwd())
    api_key = os.environ.get("S2_API_KEY") or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        logger.warning("no S2_API_KEY in env — using public quota (~1 RPS, "
                       "slower). Set S2_API_KEY in .env for ~100 RPS.")
        args.rate_limit_hz = min(args.rate_limit_hz, 1.0)

    # Read the gold paper IDs
    if not args.gold_csv.is_file():
        raise SystemExit(f"missing {args.gold_csv}")
    paper_ids: List[str] = []
    with args.gold_csv.open() as fh:
        for row in csv.DictReader(fh):
            pid = (row.get("anthology_id") or "").strip()
            if pid:
                paper_ids.append(pid)
    logger.info("loaded %d gold paper_ids from %s", len(paper_ids), args.gold_csv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sess = _session(api_key)
    delay = 1.0 / max(args.rate_limit_hz, 0.1)
    n_hit = 0; n_miss = 0; n_err = 0
    t_start = time.time()
    with args.out.open("w", encoding="utf-8") as fh:
        for i, pid in enumerate(paper_ids):
            t0 = time.perf_counter()
            err = None
            rec = None
            try:
                rec = _fetch_paper(sess, pid)
                if rec is None:
                    n_miss += 1
                else:
                    n_hit += 1
            except Exception as e:
                err = f"{type(e).__name__}: {e}"
                n_err += 1
            wall = time.perf_counter() - t0
            row = _to_prediction_row(pid, rec, wall, err)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if (i + 1) % 100 == 0:
                logger.info("  fetched %d/%d  hits=%d  miss=%d  err=%d  "
                            "rate=%.1f req/s",
                            i + 1, len(paper_ids), n_hit, n_miss, n_err,
                            (i + 1) / (time.time() - t_start))
            time.sleep(delay)
    elapsed = time.time() - t_start
    logger.info("done in %.1fs: %d hits, %d misses, %d errors",
                elapsed, n_hit, n_miss, n_err)
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
