"""Sample N papers from the enriched affiliation set and check how
often OpenAlex returns usable affiliation data for them. Output is a
coverage report telling us:

  - what fraction of ACL papers have an OpenAlex record (via DOI)
  - of those, how many have at least one author with raw_affiliation_strings
  - of those, how many have at least one ROR-linked institution
  - of those, how many have a country_code

These four numbers bound the training-data yield. If only 30% of
ACL papers have ROR-linked OA authorships, building a 20k-sample
training set means scanning ~70k papers. If 80% do, we only need ~25k.

Decision is whether to proceed with the OpenAlex+pymu_llm fetcher.

Usage:
    python scripts/check_openalex_aff_coverage.py --sample 1000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger("oa_coverage")
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
    retry = Retry(total=4, backoff_factor=1.0,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = (
        f"acl-processor/check_oa_coverage (mailto:{mailto})"
        if mailto else "acl-processor/check_oa_coverage"
    )
    return s


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


def _classify(paper_id: str, enriched_path: Path,
              sess: requests.Session, mailto: Optional[str]
              ) -> Dict[str, Any]:
    """Return a per-paper coverage dict:
        no_doi / oa_miss / no_authorships /
        n_authors / n_with_raw_str / n_with_ror / n_with_country
    """
    out: Dict[str, Any] = {
        "paper_id": paper_id,
        "no_doi": False, "oa_miss": False, "no_authorships": False,
        "n_authors": 0, "n_with_raw_str": 0,
        "n_with_ror": 0, "n_with_country": 0,
        "n_with_institution": 0,
    }
    try:
        d = json.loads(enriched_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        out["no_doi"] = True
        return out
    doi = (d.get("doi") or "").strip().lower()
    if not doi:
        out["no_doi"] = True
        return out
    oa = _lookup(doi, sess, mailto)
    if oa is None:
        out["oa_miss"] = True
        return out
    authorships = oa.get("authorships") or []
    if not authorships:
        out["no_authorships"] = True
        return out
    for a in authorships:
        out["n_authors"] += 1
        if a.get("raw_affiliation_strings"):
            out["n_with_raw_str"] += 1
        institutions = a.get("institutions") or []
        if any(i.get("display_name") for i in institutions):
            out["n_with_institution"] += 1
        if any(i.get("ror") for i in institutions):
            out["n_with_ror"] += 1
        if any(i.get("country_code") for i in institutions):
            out["n_with_country"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--enriched-affs-dir", type=Path,
                    default=Path("/share/moritz.staudinger/acl_data/"
                                 "ror_parsed_affiliations_90_with_topics_and_bibmeta"))
    ap.add_argument("--sample", type=int, default=1000,
                    help="Random sample of papers to check (default 1000 "
                         "= ~2 min at 10 RPS).")
    ap.add_argument("--mailto", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("reports/openalex_aff_coverage.json"),
                    help="Where to dump the full per-paper classification.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    if not args.enriched_affs_dir.is_dir():
        raise SystemExit(f"missing {args.enriched_affs_dir}")
    _load_env(Path.cwd())
    mailto = args.mailto or os.environ.get("OPENALEX_MAILTO")
    if not mailto:
        logger.warning("no --mailto / OPENALEX_MAILTO — using OA public "
                       "pool (slow). Set for 10 RPS.")

    files = sorted(args.enriched_affs_dir.glob("*.json"))
    logger.info("total enriched JSONs: %d", len(files))
    random.seed(args.seed)
    sample = random.sample(files, min(args.sample, len(files)))
    logger.info("sampling %d papers for OpenAlex lookup", len(sample))

    sess = _session(mailto)
    results: List[Dict[str, Any]] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_classify, fp.stem, fp, sess, mailto): fp
                for fp in sample}
        n_done = 0
        for fut in as_completed(futs):
            n_done += 1
            try:
                results.append(fut.result())
            except Exception as e:
                logger.debug("err: %s", e)
            if n_done % 100 == 0:
                rate = n_done / (time.time() - t0)
                logger.info("  done %d/%d  (%.1f papers/sec)",
                            n_done, len(sample), rate)
    elapsed = time.time() - t0
    logger.info("lookups done in %.1fs", elapsed)

    # Aggregate
    n_total = len(results)
    n_no_doi = sum(1 for r in results if r["no_doi"])
    n_oa_miss = sum(1 for r in results if r["oa_miss"])
    n_no_authorships = sum(1 for r in results if r["no_authorships"])
    n_has_oa = n_total - n_no_doi - n_oa_miss
    n_has_authorships = n_has_oa - n_no_authorships
    n_has_any_ror = sum(1 for r in results if r["n_with_ror"] > 0)
    n_has_any_raw_str = sum(1 for r in results if r["n_with_raw_str"] > 0)
    n_has_any_country = sum(1 for r in results if r["n_with_country"] > 0)
    n_has_any_institution = sum(1 for r in results
                                if r["n_with_institution"] > 0)

    print()
    print("=" * 70)
    print(f"OpenAlex affiliation coverage on {n_total} sampled ACL papers")
    print("=" * 70)
    print(f"{'has DOI in enriched affs':<45} {n_total - n_no_doi:>6} "
          f"({100*(n_total-n_no_doi)/n_total:5.1f}%)")
    print(f"{'has OpenAlex record (DOI lookup hit)':<45} {n_has_oa:>6} "
          f"({100*n_has_oa/n_total:5.1f}%)")
    print(f"{'has ≥1 authorship in OA':<45} {n_has_authorships:>6} "
          f"({100*n_has_authorships/n_total:5.1f}%)")
    print(f"{'≥1 author with institution display_name':<45} "
          f"{n_has_any_institution:>6} "
          f"({100*n_has_any_institution/n_total:5.1f}%)")
    print(f"{'≥1 author with raw_affiliation_strings':<45} "
          f"{n_has_any_raw_str:>6} "
          f"({100*n_has_any_raw_str/n_total:5.1f}%)")
    print(f"{'≥1 author with ROR-linked institution':<45} "
          f"{n_has_any_ror:>6} "
          f"({100*n_has_any_ror/n_total:5.1f}%)")
    print(f"{'≥1 author with country_code':<45} "
          f"{n_has_any_country:>6} "
          f"({100*n_has_any_country/n_total:5.1f}%)")
    print()
    # Per-author distribution (over papers that have OA authorships)
    counts_ror = Counter()
    counts_raw = Counter()
    counts_total_authors = Counter()
    for r in results:
        if r["n_authors"] == 0:
            continue
        counts_total_authors[min(r["n_authors"], 15)] += 1
        if r["n_with_ror"] > 0:
            frac = r["n_with_ror"] / r["n_authors"]
            counts_ror[round(frac * 10)] += 1
        if r["n_with_raw_str"] > 0:
            frac = r["n_with_raw_str"] / r["n_authors"]
            counts_raw[round(frac * 10)] += 1
    print("Per-paper: of papers with OA authorships, what fraction of "
          "authors have a ROR-linked institution?")
    print("(0.0 = none, 1.0 = all)")
    for bin_, n in sorted(counts_ror.items()):
        print(f"  ≈{bin_/10:.1f}  {n:>5}  {'#' * (n // 5)}")
    print()
    print("Training-data yield extrapolation:")
    if n_total:
        train_yield_rate = n_has_any_ror / n_total
        print(f"  ~{train_yield_rate * 100:.1f}% of ACL papers will yield a "
              f"usable training row (has DOI + OA + ≥1 ROR-linked author).")
        for target in (5000, 10000, 20000):
            need = int(target / train_yield_rate) if train_yield_rate else float('inf')
            print(f"  target {target} rows → need to scan ~{need:,} papers")
    print()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "n_sampled": n_total,
        "summary": {
            "no_doi": n_no_doi,
            "oa_miss": n_oa_miss,
            "no_authorships": n_no_authorships,
            "has_oa_record": n_has_oa,
            "has_authorships": n_has_authorships,
            "with_institution": n_has_any_institution,
            "with_raw_aff_str": n_has_any_raw_str,
            "with_ror": n_has_any_ror,
            "with_country": n_has_any_country,
        },
        "per_paper": results,
    }, ensure_ascii=False, indent=2))
    print(f"full per-paper classification → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
