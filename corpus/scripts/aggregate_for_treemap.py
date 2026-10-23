"""Aggregate the bibmeta-enriched corpus into long-form CSVs ready for
squarified treemaps and other group-by visualisations.

Walks the 117k per-paper JSONs once and emits three long-form CSVs:

  1) institutions_by_country.csv
       country_code, country_label, ror_id, canonical_name, paper_count
       — one row per (country, institution); paper counted at most
         once per (paper, institution).

  2) topics_by_country.csv
       country_code, country_label, cso_label, paper_count
       — one row per (country, CSO label); paper counted once per
         (paper, country, label).

  3) institutions_by_topic.csv
       cso_label, ror_id, canonical_name, country_code, country_label,
       paper_count
       — one row per (CSO label, institution); paper counted once per
         (paper, label, institution).

All counts dedupe at the paper level so multi-author papers don't
double-count a single institution, and multi-CSO papers don't
double-count a single (country, institution) pair.

After running on the cluster, scp the three CSVs to your laptop and
use the snippet at the bottom of this file to convert any of them into
the (group, [(item, count), ...]) shape your squarify code expects.

Usage:
    python scripts/aggregate_for_treemap.py \\
        --corpus-dir /share/.../ror_parsed_affiliations_90_with_topics_and_bibmeta \\
        --out-dir reports/aggregates
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("aggregate_treemap")


# Friendly country labels for the top countries in the corpus
# (per reports/corpus_stats.md). Anything not in this dict falls
# back to the ISO code itself in the CSV — extend as needed.
COUNTRY_LABELS: Dict[str, str] = {
    "US": "United States", "CN": "China", "GB": "United Kingdom",
    "DE": "Germany", "JP": "Japan", "FR": "France", "IN": "India",
    "CA": "Canada", "IT": "Italy", "NL": "Netherlands",
    "AU": "Australia", "HK": "Hong Kong", "TW": "Taiwan",
    "SG": "Singapore", "KR": "South Korea", "ES": "Spain",
    "CH": "Switzerland", "SE": "Sweden", "IE": "Ireland", "IL": "Israel",
    "BE": "Belgium", "AT": "Austria", "DK": "Denmark", "FI": "Finland",
    "NO": "Norway", "PL": "Poland", "RU": "Russia", "BR": "Brazil",
    "PT": "Portugal", "GR": "Greece", "CZ": "Czechia",
    "TR": "Turkey", "MX": "Mexico", "AE": "United Arab Emirates",
    "SA": "Saudi Arabia", "ZA": "South Africa", "QA": "Qatar",
    "AR": "Argentina", "CL": "Chile", "HU": "Hungary", "RO": "Romania",
    "TH": "Thailand", "MY": "Malaysia", "ID": "Indonesia",
    "VN": "Vietnam", "NZ": "New Zealand", "EG": "Egypt", "IR": "Iran",
}


def _label(cc: str) -> str:
    return COUNTRY_LABELS.get(cc, cc)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--corpus-dir", type=Path,
        default=Path("/share/moritz.staudinger/acl_data/"
                     "ror_parsed_affiliations_90_with_topics_and_bibmeta"),
        help="Per-paper JSON corpus.")
    ap.add_argument("--out-dir", type=Path,
                    default=Path("reports/aggregates"))
    ap.add_argument("--year-min", type=int, default=None,
                    help="Include only papers with year >= this value. "
                         "Papers without a usable year field are dropped "
                         "if either --year-min or --year-max is set.")
    ap.add_argument("--year-max", type=int, default=None,
                    help="Include only papers with year <= this value.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    year_filtering = args.year_min is not None or args.year_max is not None

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s")

    if not args.corpus_dir.is_dir():
        raise SystemExit(f"corpus dir not found: {args.corpus_dir}")

    files = sorted(args.corpus_dir.glob("*.json"))
    logger.info("scanning %d JSONs in %s", len(files), args.corpus_dir)

    # Aggregates
    inst_country_count: Counter = Counter()          # (cc, ror) → papers
    inst_name: Dict[str, str] = {}                   # ror   → display name
    inst_country: Dict[str, str] = {}                # ror   → ISO cc
    topic_country_count: Counter = Counter()         # (cc, label) → papers
    topic_inst_count: Counter = Counter()            # (label, ror) → papers

    n_papers = 0
    n_skipped_year = 0
    for fp in files:
        try:
            d = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue

        if year_filtering:
            try:
                year = int(d.get("year")) if d.get("year") is not None else None
            except (TypeError, ValueError):
                year = None
            if year is None:
                n_skipped_year += 1
                continue
            if args.year_min is not None and year < args.year_min:
                n_skipped_year += 1
                continue
            if args.year_max is not None and year > args.year_max:
                n_skipped_year += 1
                continue

        n_papers += 1
        if n_papers % 10_000 == 0:
            logger.info("  processed %d / %d papers", n_papers, len(files))

        # ---- gather per-paper sets first (dedupe within a paper) ----
        paper_insts: Dict[str, str] = {}    # ror → cc (first one wins)
        for author in (d.get("authors") or []):
            for aff in (author.get("affiliations") or []):
                if not isinstance(aff, dict):
                    continue
                ror = (aff.get("ror_id") or "").strip()
                if not ror:
                    continue
                cc = (aff.get("country_code") or "").strip().upper()
                if cc == "??":
                    cc = ""
                paper_insts.setdefault(ror, cc)
                name = (aff.get("clean_institution_name") or "").strip()
                if name and ror not in inst_name:
                    inst_name[ror] = name
                if cc and ror not in inst_country:
                    inst_country[ror] = cc

        # Countries seen on this paper (for topics×country)
        paper_countries = {cc for cc in paper_insts.values() if cc}

        cso_union = (((d.get("topics") or {}).get("cso") or {})
                     .get("union")) or []
        cso_labels = {str(t).lower() for t in cso_union if t}

        # ---- (1) institutions × country ----
        for ror, cc in paper_insts.items():
            inst_country_count[(cc, ror)] += 1

        # ---- (2) topics × country ----
        for cc in paper_countries:
            for label in cso_labels:
                topic_country_count[(cc, label)] += 1

        # ---- (3) institutions × topic ----
        for ror in paper_insts.keys():
            for label in cso_labels:
                topic_inst_count[(label, ror)] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- write CSV (1): institutions_by_country.csv ----
    path1 = args.out_dir / "institutions_by_country.csv"
    with path1.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["country_code", "country_label", "ror_id",
                    "canonical_name", "paper_count"])
        rows = sorted(inst_country_count.items(),
                      key=lambda kv: (-kv[1], kv[0]))
        for (cc, ror), n in rows:
            w.writerow([cc, _label(cc), ror,
                        inst_name.get(ror, ""), n])
    logger.info("wrote %s (%d rows)", path1, len(inst_country_count))

    # ---- write CSV (2): topics_by_country.csv ----
    path2 = args.out_dir / "topics_by_country.csv"
    with path2.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["country_code", "country_label",
                    "cso_label", "paper_count"])
        rows = sorted(topic_country_count.items(),
                      key=lambda kv: (-kv[1], kv[0]))
        for (cc, label), n in rows:
            w.writerow([cc, _label(cc), label, n])
    logger.info("wrote %s (%d rows)", path2, len(topic_country_count))

    # ---- write CSV (3): institutions_by_topic.csv ----
    path3 = args.out_dir / "institutions_by_topic.csv"
    with path3.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cso_label", "ror_id", "canonical_name",
                    "country_code", "country_label", "paper_count"])
        rows = sorted(topic_inst_count.items(),
                      key=lambda kv: (-kv[1], kv[0]))
        for (label, ror), n in rows:
            cc = inst_country.get(ror, "")
            w.writerow([label, ror, inst_name.get(ror, ""),
                        cc, _label(cc), n])
    logger.info("wrote %s (%d rows)", path3, len(topic_inst_count))

    if year_filtering:
        logger.info("done. processed %d papers in year range "
                    "(%s..%s); skipped %d outside range or year-less.",
                    n_papers,
                    args.year_min if args.year_min is not None else "−∞",
                    args.year_max if args.year_max is not None else "+∞",
                    n_skipped_year)
    else:
        logger.info("done. processed %d papers.", n_papers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
