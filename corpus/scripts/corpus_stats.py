"""Compute corpus statistics from the bibmeta-enriched affiliations
corpus and emit a markdown report suitable for paper use.

Walks <corpus-dir>/*.json (one per paper) and aggregates:
  - Total papers / year coverage / per-decade distribution
  - Venue / DOI / abstract / language coverage rates
  - Authors (total, mean per paper)
  - Affiliations: total raw strings, ROR-link rate, unique ROR ids,
    unique countries, per-paper means
  - CSO topic coverage: rate, total unique labels, top-N labels
  - ROR confidence distribution (≥0.9, ≥0.95, =1.0)

Output:
  - reports/corpus_stats.md  (markdown report)
  - reports/corpus_stats.json (machine-readable, same numbers)

Usage:
    python scripts/corpus_stats.py
    python scripts/corpus_stats.py \\
        --corpus-dir /share/.../ror_parsed_affiliations_90_with_topics_and_bibmeta \\
        --out-md reports/corpus_stats.md
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("corpus_stats")


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100 * n / total:.1f}%"


def _safe_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", type=Path,
                    default=Path("/share/moritz.staudinger/acl_data/"
                                 "ror_parsed_affiliations_90_with_topics_and_bibmeta"),
                    help="Bibmeta-enriched per-paper JSON corpus.")
    ap.add_argument("--out-md", type=Path,
                    default=Path("reports/corpus_stats.md"))
    ap.add_argument("--out-json", type=Path,
                    default=Path("reports/corpus_stats.json"))
    ap.add_argument("--top-n-topics", type=int, default=30,
                    help="How many top CSO labels to show in the report.")
    ap.add_argument("--top-n-institutions", type=int, default=30,
                    help="How many top institutions (by paper count) to show.")
    ap.add_argument("--top-n-countries", type=int, default=20)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    if not args.corpus_dir.is_dir():
        raise SystemExit(f"corpus dir not found: {args.corpus_dir}")

    files = sorted(args.corpus_dir.glob("*.json"))
    logger.info("scanning %d JSONs in %s", len(files), args.corpus_dir)

    # Top-line counters
    n_papers = 0
    n_with_year = 0
    n_with_venue = 0
    n_with_doi = 0
    n_with_abstract = 0
    n_with_language = 0
    n_with_any_affs = 0
    n_with_topics = 0

    year_hist: Counter = Counter()
    venue_hist: Counter = Counter()
    language_hist: Counter = Counter()

    # Author / affiliation aggregates
    total_authors = 0
    total_affs_raw = 0
    total_affs_linked = 0
    authors_per_paper: List[int] = []
    affs_per_paper: List[int] = []

    inst_hist: Counter = Counter()  # ror_id → paper count (one paper counted
                                    # at most once per ror_id)
    country_hist: Counter = Counter()
    ror_confidence_hist: Counter = Counter()  # bucket → count

    # CSO topic aggregates
    cso_label_hist: Counter = Counter()  # label → paper count
    cso_union_sizes: List[int] = []

    for fp in files:
        try:
            d = json.loads(fp.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        n_papers += 1

        # Bibmeta
        year = _safe_int(d.get("year"))
        if year is not None:
            n_with_year += 1
            year_hist[year] += 1
        venue = (d.get("venue") or "").strip()
        if venue:
            n_with_venue += 1
            venue_hist[venue] += 1
        if (d.get("doi") or "").strip():
            n_with_doi += 1
        if (d.get("abstract") or "").strip():
            n_with_abstract += 1
        lang = (d.get("language") or "").strip()
        if lang:
            n_with_language += 1
            language_hist[lang] += 1

        # Authors + affiliations
        authors = d.get("authors") or []
        total_authors += len(authors)
        authors_per_paper.append(len(authors))
        paper_inst_seen: set = set()
        paper_country_seen: set = set()
        paper_aff_count = 0
        paper_has_any_aff = False
        for author in authors:
            for aff in (author.get("affiliations") or []):
                if not isinstance(aff, dict):
                    continue
                raw = (aff.get("raw_affiliation") or "").strip()
                if raw:
                    total_affs_raw += 1
                    paper_aff_count += 1
                    paper_has_any_aff = True
                ror_id = (aff.get("ror_id") or "").strip()
                if ror_id:
                    total_affs_linked += 1
                    if ror_id not in paper_inst_seen:
                        paper_inst_seen.add(ror_id)
                        inst_hist[ror_id] += 1
                    score = aff.get("ror_score") or aff.get("ror_confidence")
                    try:
                        s = float(score) if score is not None else 0
                    except (TypeError, ValueError):
                        s = 0
                    if s >= 1.0:
                        ror_confidence_hist["=1.0"] += 1
                    elif s >= 0.95:
                        ror_confidence_hist[">=0.95"] += 1
                    elif s >= 0.90:
                        ror_confidence_hist[">=0.90"] += 1
                    else:
                        ror_confidence_hist["<0.90"] += 1
                cc = (aff.get("country_code") or "").strip().upper()
                if cc and cc != "??" and cc not in paper_country_seen:
                    paper_country_seen.add(cc)
                    country_hist[cc] += 1
        if paper_has_any_aff:
            n_with_any_affs += 1
        affs_per_paper.append(paper_aff_count)

        # CSO topics
        cso_union = (((d.get("topics") or {}).get("cso") or {}).get("union")) or []
        if cso_union:
            n_with_topics += 1
            cso_union_sizes.append(len(cso_union))
            for label in cso_union:
                cso_label_hist[str(label).lower()] += 1

    # ---- Derived stats ----
    def _pp_mean(xs: List[int]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    def _pp_median(xs: List[int]) -> int:
        if not xs:
            return 0
        s = sorted(xs)
        return s[len(s) // 2]

    decade_hist: Counter = Counter()
    for y, c in year_hist.items():
        decade_hist[f"{y // 10 * 10}s"] += c
    year_min = min(year_hist) if year_hist else None
    year_max = max(year_hist) if year_hist else None

    # ---- Assemble report ----
    md_lines: List[str] = []
    md_lines.append("# ACL Anthology corpus statistics\n")
    md_lines.append(f"Generated from `{args.corpus_dir}` "
                    f"({n_papers} per-paper JSONs).\n")

    md_lines.append("## Top-line coverage\n")
    md_lines.append("| metric | count | rate |")
    md_lines.append("|---|---:|---:|")
    rows = [
        ("papers (total)", n_papers, n_papers),
        ("with `year`", n_with_year, n_papers),
        ("with `venue`", n_with_venue, n_papers),
        ("with `doi`", n_with_doi, n_papers),
        ("with `abstract`", n_with_abstract, n_papers),
        ("with `language`", n_with_language, n_papers),
        ("with at least one affiliation", n_with_any_affs, n_papers),
        ("with CSO topics", n_with_topics, n_papers),
    ]
    for label, n, tot in rows:
        md_lines.append(f"| {label} | {n:,} | {_pct(n, tot)} |")
    md_lines.append("")

    md_lines.append("## Temporal coverage\n")
    md_lines.append(f"Year range: **{year_min}–{year_max}**  "
                    f"(distinct years = {len(year_hist)})\n")
    md_lines.append("| decade | papers |")
    md_lines.append("|---|---:|")
    for dec, c in sorted(decade_hist.items()):
        md_lines.append(f"| {dec} | {c:,} |")
    md_lines.append("")

    md_lines.append("## Authors & affiliations\n")
    md_lines.append("| metric | value |")
    md_lines.append("|---|---:|")
    md_lines.append(f"| total author entries | {total_authors:,} |")
    md_lines.append(f"| total raw affiliation strings | {total_affs_raw:,} |")
    md_lines.append(f"| total ROR-linked affiliations | {total_affs_linked:,} "
                    f"({_pct(total_affs_linked, total_affs_raw)} of raw) |")
    md_lines.append(f"| unique ROR ids in corpus | {len(inst_hist):,} |")
    md_lines.append(f"| unique countries | {len(country_hist)} |")
    md_lines.append(f"| mean authors / paper | {_pp_mean(authors_per_paper):.2f} "
                    f"(median {_pp_median(authors_per_paper)}) |")
    md_lines.append(f"| mean raw affs / paper | {_pp_mean(affs_per_paper):.2f} "
                    f"(median {_pp_median(affs_per_paper)}) |")
    md_lines.append("")

    md_lines.append("### ROR-match confidence distribution\n")
    md_lines.append("| confidence bucket | count |")
    md_lines.append("|---|---:|")
    for bucket in ("=1.0", ">=0.95", ">=0.90", "<0.90"):
        md_lines.append(f"| {bucket} | {ror_confidence_hist.get(bucket, 0):,} |")
    md_lines.append("")

    md_lines.append("## CSO topic coverage\n")
    if cso_union_sizes:
        s = sorted(cso_union_sizes)
        med = s[len(s) // 2]
        p90 = s[int(len(s) * 0.9)] if s else 0
        md_lines.append(f"- Papers with topics: **{n_with_topics:,}** "
                        f"({_pct(n_with_topics, n_papers)})")
        md_lines.append(f"- Mean CSO labels / paper: **{_pp_mean(cso_union_sizes):.1f}**  "
                        f"(median {med}, p90 {p90})")
        md_lines.append(f"- Unique CSO labels in corpus: **{len(cso_label_hist):,}**\n")
        md_lines.append(f"### Top {args.top_n_topics} CSO labels (by paper count)\n")
        md_lines.append("| CSO label | papers |")
        md_lines.append("|---|---:|")
        for label, c in cso_label_hist.most_common(args.top_n_topics):
            md_lines.append(f"| {label} | {c:,} |")
        md_lines.append("")
    else:
        md_lines.append("(no CSO topics found in corpus — `enrich_with_topics.py` "
                        "may not have run yet)\n")

    md_lines.append(f"## Top {args.top_n_countries} countries (by paper count)\n")
    md_lines.append("| ISO code | papers |")
    md_lines.append("|---|---:|")
    for cc, c in country_hist.most_common(args.top_n_countries):
        md_lines.append(f"| {cc} | {c:,} |")
    md_lines.append("")

    md_lines.append(f"## Top {args.top_n_institutions} institutions "
                    f"(by paper count)\n")
    md_lines.append("Note: counts a paper at most once per institution; "
                    "co-author papers contribute once per distinct ROR id.\n")
    md_lines.append("| ROR id | papers |")
    md_lines.append("|---|---:|")
    for ror_id, c in inst_hist.most_common(args.top_n_institutions):
        md_lines.append(f"| {ror_id} | {c:,} |")
    md_lines.append("")

    md_lines.append("## Per-venue distribution (top 30)\n")
    md_lines.append("| venue | papers |")
    md_lines.append("|---|---:|")
    for v, c in venue_hist.most_common(30):
        md_lines.append(f"| {v} | {c:,} |")
    md_lines.append("")

    md = "\n".join(md_lines) + "\n"
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")
    logger.info("wrote markdown report: %s", args.out_md)

    # JSON snapshot for downstream use
    snapshot: Dict[str, Any] = {
        "corpus_dir": str(args.corpus_dir),
        "n_papers": n_papers,
        "coverage": {
            "with_year": n_with_year,
            "with_venue": n_with_venue,
            "with_doi": n_with_doi,
            "with_abstract": n_with_abstract,
            "with_language": n_with_language,
            "with_any_affs": n_with_any_affs,
            "with_topics": n_with_topics,
        },
        "year_range": [year_min, year_max],
        "year_hist": dict(year_hist),
        "decade_hist": dict(decade_hist),
        "total_authors": total_authors,
        "total_affs_raw": total_affs_raw,
        "total_affs_linked": total_affs_linked,
        "unique_ror_ids": len(inst_hist),
        "unique_countries": len(country_hist),
        "mean_authors_per_paper": _pp_mean(authors_per_paper),
        "mean_affs_per_paper": _pp_mean(affs_per_paper),
        "ror_confidence_hist": dict(ror_confidence_hist),
        "n_unique_cso_labels": len(cso_label_hist),
        "top_cso_labels": dict(cso_label_hist.most_common(args.top_n_topics)),
        "top_countries": dict(country_hist.most_common(args.top_n_countries)),
        "top_institutions": dict(inst_hist.most_common(args.top_n_institutions)),
        "top_venues": dict(venue_hist.most_common(30)),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2),
                             encoding="utf-8")
    logger.info("wrote json snapshot: %s", args.out_json)

    # Also print the report to stdout so a sbatch tail shows everything
    print()
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
