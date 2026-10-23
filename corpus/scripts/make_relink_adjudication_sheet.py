"""Build a human adjudication sheet for ROR linking (current vs single_search).

The auto-linked reference is not a true gold (it was produced by ROR linking,
not manual annotation), so strategy F1 vs it is circular. This sheet lets you
build a REAL per-string gold quickly: for every unique affiliation string it
shows the two contested candidates side by side with org NAMES, and you fill a
`correct_ror` column.

Output CSV columns:
    string, n_papers, agree,
    multi_ror, multi_name, multi_country, multi_score,
    single_ror, single_name, single_country, single_score,
    correct_ror, notes
Rows are ordered: DISAGREEMENTS first (sorted by n_papers desc — highest impact
first, so a partial pass still covers the most-used strings), then agreements
(with correct_ror pre-filled = the agreed id; spot-check only).

correct_ror conventions:
    - copy multi_ror or single_ror, or type a different bare ROR id / URL;
    - semicolon-separate if the string maps to several orgs (compound "A & B");
    - leave blank / write "none" if no ROR applies.

Needs a bounded org-by-id fetch for candidate names (cached to
reports/ror_name_cache.json; resumable).

Usage:
    python scripts/make_relink_adjudication_sheet.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from relink_strategy_compare import load_gold, string_ror_set, norm_ror  # noqa: E402

ORG_BY_ID = "https://api.ror.org/v2/organizations/"
DEFAULT_GOLD = Path("/home/moritz/Downloads/11688_Singing_from_the_Same_AN_Data/"
                    "sample_data/affiliation_ground_truth.csv")


def top_candidate(items):
    """Highest-score item (ror, score) or (None, None)."""
    best, bs = None, -1.0
    for it in items or []:
        s = float(it.get("score") or 0)
        if it.get("ror") and s > bs:
            best, bs = it, s
    return (best["ror"], best["score"]) if best else (None, None)


def fetch_name(bare: str, timeout: int) -> Tuple[str, str]:
    for attempt in range(5):
        try:
            req = urllib.request.Request(ORG_BY_ID + bare,
                                         headers={"User-Agent": "acl-adjud/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                org = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429 or 500 <= e.code < 600:
                time.sleep(2 ** attempt); continue
            return ("<http error>", "")
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt); continue
    else:
        return ("<fetch failed>", "")
    name = ""
    names = org.get("names") or []
    for n in names:
        if "ror_display" in (n.get("types") or []):
            name = n.get("value") or ""
            break
    if not name and names:
        name = names[0].get("value") or ""
    country = ""
    locs = org.get("locations") or []
    if locs:
        gd = (locs[0].get("geonames_details") or {})
        country = gd.get("country_code") or gd.get("country_name") or ""
    return (name, country)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold-csv", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--items-cache", type=Path,
                    default=REPO / "reports/ror_items_raw_cache.json")
    ap.add_argument("--name-cache", type=Path, default=REPO / "reports/ror_name_cache.json")
    ap.add_argument("--min-score", type=float, default=0.7)
    ap.add_argument("--rate-hz", type=float, default=5.0)
    ap.add_argument("--timeout", type=int, default=20)
    ap.add_argument("--out", type=Path, default=REPO / "reports/relink_compare/adjudication_sheet.csv")
    args = ap.parse_args()

    papers, uniq = load_gold(args.gold_csv)
    cache = json.loads(args.items_cache.read_text())
    freq = Counter()
    for p in papers.values():
        for s in set(p["raw_strings"]):
            freq[s] += 1

    # Per-string contested candidates.
    rows = []
    cand_ids: Set[str] = set()
    for s in uniq:
        m_items = cache.get(f"multi||{s}") or []
        sg_items = cache.get(f"single||{s}") or []
        m_ror, m_sc = top_candidate(m_items)
        # current rule accepts only score>=thr; reflect that
        if m_sc is not None and m_sc < args.min_score:
            m_ror = None
        s_ror, s_sc = top_candidate(sg_items)
        if s_sc is not None and s_sc < args.min_score:
            s_ror = None
        agree = (m_ror == s_ror)
        if m_ror:
            cand_ids.add(m_ror)
        if s_ror:
            cand_ids.add(s_ror)
        rows.append({"string": s, "n_papers": freq[s], "agree": agree,
                     "multi_ror": m_ror or "", "multi_score": m_sc,
                     "single_ror": s_ror or "", "single_score": s_sc})

    # Names for candidate ids (cached, resumable).
    names: Dict[str, Tuple[str, str]] = {}
    if args.name_cache.exists():
        names = {k: tuple(v) for k, v in json.loads(args.name_cache.read_text()).items()}
    missing = sorted(c for c in cand_ids if c not in names)
    print(f"# {len(cand_ids)} unique candidate ROR ids | {len(missing)} names to fetch",
          flush=True)
    delay = 1.0 / args.rate_hz if args.rate_hz > 0 else 0.0
    for i, b in enumerate(missing):
        names[b] = fetch_name(b, args.timeout)
        if delay:
            time.sleep(delay)
        if (i + 1) % 50 == 0:
            args.name_cache.write_text(json.dumps({k: list(v) for k, v in names.items()},
                                                  ensure_ascii=False))
            print(f"# fetched {i+1}/{len(missing)} names", flush=True)
    args.name_cache.write_text(json.dumps({k: list(v) for k, v in names.items()},
                                          ensure_ascii=False))

    def nm(rid):
        return names.get(rid, ("", ""))[0] if rid else ""

    def cc(rid):
        return names.get(rid, ("", ""))[1] if rid else ""

    # Order: disagreements first (n_papers desc), then agreements (n_papers desc).
    rows.sort(key=lambda r: (r["agree"], -r["n_papers"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["string", "n_papers", "agree", "multi_ror", "multi_name", "multi_country",
            "multi_score", "single_ror", "single_name", "single_country", "single_score",
            "correct_ror", "notes"]
    n_dis = 0
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            dis = not r["agree"]
            n_dis += int(dis)
            # pre-fill correct_ror for agreements (spot-check); blank for disagreements
            prefill = "" if dis else r["multi_ror"]
            w.writerow([
                r["string"], r["n_papers"], "" if dis else "AGREE",
                r["multi_ror"], nm(r["multi_ror"]), cc(r["multi_ror"]),
                f'{r["multi_score"]:.3f}' if r["multi_score"] is not None else "",
                r["single_ror"], nm(r["single_ror"]), cc(r["single_ror"]),
                f'{r["single_score"]:.3f}' if r["single_score"] is not None else "",
                prefill, "",
            ])
    print(f"# wrote {args.out}", flush=True)
    print(f"#   {len(rows)} strings | {n_dis} disagreements (blank correct_ror, do these) "
          f"| {len(rows)-n_dis} agreements (pre-filled, spot-check)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
