"""Bulk ROR-link the raw affiliation strings in every predictions_*.jsonl
file in a comparison output dir. Run AFTER all extraction pipelines have
finished and BEFORE scoring/aggregation.

Why: per-pipeline ROR linking serialises poorly across parallel slurm
jobs — each pipeline contests the same ROR API rate limit (~7 req/sec).
Doing it ONCE here, with a single resolver hitting the shared cache,
exploits two cheats at once:
   - de-duplication: 24 pipelines × 781 papers might emit 30k affiliation
     strings total, but only ~2,500 are unique → 12x reduction in API calls
   - cache amortisation: lookups done while processing pipeline #1 are
     instant hits for pipelines #2..#24

Skips predictions that already have ror_id set (e.g., the passthrough
'ours' pipeline reads pre-linked records from disk and we don't want to
overwrite those).

Usage:
    python scripts/ror_link_predictions.py \\
        --predictions-dir reports/pipeline_comparison/ \\
        --ror-cache cache/ror_lookup_cache.json \\
        --hierarchy-cache cache/ror_hierarchy_cache.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from aff_pipelines.ror import RorResolver  # noqa: E402

logger = logging.getLogger("ror_link_preds")


def _collect_unique_strings(predictions_dir: Path) -> Dict[str, List[str]]:
    """Return {raw_aff_string: [pipeline_ids that emitted it]}.
    De-duplicates so the resolver does one lookup per unique string."""
    out: Dict[str, set] = {}
    for fp in sorted(predictions_dir.glob("predictions_*.jsonl")):
        pid = fp.stem.removeprefix("predictions_")
        with fp.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for a in (rec.get("authors") or []):
                    for af in (a.get("affiliations") or []):
                        raw = (af.get("raw_affiliation") or "").strip()
                        if not raw:
                            continue
                        # If the prediction is already ROR-linked, no need
                        # to re-look-up (preserves the passthrough 'ours'
                        # pipeline's existing high-quality links).
                        if af.get("ror_id"):
                            continue
                        out.setdefault(raw, set()).add(pid)
    return {k: sorted(v) for k, v in out.items()}


def _link_one_prediction_file(fp: Path, resolver) -> int:
    """Open one predictions JSONL, attach ror_id/clean_institution_name/
    country_code to every affiliation that doesn't already have a ror_id.
    Writes back in place. Returns count of new links added."""
    new_links = 0
    with fp.open() as f:
        lines = f.readlines()
    updated_lines: List[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            updated_lines.append("\n")
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            updated_lines.append(line + "\n")
            continue
        changed = False
        for a in (rec.get("authors") or []):
            for af in (a.get("affiliations") or []):
                if af.get("ror_id"):
                    continue  # already linked — preserve
                raw = (af.get("raw_affiliation") or "").strip()
                if not raw:
                    continue
                resolved = resolver(raw)
                if not resolved:
                    continue
                af["ror_id"] = resolved.get("ror_id")
                af["clean_institution_name"] = resolved.get("clean_institution_name")
                af["country_code"] = resolved.get("country_code")
                new_links += 1
                changed = True
        if changed:
            updated_lines.append(json.dumps(rec, ensure_ascii=False) + "\n")
        else:
            updated_lines.append(line + "\n")
    fp.write_text("".join(updated_lines))
    return new_links


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions-dir", type=Path, required=True)
    ap.add_argument("--ror-cache", type=Path,
                    default=Path("cache/ror_lookup_cache.json"))
    ap.add_argument("--hierarchy-cache", type=Path,
                    default=Path("cache/ror_hierarchy_cache.json"))
    ap.add_argument("--min-confidence", type=float, default=0.7)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    if not args.predictions_dir.is_dir():
        raise SystemExit(f"not a directory: {args.predictions_dir}")

    args.ror_cache.parent.mkdir(parents=True, exist_ok=True)
    resolver = RorResolver(
        cache_path=args.ror_cache,
        hierarchy_cache_path=args.hierarchy_cache,
        min_confidence=args.min_confidence,
    )

    # 1. Collect unique strings across all predictions
    t0 = time.time()
    unique = _collect_unique_strings(args.predictions_dir)
    logger.info("found %d unique unlinked affiliation strings across "
                "%d predictions files (%.1fs to scan)",
                len(unique),
                len(list(args.predictions_dir.glob("predictions_*.jsonl"))),
                time.time() - t0)

    # 2. Resolve serially — exploits the cache + amortises 429 risk
    t1 = time.time()
    n_resolved = 0
    n_cache_hits = 0
    new_lookups_before = resolver._new_lookups
    for i, raw in enumerate(sorted(unique.keys())):
        before = resolver._new_lookups
        r = resolver(raw)
        if resolver._new_lookups == before:
            n_cache_hits += 1
        if r:
            n_resolved += 1
        if (i + 1) % 100 == 0:
            logger.info("resolved %d/%d unique strings  (cache_hits=%d, "
                        "new_lookups=%d)",
                        i + 1, len(unique), n_cache_hits,
                        resolver._new_lookups - new_lookups_before)
    resolver.save()
    logger.info("resolution pass done: %d/%d resolved, cache_hits=%d, "
                "new_lookups=%d, %.1fs",
                n_resolved, len(unique), n_cache_hits,
                resolver._new_lookups - new_lookups_before,
                time.time() - t1)

    # 3. Patch every predictions_*.jsonl in place
    t2 = time.time()
    total_new_links = 0
    for fp in sorted(args.predictions_dir.glob("predictions_*.jsonl")):
        added = _link_one_prediction_file(fp, resolver)
        logger.info("  %s: +%d ROR links written", fp.name, added)
        total_new_links += added
    logger.info("patched all prediction files in %.1fs (+%d ROR links total)",
                time.time() - t2, total_new_links)

    print(f"\n=== ROR LINKING DONE ===")
    print(f"  unique strings:      {len(unique)}")
    print(f"  resolved (matched):  {n_resolved}")
    print(f"  cache hits:          {n_cache_hits}")
    print(f"  new API lookups:     {resolver._new_lookups - new_lookups_before}")
    print(f"  total links added:   {total_new_links}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
