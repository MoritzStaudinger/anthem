"""Precompute per-question gold paper_id sets so eval_scico.py doesn't
have to walk 92k JSON files over NFS on every run.

For each row in the testset, derive the gold paper_ids using the same
logic eval_scico.py uses at runtime (compute_gold_paper_ids in
backend.eval.gold). Save as one small JSON: {row_id: [paper_id, ...]}.

Run this ONCE per (testset, corpus) pair. ~5-15 min depending on NFS;
the resulting file is ~5-20 MB and loads in <1 sec at eval time.

Usage:
    python scripts/precompute_gold_paper_ids.py \\
        --affiliations-dir /share/.../ror_parsed_affiliations_90_with_topics \\
        --testset data/eval/scico_testset.jsonl \\
        --out data/eval/scico_testset.gold_paper_ids.json

Then in any eval_scico.py invocation, pass:
    --gold-paper-ids data/eval/scico_testset.gold_paper_ids.json
…instead of --affiliations-dir.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("precompute_gold")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--affiliations-dir", required=True, type=Path,
                    help="Affiliations corpus dir (the slow NFS path).")
    ap.add_argument("--testset", required=True, type=Path,
                    help="The eval testset JSONL.")
    ap.add_argument("--out", required=True, type=Path,
                    help="Where to write the precomputed gold lookup.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    # Find the project root + add to sys.path so backend.* imports work
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here.parent))
    from backend.eval.gold import (
        compute_gold_paper_ids, load_affiliations_corpus,
    )

    logger.info("loading affiliations corpus (this is the slow part)...")
    corpus = load_affiliations_corpus(args.affiliations_dir)
    logger.info("loaded %d papers", len(corpus))

    logger.info("loading testset %s", args.testset)
    rows = []
    with args.testset.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    logger.info("loaded %d testset rows", len(rows))

    logger.info("computing gold paper_id sets...")
    out_map: dict = {}
    for r in rows:
        rid = r.get("id")
        if not rid:
            continue
        # sorted list so the JSON file is deterministic across runs
        out_map[rid] = sorted(compute_gold_paper_ids(r, corpus))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_map, ensure_ascii=False))
    sizes = [len(v) for v in out_map.values()]
    logger.info("wrote %s — %d rows, gold sizes min=%d median=%d max=%d",
                args.out, len(out_map),
                min(sizes) if sizes else 0,
                sorted(sizes)[len(sizes) // 2] if sizes else 0,
                max(sizes) if sizes else 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
