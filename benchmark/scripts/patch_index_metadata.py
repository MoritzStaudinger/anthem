"""In-place patch an existing ACL vector index with new metadata fields
from enriched per-paper JSONs.

The embeddings only encode the document text — they don't depend on
metadata fields like year/venue/title. So if we want to add those
fields to the index without paying for a full re-embed, we can:

  1. Load the persisted index
  2. For each node, look up its paper_id in the enriched JSONs
  3. Mutate node.metadata in place, adding year/venue/title/doi/url/language
  4. Persist the index back to disk

~3-5 min for 235k docs vs ~20-30 min for a fresh build.

Usage:
    python scripts/patch_index_metadata.py \\
        --index-dir data/local_index_acl \\
        --enriched-dir /share/.../ror_parsed_affiliations_90_with_topics_and_bibmeta

After this completes, retrieval filters like
`year_start=2019, year_end=2022` actually narrow the result set instead
of zeroing it.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Repo root on sys.path so `backend.*` imports work when invoked directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("patch_index")


def _load_bibmeta(enriched_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Read enriched JSONs and produce {paper_id: {year, venue, title, doi, url, language}}."""
    out: Dict[str, Dict[str, Any]] = {}
    files = sorted(enriched_dir.glob("*.json"))
    logger.info("loading enriched metadata from %d files in %s",
                len(files), enriched_dir)
    n_with_year = 0
    for fp in files:
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        pid = d.get("paper_id") or fp.stem
        if not pid:
            continue
        if d.get("year") is not None:
            n_with_year += 1
        out[pid] = {
            "year": d.get("year"),
            "venue": d.get("venue") or "",
            "title": d.get("title") or "",
            "doi": d.get("doi"),
            "url": d.get("url"),
            "language": d.get("language"),
        }
    logger.info("loaded %d paper records (%d have year)", len(out), n_with_year)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index-dir", type=Path,
                    default=Path("data/local_index_acl"),
                    help="Persisted llama-index directory to patch in place.")
    ap.add_argument("--enriched-dir", type=Path, required=True,
                    help="Directory of enriched per-paper JSONs (output of "
                         "scripts/enrich_affiliations_with_acl_metadata.py).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change but don't persist.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    if not args.index_dir.is_dir():
        raise SystemExit(f"missing index dir: {args.index_dir}")
    if not args.enriched_dir.is_dir():
        raise SystemExit(f"missing enriched dir: {args.enriched_dir}")

    # Load enriched metadata first (fail fast if path wrong)
    t0 = time.time()
    bibmeta = _load_bibmeta(args.enriched_dir)
    logger.info("metadata load wall: %.1fs", time.time() - t0)

    # Heavy imports
    from llama_index.core import StorageContext, load_index_from_storage
    from backend.rag.local_acl_indexing import _acl_embed_model

    logger.info("loading persisted index from %s ...", args.index_dir)
    t1 = time.time()
    storage = StorageContext.from_defaults(persist_dir=str(args.index_dir))
    index = load_index_from_storage(storage, embed_model=_acl_embed_model())
    logger.info("index loaded in %.1fs; docs=%d",
                time.time() - t1, len(index.docstore.docs))

    # Walk the docstore, mutate metadata in place
    t2 = time.time()
    n_total = 0
    n_patched = 0
    n_no_paper_id = 0
    n_no_bibmeta = 0
    fields = ("year", "venue", "title", "doi", "url", "language")
    for doc_id, node in index.docstore.docs.items():
        n_total += 1
        meta = node.metadata or {}
        pid = meta.get("paper_id")
        if not pid:
            n_no_paper_id += 1
            continue
        bib = bibmeta.get(pid)
        if not bib:
            n_no_bibmeta += 1
            continue
        # Only overwrite when the bib value is non-empty AND the existing
        # value is missing/empty. Keeps the index idempotent across reruns.
        changed = False
        for f in fields:
            new = bib.get(f)
            if new in (None, ""):
                continue
            if meta.get(f) in (None, ""):
                meta[f] = new
                changed = True
            elif meta.get(f) != new:
                # If both populated but different, prefer the canonical
                # bibmeta — paper_id is authoritative.
                meta[f] = new
                changed = True
        if changed:
            node.metadata = meta
            n_patched += 1
        if n_total % 50000 == 0:
            logger.info("  walked %d/%d  patched=%d  no_bibmeta=%d",
                        n_total, len(index.docstore.docs), n_patched, n_no_bibmeta)

    logger.info("walk done in %.1fs", time.time() - t2)
    logger.info("  total docs            : %d", n_total)
    logger.info("  patched               : %d", n_patched)
    logger.info("  no paper_id on node   : %d", n_no_paper_id)
    logger.info("  paper_id not in bib   : %d", n_no_bibmeta)

    if args.dry_run:
        logger.info("DRY RUN — not persisting")
        return 0

    # Persist ONLY the docstore. The vector store (~5 GB embeddings) and
    # index_store didn't change — rewriting them adds 30-90s of pure
    # disk I/O for no reason. SimpleVectorStore reads node metadata from
    # the docstore at query time, so this is enough.
    t3 = time.time()
    docstore_path = args.index_dir / "docstore.json"
    logger.info("persisting docstore to %s ...", docstore_path)
    index.docstore.persist(persist_path=str(docstore_path))
    logger.info("persisted in %.1fs", time.time() - t3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
