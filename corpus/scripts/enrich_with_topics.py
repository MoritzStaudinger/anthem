"""Topic enrichment for SciCo affiliations JSONs (v3 — CSO-direct, no projection).

For each <paper_id>.json under <root>/affiliations/, adds a `topics` block:

    "topics": {
      "cso": {"syntactic": [...], "semantic": [...], "union": [...], "enhanced": [...]},
      "openalex_work_id": "<https://openalex.org/W...>",
      "_provenance": {...}
    }

The CSO classification (union list) is the topic ground truth. No custom
SciCo taxonomy projection — downstream eval and question generation read
the raw CSO labels and compute closure matches against the chosen
question-generation topics defined in configs/scico_topic_mapping.json.

OpenAlex's role is narrow: title backfill and work-id provenance only.
Its 4500-topic taxonomy is too coarse for NLP sub-areas (every NLP paper
gets ~the same trio) so it is not consulted for per-topic labeling.

Idempotent: re-running skips papers that already have a `topics._provenance`
block unless --force is passed.

Usage:
  python scripts/enrich_with_topics.py \\
      --affiliations-dir /path/to/datastructure/affiliations \\
      --mailto you@example.com

Requirements:
  pip install cso-classifier requests tqdm
  python -m spacy download en_core_web_sm
  python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

OPENALEX_BASE = "https://api.openalex.org/works"
ENRICHMENT_VERSION = "3.2"

# Most papers' abstracts live as structured fields in the ACL Anthology
# metadata XML. For the long tail not in anthology metadata (very old / some
# workshops), the PyMu_LLM Markdown export is the secondary source.


def load_anthology_abstracts(anthology_repo: Path) -> Dict[str, str]:
    """Build paper_id -> abstract from the cloned acl-org/acl-anthology repo
    (its data/xml/ directory). Uses ElementTree directly to avoid pulling
    in the acl-anthology Python package.
    """
    import xml.etree.ElementTree as ET

    xml_dir = anthology_repo / "data" / "xml"
    if not xml_dir.is_dir():
        # Repos sometimes don't have data/xml when the user passed the xml
        # dir itself or a non-standard layout — fall back to whatever they
        # gave us.
        xml_dir = anthology_repo
    out: Dict[str, str] = {}
    for xml_file in sorted(xml_dir.glob("*.xml")):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            continue
        root = tree.getroot()
        collection_id = root.attrib.get("id") or xml_file.stem  # e.g. "2021.acl"
        for volume in root.findall("volume"):
            vol_id = volume.attrib.get("id") or ""
            vol_full = f"{collection_id}-{vol_id}" if vol_id else collection_id
            for paper in volume.findall("paper"):
                pid_local = paper.attrib.get("id") or ""
                # ACL Anthology paper_id format is "<collection>-<volume>.<paper>" pre-2020,
                # and "<collection>.<paper>" 2020+ (collection encodes year+venue+volume).
                paper_id = f"{vol_full}.{pid_local}" if "." in collection_id else f"{collection_id}-{vol_id}{pid_local}"
                abstract_el = paper.find("abstract")
                if abstract_el is None:
                    continue
                # Flatten any inline markup (e.g., <fixed-case>) to plain text.
                text = "".join(abstract_el.itertext())
                text = re.sub(r"\s+", " ", text).strip()
                if text:
                    out[paper_id] = text
    return out


def _extract_abstract_from_md(path: Path) -> Optional[str]:
    """Pull the abstract section from a PyMu_LLM Markdown file. Falls back
    to the first ~3000 chars if no explicit Abstract header is present."""
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    m = re.search(
        r"(?:^|\n)#{0,6}\s*abstract\b[:.\s]*\n+(.+?)(?=\n#{1,6}\s|\Z)",
        text, re.IGNORECASE | re.DOTALL,
    )
    if m:
        body = re.sub(r"\s+", " ", m.group(1)).strip()
        return body[:5000] if body else None
    body = re.sub(r"\s+", " ", text).strip()
    return body[:3000] if body else None


def lookup_abstract(
    paper_id: str,
    anthology_abstracts: Optional[Dict[str, str]],
    md_dir: Optional[Path],
    min_chars: int = 100,
) -> Tuple[Optional[str], Optional[str]]:
    """Returns (abstract_text, source_tag) — source_tag is one of
    'anthology' / 'pymu_md' / None."""
    if anthology_abstracts and paper_id in anthology_abstracts:
        ab = anthology_abstracts[paper_id]
        if ab and len(ab) >= min_chars:
            return ab, "anthology"
    if md_dir:
        ab = _extract_abstract_from_md(md_dir / f"{paper_id}.md")
        if ab and len(ab) >= min_chars:
            return ab, "pymu_md"
    return None, None


# ---------------------------------------------------------------------------
# OpenAlex (title + work_id backfill only)
# ---------------------------------------------------------------------------

def _acl_doi_from_paper_id(paper_id: str) -> Optional[str]:
    """ACL Anthology assigns DOIs of the form 10.18653/v1/<paper_id> to
    most papers from ~2017 onward. Pre-2017 IDs (legacy format like
    'P80-1001') generally don't have DOIs; return None."""
    if not paper_id:
        return None
    if paper_id[:4].isdigit():
        return f"10.18653/v1/{paper_id}"
    return None


def fetch_openalex_meta(
    paper_id: str,
    doi: Optional[str],
    abstract: Optional[str],
    year: Optional[int],
    mailto: Optional[str],
    session: requests.Session,
    retries: int = 3,
) -> Tuple[Optional[str], Optional[str]]:
    """Return (openalex_work_id, display_name) or (None, None)."""
    headers = {"User-Agent": f"scico-enrichment (mailto:{mailto})" if mailto else "scico-enrichment"}
    select = "id,display_name"

    candidates: List[Tuple[str, str, Dict[str, Any]]] = []
    for d in [doi, _acl_doi_from_paper_id(paper_id)]:
        if not d:
            continue
        d_norm = d.lower().replace("https://doi.org/", "")
        candidates.append((
            "direct_doi",
            f"{OPENALEX_BASE}/https://doi.org/{d_norm}",
            {"select": select, **({"mailto": mailto} if mailto else {})},
        ))

    if abstract:
        snippet = " ".join(abstract.split()[:25])
        params: Dict[str, Any] = {"search": snippet, "per-page": 1, "select": select}
        if year:
            params["filter"] = f"publication_year:{year}"
        if mailto:
            params["mailto"] = mailto
        candidates.append(("search_abstract", OPENALEX_BASE, params))

    for kind, url, params in candidates:
        for attempt in range(retries):
            try:
                r = session.get(url, params=params, headers=headers, timeout=20)
                if r.status_code == 200:
                    data = r.json()
                    if kind == "direct_doi":
                        if data.get("id"):
                            return data.get("id"), data.get("display_name")
                    else:
                        res = data.get("results") or []
                        if res:
                            return res[0].get("id"), res[0].get("display_name")
                    break
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                if r.status_code == 404:
                    break
                break
            except requests.RequestException:
                time.sleep(1 + attempt)
                continue
    return None, None


# ---------------------------------------------------------------------------
# CSO classification (batched)
# ---------------------------------------------------------------------------

def run_cso_on_papers(
    papers: List[Tuple[str, str, str]],
    workers: int = 1,
) -> Dict[str, Dict[str, List[str]]]:
    """papers: list of (paper_id, title, abstract). Returns paper_id ->
    {syntactic, semantic, union, enhanced}."""
    from cso_classifier import CSOClassifier

    clf = CSOClassifier(modules="both", enhancement="first", explanation=False)
    payload = {
        pid: {"title": title or "", "abstract": abstract or ""}
        for pid, title, abstract in papers
    }
    if not payload:
        return {}
    result = clf.batch_run(papers=payload, workers=workers)
    out: Dict[str, Dict[str, List[str]]] = {}
    for pid, r in result.items():
        out[pid] = {
            "syntactic": list(r.get("syntactic") or []),
            "semantic":  list(r.get("semantic") or []),
            "union":     list(r.get("union") or []),
            "enhanced":  list(r.get("enhanced") or []),
        }
    return out


# ---------------------------------------------------------------------------
# Per-paper assembly + main
# ---------------------------------------------------------------------------

def build_topics_block(
    cso_block: Optional[Dict[str, List[str]]],
    openalex_work_id: Optional[str],
) -> Dict[str, Any]:
    cso_block = cso_block or {"syntactic": [], "semantic": [], "union": [], "enhanced": []}
    return {
        "cso": cso_block,
        "openalex_work_id": openalex_work_id,
        "_provenance": {
            "cso_ran": bool(cso_block.get("union")),
            "openalex_found": bool(openalex_work_id),
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "enrichment_version": ENRICHMENT_VERSION,
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description="CSO topic enrichment for SciCo paper JSONs.")
    p.add_argument("--affiliations-dir", type=Path, required=True)
    p.add_argument("--mailto", type=str, default=None,
                   help="Email for OpenAlex polite pool.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N papers (smoke tests).")
    p.add_argument("--force", action="store_true",
                   help="Re-run on papers that already have a topics block.")
    p.add_argument("--cso-batch", type=int, default=200)
    p.add_argument("--cso-workers", type=int, default=1)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="If set, enriched JSONs are written to this directory "
                        "as <paper_id>.json. Use this for canary/test runs to "
                        "keep the source affiliations dir clean. If unset, "
                        "the script overwrites the source files in place "
                        "(the production behaviour for the full corpus).")
    p.add_argument("--out-suffix", type=str, default="",
                   help="(deprecated, prefer --out-dir) If set, write to "
                        "<id>.<suffix>.json next to the source file.")
    p.add_argument("--skip-openalex", action="store_true",
                   help="Skip the OpenAlex title/work_id backfill (CSO only).")
    p.add_argument("--anthology-repo", type=Path, default=None,
                   help="Local clone of acl-org/acl-anthology. The XML files "
                        "under data/xml/ supply author-provided abstracts for "
                        "most papers — this is the primary abstract source "
                        "when the JSON has none.")
    p.add_argument("--md-dir", type=Path, default=None,
                   help="Directory of PyMu_LLM Markdown files (<paper_id>.md). "
                        "Used as a fallback for papers whose abstract isn't "
                        "in the anthology metadata.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    files = sorted(args.affiliations_dir.glob("*.json"))
    files = [f for f in files if not f.stem.endswith(".enriched")]
    if args.limit:
        files = files[: args.limit]
    logger.info("found %d affiliation JSONs", len(files))

    todo: List[Tuple[Path, Dict[str, Any]]] = []
    for fp in files:
        try:
            meta = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("skipping unreadable %s", fp.name)
            continue
        if not args.force and meta.get("topics", {}).get("_provenance"):
            continue
        todo.append((fp, meta))
    logger.info("will enrich %d papers (%d skipped, already done)",
                len(todo), len(files) - len(todo))

    # ---- OpenAlex shared session ---------------------------------------
    session = requests.Session() if not args.skip_openalex else None

    # Process in batches so a crash at batch N keeps the first N-1 batches.
    # Resumability comes from the topics._provenance check on disk.
    total_written = 0
    total_backfilled = 0
    n_batches = (len(todo) + args.cso_batch - 1) // args.cso_batch

    # Load anthology abstracts once up front.
    anthology_abstracts: Optional[Dict[str, str]] = None
    if args.anthology_repo:
        logger.info("loading anthology abstracts from %s ...", args.anthology_repo)
        anthology_abstracts = load_anthology_abstracts(args.anthology_repo)
        logger.info("indexed %d abstracts from anthology metadata", len(anthology_abstracts))

    backfill_counts: Dict[str, int] = {"anthology": 0, "pymu_md": 0}
    no_text_papers = 0
    for batch_idx in range(n_batches):
        batch = todo[batch_idx * args.cso_batch : (batch_idx + 1) * args.cso_batch]
        inputs: List[Tuple[str, str, str]] = []
        for fp, m in batch:
            pid = m.get("paper_id") or fp.stem
            title = m.get("title") or ""
            abstract = m.get("abstract") or ""
            if not abstract:
                fallback, source = lookup_abstract(pid, anthology_abstracts, args.md_dir)
                if fallback:
                    abstract = fallback
                    backfill_counts[source] = backfill_counts.get(source, 0) + 1
                    m["abstract"] = abstract       # persist into the written JSON
                    m.setdefault("_provenance", {})
                    m["_provenance"]["abstract_source"] = source
            if title or abstract:
                inputs.append((pid, title, abstract))
            else:
                no_text_papers += 1
        cso_results: Dict[str, Dict[str, List[str]]] = {}
        if inputs:
            logger.info("CSO batch %d/%d (%d papers with text)",
                        batch_idx + 1, n_batches, len(inputs))
            cso_results = run_cso_on_papers(inputs, workers=args.cso_workers)

        oa_id_results: Dict[str, Optional[str]] = {}
        oa_title_results: Dict[str, Optional[str]] = {}
        if session is not None:
            for fp, meta in tqdm(batch, desc=f"OpenAlex b{batch_idx+1}/{n_batches}"):
                pid = meta.get("paper_id") or fp.stem
                oa_id, title = fetch_openalex_meta(
                    paper_id=pid,
                    doi=meta.get("doi"),
                    abstract=meta.get("abstract"),
                    year=meta.get("year"),
                    mailto=args.mailto,
                    session=session,
                )
                oa_id_results[pid] = oa_id
                oa_title_results[pid] = title

        # Write the batch immediately — crash loses at most one batch.
        batch_written = 0
        batch_backfilled = 0
        for fp, meta in batch:
            pid = meta.get("paper_id") or fp.stem
            if not meta.get("title") and oa_title_results.get(pid):
                meta["title"] = oa_title_results[pid]
                batch_backfilled += 1
            meta["topics"] = build_topics_block(
                cso_block=cso_results.get(pid),
                openalex_work_id=oa_id_results.get(pid),
            )
            if args.out_dir:
                args.out_dir.mkdir(parents=True, exist_ok=True)
                target = args.out_dir / f"{fp.stem}.json"
            elif args.out_suffix:
                target = fp.parent / f"{fp.stem}.{args.out_suffix}.json"
            else:
                target = fp
            target.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            batch_written += 1
        total_written += batch_written
        total_backfilled += batch_backfilled
        logger.info("checkpoint: batch %d/%d done — %d files written (cumulative %d/%d)",
                    batch_idx + 1, n_batches, batch_written, total_written, len(todo))

    logger.info(
        "done — %d enriched, %d titles backfilled, abstracts backfilled: %s, %d papers had no usable text",
        total_written, total_backfilled, backfill_counts, no_text_papers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
