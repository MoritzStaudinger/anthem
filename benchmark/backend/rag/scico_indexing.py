"""SciCo dataset adapter — turns the data/datastructure layout into a
llama-index VectorStoreIndex over abstracts (and full text when available).

SciCo layout:
    <root>/affiliations/<paper_id>.json   — ROR-normalized affiliations + abstract
    <root>/pymu_llm/<paper_id>.md         — full text (often only stubs)
    <root>/grobid/<paper_id>.tei.xml      — alt parser output (unused here)
    <root>/ocr/<paper_id>.jsonl           — alt parser output (unused here)
    <root>/ground_truth.csv               — gold standard (used by eval, not indexing)

Each Document is the abstract (or full text when present), with metadata:
    paper_id, title, publication_year, venue, url, authors,
    affiliations, affiliations_string, countries, countries_string,
    ror_ids, institution_types, doi.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

from backend.rag.local_acl_indexing import (
    DEFAULT_ACL_PERSIST_DIR,
    _acl_embed_model,
)

logger = logging.getLogger(__name__)

DEFAULT_SCICO_ROOT = "data/datastructure"
# Two persist dirs — one for the abstracts-only index (fast institution/
# topic queries) and one for the full-text index (paper-vs-paper
# methodology comparisons). The abstracts dir aliases the original ACL
# persist dir for backward compatibility with the existing
# search_local_acl_papers agent tool.
DEFAULT_SCICO_ABSTRACTS_PERSIST_DIR = DEFAULT_ACL_PERSIST_DIR
DEFAULT_SCICO_FULLTEXT_PERSIST_DIR = "data/local_index_scico_fulltext"
DEFAULT_SCICO_PERSIST_DIR = DEFAULT_SCICO_ABSTRACTS_PERSIST_DIR  # legacy alias

# Text strategies for the document constructor.
ABSTRACT_ONLY = "abstract_only"
PREFER_FULLTEXT = "prefer_fulltext"


def _safe_read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _flatten_affiliations(meta: Dict[str, Any]):
    """Walk a SciCo affiliations JSON and produce flat lists of clean
    institution names, country codes, ROR ids, and institution types
    (deduplicated, stable order)."""
    affs: List[str] = []
    countries: List[str] = []
    rors: List[str] = []
    types: List[str] = []
    for author in meta.get("authors", []) or []:
        for aff in author.get("affiliations", []) or []:
            inst = aff.get("clean_institution_name") or aff.get("raw_affiliation")
            if inst and inst not in affs:
                affs.append(inst)
            cc = aff.get("country_code")
            if cc and cc not in countries:
                countries.append(cc)
            r = aff.get("ror_id")
            if r and r not in rors:
                rors.append(r)
            t = aff.get("institution_type")
            if t and t not in types:
                types.append(t)
    return affs, countries, rors, types


def _author_names(meta: Dict[str, Any]) -> List[str]:
    return [a.get("name") for a in (meta.get("authors") or []) if a.get("name")]


# Per-paper, we emit two Documents with the SAME paper_id metadata:
#   doc_type=abstract — text is the abstract (or full text in PREFER_FULLTEXT)
#   doc_type=people   — text is "Authors: ...\nAffiliations: ..."
# Same metadata (so MetadataFilters work uniformly), different embed text.
# The agent's search_local_acl_papers tool dedupes by paper_id so the
# user-visible result set still has one row per paper.
DOC_TYPE_ABSTRACT = "abstract"
DOC_TYPE_PEOPLE = "people"

# Metadata fields excluded from the MiniLM embed string (and the LLM
# context string). They stay in node.metadata for filtering and direct
# access by the agent.
#   id-only       : paper_id, doi, url        — random tokens, embed noise
#   filter-only   : publication_year, countries(_string), has_full_text
#                   — accessed via MetadataFilter, not text similarity
#   bloat         : authors / affiliations / ror_ids / institution_types
#                   — go in the people doc's text instead
#   redundant     : abstract                  — already the text body
#                   affiliations_string        — already in people doc text
_EXCLUDED_FROM_EMBED = [
    "doc_type",
    "paper_id", "doi", "url",
    "publication_year", "countries", "countries_string", "has_full_text",
    "authors", "affiliations", "affiliations_string",
    "ror_ids", "institution_types",
    "abstract",
]


def _shared_metadata(
    meta: Dict[str, Any], full_text: Optional[str],
    affs: List[str], countries: List[str], rors: List[str], types: List[str],
) -> Dict[str, Any]:
    paper_id = meta.get("paper_id") or ""
    return {
        "paper_id": paper_id,
        "title": meta.get("title") or paper_id,
        "publication_year": meta.get("year"),
        "venue": meta.get("venue") or "",
        "url": meta.get("url") or (f"https://aclanthology.org/{paper_id}/" if paper_id else ""),
        "doi": meta.get("doi") or meta.get("DOI") or "",
        "authors": _author_names(meta),
        "abstract": (meta.get("abstract") or "").strip(),
        "affiliations": affs,
        "affiliations_string": ", ".join(affs),
        "countries": countries,
        "countries_string": ", ".join(countries),
        "ror_ids": rors,
        "institution_types": types,
        "has_full_text": bool(full_text and full_text.strip()),
    }


def _documents_for_paper(
    meta: Dict[str, Any],
    full_text: Optional[str],
    text_strategy: str = PREFER_FULLTEXT,
) -> List[Document]:
    paper_id = meta.get("paper_id") or ""
    abstract = (meta.get("abstract") or "").strip()
    if text_strategy == ABSTRACT_ONLY:
        body = abstract or paper_id
    else:  # PREFER_FULLTEXT
        body = (full_text or "").strip() or abstract or paper_id

    affs, countries, rors, types = _flatten_affiliations(meta)
    base_md = _shared_metadata(meta, full_text, affs, countries, rors, types)

    docs: List[Document] = []
    # Doc A — content (abstract or full text)
    docs.append(Document(
        text=body,
        metadata={**base_md, "doc_type": DOC_TYPE_ABSTRACT},
        excluded_embed_metadata_keys=_EXCLUDED_FROM_EMBED,
        excluded_llm_metadata_keys=_EXCLUDED_FROM_EMBED,
    ))
    # Doc B — people (authors + affiliations as text, for semantic match
    # against queries like "papers by John Smith" or "Carnegie Mellon NLP").
    authors = _author_names(meta)
    people_lines = []
    if authors:
        people_lines.append("Authors: " + ", ".join(authors))
    if affs:
        people_lines.append("Affiliations: " + ", ".join(affs))
    if countries:
        people_lines.append("Countries: " + ", ".join(countries))
    if people_lines:
        docs.append(Document(
            text="\n".join(people_lines),
            metadata={**base_md, "doc_type": DOC_TYPE_PEOPLE},
            excluded_embed_metadata_keys=_EXCLUDED_FROM_EMBED,
            excluded_llm_metadata_keys=_EXCLUDED_FROM_EMBED,
        ))
    return docs


def load_scico_papers(scico_root: str | Path) -> List[Dict[str, Any]]:
    """Return the raw affiliation records (paper_id → meta dict) sorted by id.
    Used both by the indexer and the eval grader.

    Accepts either:
      - <scico_root>/affiliations/<paper_id>.json   (classic SciCo layout)
      - <scico_root>/<paper_id>.json                 (flat — when the
        caller points us straight at the enriched dir)
    """
    root = Path(scico_root)
    nested = root / "affiliations"
    if nested.is_dir():
        aff_dir = nested
    elif any(root.glob("*.json")):
        aff_dir = root
    else:
        raise FileNotFoundError(
            f"No affiliation JSONs found under {root} or {nested}."
        )
    out = []
    for jp in sorted(aff_dir.glob("*.json")):
        try:
            out.append(json.loads(_safe_read(jp)))
        except json.JSONDecodeError:
            logger.warning("could not parse %s, skipping", jp)
    return out


def build_scico_documents(
    scico_root: str | Path,
    text_strategy: str = PREFER_FULLTEXT,
) -> List[Document]:
    """Load every affiliations JSON; attach full text from pymu_llm/<id>.md
    when present.

    text_strategy:
      - PREFER_FULLTEXT: text = full markdown if present, else abstract.
      - ABSTRACT_ONLY:   text = abstract regardless (smaller index, faster
                          to build, cheaper to embed; ideal for institution-
                          and topic-level queries).
    """
    root = Path(scico_root)
    text_dir = root / "pymu_llm"
    papers = load_scico_papers(root)

    documents: List[Document] = []
    with_full_text = 0
    abstract_only_count = 0
    skipped = 0

    for meta in papers:
        paper_id = meta.get("paper_id")
        if not paper_id:
            skipped += 1
            continue

        full_text = None
        if text_strategy == PREFER_FULLTEXT:
            md_path = text_dir / f"{paper_id}.md"
            if md_path.exists():
                full_text = _safe_read(md_path)
                with_full_text += 1
            else:
                abstract_only_count += 1
        # ABSTRACT_ONLY ignores the markdown directory entirely.

        if not (full_text or meta.get("abstract")):
            skipped += 1
            continue

        documents.extend(_documents_for_paper(meta, full_text, text_strategy=text_strategy))

    if text_strategy == PREFER_FULLTEXT:
        logger.info(
            "SciCo: built %d documents (%d with full text, %d abstract-only "
            "fallback, %d skipped for missing content)",
            len(documents), with_full_text, abstract_only_count, skipped,
        )
    else:
        logger.info(
            "SciCo: built %d documents (abstract-only mode, %d skipped for "
            "missing abstract)",
            len(documents), skipped,
        )
    return documents


def _atomic_persist(index, persist_path: Path) -> None:
    """Persist the index to disk via write-then-rename, so a job killed
    mid-write doesn't leave the JSON files truncated and the next
    resume crashing on JSONDecodeError.

    llama-index's SimpleVectorStore writes top-to-bottom into the live
    file. Kill the job between truncate and final write and you're left
    with empty/partial JSON. We persist into a sibling tmp dir, then
    atomically swap files into the canonical persist dir.
    """
    import os, shutil
    persist_path.mkdir(parents=True, exist_ok=True)
    tmp = persist_path.with_name(persist_path.name + ".tmp_persist")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(tmp))
    # Atomic per-file rename. We can't atomically swap entire directories
    # (rename(dir, dir) only works on empty target on POSIX), so we move
    # files one at a time. Each individual file rename IS atomic, so a
    # crash mid-loop leaves the persist dir mixed-version but not
    # truncated — still readable, possibly out-of-sync between
    # index_store/docstore/vector_store, which load_index handles.
    for src in tmp.iterdir():
        dst = persist_path / src.name
        os.replace(str(src), str(dst))
    shutil.rmtree(tmp, ignore_errors=True)


def _load_processed(progress_file: Path) -> set:
    if not progress_file.exists():
        return set()
    return {line.strip() for line in progress_file.read_text().splitlines() if line.strip()}


def _append_processed(progress_file: Path, paper_ids: Iterable[str]) -> None:
    with progress_file.open("a", encoding="utf-8") as fh:
        for pid in paper_ids:
            fh.write(f"{pid}\n")


def _documents_for(
    meta: Dict[str, Any], text_dir: Path, text_strategy: str
) -> List[Document]:
    paper_id = meta.get("paper_id")
    if not paper_id:
        return []
    full_text = None
    if text_strategy == PREFER_FULLTEXT:
        md_path = text_dir / f"{paper_id}.md"
        if md_path.exists():
            full_text = _safe_read(md_path)
    if not (full_text or meta.get("abstract")):
        return []
    return _documents_for_paper(meta, full_text, text_strategy=text_strategy)


def build_scico_index_resumable(
    scico_root: str | Path = DEFAULT_SCICO_ROOT,
    persist_dir: str | Path | None = None,
    *,
    text_strategy: str = ABSTRACT_ONLY,
    batch_size: int = 500,
    show_progress: bool = True,
    limit_papers: Optional[int] = None,
    no_checkpoint: bool = False,
) -> VectorStoreIndex:
    """Build (or resume) a SciCo index in batches, checkpointing after each.

    On restart, loads the persisted index + `processed.txt` and skips any
    paper_id already embedded. Crashes mid-batch lose at most one batch
    of work, not the whole run. Designed for the 100k-paper corpus where
    fresh-build wall time runs into hours.

    text_strategy defaults to ABSTRACT_ONLY (the right v1 for the big
    corpus — fast, cheap, and the drill_into_papers_fulltext agent tool
    handles the rare paper-comparison cases on demand).
    """
    if persist_dir is None:
        persist_dir = (
            DEFAULT_SCICO_ABSTRACTS_PERSIST_DIR
            if text_strategy == ABSTRACT_ONLY
            else DEFAULT_SCICO_FULLTEXT_PERSIST_DIR
        )
    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    progress_file = persist_path / "processed.txt"

    processed = _load_processed(progress_file)
    text_dir = Path(scico_root) / "pymu_llm"
    embed_model = _acl_embed_model()

    # Resume an existing index if there's evidence of prior progress.
    if processed and (persist_path / "index_store.json").exists():
        logger.info(
            "resuming SciCo build at %s — %d papers already embedded",
            persist_path, len(processed),
        )
        storage = StorageContext.from_defaults(persist_dir=str(persist_path))
        index = load_index_from_storage(storage, embed_model=embed_model)
    else:
        logger.info("starting fresh SciCo build at %s (%s)", persist_path, text_strategy)
        index = VectorStoreIndex.from_documents([], embed_model=embed_model)

    papers = load_scico_papers(scico_root)
    if limit_papers is not None:
        logger.info("LIMIT mode: only processing first %d papers", limit_papers)
        papers = papers[:limit_papers]
    total_seen = 0
    skipped = 0
    batch: List[Document] = []
    batch_ids: List[str] = []
    embedded = 0

    # Pull the splitter once. `index.insert(doc)` re-walks the
    # transformation pipeline ONE DOC AT A TIME, which forces the
    # embedder to run with batch_size=1 — wasting the GPU. We pre-split
    # the whole flush batch ourselves and hand the splitter's nodes to
    # `index.insert_nodes()`, which batches across the embedder
    # (controlled by HuggingFaceEmbedding(embed_batch_size=128)).
    from llama_index.core.node_parser import SentenceSplitter
    _splitter = SentenceSplitter(
        chunk_size=Settings.chunk_size or 1024,
        chunk_overlap=Settings.chunk_overlap or 20,
    )

    def _flush_batch() -> None:
        nonlocal batch, batch_ids, embedded
        if not batch:
            return
        # 1) Split all docs in the batch into nodes in one pass.
        nodes = _splitter.get_nodes_from_documents(batch, show_progress=False)
        # 2) One bulk insert — the embedder sees `len(nodes)` items and
        #    actually fills its 128-doc batch.
        index.insert_nodes(nodes)
        # Persist every batch unless --no-checkpoint was requested. The
        # 5 GB vector_store rewrite dominates wall time, so skipping
        # intermediate persists ~halves the build budget. Crash recovery
        # in no_checkpoint mode = re-run from scratch.
        if not no_checkpoint:
            _atomic_persist(index, persist_path)
        _append_processed(progress_file, batch_ids)
        # embedded counts papers (matches processed.txt); each paper
        # contributes 1-2 docs to the index but we report paper-level
        # progress for parity with len(papers).
        embedded += len(batch_ids)
        logger.info("checkpointed %d / %d papers (this run: %d papers, %d docs, %d nodes)",
                    len(processed) + embedded, len(papers), embedded,
                    len(batch), len(nodes))
        batch.clear()
        batch_ids.clear()

    for meta in papers:
        total_seen += 1
        pid = meta.get("paper_id")
        if not pid or pid in processed:
            skipped += 1
            continue
        docs = _documents_for(meta, text_dir, text_strategy)
        if not docs:
            skipped += 1
            continue
        # Both docs (abstract + people) for one paper go in together so
        # `processed.txt` only marks the paper as done once both are
        # safely persisted; a mid-paper crash re-emits both on resume.
        batch.extend(docs)
        batch_ids.append(pid)
        # batch_size counts papers, not docs — keep checkpoint cadence
        # roughly the same as before (≈500 papers ≈ ≤1000 docs).
        if len(batch_ids) >= batch_size:
            _flush_batch()

    _flush_batch()  # final partial batch
    # In no_checkpoint mode we skipped the per-batch persist — write
    # everything to disk once now that we're done.
    if no_checkpoint:
        logger.info("no_checkpoint mode — persisting final index ...")
        _atomic_persist(index, persist_path)
    logger.info(
        "SciCo resumable build done — %d new docs (%d total processed, %d skipped, %d papers seen)",
        embedded, len(processed) + embedded, skipped, total_seen,
    )
    return index


def build_scico_index(
    scico_root: str | Path = DEFAULT_SCICO_ROOT,
    persist_dir: str | Path | None = None,
    *,
    text_strategy: str = PREFER_FULLTEXT,
    show_progress: bool = True,
) -> VectorStoreIndex:
    """Build + persist a SciCo VectorStoreIndex using MiniLM embeddings.

    text_strategy picks between abstracts-only and full-text indexing.
    persist_dir defaults based on strategy (abstracts → DEFAULT_SCICO_
    ABSTRACTS_PERSIST_DIR, fulltext → DEFAULT_SCICO_FULLTEXT_PERSIST_DIR).
    """
    if persist_dir is None:
        persist_dir = (
            DEFAULT_SCICO_ABSTRACTS_PERSIST_DIR
            if text_strategy == ABSTRACT_ONLY
            else DEFAULT_SCICO_FULLTEXT_PERSIST_DIR
        )
    documents = build_scico_documents(scico_root, text_strategy=text_strategy)
    if not documents:
        raise RuntimeError(f"No SciCo documents found under {scico_root!r}.")
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=_acl_embed_model(),
        show_progress=show_progress,
    )
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    logger.info(
        "SciCo %s index persisted to %s",
        "fulltext" if text_strategy == PREFER_FULLTEXT else "abstracts",
        persist_dir,
    )
    return index
