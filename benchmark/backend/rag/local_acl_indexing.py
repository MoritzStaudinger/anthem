from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from llama_index.core import (
    Document,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)

# Pin embedding to a small, fast, local model. The corpus is fixed (ACL
# anthology) and we don't want to pay OpenAI per re-index. The same model
# has to be used at build *and* query time — see _acl_embed_model().
ACL_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# The persist dir defaults to the repo-relative `data/local_index_acl/`,
# but can be overridden via the ACL_INDEX_DIR env var. Useful on slurm
# nodes where the canonical index lives on slow shared NFS — stage to
# /tmp at job start and point ACL_INDEX_DIR there to cut load wall from
# ~9 min to ~30 sec.
import os as _os
DEFAULT_ACL_PERSIST_DIR = _os.environ.get("ACL_INDEX_DIR") or "data/local_index_acl"

_ACL_EMBED_MODEL = None
_ACL_INDEX_CACHE: dict = {}  # persist_dir -> VectorStoreIndex


def _acl_embed_model() -> HuggingFaceEmbedding:
    """Return a process-wide MiniLM embedder. First call downloads the
    model into the HF cache (~80MB); subsequent calls reuse it.

    embed_batch_size=128: MiniLM's default of 10 wastes the GPU. On an
    A40 a batch of 128 takes <50ms and gets us ~3000 docs/sec instead of
    ~100. Costs ~200MB of extra VRAM, negligible vs the 7GB headroom.
    """
    global _ACL_EMBED_MODEL
    if _ACL_EMBED_MODEL is None:
        _ACL_EMBED_MODEL = HuggingFaceEmbedding(
            model_name=ACL_EMBED_MODEL_NAME,
            embed_batch_size=128,
        )
    return _ACL_EMBED_MODEL


def _safe_read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_authors(meta: dict) -> List[str]:
    authors = meta.get("authors", [])
    names: List[str] = []
    for a in authors:
        n = a.get("name")
        if n:
            names.append(n)
    return names


def _extract_affiliations_and_countries(meta: dict) -> Tuple[List[str], List[str], List[str]]:
    """
    Returns:
      affiliations: list of clean institution names (unique, stable order)
      country_codes: list of country codes (unique, stable order)
      ror_ids: list of ror ids (unique, stable order)
    """
    affs: List[str] = []
    ccs: List[str] = []
    rors: List[str] = []

    for a in meta.get("authors", []):
        for aff in a.get("affiliations", []) or []:
            inst = aff.get("clean_institution_name") or aff.get("raw_affiliation")
            if inst and inst not in affs:
                affs.append(inst)

            cc = aff.get("country_code")
            if cc and cc not in ccs:
                ccs.append(cc)

            r = aff.get("ror_id")
            if r and r not in rors:
                rors.append(r)

    return affs, ccs, rors


def build_documents_from_folders(
    text_dir: str | Path,
    meta_dir: str | Path,
    *,
    text_glob: str = "*.md",
) -> List[Document]:
    text_dir = Path(text_dir)
    meta_dir = Path(meta_dir)

    # Index metadata by paper_id AND by basename for robustness
    meta_by_key: Dict[str, dict] = {}
    for jp in meta_dir.glob("*.json"):
        meta = _load_json(jp)
        paper_id = meta.get("paper_id") or jp.stem
        meta_by_key[paper_id] = meta
        meta_by_key[jp.stem] = meta  # e.g. "2021.acl-long.541"

    documents: List[Document] = []
    missing_meta = 0

    for mp in text_dir.glob(text_glob):
        key = mp.stem  # "2021.acl-long.541"
        meta = meta_by_key.get(key)
        if not meta:
            missing_meta += 1
            # Still index text with minimal metadata
            text = _safe_read_text(mp)
            documents.append(Document(
                text=text,
                metadata={
                    "paper_id": key,
                    "title": key,
                    "publication_year": None,
                    "authors": [],
                    "doi": "",
                    "affiliations": [],
                    "affiliations_string": "",
                    "countries": [],
                    "countries_string": "",
                }
            ))
            continue

        text = _safe_read_text(mp)

        year = meta.get("year")
        if year is None:
            # fallback: parse first 4 chars of key if it looks like a year
            try:
                year = int(key.split(".", 1)[0])
            except Exception:
                year = None

        authors = _extract_authors(meta)
        affiliations, country_codes, ror_ids = _extract_affiliations_and_countries(meta)

        title = meta.get("title") or meta.get("paper_title") or meta.get("paper_id") or key
        doi = meta.get("doi") or meta.get("DOI") or ""

        documents.append(Document(
            text=text,
            metadata={
                "paper_id": meta.get("paper_id", key),
                "title": title,
                "publication_year": year,
                "authors": authors,
                "doi": doi,
                "affiliations": affiliations,
                "affiliations_string": ", ".join(affiliations),
                "countries": country_codes,
                "countries_string": ", ".join(country_codes),
                "ror_ids": ror_ids,
            }
        ))

    logger.info("loaded %d documents (missing metadata for %d texts)", len(documents), missing_meta)
    return documents


def build_acl_index(
    text_dir: str | Path,
    meta_dir: str | Path,
    persist_dir: str | Path = DEFAULT_ACL_PERSIST_DIR,
    *,
    text_glob: str = "*.md",
    show_progress: bool = True,
) -> VectorStoreIndex:
    """Build and persist a VectorStoreIndex over local ACL papers using
    MiniLM. Self-contained — does not touch llama_index.core.Settings, so
    it doesn't interfere with whatever LLM the request pipeline has set."""
    documents = build_documents_from_folders(text_dir, meta_dir, text_glob=text_glob)
    if not documents:
        raise RuntimeError(
            f"No ACL documents found under text_dir={text_dir!r} "
            f"(glob {text_glob!r}). Nothing to index."
        )
    logger.info(
        "building ACL index over %d documents with %s",
        len(documents), ACL_EMBED_MODEL_NAME,
    )
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=_acl_embed_model(),
        show_progress=show_progress,
    )
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    logger.info("ACL index persisted to %s", persist_dir)
    return index


def load_acl_index(persist_dir: str | Path = DEFAULT_ACL_PERSIST_DIR) -> VectorStoreIndex:
    """Load the persisted ACL index, pinning the same MiniLM embedder used
    to build it. Cached per persist_dir."""
    key = str(persist_dir)
    cached = _ACL_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    p = Path(persist_dir)
    if not p.exists() or not (p / "index_store.json").exists():
        raise FileNotFoundError(
            f"No ACL index at {p}. Run scripts/build_acl_index.py first "
            f"(text-dir + meta-dir → {p})."
        )
    storage = StorageContext.from_defaults(persist_dir=str(p))
    index = load_index_from_storage(storage, embed_model=_acl_embed_model())
    _ACL_INDEX_CACHE[key] = index
    return index


def reset_acl_index_cache() -> None:
    """Drop the cached index so the next call re-loads from disk. Used
    after rebuilding the index in the same process."""
    _ACL_INDEX_CACHE.clear()
