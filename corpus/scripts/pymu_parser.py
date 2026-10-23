#!/usr/bin/env python3
"""
PDF Library Parser & Cleaner

- Recursively scans an input directory for PDFs
- Extracts title, abstract, and fulltext using PyMuPDF (fitz)
- Cleans the fulltext for downstream summarization / NLP
- Writes results to JSONL (one record per PDF) and optional per-file TXT

Usage:
  python pdf_library_parser_and_cleaner.py \
      --input /path/to/pdfs \
      --output /path/to/out \
      --workers 8 \
      --write-txt

Requirements:
  - Python 3.9+
  - PyMuPDF (pip install pymupdf)
  - (Optional) tqdm for progress bar (pip install tqdm)

Notes:
  - Encrypted or image-only PDFs will be skipped (unless text can be extracted by fitz).
  - Heuristics are conservative; feel free to tweak `clean_fulltext_for_summarization` and
    `extract_title_abstract_fulltext` for your corpus.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import re
import sys
from contextlib import contextmanager
import io
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional, Tuple

# Make stdout/stderr UTF-8 to avoid surprises on some systems
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import fitz  # PyMuPDF
except Exception as e:
    print("ERROR: PyMuPDF (fitz) is required. Install with: pip install pymupdf", file=sys.stderr)
    raise

try:
    from tqdm import tqdm  # optional
except Exception:  # pragma: no cover
    tqdm = None


# ------------------------------
# MuPDF noise / error suppression
# ------------------------------
@contextmanager
def suppress_mupdf_stderr():
    """Temporarily silence MuPDF's stderr warnings/errors for broken PDFs.
    This does not hide Python exceptions — we still catch/handle those.
    """
    old_stderr = sys.stderr
    try:
        sys.stderr = io.StringIO()
        yield
    finally:
        sys.stderr = old_stderr


# ------------------------------
# Unicode sanitation (fix for surrogate-related encoding errors)
# ------------------------------
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

def sanitize_unicode(s: str) -> str:
    """Replace unpaired surrogates and normalize odd spaces."""
    if not s:
        return ""
    s = _SURROGATE_RE.sub("\uFFFD", s)
    return _normalize_unicode_spaces(s)


# ------------------------------
# Cleaning heuristic (adapted)
# ------------------------------
HEADER_FOOTER_PATTERNS = [
    r"Proceedings of .*?\n",
    r"©\s*\d{4}.*?\n",
    r"All rights reserved.*?\n",
    r"arXiv:\S+",
    r"CC-BY.*?\n",
]


def _normalize_unicode_spaces(text: str) -> str:
    # Replace non-breaking spaces and other odd whitespace with regular spaces/newlines
    text = text.replace("\u00A0", " ")  # NBSP
    text = text.replace("\u2009", " ")  # thin space
    text = text.replace("\u2002", " ")  # en space
    text = text.replace("\u2003", " ")  # em space
    text = text.replace("\u2028", "\n")  # line sep
    text = text.replace("\u2029", "\n\n")  # paragraph sep
    return text


def clean_fulltext_for_summarization(text: str) -> str:
    """Lightweight cleanup tailored for typical scientific PDFs.

    You can tweak or simplify as needed. This version:
      - normalizes spaces
      - strips common headers/footers and figure/table captions (heuristic)
      - removes isolated page numbers
      - fixes hyphenation at line breaks
      - joins broken mid-sentence lines
      - collapses extra whitespace
      - (optional) truncates after References if desired (disabled by default)
    """
    if not text:
        return ""

    text = _normalize_unicode_spaces(text)

    # Remove isolated page numbers on their own line
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)

    # Remove figure/table captions heuristically (single greedy block up to next block starting with capital)
    text = re.sub(r"(?is)(?:Figure|Table)\s+\d+[:.].*?(?=\n[A-Z])", "", text)

    # Remove common headers/footers
    for pat in HEADER_FOOTER_PATTERNS:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # Remove running headers like "Author — Short Title" if repeated across many pages (simple heuristic)
    counts = {}
    for line in text.splitlines():
        line_s = line.strip()
        if 5 <= len(line_s) <= 100:
            counts[line_s] = counts.get(line_s, 0) + 1
    repeated = {k for k, v in counts.items() if v >= 5}
    if repeated:
        pat = r"^(?:" + r"|".join(re.escape(x) for x in sorted(repeated, key=len, reverse=True)) + r")\s*$"
        text = re.sub(pat, "", text, flags=re.MULTILINE)

    # Fix hyphenated line breaks: e.g., "hy-
    # phenation" -> "hyphenation"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    # Join mid-sentence lines (line breaks that are not paragraph boundaries)
    text = re.sub(r"(?<![.?!:])\n(?!\n)", " ", text)

    # Collapse multiple newlines to single paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove lingering multiple spaces
    text = re.sub(r"[ ]{2,}", " ", text)

    return text.strip()


# --------------------------------------------
# Title/Abstract/Fulltext extraction heuristic
# --------------------------------------------
def extract_title_abstract_fulltext(pdf_path: str) -> Tuple[str, str, str, int]:
    """Return (title, abstract, fulltext, num_pages) using PyMuPDF.

    Improvements over the baseline:
      - recognizes an explicit ABSTRACT block if present on page 1
      - guards against empty/locked documents
      - keeps the original fulltext before cleaning
    """
    doc = fitz.open(pdf_path)
    try:
        if doc.is_encrypted:
            # Try to open without password; if still locked, skip
            if not doc.authenticate(""):
                raise RuntimeError("Encrypted PDF (cannot extract text)")

        num_pages = len(doc)
        if num_pages == 0:
            return "", "", "", 0

        # Fulltext (raw)
        fulltext = "\n".join(page.get_text() or "" for page in doc)

        title = ""
        abstract = ""

        # Only analyze first page for title/abstract
        page0 = doc[0]
        blocks = page0.get_text("dict").get("blocks", [])
        text_blocks: List[Tuple[str, float]] = []

        for b in blocks:
            if "lines" in b:
                block_text = []
                max_fontsize = 0.0
                for line in b["lines"]:
                    for span in line.get("spans", []):
                        t = (span.get("text") or "").strip("\n")
                        if t:
                            block_text.append(t)
                            try:
                                max_fontsize = max(max_fontsize, float(span.get("size", 0) or 0))
                            except Exception:
                                pass
                joined = " ".join(block_text).strip()
                if joined:
                    text_blocks.append((joined, max_fontsize))

        # Sort by font size descending to guess title
        sorted_blocks = sorted(text_blocks, key=lambda x: -x[1])
        if sorted_blocks:
            candidate_title = sorted_blocks[0][0].strip()
            # Avoid absurdly long titles (fallback to first non-empty line)
            if 5 <= len(candidate_title) <= 300:
                title = candidate_title
            else:
                title = text_blocks[0][0]

        # Find explicit ABSTRACT block if present
        abstract_idx = None
        for i, (block, _) in enumerate(text_blocks):
            low = block.lower()
            if re.match(r"^abstract\b[:\s-]*", low):
                abstract_idx = i
                break

        if abstract_idx is not None:
            # Include this block minus the heading, and possibly the following block if it's substantive
            abstract_text = re.sub(r"(?i)^abstract\b[:\s-]*", "", text_blocks[abstract_idx][0]).strip()
            # If next block is big and not Introduction, append
            if abstract_idx + 1 < len(text_blocks):
                nxt = text_blocks[abstract_idx + 1][0]
                if len(nxt.split()) > 20 and not re.match(r"(?i)^(1\s+)?introduction\b", nxt):
                    abstract_text += " " + nxt
            abstract = abstract_text.strip()
        else:
            # Heuristic: between title and introduction
            intro_index = None
            for i, (block, _) in enumerate(text_blocks):
                low = block.lower()
                if re.search(r"(?i)^(?:1\s+)?introduction\b", low) or re.match(r"^1\s+", low):
                    intro_index = i
                    break
            if intro_index is None:
                intro_index = len(text_blocks)

            abstract_candidates = [
                text_blocks[i][0] for i in range(1, intro_index)
                if len(text_blocks[i][0].split()) > 30
            ]
            abstract = " ".join(abstract_candidates).strip()

        return title.strip(), abstract.strip(), fulltext.strip(), num_pages
    finally:
        doc.close()


# --------------------------------------------
# Resilient wrapper around extraction (suppresses MuPDF stderr; skips bad pages)
# --------------------------------------------
def extract_title_abstract_fulltext_resilient(pdf_path: str) -> Tuple[str, str, str, int]:
    with suppress_mupdf_stderr():
        doc = fitz.open(pdf_path)
    try:
        if doc.is_encrypted:
            with suppress_mupdf_stderr():
                if not doc.authenticate(""):
                    raise RuntimeError("Encrypted PDF (cannot extract text)")

        num_pages = len(doc)
        if num_pages == 0:
            return "", "", "", 0

        raw_pages: List[str] = []
        for i in range(num_pages):
            try:
                with suppress_mupdf_stderr():
                    page = doc[i]
                    txt = page.get_text("text") or ""
                raw_pages.append(txt)
            except Exception:
                continue
        fulltext = "\n".join(raw_pages)

        title = ""
        abstract = ""

        with suppress_mupdf_stderr():
            page0 = doc[0]
            blocks = page0.get_text("dict").get("blocks", [])
        text_blocks: List[Tuple[str, float]] = []

        for b in blocks:
            if "lines" in b:
                block_text = []
                max_fontsize = 0.0
                for line in b["lines"]:
                    for span in line.get("spans", []):
                        t = (span.get("text") or "").strip("\n")
                        if t:
                            block_text.append(t)
                            try:
                                max_fontsize = max(max_fontsize, float(span.get("size", 0) or 0))
                            except Exception:
                                pass
                joined = " ".join(block_text).strip()
                if joined:
                    text_blocks.append((joined, max_fontsize))

        sorted_blocks = sorted(text_blocks, key=lambda x: -x[1])
        if sorted_blocks:
            candidate_title = sorted_blocks[0][0].strip()
            if 5 <= len(candidate_title) <= 300:
                title = candidate_title
            else:
                title = text_blocks[0][0]

        abstract_idx = None
        for i, (block, _) in enumerate(text_blocks):
            low = block.lower()
            if re.match(r"^abstract\b[:\s-]*", low):
                abstract_idx = i
                break

        if abstract_idx is not None:
            abstract_text = re.sub(r"(?i)^abstract\b[:\s-]*", "", text_blocks[abstract_idx][0]).strip()
            if abstract_idx + 1 < len(text_blocks):
                nxt = text_blocks[abstract_idx + 1][0]
                if len(nxt.split()) > 20 and not re.match(r"(?i)^(1\s+)?introduction\b", nxt):
                    abstract_text += " " + nxt
            abstract = abstract_text.strip()
        else:
            intro_index = None
            for i, (block, _) in enumerate(text_blocks):
                low = block.lower()
                if re.search(r"(?i)^(?:1\s+)?introduction\b", low) or re.match(r"^1\s+", low):
                    intro_index = i
                    break
            if intro_index is None:
                intro_index = len(text_blocks)

            abstract_candidates = [
                text_blocks[i][0] for i in range(1, intro_index)
                if len(text_blocks[i][0].split()) > 30
            ]
            abstract = " ".join(abstract_candidates).strip()

        return title.strip(), abstract.strip(), fulltext.strip(), num_pages
    finally:
        with suppress_mupdf_stderr():
            doc.close()


# ------------------------------
# Utilities
# ------------------------------
@dataclass
class Record:
    source_path: str
    sha256: str
    pages: int
    title: str
    abstract: str
    fulltext_raw_len: int
    fulltext_clean_len: int
    fulltext_clean: str


def sha256_file(path: str, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iter_pdfs(root: str) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(".pdf"):
                yield os.path.join(dirpath, name)


def process_pdf(path: str, write_txt_dir: Optional[str] = None) -> Optional[Record]:
    try:
        digest = sha256_file(path)
        title, abstract, fulltext_raw, pages = extract_title_abstract_fulltext_resilient(path)

        # --- Sanitize BEFORE any writing/JSON ---
        title = sanitize_unicode(title)
        abstract = sanitize_unicode(abstract)
        fulltext_raw = sanitize_unicode(fulltext_raw)

        if not fulltext_raw:
            # No text -> likely scanned or empty
            return None

        cleaned = clean_fulltext_for_summarization(fulltext_raw)
        cleaned = sanitize_unicode(cleaned)  # extra safety

        rec = Record(
            source_path=os.path.abspath(path),
            sha256=digest,
            pages=pages,
            title=title,
            abstract=abstract,
            fulltext_raw_len=len(fulltext_raw),
            fulltext_clean_len=len(cleaned),
            fulltext_clean=cleaned,
        )

        if write_txt_dir:
            base = os.path.splitext(os.path.basename(path))[0]
            out_txt = os.path.join(write_txt_dir, f"{base}.txt")
            os.makedirs(os.path.dirname(out_txt), exist_ok=True)
            # errors='replace' provides a final guardrail
            with open(out_txt, "w", encoding="utf-8", errors="replace") as f:
                f.write(rec.fulltext_clean)

        return rec
    except Exception:
        # Return None to signal failure; caller logs it
        return None


# ------------------------------
# Main pipeline
# ------------------------------
def main():
    ap = argparse.ArgumentParser(description="Parse and clean a PDF library.")
    ap.add_argument("--input", required=True, help="Root directory containing PDFs (scanned recursively)")
    ap.add_argument("--output", required=True, help="Output directory (JSONL + optional TXT)")
    ap.add_argument("--jsonl-name", default="library_clean.jsonl", help="Name of the JSONL file to create")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Parallel workers")
    ap.add_argument("--write-txt", action="store_true", help="Also write per-document cleaned TXT files")
    ap.add_argument("--skip-small", type=int, default=0, help="Skip PDFs smaller than N KB (0 = no skip)")

    args = ap.parse_args()

    in_root = os.path.abspath(args.input)
    out_root = os.path.abspath(args.output)
    os.makedirs(out_root, exist_ok=True)

    jsonl_path = os.path.join(out_root, args.jsonl_name)
    txt_dir = os.path.join(out_root, "texts") if args.write_txt else None
    if txt_dir:
        os.makedirs(txt_dir, exist_ok=True)

    pdfs = list(iter_pdfs(in_root))
    print(f"Found {len(pdfs)} PDFs to process.")
    if args.skip_small > 0:
        kb = args.skip_small * 1024
        pdfs = [p for p in pdfs if os.path.getsize(p) >= kb]

    if not pdfs:
        print("No PDFs found.")
        return 0

    # Process in parallel
    iterator = pdfs
    if tqdm is not None:
        iterator = tqdm(pdfs, desc="Processing PDFs")

    n_ok = 0
    n_fail = 0
    # Open JSONL with UTF-8; sanitized strings ensure validity for json + write
    with open(jsonl_path, "w", encoding="utf-8") as jf:
        if args.workers and args.workers > 1:
            with futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
                fut_to_path = {ex.submit(process_pdf, p, txt_dir): p for p in pdfs}
                for fut in futures.as_completed(fut_to_path):
                    rec = fut.result()
                    if rec is None:
                        n_fail += 1
                    else:
                        jf.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                        n_ok += 1
        else:
            for p in iterator:
                rec = process_pdf(p, txt_dir)
                if rec is None:
                    n_fail += 1
                else:
                    jf.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                    n_ok += 1

    print(f"Done. OK: {n_ok} | Failed/Skipped: {n_fail} | Output: {jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
