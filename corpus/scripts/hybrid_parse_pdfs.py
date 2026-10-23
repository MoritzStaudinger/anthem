#!/usr/bin/env python3
"""
Hybrid PDF parser: PyMuPDF text extraction first, OCR fallback if needed.

- Input: directory of PDFs (recursive) OR list_file with paths
- Output:
  - md    : one Markdown per PDF
  - jsonl : one JSON object per page
- Skips existing outputs (unless --overwrite)
- Logs failures + summary counters
- Detects output collisions (two PDFs mapping to same output path)

Text path:
  - Extract text via PyMuPDF page.get_text("text") (fast, best for born-digital PDFs)
  - If page text is too small (chars < threshold), run OCR:
      - render page image at --ocr_dpi
      - pytesseract.image_to_string(...)

Requirements:
  pip install pymupdf pytesseract pillow tqdm
System:
  tesseract-ocr installed and on PATH
"""

from __future__ import annotations

import os
import sys
from collections import deque
import multiprocessing as mp
from concurrent.futures.process import BrokenProcessPool


import io
import json
import argparse
import datetime
import hashlib
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from tqdm import tqdm

import pymupdf as fitz  # PyMuPDF (module renamed from `fitz` to `pymupdf` in v1.24; alias removed in 1.26+)
import pytesseract
from PIL import Image


# ------------------------------
# Logging helpers
# ------------------------------
def _ts() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def _append_line(path: str, line: str):
    try:
        ensure_dir(os.path.dirname(path))
        with open(path, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
    except Exception:
        pass


def _safe_read_last_pdf(output_dir: str) -> str | None:
    p = os.path.join(output_dir, "hybrid_last_pdf.txt")
    try:
        with open(p, "r", encoding="utf-8") as f:
            return f.read().strip() or None
    except Exception:
        return None


def ocr_log(msg: str):
    ts = _ts()
    pid = os.getpid()
    print(f"[{ts}][pid={pid}][OCR] {msg}", flush=True)


def _info(msg: str):
    ts = _ts()
    pid = os.getpid()
    print(f"[{ts}][pid={pid}][INFO] {msg}", flush=True)


def _warn(msg: str):
    ts = _ts()
    pid = os.getpid()
    print(f"[{ts}][pid={pid}][WARN] {msg}", flush=True)


# ------------------------------
# MuPDF noise / error suppression
# ------------------------------
@contextmanager
def suppress_mupdf_stderr():
    """
    Best-effort suppression of Python-level stderr.
    Note: MuPDF may still emit to OS-level stderr unless we also disable display errors.
    """
    yield
    #old_stderr = sys.stderr
    #try:
    #    sys.stderr = io.StringIO()
    #    yield
    #finally:
    #    sys.stderr = old_stderr

def _write_breadcrumb(output_dir: str, pdf_path: str):
    """
    Best-effort breadcrumb to identify the last PDF started before a crash/segfault.
    """
    try:
        ensure_dir(output_dir)
        p = os.path.join(output_dir, "hybrid_last_pdf.txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write(f"{_ts()}\tpid={os.getpid()}\t{pdf_path}\n")
    except Exception:
        pass

def _mupdf_quiet(enable: bool = True):
    """
    Disable MuPDF diagnostic spew to stderr/stdout.
    Works for many PyMuPDF builds.
    """
    try:
        fitz.TOOLS.mupdf_display_errors(not enable)  # False => suppress
    except Exception:
        # Older/newer builds may not expose it; ignore.
        pass


def _mupdf_warnings_text() -> str:
    """
    Retrieve accumulated MuPDF warnings (if available).
    """
    try:
        return (fitz.TOOLS.mupdf_warnings() or "").strip()
    except Exception:
        return ""


# ------------------------------
# File / CLI helpers
# ------------------------------
def list_pdfs(input_dir: str):
    pdfs = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, fn))
    pdfs.sort()
    return pdfs


def parse_pages_arg(pages: str):
    """
    Accept:
      - None
      - "0,1,2"
      - "0-4"
      - "0-4,7,9-10"
    Returns sorted list[int] (0-based) or None.
    """
    if not pages:
        return None

    out = set()
    parts = [p.strip() for p in pages.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            a, b = int(a.strip()), int(b.strip())
            if b < a:
                a, b = b, a
            for x in range(a, b + 1):
                out.add(x)
        else:
            out.add(int(p))
    return sorted(out)


def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def _write_text(path: str, text: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        f.write(text)


def _write_jsonl(path: str, rows: list[dict]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", errors="replace") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stable_path_hash(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:n]


def _derive_out_paths(pdf_path: str, output_dir: str, mode: str):
    """
    Default: keep basename-based output (as you had), but also make collisions diagnosable.
    If you ever discover collisions in practice, switch to the hash-based filename below.
    """
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    if mode == "md":
        return os.path.join(output_dir, f"{base}.md")
    if mode == "jsonl":
        return os.path.join(output_dir, f"{base}.jsonl")
    raise ValueError(f"Unknown mode: {mode}")

    # If you want collision-proof outputs, use:
    # h = _stable_path_hash(pdf_path)
    # ext = "md" if mode == "md" else "jsonl"
    # return os.path.join(output_dir, f"{base}__{h}.{ext}")


# ------------------------------
# OCR helpers
# ------------------------------
def _render_page_to_pil(page, dpi: int) -> Image.Image:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _extract_page_text_hybrid(
    page,
    *,
    min_chars_for_text: int,
    ocr_dpi: int,
    ocr_lang: str,
    ocr_psm: int,
    ocr_oem: int,
    ocr_timeout_sec: int,
    pdf_path: str,
    page_index: int,
) -> tuple[str, str]:
    """
    Returns (text, method) where method in {"pymupdf", "ocr", "ocr_timeout"}.
    """
    txt = page.get_text("text") or ""
    txt_compact = "".join(txt.split())
    if len(txt_compact) >= min_chars_for_text:
        return txt.strip(), "pymupdf"

    # ---- OCR fallback ----
    ocr_log(f"triggered pdf={os.path.basename(pdf_path)} page={page_index}")

    try:
        ocr_log(f"render start pdf={os.path.basename(pdf_path)} page={page_index}")
        img = _render_page_to_pil(page, dpi=ocr_dpi)
        ocr_log(f"render done pdf={os.path.basename(pdf_path)} page={page_index}")

        ocr_log(f"tesseract start pdf={os.path.basename(pdf_path)} page={page_index}")
        text = pytesseract.image_to_string(
            img,
            lang=ocr_lang,
            config=f"--oem {ocr_oem} --psm {ocr_psm}",
            timeout=ocr_timeout_sec,
        )
        ocr_log(f"tesseract done pdf={os.path.basename(pdf_path)} page={page_index}")

        return (text or "").strip(), "ocr"

    except RuntimeError as e:
        # Tesseract timeout or crash
        ocr_log(
            f"tesseract FAILED pdf={os.path.basename(pdf_path)} "
            f"page={page_index} err={e}"
        )
        return "", "ocr_timeout"


# ------------------------------
# Main worker: parse one PDF
# ------------------------------
def _parse_one_pdf(
    pdf_path: str,
    output_dir: str,
    mode: str,
    overwrite: bool,
    pages: list[int] | None,
    min_chars_for_text: int,
    ocr_dpi: int,
    ocr_lang: str,
    ocr_psm: int,
    ocr_oem: int,
    ocr_timeout_sec: int,
    quiet_mupdf: bool,
    collect_mupdf_warnings: bool,
):
    # In process pools each worker has its own state, so apply quieting here too.
    if quiet_mupdf:
        _mupdf_quiet(True)

    out_path = _derive_out_paths(pdf_path, output_dir, mode)

    if (not overwrite) and os.path.exists(out_path):
        return ("skipped", pdf_path, out_path, None, None)

    mupdf_warn = None

    # Breadcrumb for post-mortem debugging of segfaults
    _write_breadcrumb(output_dir, pdf_path)

    try:
        # Clear warnings buffer if available
        if collect_mupdf_warnings:
            _ = _mupdf_warnings_text()

        # NOTE: suppress_mupdf_stderr is now a no-op by default (recommended)
        with suppress_mupdf_stderr():
            doc = fitz.open(pdf_path)

        try:
            if doc.is_encrypted:
                with suppress_mupdf_stderr():
                    if not doc.authenticate(""):
                        raise RuntimeError("Encrypted PDF (cannot extract text)")

            page_indices = pages if pages is not None else list(range(doc.page_count))

            if mode == "md":
                parts = []
                for i in page_indices:
                    with suppress_mupdf_stderr():
                        page = doc.load_page(i)

                    text, method = _extract_page_text_hybrid(
                        page,
                        min_chars_for_text=min_chars_for_text,
                        ocr_dpi=ocr_dpi,
                        ocr_lang=ocr_lang,
                        ocr_psm=ocr_psm,
                        ocr_oem=ocr_oem,
                        ocr_timeout_sec=ocr_timeout_sec,
                        pdf_path=pdf_path,
                        page_index=i,
                    )

                    parts.append(f"\n\n---\n\n## Page {i} ({method})\n\n")
                    parts.append(text)

                _write_text(out_path, "".join(parts))

            else:
                # mode == "jsonl"
                rows = []
                for i in page_indices:
                    with suppress_mupdf_stderr():
                        page = doc.load_page(i)

                    text, method = _extract_page_text_hybrid(
                        page,
                        min_chars_for_text=min_chars_for_text,
                        ocr_dpi=ocr_dpi,
                        ocr_lang=ocr_lang,
                        ocr_psm=ocr_psm,
                        ocr_oem=ocr_oem,
                        ocr_timeout_sec=ocr_timeout_sec,
                        pdf_path=pdf_path,
                        page_index=i,
                    )

                    rows.append(
                        {
                            "metadata": {
                                "source_pdf": os.path.basename(pdf_path),
                                "pdf_path": pdf_path,
                                "page_index": i,
                                "page_count": doc.page_count,
                                "method": method,
                                "min_chars_for_text": min_chars_for_text,
                                "ocr": {
                                    "dpi": ocr_dpi,
                                    "lang": ocr_lang,
                                    "psm": ocr_psm,
                                    "oem": ocr_oem,
                                }
                                if method in {"ocr", "ocr_timeout"}
                                else None,
                            },
                            "text": text,
                        }
                    )

                _write_jsonl(out_path, rows)

            if collect_mupdf_warnings:
                mupdf_warn = _mupdf_warnings_text() or None

            return ("parsed", pdf_path, out_path, None, mupdf_warn)

        finally:
            with suppress_mupdf_stderr():
                doc.close()

    except Exception as e:
        if collect_mupdf_warnings:
            mupdf_warn = _mupdf_warnings_text() or None
        return ("failed", pdf_path, out_path, repr(e), mupdf_warn)



# ------------------------------
# CLI entry point
# ------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Parse PDFs in parallel using PyMuPDF text extraction with OCR fallback."
    )

    ap.add_argument("--input_dir", required=False, help="Directory containing PDFs.")
    ap.add_argument("--list_file", default=None, help="File with PDF paths (one per line).")
    ap.add_argument("--output_dir", required=True, help="Directory to write outputs.")
    ap.add_argument("--log_file", default=None, help="Failure log file.")

    ap.add_argument("--executor", choices=["process", "thread"], default="process")
    ap.add_argument("--max_workers", type=int, default=1)

    ap.add_argument("--mode", choices=["md", "jsonl"], default="md")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--pages", default=None)

    ap.add_argument("--min_chars_for_text", type=int, default=200)
    ap.add_argument("--ocr_dpi", type=int, default=300)
    ap.add_argument("--ocr_lang", default="eng")
    ap.add_argument("--ocr_psm", type=int, default=6)
    ap.add_argument("--ocr_oem", type=int, default=3)
    ap.add_argument("--ocr_timeout_sec", type=int, default=180)

    ap.add_argument("--quiet_mupdf", action="store_true")
    ap.add_argument("--collect_mupdf_warnings", action="store_true")

    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    log_file = args.log_file or os.path.join(args.output_dir, "hybrid_parse_errors.log")
    warnings_log = os.path.join(args.output_dir, "hybrid_parse_mupdf_warnings.log")
    parsed_list = os.path.join(args.output_dir, "hybrid_parsed_paths.txt")
    skipped_list = os.path.join(args.output_dir, "hybrid_skipped_paths.txt")
    failed_list = os.path.join(args.output_dir, "hybrid_failed_paths.txt")
    collision_log = os.path.join(args.output_dir, "hybrid_output_collisions.log")
    crash_log = os.path.join(args.output_dir, "hybrid_pool_crashes.log")

    if args.quiet_mupdf:
        _mupdf_quiet(True)

    # -------- load PDFs --------
    if args.list_file:
        with open(args.list_file, "r", encoding="utf-8") as f:
            pdfs = [ln.strip() for ln in f if ln.strip()]
        _info(f"Found {len(pdfs)} PDFs from list_file.")
    else:
        if not args.input_dir:
            raise ValueError("Either --input_dir or --list_file must be provided.")
        pdfs = list_pdfs(args.input_dir)
        _info(f"Found {len(pdfs)} PDFs under input_dir.")

    pages = parse_pages_arg(args.pages)

    # -------- collision preflight --------
    seen = {}
    collisions = []
    for p in pdfs:
        outp = _derive_out_paths(p, args.output_dir, args.mode)
        if outp in seen and seen[outp] != p:
            collisions.append((outp, seen[outp], p))
        else:
            seen[outp] = p

    if collisions:
        _warn(f"Detected {len(collisions)} output collisions.")
        with open(collision_log, "w", encoding="utf-8") as f:
            for outp, a, b in collisions:
                f.write(f"{outp}\n  A={a}\n  B={b}\n\n")

    # -------- counters --------
    parsed = skipped = failed = warned = 0

    # -------- executor config --------
    executor_cls = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor

    mp_ctx = None
    if args.executor == "process":
        try:
            mp_ctx = mp.get_context("spawn")
        except Exception:
            mp_ctx = None

    executor_kwargs = {"max_workers": args.max_workers}
    if mp_ctx:
        executor_kwargs["mp_context"] = mp_ctx
    if sys.version_info >= (3, 11):
        executor_kwargs["max_tasks_per_child"] = 25

    def submit_one(ex, pdf_path):
        return ex.submit(
            _parse_one_pdf,
            pdf_path,
            args.output_dir,
            args.mode,
            args.overwrite,
            pages,
            args.min_chars_for_text,
            args.ocr_dpi,
            args.ocr_lang,
            args.ocr_psm,
            args.ocr_oem,
            args.ocr_timeout_sec,
            args.quiet_mupdf,
            args.collect_mupdf_warnings,
        )

    # -------- logging --------
    with open(log_file, "a", encoding="utf-8") as flog, \
         open(warnings_log, "a", encoding="utf-8") as wlog, \
         open(parsed_list, "a", encoding="utf-8") as plog, \
         open(skipped_list, "a", encoding="utf-8") as slog, \
         open(failed_list, "a", encoding="utf-8") as blog:

        remaining = deque(pdfs)
        in_flight_target = max(2, args.max_workers * 2)

        pbar = tqdm(total=len(pdfs), desc="Parsing PDFs")

        def handle(status, pdf_path, out_path, reason, mupdf_warn):
            nonlocal parsed, skipped, failed, warned
            if status == "parsed":
                parsed += 1
                plog.write(f"{pdf_path}\t{out_path}\n")
            elif status == "skipped":
                skipped += 1
                slog.write(f"{pdf_path}\t{out_path}\n")
            else:
                failed += 1
                flog.write(f"Failed {pdf_path} -> {out_path}: {reason}\n")
                blog.write(f"{pdf_path}\t{out_path}\t{reason}\n")
            if mupdf_warn:
                warned += 1
                wlog.write(f"[pdf]={pdf_path}\n{mupdf_warn}\n\n")

        # -------- main loop --------
        while remaining:
            pending = deque()
            pending_paths = deque()

            try:
                with executor_cls(**executor_kwargs) as ex:
                    while remaining and len(pending) < in_flight_target:
                        p = remaining.popleft()
                        pending.append(submit_one(ex, p))
                        pending_paths.append(p)

                    while pending:
                        fut = pending.popleft()
                        p_assigned = pending_paths.popleft()

                        try:
                            res = fut.result()
                            handle(*res)
                            pbar.update(1)

                        except BrokenProcessPool as e:
                            last_pdf = _safe_read_last_pdf(args.output_dir)
                            _append_line(
                                crash_log,
                                f"{_ts()} BrokenProcessPool | last_pdf={last_pdf} | requeue={p_assigned}",
                            )

                            remaining.appendleft(p_assigned)
                            while pending_paths:
                                remaining.appendleft(pending_paths.pop())
                            pending.clear()
                            pending_paths.clear()
                            break

                        except Exception as e:
                            outp = _derive_out_paths(p_assigned, args.output_dir, args.mode)
                            handle("failed", p_assigned, outp, repr(e), None)
                            pbar.update(1)

                        while remaining and len(pending) < in_flight_target:
                            p = remaining.popleft()
                            pending.append(submit_one(ex, p))
                            pending_paths.append(p)

            except BrokenProcessPool as e:
                last_pdf = _safe_read_last_pdf(args.output_dir)
                _append_line(
                    crash_log,
                    f"{_ts()} BrokenProcessPool during setup | last_pdf={last_pdf}",
                )
                continue

        pbar.close()

    _info("Parsing completed.")
    _info(f"Summary: parsed={parsed} skipped={skipped} failed={failed} warnings={warned}")
    _info(f"Outputs in: {os.path.abspath(args.output_dir)}")
    _info(f"Failure log: {os.path.abspath(log_file)}")



if __name__ == "__main__":
    main()
