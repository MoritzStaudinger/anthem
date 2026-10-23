#!/usr/bin/env python3
"""
Parse PDFs -> Markdown (or page-chunk JSONL) using PyMuPDF4LLM, in parallel.

- Input: a directory of PDFs (e.g., downloaded ACL anthology PDFs named <paper_id>.pdf)
- Output:
  - Default: one <paper_id>.md per PDF
  - If --page_chunks: one <paper_id>.jsonl per PDF (one JSON object per page)
- Skips outputs that already exist (unless --overwrite)
- Logs failures to a log file

PyMuPDF4LLM docs:
  - pymupdf4llm.to_markdown(doc, ..., page_chunks=bool, pages=[...], write_images=bool, ...)
"""
import os
import argparse
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import pymupdf4llm  # pip install pymupdf4llm


def list_pdfs(input_dir: str):
    pdfs = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, fn))
    pdfs.sort()
    return pdfs


def parse_pages_arg(pages: str ):
    """
    Accept:
      - None
      - "0,1,2"
      - "0-4"
      - "0-4,7,9-10"
    Returns a sorted list[int] (0-based), or None.
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
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_jsonl(path: str, rows: list[dict]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _derive_out_paths(pdf_path: str, output_dir: str, mode: str):
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    if mode == "md":
        return os.path.join(output_dir, f"{base}.md")
    elif mode == "jsonl":
        return os.path.join(output_dir, f"{base}.jsonl")
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _parse_one_pdf(
    pdf_path: str,
    output_dir: str,
    mode: str,
    overwrite: bool,
    pages: list[int],
    # PyMuPDF4LLM options (a practical subset; add more as needed)
    write_images: bool,
    embed_images: bool,
    image_path: str,
    image_format: str,
    dpi: int,
    use_ocr: bool,
    ocr_dpi: int,
    table_strategy: str,
    force_text: bool,
    header: bool,
    footer: bool,
):
    out_path = _derive_out_paths(pdf_path, output_dir, mode)

    if (not overwrite) and os.path.exists(out_path):
        return ("skipped", pdf_path, out_path, None)

    try:
        if mode == "md":
            md = pymupdf4llm.to_markdown(
                pdf_path,
                pages=pages,
                page_chunks=False,
                write_images=write_images,
                embed_images=embed_images,
                image_path=image_path,
                image_format=image_format,
                dpi=dpi,
                use_ocr=use_ocr,
                ocr_dpi=ocr_dpi,
                table_strategy=table_strategy,
                force_text=force_text,
                header=header,
                footer=footer,
                show_progress=False,
            )
            _write_text(out_path, md)
            return ("parsed", pdf_path, out_path, None)

        # mode == "jsonl"
        chunks = pymupdf4llm.to_markdown(
            pdf_path,
            pages=pages,
            page_chunks=True,  # returns list[dict] with metadata + text (per docs)
            write_images=write_images,
            embed_images=embed_images,
            image_path=image_path,
            image_format=image_format,
            dpi=dpi,
            use_ocr=use_ocr,
            ocr_dpi=ocr_dpi,
            table_strategy=table_strategy,
            force_text=force_text,
            header=header,
            footer=footer,
            show_progress=False,
        )
        # Write one JSON object per page to JSONL.
        # The dicts already contain "metadata" + extracted markdown per page (per API docs).
        _write_jsonl(out_path, chunks)
        return ("parsed", pdf_path, out_path, None)

    except Exception as e:
        return ("failed", pdf_path, out_path, repr(e))


def main():
    ap = argparse.ArgumentParser(
        description="Parse PDFs in parallel using PyMuPDF4LLM (to Markdown or JSONL page chunks)."
    )
    ap.add_argument("--input_dir", required=False, help="Directory containing PDFs (recursively scanned).")
    ap.add_argument("--output_dir", required=True, help="Directory to write parsed outputs.")
    ap.add_argument("--log_file", default="logs/pymupdf4llm_parse_errors.log", help="Failure log file.")
    ap.add_argument("--max_workers", type=int, default=8, help="Parallel workers.")
    ap.add_argument("--list_file", default=None,
                    help="Optional file with PDF paths (one per line). If set, input_dir is ignored.")
    ap.add_argument(
        "--executor",
        choices=["process", "thread"],
        default="process",
        help="Use process pool (default) or thread pool.",
    )
    ap.add_argument(
        "--mode",
        choices=["md", "jsonl"],
        default="md",
        help="Output mode: md = one Markdown file per PDF; jsonl = per-page dicts (one JSON line per page).",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs (default: skip).")
    ap.add_argument(
        "--pages",
        default=None,
        help='Optional pages (0-based) like "0-2,5,7-9". If omitted, all pages.',
    )

    # Practical PyMuPDF4LLM knobs (subset of to_markdown params)
    ap.add_argument("--write_images", action="store_true", help="Write images to disk and reference them in output.")
    ap.add_argument("--embed_images", action="store_true", help="Embed images as base64 in markdown/JSONL (can be huge).")
    ap.add_argument("--image_path", default="", help="Folder to store images if --write_images is set.")
    ap.add_argument("--image_format", default="png", help="Image format extension, e.g. png/jpg.")
    ap.add_argument("--dpi", type=int, default=150, help="Image DPI for extracted images.")
    ap.add_argument("--use_ocr", action="store_true", help="Enable OCR assistance (only effective with PyMuPDF Layout).")
    ap.add_argument("--ocr_dpi", type=int, default=400, help="OCR DPI (only relevant if --use_ocr).")
    ap.add_argument(
        "--table_strategy",
        default="lines_strict",
        help="Table extraction strategy (per PyMuPDF4LLM API).",
    )
    ap.add_argument("--no_force_text", action="store_true", help="Disable force_text (default: enabled).")
    ap.add_argument("--no_header", action="store_true", help="Exclude header text (default: include).")
    ap.add_argument("--no_footer", action="store_true", help="Exclude footer text (default: include).")

    args = ap.parse_args()

    if args.list_file:
        with open(args.list_file, "r", encoding="utf-8") as f:
            pdfs = [ln.strip() for ln in f if ln.strip()]
    else:
        pdfs = list_pdfs(args.input_dir)
    print(f"Found {len(pdfs)} PDFs under: {os.path.abspath(args.input_dir)}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)

    pages = parse_pages_arg(args.pages)

    def submit_all(executor_cls):
        failures = 0
        with open(args.log_file, "a", encoding="utf-8") as flog:
            with executor_cls(max_workers=args.max_workers) as ex:
                futures = [
                    ex.submit(
                        _parse_one_pdf,
                        pdf_path,
                        args.output_dir,
                        args.mode,
                        args.overwrite,
                        pages,
                        args.write_images,
                        args.embed_images,
                        args.image_path,
                        args.image_format,
                        args.dpi,
                        args.use_ocr,
                        args.ocr_dpi,
                        args.table_strategy,
                        (not args.no_force_text),
                        (not args.no_header),
                        (not args.no_footer),
                    )
                    for pdf_path in pdfs
                ]

                for fut in tqdm(as_completed(futures), total=len(futures), desc="Parsing PDFs"):
                    status, pdf_path, out_path, reason = fut.result()
                    if status == "failed":
                        failures += 1
                        flog.write(f"Failed {pdf_path} -> {out_path}: {reason}\n")

        return failures

    executor_cls = ProcessPoolExecutor if args.executor == "process" else ThreadPoolExecutor
    failures = submit_all(executor_cls)

    print("\nParsing completed.")
    if failures:
        print(f"Failures logged: {failures} -> {os.path.abspath(args.log_file)}")
    print(f"Outputs in: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
