"""Pack a tiny demo of the dataset structure: N papers' worth of
parsed PDFs (3 variants) + final metadata, mirroring the actual
cluster directory layout.

Output (default N=20):

    sample/
    ├── README.md
    ├── ror_parsed_affiliations_90_with_topics_and_bibmeta/   (final metadata)
    │   ├── <paper_id>.json   (× N)
    │   └── ...
    ├── acl_pdfs_parsed/                                       (GROBID TEI)
    │   ├── <paper_id>.xml    (× N)
    │   └── ...
    ├── acl_pdfs_pymu_llm/                                     (PyMu+LLM .md)
    │   └── ...
    └── acl_pdfs_hybrid/                                       (Hybrid OCR)
        └── ...

Usage on the cluster:

    python scripts/package_sample_release.py \\
        --n 20 \\
        --final-dir   /share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta \\
        --grobid-dir  $HOME/data/acl_pdfs_parsed \\
        --pymu-dir    /share/moritz.staudinger/acl_data/acl_pdfs_pymu_llm \\
        --hybrid-dir  $HOME/data/acl_pdfs_hybrid \\
        --out         /tmp/acl_dataset_sample
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import tarfile
from pathlib import Path

logger = logging.getLogger("pkg")


def _safe_year(d):
    y = d.get("year")
    try:
        return int(y) if y is not None else None
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20,
                    help="Number of papers in the sample (default 20).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--year-min", type=int, default=None,
                    help="Restrict candidates to year >= this (optional).")
    ap.add_argument("--year-max", type=int, default=None)
    ap.add_argument("--final-dir",  type=Path, required=True,
                    help="Bibmeta-enriched JSON corpus dir on the cluster.")
    ap.add_argument("--grobid-dir", type=Path, required=True,
                    help="GROBID TEI dir.")
    ap.add_argument("--pymu-dir",   type=Path, required=True,
                    help="PyMu+LLM markdown dir.")
    ap.add_argument("--hybrid-dir", type=Path, required=True,
                    help="Hybrid OCR JSONL dir.")
    ap.add_argument("--require-all-parsers", action="store_true",
                    help="Only pick papers that have ALL three parser "
                         "variants present (recommended for a clean demo).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory (must not exist).")
    ap.add_argument("--no-tar", action="store_true",
                    help="Skip the .tar.gz, just leave the tree on disk.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s")

    if args.out.exists():
        raise SystemExit(f"--out already exists: {args.out}")
    for d in (args.final_dir, args.grobid_dir, args.pymu_dir, args.hybrid_dir):
        if not d.is_dir():
            raise SystemExit(f"missing source dir: {d}")

    # Enumerate candidates from the final dir
    rng = random.Random(args.seed)
    candidates = sorted([fp.stem for fp in args.final_dir.glob("*.json")])
    rng.shuffle(candidates)

    # Per-parser candidate extensions, tried in order. The first one
    # that resolves to an existing file is used. Add more if your
    # cluster uses a different convention.
    PARSER_EXTS = {
        "grobid": [".tei.xml", ".xml", ".tei"],
        "pymu":   [".md"],
        "hybrid": [".jsonl", ".json"],
    }

    def _find(src: Path, pid: str, exts) -> Path | None:
        for ext in exts:
            p = src / f"{pid}{ext}"
            if p.is_file():
                return p
        return None

    picked = []
    for pid in candidates:
        if len(picked) >= args.n:
            break
        fjson = args.final_dir / f"{pid}.json"
        try:
            d = json.loads(fjson.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            continue
        y = _safe_year(d)
        if args.year_min is not None and (y is None or y < args.year_min):
            continue
        if args.year_max is not None and (y is None or y > args.year_max):
            continue
        if args.require_all_parsers:
            if not _find(args.grobid_dir, pid, PARSER_EXTS["grobid"]):
                continue
            if not _find(args.pymu_dir,   pid, PARSER_EXTS["pymu"]):
                continue
            if not _find(args.hybrid_dir, pid, PARSER_EXTS["hybrid"]):
                continue
        picked.append(pid)

    if len(picked) < args.n:
        logger.warning("only found %d papers matching criteria (asked for %d)",
                       len(picked), args.n)

    # Lay out mirrored subdirs
    out = args.out
    sub_final  = out / args.final_dir.name
    sub_grobid = out / args.grobid_dir.name
    sub_pymu   = out / args.pymu_dir.name
    sub_hybrid = out / args.hybrid_dir.name
    for d in (sub_final, sub_grobid, sub_pymu, sub_hybrid):
        d.mkdir(parents=True)

    counts = {"final": 0, "grobid": 0, "pymu": 0, "hybrid": 0}
    for pid in picked:
        # final: fixed .json
        f = args.final_dir / f"{pid}.json"
        if f.is_file():
            shutil.copy2(f, sub_final / f.name)
            counts["final"] += 1
        # parsers: extension-tolerant
        for src, exts, dst, key in [
            (args.grobid_dir, PARSER_EXTS["grobid"], sub_grobid, "grobid"),
            (args.pymu_dir,   PARSER_EXTS["pymu"],   sub_pymu,   "pymu"),
            (args.hybrid_dir, PARSER_EXTS["hybrid"], sub_hybrid, "hybrid"),
        ]:
            f = _find(src, pid, exts)
            if f is not None:
                shutil.copy2(f, dst / f.name)
                counts[key] += 1

    # README
    readme = f"""# ACL Anthology — dataset structure sample

A {len(picked)}-paper sample showing the on-disk layout of the dataset
described in the accompanying submission. Each subdirectory mirrors the
corresponding location on the cluster.

## Layout

```
{args.final_dir.name}/        final bibmeta-enriched JSON   ({counts['final']} files)
{args.grobid_dir.name}/       GROBID 0.8.x TEI XML          ({counts['grobid']} files)
{args.pymu_dir.name}/         PyMuPDF + LLM-cleaned Markdown ({counts['pymu']} files)
{args.hybrid_dir.name}/       PyMuPDF + Tesseract OCR JSONL  ({counts['hybrid']} files)
```

File-naming convention: `<anthology_paper_id>.<ext>` across every subdir,
so a given paper id lines up across all four variants.

## Final JSON schema (one example)

```
{json.dumps(_load_first(picked, sub_final), indent=2, ensure_ascii=False)[:1500]}
...
```

The full corpus contains 117,386 papers spanning 1952-2025; this sample
just shows the structure.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    logger.info("%d papers; %.1f MB on disk", len(picked), total/1024/1024)

    if not args.no_tar:
        archive = out.parent / f"{out.name}.tar.gz"
        with tarfile.open(archive, "w:gz") as t:
            t.add(out, arcname=out.name)
        logger.info("tar.gz: %s  (%.1f MB)", archive,
                    archive.stat().st_size / 1024 / 1024)
    return 0


def _load_first(picked, sub_final):
    for pid in picked:
        f = sub_final / f"{pid}.json"
        if f.is_file():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
