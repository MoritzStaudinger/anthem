"""Convert raw S2 / S2ORC records into the SciCo per-paper JSON schema.

Reads:
  <input-dir>/papers/<acl_id>.json    raw `papers` dataset record
  <input-dir>/s2orc/<acl_id>.json     raw `s2orc` dataset record (optional)

Writes:
  <output-dir>/affiliations/<acl_id>.json   schema-matching record
  <output-dir>/pymu_llm/<acl_id>.md         body text extracted from S2ORC

The output schema mirrors what `backend/rag/scico_indexing.py::load_scico_papers`
expects so the eval harness can swap corpora with just `--affiliations-dir`.

S2ORC annotations are character-offset spans into `content.text`. We resolve
each annotation type ("author", "authoraffiliation", "abstract", etc.) into
strings by slicing the text. The "authoraffiliation" annotation gives us
S2's view of each paper's affiliations — that's the direct comparison
target for the reviewer rebuttal.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("s2orc_normalize")


# -------------------------- S2ORC annotation slicing --------------------------

def _resolve_spans(text: str, ann_field: Any) -> List[str]:
    """S2ORC annotations come as either a JSON-encoded string or a list of
    {start,end,...} dicts. Return the spans' text content as a list."""
    if ann_field is None:
        return []
    if isinstance(ann_field, str):
        try:
            ann_field = json.loads(ann_field)
        except json.JSONDecodeError:
            return []
    if not isinstance(ann_field, list):
        return []
    out = []
    for a in ann_field:
        try:
            s = int(a["start"])
            e = int(a["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= s < e <= len(text):
            out.append(text[s:e].strip())
    return [x for x in out if x]


def _extract_affiliations_from_s2orc(rec: dict) -> List[str]:
    """Extract S2ORC's parsed affiliation strings.

    These come from S2ORC's structure detection in front matter. May be
    very noisy (sometimes captures emails or city names alongside the
    institution). De-dup and trim.
    """
    content = rec.get("content") or {}
    text = content.get("text") or ""
    annotations = content.get("annotations") or {}
    spans = _resolve_spans(text, annotations.get("authoraffiliation"))
    # Deduplicate while preserving order
    seen, out = set(), []
    for s in spans:
        # Strip trailing punctuation, collapse whitespace, drop tiny ones
        s = re.sub(r"\s+", " ", s).strip(" ,;.")
        if len(s) < 3:
            continue
        if s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return out


def _extract_body_text(rec: dict) -> str:
    """Pull the main body text from S2ORC, excluding bibliography if labeled."""
    content = rec.get("content") or {}
    text = content.get("text") or ""
    annotations = content.get("annotations") or {}
    # If bibliography annotations exist, trim everything after the earliest one.
    bib_spans = _resolve_spans(text, annotations.get("bibentry"))
    if bib_spans:
        # Easier: find earliest bibentry in raw text, slice before it.
        # bib_spans values are strings; we re-locate to get the offset.
        first_offset = None
        ann = annotations.get("bibentry")
        if isinstance(ann, str):
            try: ann = json.loads(ann)
            except json.JSONDecodeError: ann = []
        if isinstance(ann, list):
            offsets = [int(a["start"]) for a in ann if isinstance(a, dict) and "start" in a]
            if offsets:
                first_offset = min(offsets)
        if first_offset is not None:
            text = text[:first_offset]
    return text.strip()


# -------------------------- schema mapping --------------------------

def _map_to_scico_schema(
    paper_rec: dict,
    s2orc_rec: Optional[dict],
    acl_id: str,
) -> dict:
    """Map S2's `papers` (+optional `s2orc`) into our per-paper JSON schema.

    Schema fields we populate (subset — extras kept under `_s2_raw_*`):
      paper_id, title, abstract, year, authors, affiliations,
      countries, ror_ids, institution_types, full_text,
      _s2_corpus_id, _s2_fields_of_study
    """
    title = paper_rec.get("title") or ""
    abstract = paper_rec.get("abstract") or ""
    year = paper_rec.get("year")
    if isinstance(year, str) and year.isdigit():
        year = int(year)
    elif not isinstance(year, int):
        year = None

    authors = []
    for a in (paper_rec.get("authors") or []):
        if not a:
            continue
        authors.append({
            "name": a.get("name") or "",
            "author_id": a.get("authorId"),
        })

    # Affiliations: prefer S2ORC's structured annotations, fall back to nothing.
    # (S2's `papers` dataset doesn't carry per-author affiliations.)
    affiliations: List[str] = []
    if s2orc_rec:
        affiliations = _extract_affiliations_from_s2orc(s2orc_rec)

    # Body text from S2ORC (excluding bibliography). Empty when we don't
    # have a matching s2orc record for this paper.
    full_text = ""
    if s2orc_rec:
        full_text = _extract_body_text(s2orc_rec)

    return {
        "paper_id": acl_id,
        "title": title,
        "abstract": abstract,
        "year": year,
        "authors": authors,
        "affiliations": affiliations,
        # The following are placeholders so the schema is uniform with our
        # custom corpus. They'll be backfilled by downstream enrichment
        # (ROR linking, CSO classification) the same way.
        "countries": [],
        "ror_ids": [],
        "institution_types": [],
        "full_text": full_text,
        # Provenance & extra S2 metadata kept for traceability
        "_provenance": {
            "source": "s2orc" if s2orc_rec else "s2_papers_only",
            "s2_corpus_id": paper_rec.get("corpusId"),
        },
        "_s2_fields_of_study": paper_rec.get("fieldsOfStudy") or [],
        "_s2_s2fieldsOfStudy": paper_rec.get("s2FieldsOfStudy") or [],
    }


# -------------------------- main --------------------------

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-dir", required=True,
                   help="Directory holding `papers/`, `s2orc/` from s2orc_download.py")
    p.add_argument("--output-dir", required=True,
                   help="Where to write affiliations/<paper_id>.json + pymu_llm/<paper_id>.md")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N papers (for testing).")
    p.add_argument("--write-fulltext-md", action="store_true",
                   help="Also emit a pymu_llm/<paper_id>.md file containing the "
                        "S2ORC body text. Lets the existing indexer pick it up "
                        "the same way it picks up your PyMu_LLM markdown.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    in_dir = Path(args.input_dir)
    papers_dir = in_dir / "papers"
    s2orc_dir = in_dir / "s2orc"

    if not papers_dir.is_dir():
        raise SystemExit(f"papers/ dir not found under {in_dir}. "
                         "Run s2orc_download.py papers first.")

    out_dir = Path(args.output_dir)
    aff_out = out_dir / "affiliations"
    md_out = out_dir / "pymu_llm"
    aff_out.mkdir(parents=True, exist_ok=True)
    if args.write_fulltext_md:
        md_out.mkdir(parents=True, exist_ok=True)

    paper_files = sorted(papers_dir.glob("*.json"))
    if args.limit:
        paper_files = paper_files[: args.limit]

    coverage = Counter()
    for pf in paper_files:
        acl_id = pf.stem
        try:
            paper_rec = json.loads(pf.read_text())
        except json.JSONDecodeError as e:
            logger.warning("skipping malformed %s: %s", pf, e)
            coverage["malformed_papers_record"] += 1
            continue

        s2orc_rec = None
        s2_path = s2orc_dir / f"{acl_id}.json"
        if s2_path.is_file():
            try:
                s2orc_rec = json.loads(s2_path.read_text())
                coverage["with_s2orc_fulltext"] += 1
            except json.JSONDecodeError:
                coverage["s2orc_malformed"] += 1
        else:
            coverage["metadata_only"] += 1

        mapped = _map_to_scico_schema(paper_rec, s2orc_rec, acl_id)
        (aff_out / f"{acl_id}.json").write_text(
            json.dumps(mapped, ensure_ascii=False, indent=2)
        )

        if args.write_fulltext_md and mapped["full_text"]:
            (md_out / f"{acl_id}.md").write_text(mapped["full_text"])

    logger.info("normalized %d papers → %s", len(paper_files), aff_out)
    for k, v in coverage.most_common():
        logger.info("  %s: %d", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
