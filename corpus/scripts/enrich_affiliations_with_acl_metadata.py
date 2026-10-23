"""Build a NEW directory of per-paper JSONs that joins the existing
affiliation data with the canonical ACL Anthology bibliographic
metadata (year, venue, title, authors, doi, abstract, …).

The original affiliation enrichment pipeline carried only what the
ROR linker needed (authors + affiliations + topics + abstract). Year
and venue weren't preserved, so the local ACL vector index has
`publication_year=None` for every paper and year filters in retrieve()
always return zero. We fix that by joining against the ACL Anthology
repo XML.

Output goes to a NEW directory (--out-dir). The source directory is
never modified. Once enriched, point downstream tools at --out-dir
instead of the original.

Per-paper output schema:
    paper_id, authors, affiliation_index, conflicts, _provenance,
    abstract, topics            ← copied from the source JSON verbatim

    year:   int                 ← from ACL Anthology
    venue:  str
    title:  str
    doi:    str | null
    url:    str | null
    bibmeta: { full ACL Anthology paper record }
                                ← authors (canonical), editors,
                                  bibkey, pages, month, ee, abstract,
                                  publisher, address, collection_id,
                                  volume_id, extras (attachments/etc.)

Two metadata sources are supported:

  --acl-repo <path>        : parse XML files from a local clone of
                              https://github.com/acl-org/acl-anthology.
                              Pass the path to the repo's `data/` dir.
  --use-acl-pkg            : use the `acl-anthology` PyPI package
                              (auto-downloads + caches the repo).

Idempotent: rerunning skips files already present in --out-dir.
Use --force to overwrite.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("enrich_aff")


# ---------------------------------------------------------------------------
# Metadata sources
# ---------------------------------------------------------------------------

def _load_metadata_via_pkg() -> Dict[str, Dict[str, Any]]:
    """Use the acl-anthology PyPI package. Minimal field set, matches
    the XML parser's output schema."""
    try:
        from acl_anthology import Anthology
    except ImportError as e:
        raise SystemExit(
            "acl-anthology package not installed. Run:\n"
            "    pip install acl-anthology\n"
            "or pass --acl-repo <path-to-cloned-anthology/data>\n"
            f"(original error: {e})"
        )
    logger.info("loading acl-anthology repo (may take a minute on first call)")
    a = Anthology.from_repo()
    out: Dict[str, Dict[str, Any]] = {}
    for paper in a.papers():
        pid = paper.full_id
        try:
            year = int(paper.year)
        except (ValueError, TypeError, AttributeError):
            year = None
        try:
            venue = paper.parent.parent.id
        except AttributeError:
            venue = ""
        try:
            title = str(paper.title)
        except Exception:
            title = ""
        try:
            doi = paper.doi
        except AttributeError:
            doi = None
        try:
            url = paper.web_url
        except AttributeError:
            url = None
        try:
            language = paper.language
        except AttributeError:
            language = None
        try:
            authors = [{"first": a.first, "last": a.last,
                        "name": str(a.name), "id": getattr(a, "id", None)}
                       for a in paper.authors]
        except Exception:
            authors = []
        out[pid] = {
            "year": year,
            "venue": venue or "",
            "title": title or "",
            "doi": doi or None,
            "url": url or None,
            "language": language or None,
            "bib_authors": authors,
        }
    logger.info("loaded %d papers from acl-anthology package", len(out))
    return out


def _xml_text(elem) -> str:
    """Flatten an XML element's text + tail recursively into a plain
    string. Handles ACL Anthology's mixed-content fields (title /
    abstract with inline <i>, <b>, <fixed-case>, etc.)."""
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()


def _parse_author(elem) -> Dict[str, Any]:
    """Extract one author/editor from the XML. ACL Anthology uses
    <first>, <last>, and optionally <affiliation> children."""
    first = _xml_text(elem.find("first"))
    last = _xml_text(elem.find("last"))
    name_parts = [p for p in (first, last) if p]
    return {
        "first": first or None,
        "last": last or None,
        "name": " ".join(name_parts) if name_parts else _xml_text(elem),
        "id": elem.get("id") or None,
    }


def _load_metadata_via_xml(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Parse XML files in a cloned acl-anthology/data directory.

    Extracts only the fields we actually need downstream:
        year, venue, title, doi, url, language, bib_authors
    Everything else (bibkey, pages, month, editors, attachments…)
    is dropped to keep the per-paper JSON small.
    """
    import xml.etree.ElementTree as ET
    xml_dir = data_dir / "xml"
    if not xml_dir.is_dir():
        raise SystemExit(f"expected {xml_dir}; pass the parent of `xml/`")

    out: Dict[str, Dict[str, Any]] = {}
    for fp in sorted(xml_dir.glob("*.xml")):
        try:
            tree = ET.parse(fp)
        except ET.ParseError as e:
            logger.warning("parse error in %s: %s", fp.name, e)
            continue
        root = tree.getroot()
        collection_id = root.get("id") or fp.stem
        for vol in root.findall("volume"):
            vol_id = vol.get("id") or ""
            meta = vol.find("meta")
            vol_year = None
            vol_venue = ""
            vol_language = None
            if meta is not None:
                ye = meta.find("year")
                if ye is not None and ye.text:
                    try:
                        vol_year = int(ye.text)
                    except ValueError:
                        pass
                ven = meta.find("venue")
                if ven is not None:
                    vol_venue = (ven.text or "").strip()
                lang = meta.find("language")
                if lang is not None and lang.text:
                    vol_language = lang.text.strip() or None

            # Build a list of (paper_element, paper_id_in_xml) so we can
            # also handle <frontmatter> as paper_id "0".
            paper_elements = [(p, p.get("id") or "") for p in vol.findall("paper")]
            front_el = vol.find("frontmatter")
            if front_el is not None:
                paper_elements.append((front_el, "0"))

            for paper, pid_part in paper_elements:
                # ACL Anthology canonical paper id construction.
                #
                # Modern (year-prefixed): "<collection_id>-<vol_id>.<paper>"
                #   e.g. 2024.acl + long + 123 -> "2024.acl-long.123"
                #   Frontmatter: paper "0" -> "2024.acl-long.0"
                #
                # Legacy (letter-prefixed): "<collection_id>-<vol_id><paper>"
                #   The suffix after the dash is always 4 digits total.
                #   Volumes 1-9 take 1 char + paper takes 3 (zero-padded).
                #   Volumes 10+ take 2 chars + paper takes 2 (zero-padded).
                #   e.g. P85 + 1 + 10  -> "P85-1010"
                #        W14 + 38 + 1 -> "W14-3801"
                if collection_id and "." in collection_id:
                    pid = f"{collection_id}-{vol_id}.{pid_part}"
                else:
                    try:
                        vol_int = int(vol_id)
                        paper_int = int(pid_part)
                    except ValueError:
                        # Falls back to simple concat for non-numeric vol/paper
                        pid = f"{collection_id}-{vol_id}{pid_part.zfill(3)}"
                    else:
                        if vol_int < 10:
                            suffix = f"{vol_int}{paper_int:03d}"
                        else:
                            suffix = f"{vol_int:02d}{paper_int:02d}"
                        pid = f"{collection_id}-{suffix}"

                def _t(tag):
                    return _xml_text(paper.find(tag))

                title = _t("title")
                authors = [_parse_author(a) for a in paper.findall("author")]
                doi = _t("doi") or None
                url = _t("url") or None
                # Language: paper-level overrides volume-level; both
                # rarely populated outside LREC / multilingual venues.
                lang_el = paper.find("language")
                paper_lang = (lang_el.text.strip() if lang_el is not None and lang_el.text else None)
                language = paper_lang or vol_language

                out[pid] = {
                    "year": vol_year,
                    "venue": vol_venue or collection_id.split(".", 1)[-1],
                    "title": title or "",
                    "doi": doi,
                    "url": url,
                    "language": language,
                    "bib_authors": authors,
                }
    logger.info("loaded %d papers from %s", len(out), xml_dir)
    return out


# ---------------------------------------------------------------------------
# Enrich each affiliation JSON in place
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--affs-dir", type=Path,
                    default=Path("/share/moritz.staudinger/acl_data/"
                                 "ror_parsed_affiliations_90_with_topics"),
                    help="SOURCE directory of per-paper affiliation JSONs. "
                         "Never modified.")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="OUTPUT directory. Each <paper_id>.json gets "
                         "written here with the original fields plus the "
                         "ACL Anthology fields merged in. The source dir "
                         "is never touched.")
    ap.add_argument("--acl-repo", type=Path, default=None,
                    help="Path to a cloned acl-anthology/data dir. If unset, "
                         "use the acl-anthology PyPI package as the source.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite output files that already exist.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change but don't write any files.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    if not args.affs_dir.is_dir():
        raise SystemExit(f"missing source dir: {args.affs_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: build paper_id → bib metadata map
    if args.acl_repo:
        meta_map = _load_metadata_via_xml(args.acl_repo)
    else:
        meta_map = _load_metadata_via_pkg()

    # Step 2: iterate source JSONs, merge bib fields, write to OUT dir.
    files = sorted(args.affs_dir.glob("*.json"))
    logger.info("scanning %d source JSONs in %s", len(files), args.affs_dir)
    logger.info("writing enriched JSONs to %s", args.out_dir)

    # Build a dash-tolerant secondary lookup so non-canonical paper_ids
    # in the affiliation source can still match the ACL Anthology
    # canonical IDs. Two common drift patterns observed in the source:
    #   1. Dashes replaced by dots:
    #        source "1993.iwpt.1.13"  vs  canonical "1993.iwpt-1.13"
    #   2. Single-volume venues with the volume dropped entirely:
    #        source "1993.iwpt.14"    vs  canonical "1993.iwpt-1.14"
    # We index each canonical id under (a) a dots-only normalised form
    # and (b) a "without volume" form for collections that have only
    # volume "1". On lookup we try exact → dots-only → without-vol.
    norm_map: Dict[str, Dict[str, Any]] = {}
    no_vol_map: Dict[str, Dict[str, Any]] = {}
    for cpid, bib in meta_map.items():
        norm_map[cpid.replace("-", ".")] = bib
        # "1993.iwpt-1.14" -> base "1993.iwpt", suffix "14"  →  "1993.iwpt.14"
        if "-" in cpid:
            base, _, suffix = cpid.rpartition("-")
            if "." in suffix:
                _vol, _, paper_part = suffix.partition(".")
                no_vol_map[f"{base}.{paper_part}"] = bib

    def _resolve(pid: str):
        b = meta_map.get(pid)
        if b is not None:
            return b
        b = norm_map.get(pid.replace("-", "."))
        if b is not None:
            return b
        b = no_vol_map.get(pid)
        if b is not None:
            return b
        return None

    n_total = 0
    n_skipped_existing = 0
    n_no_match = 0
    n_written = 0
    sample_missing: list = []

    for fp in files:
        n_total += 1
        out_fp = args.out_dir / fp.name
        if out_fp.exists() and not args.force:
            n_skipped_existing += 1
            continue
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("skip %s: %s", fp.name, e)
            continue
        pid = d.get("paper_id") or fp.stem
        bib = _resolve(pid)
        if not bib:
            n_no_match += 1
            if len(sample_missing) < 30:
                sample_missing.append(pid)
            # Still copy the source to the output dir so downstream tools
            # see a complete set of files; they'll just have year=None
            # for these papers (rare unless paper_id format mismatches).
        else:
            d["year"] = bib["year"]
            d["venue"] = bib["venue"]
            d["title"] = bib["title"]
            d["doi"] = bib.get("doi")
            d["url"] = bib.get("url")
            d["language"] = bib.get("language")
            d["bib_authors"] = bib.get("bib_authors") or []
        if not args.dry_run:
            tmp = out_fp.with_suffix(out_fp.suffix + ".tmp")
            tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                           encoding="utf-8")
            tmp.replace(out_fp)
            n_written += 1
        if n_total % 5000 == 0:
            logger.info("  scanned %d/%d  written=%d  no_match=%d",
                        n_total, len(files), n_written, n_no_match)

    logger.info("=" * 60)
    logger.info("enrichment summary:")
    logger.info("  total source JSONs scanned: %d", n_total)
    logger.info("  skipped (out file exists) : %d", n_skipped_existing)
    logger.info("  written to %s : %d", args.out_dir, n_written)
    logger.info("  no match in ACL repo      : %d (passed through unchanged)", n_no_match)
    if sample_missing:
        logger.info("  sample paper_ids with no match (first 10):")
        for pid in sample_missing:
            logger.info("    - %s", pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
