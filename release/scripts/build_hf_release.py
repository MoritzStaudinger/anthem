"""Build the ANTHEM HuggingFace release from the exported corpus + parsed fulltext.

Takes the gzipped JSONL produced by `export_corpus_jsonl.py` plus (optionally)
the three extracted parse trees, and writes sharded Parquet configs ready to
`hf upload`:

    <out>/
    ├── README.md                dataset card (counts filled in)
    ├── papers/*.parquet         1 row/paper, nested fields as JSON strings
    ├── affiliations/*.parquet   FLAT: 1 row per paper x author x affiliation
    ├── topics/*.parquet         FLAT: 1 row per paper x CSO label
    └── fulltext/*.parquet       TEI + Markdown + OCR text per paper

ANTHEM's own layer -- the extracted affiliations, ROR links and CSO labels --
goes out under CC-BY-NC-SA-4.0, as the paper states. The `license` column is
provenance: it records the terms the *underlying paper* carries in the
Anthology (https://aclanthology.org/faq/copyright/):

  * year < 2016 or unknown   -> CC-BY-NC-SA-3.0
  * year >= 2016             -> CC-BY-4.0
  * ELRA / third-party venue -> not covered by the ACL policy, so the parsed
                                full text is withheld; metadata is kept

Source PDFs are deliberately not redistributed -- the Anthology serves those.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    print("# ERROR: pyarrow is required -> pip install pyarrow", file=sys.stderr)
    raise SystemExit(1)

# Venue slugs published by ELRA / sister societies under CC-BY-NC. The ACL
# copyright policy explicitly does not cover third-party materials.
ELRA_DEFAULT = ("lrec", "lrec-main", "lrec-coling", "signlang",
                "sigul", "osact", "bucc", "legal")

# Absolute paths that must not reach a public release.
PATH_RX = re.compile(r"(/(?:home|work|share|scratch|tmp)/[^\s\"'<>]+)")

# GROBID lifts author contact details out of the PDF header, so the TEI (and
# the OCR text) carry e-mail addresses. They are already printed in the public
# papers, so they are kept by default; --scrub-emails removes them.
MAIL_RX = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

SCHEMAS = {
    "papers": pa.schema([
        ("paper_id", pa.string()), ("title", pa.string()), ("abstract", pa.string()),
        ("year", pa.int32()), ("venue", pa.string()), ("doi", pa.string()),
        ("url", pa.string()), ("language", pa.string()),
        ("n_authors", pa.int32()), ("n_affiliations", pa.int32()),
        ("n_ror_linked", pa.int32()),
        ("authors_json", pa.string()), ("bib_authors_json", pa.string()),
        ("affiliation_index_json", pa.string()), ("conflicts_json", pa.string()),
        ("topics_json", pa.string()),
        ("provenance_model", pa.string()), ("license", pa.string()),
        ("third_party_venue", pa.bool_()),
    ]),
    "affiliations": pa.schema([
        ("paper_id", pa.string()), ("year", pa.int32()), ("venue", pa.string()),
        ("author_name", pa.string()), ("raw_affiliation", pa.string()),
        ("marker", pa.string()), ("country", pa.string()),
        ("country_code", pa.string()), ("ror_id", pa.string()),
        ("clean_institution_name", pa.string()), ("ror_score", pa.float64()),
        ("evidence_spans", pa.list_(pa.string())),
    ]),
    "topics": pa.schema([
        ("paper_id", pa.string()), ("year", pa.int32()), ("venue", pa.string()),
        ("label", pa.string()), ("source", pa.string()),
    ]),
    "fulltext": pa.schema([
        ("paper_id", pa.string()), ("year", pa.int32()), ("venue", pa.string()),
        ("license", pa.string()), ("third_party_venue", pa.bool_()),
        ("tei_xml", pa.string()),
        ("pymu_md", pa.string()), ("ocr_text", pa.string()), ("n_pages", pa.int32()),
    ]),
}


class ShardWriter:
    """Accumulate rows and flush fixed-size Parquet shards."""

    def __init__(self, out_dir: Path, name: str, rows_per_shard: int):
        self.dir = out_dir / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.schema = SCHEMAS[name]
        self.rows_per_shard = rows_per_shard
        self.buf: List[Dict[str, Any]] = []
        self.shard = 0
        self.total = 0

    def add(self, row: Dict[str, Any]) -> None:
        self.buf.append(row)
        self.total += 1
        if len(self.buf) >= self.rows_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        cols = {f.name: [r.get(f.name) for r in self.buf] for f in self.schema}
        table = pa.Table.from_pydict(cols, schema=self.schema)
        path = self.dir / f"{self.name}-{self.shard:05d}.parquet"
        pq.write_table(table, path, compression="zstd", compression_level=10)
        self.shard += 1
        self.buf = []

    def close(self) -> int:
        self.flush()
        if self.total == 0:
            # Leave no empty config directory behind.
            try:
                self.dir.rmdir()
            except OSError:
                pass
        return self.total


def read_records(src: Path) -> Iterator[dict]:
    """Stream records from a .jsonl(.gz) file or a directory of .json files."""
    if src.is_dir():
        for p in sorted(src.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            # setdefault is not enough: a few records carry an explicit
            # "paper_id": null, which would otherwise be dropped downstream.
            if not rec.get("paper_id"):
                rec["paper_id"] = p.stem
            yield rec
        return
    opener = gzip.open if src.suffix == ".gz" else open
    with opener(src, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def to_year(v: Any) -> Optional[int]:
    try:
        y = int(v)
    except (TypeError, ValueError):
        return None
    return y if 1900 < y < 2100 else None


# Legacy Anthology ids encode the year: W18-0902 -> 2018, C96-1058 -> 1996.
LEGACY_ID_RX = re.compile(r"^[A-Z](\d{2})-")


def year_from_id(pid: str) -> Optional[int]:
    """Recover the year from a legacy Anthology id, for records missing `year`.

    Modern ids ("2021.acl-long.541") carry the year up front and are handled by
    the metadata itself; only the pre-2020 letter-prefixed ids need this.
    """
    m = LEGACY_ID_RX.match(pid or "")
    if not m:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy >= 60 else 2000 + yy


def classify(year: Optional[int], venue: Optional[str], elra: set) -> tuple:
    """-> (upstream_license, third_party_venue).

    The licence names the terms the *underlying paper* carries in the Anthology,
    which is provenance, not the release licence: ANTHEM's own annotations go
    out under CC-BY-NC-SA-4.0 as the paper states.

    ELRA / third-party proceedings are flagged rather than dropped: the ACL
    copyright FAQ says its policy "does not cover third-party materials", but
    the Anthology serves those pages under the same policy footer, so whether to
    redistribute their parsed text is a call for the release owner. The flag
    makes it a filter, not a rebuild.
    """
    slug = (venue or "").strip().lower()
    if slug in elra:
        return "third-party venue (ELRA-published); see respective organisation", True
    if year is None:
        # No year even after id backfill, so we cannot place the paper against
        # the 2016 policy change. Assume the more restrictive pre-2016 terms.
        return "CC-BY-NC-SA-3.0", False
    if year < 2016:
        return "CC-BY-NC-SA-3.0", False
    return "CC-BY-4.0", False


# A handful of records in the corpus carry a dict where a string is expected
# (e.g. country as {"name": ..., "evidence": [...]}), and a few `evidence`
# lists hold bare strings rather than {"source", "span"} dicts. Coerce instead
# of crashing the whole build on ~1 record in 25,000.
_STR_KEYS = ("name", "value", "span", "text", "raw")


def as_str(v: Any) -> Optional[str]:
    """Best-effort scalar string, or None."""
    if v is None or isinstance(v, str):
        return v or None
    if isinstance(v, dict):
        for k in _STR_KEYS:
            inner = v.get(k)
            if isinstance(inner, str) and inner:
                return inner
        return None
    if isinstance(v, (int, float, bool)):
        return str(v)
    return None


def evidence_spans(evidence: Any) -> list:
    """Spans from an `evidence` list that may hold dicts or bare strings."""
    out = []
    for e in (evidence or []):
        if isinstance(e, dict):
            sp = as_str(e.get("span"))
        elif isinstance(e, str):
            sp = e or None
        else:
            sp = None
        if sp:
            out.append(sp)
    return out


# Configs written by build_hf_annotations.py. The card lists whatever is present
# in the output directory, so a release built in either order stays consistent.
EXTRA_CFG_DESC = {
    "benchmark_questions": "comparative-QA benchmark: question, persona, and the "
                           "ROR/topic/year gold specification",
    "affiliation_gold": "hand-annotated affiliation gold set (781 papers)",
    "linking_adjudication": "hand-verified ROR linking decisions, four strategies "
                            "side by side",
    "cso_validation": "hand-judged CSO topic labels",
    "lora_train": "LoRA fine-tuning data for the affiliation adapter "
                  "(`train` / `validation` splits)",
}

SCRUB_EMAILS = False


def scrub_text(s: str) -> str:
    s = PATH_RX.sub("<path-removed>", s)
    if SCRUB_EMAILS:
        s = MAIL_RX.sub("<email-removed>", s)
    return s


def read_tei(d: Optional[Path], pid: str) -> Optional[str]:
    if not d:
        return None
    for name in (f"{pid}.tei.xml", f"{pid}.xml"):
        p = d / name
        if p.is_file():
            return scrub_text(p.read_text(encoding="utf-8", errors="replace"))
    return None


def read_md(d: Optional[Path], pid: str) -> Optional[str]:
    if not d:
        return None
    p = d / f"{pid}.md"
    if p.is_file():
        return scrub_text(p.read_text(encoding="utf-8", errors="replace"))
    return None


def read_hybrid(d: Optional[Path], pid: str) -> tuple:
    """-> (text, n_pages). Drops the per-page metadata (it carries pdf_path)."""
    if not d:
        return None, None
    p = d / f"{pid}.jsonl"
    if not p.is_file():
        return None, None
    pages: List[tuple] = []
    with p.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = (rec.get("metadata") or {}).get("page_index")
            pages.append((idx if isinstance(idx, int) else len(pages),
                          rec.get("text") or ""))
    if not pages:
        return None, None
    pages.sort(key=lambda t: t[0])
    return scrub_text("\n\n".join(t for _, t in pages)), len(pages)


def flatten_topics(topics: Any) -> Iterator[tuple]:
    """-> (label, source) pairs. 'union' is derivable, so it is not emitted."""
    cso = (topics or {}).get("cso") if isinstance(topics, dict) else None
    if not isinstance(cso, dict):
        return
    for source in ("syntactic", "semantic"):
        for label in cso.get(source) or []:
            if isinstance(label, str) and label:
                yield label, source


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, required=True,
                    help=".jsonl(.gz) from export_corpus_jsonl.py, or a directory of .json")
    ap.add_argument("--tei-dir", type=Path, default=None)
    ap.add_argument("--md-dir", type=Path, default=None)
    ap.add_argument("--hybrid-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--elra-venues", nargs="*", default=list(ELRA_DEFAULT))
    ap.add_argument("--meta-rows-per-shard", type=int, default=20000)
    ap.add_argument("--fulltext-rows-per-shard", type=int, default=1500)
    ap.add_argument("--version", default="202605")
    ap.add_argument("--scrub-emails", action="store_true",
                    help="Replace e-mail addresses in the parsed text with "
                         "<email-removed>. Off by default: the addresses are "
                         "already printed in the public papers.")
    args = ap.parse_args()

    global SCRUB_EMAILS
    SCRUB_EMAILS = args.scrub_emails

    elra = {v.strip().lower() for v in args.elra_venues}
    want_fulltext = any((args.tei_dir, args.md_dir, args.hybrid_dir))
    args.out.mkdir(parents=True, exist_ok=True)

    w = {n: ShardWriter(args.out, n, args.meta_rows_per_shard)
         for n in ("papers", "affiliations", "topics")}
    if want_fulltext:
        w["fulltext"] = ShardWriter(args.out, "fulltext", args.fulltext_rows_per_shard)

    lic_counts: Dict[str, int] = {}
    models: Dict[str, int] = {}
    ft_found = ft_third_party = year_backfilled = 0
    n = 0

    no_id = 0
    for rec in read_records(args.metadata):
        pid = rec.get("paper_id")
        if not pid:
            # Only reachable for .jsonl input, where there is no filename to
            # fall back on. Never drop these silently.
            no_id += 1
            continue
        n += 1
        year = to_year(rec.get("year"))
        if year is None:
            year = year_from_id(pid)
            if year is not None:
                year_backfilled += 1
        venue = as_str(rec.get("venue"))
        lic, third_party = classify(year, venue, elra)
        lic_counts[lic] = lic_counts.get(lic, 0) + 1
        model = ((rec.get("_provenance") or {}) or {}).get("model")
        models[str(model)] = models.get(str(model), 0) + 1

        authors = rec.get("authors") or []
        n_aff = n_ror = 0
        for a in authors:
            if not isinstance(a, dict):
                continue
            for af in (a.get("affiliations") or []):
                if not isinstance(af, dict):
                    continue
                n_aff += 1
                if af.get("ror_id"):
                    n_ror += 1
                w["affiliations"].add({
                    "paper_id": pid, "year": year, "venue": venue,
                    "author_name": as_str(a.get("name")),
                    "raw_affiliation": as_str(af.get("raw_affiliation")),
                    "marker": as_str(af.get("marker")),
                    "country": as_str(af.get("country")),
                    "country_code": as_str(af.get("country_code")),
                    "ror_id": as_str(af.get("ror_id")),
                    "clean_institution_name": as_str(af.get("clean_institution_name")),
                    "ror_score": af.get("ror_score") if isinstance(
                        af.get("ror_score"), (int, float)) else None,
                    "evidence_spans": evidence_spans(af.get("evidence")),
                })

        for label, source in flatten_topics(rec.get("topics")):
            w["topics"].add({"paper_id": pid, "year": year, "venue": as_str(venue),
                             "label": as_str(label), "source": as_str(source)})

        def js(key: str) -> Optional[str]:
            v = rec.get(key)
            return json.dumps(v, ensure_ascii=False) if v is not None else None

        w["papers"].add({
            "paper_id": pid, "title": as_str(rec.get("title")),
            "abstract": as_str(rec.get("abstract")),
            "year": year, "venue": as_str(venue), "doi": as_str(rec.get("doi")),
            "url": as_str(rec.get("url")),
            "language": as_str(rec.get("language")),
            "n_authors": len(authors), "n_affiliations": n_aff, "n_ror_linked": n_ror,
            "authors_json": js("authors"), "bib_authors_json": js("bib_authors"),
            "affiliation_index_json": js("affiliation_index"),
            "conflicts_json": js("conflicts"), "topics_json": js("topics"),
            "provenance_model": model, "license": lic,
            "third_party_venue": third_party,
        })

        if want_fulltext:
            tei = read_tei(args.tei_dir, pid)
            md = read_md(args.md_dir, pid)
            ocr, n_pages = read_hybrid(args.hybrid_dir, pid)
            if tei or md or ocr:
                ft_found += 1
                if third_party:
                    ft_third_party += 1
                w["fulltext"].add({
                    "paper_id": pid, "year": year, "venue": venue, "license": lic,
                    "third_party_venue": third_party,
                    "tei_xml": tei, "pymu_md": md, "ocr_text": ocr,
                    "n_pages": n_pages,
                })

        if n % 10000 == 0:
            print(f"# {n} papers", flush=True)

    totals = {name: wr.close() for name, wr in w.items()}

    extra = {}
    for d in sorted(args.out.iterdir()):
        if not (d.is_dir() and d.name in EXTRA_CFG_DESC):
            continue
        shards = sorted(d.glob("*.parquet"))
        if not shards:
            continue
        rows = sum(pq.read_metadata(f).num_rows for f in shards)
        # build_hf_annotations.py names a split's shards "<split>-00000.parquet"
        # and an unsplit config's "<config>-00000.parquet".
        stems = {f.name.rsplit("-", 1)[0] for f in shards}
        extra[d.name] = (rows, sorted(stems - {d.name}))
    if extra:
        print("# also present      : "
              + ", ".join(f"{k} ({v[0]:,})" for k, v in extra.items()))

    card = args.out / "README.md"
    card.write_text(_card(args.version, n, totals, lic_counts, models,
                            want_fulltext, ft_third_party,
                            args.scrub_emails, extra),
                    encoding="utf-8")

    print(f"\n# papers read      : {n}")
    if no_id:
        print(f"# DROPPED no paper_id: {no_id}")
    for k, v in totals.items():
        print(f"# {k:20s}: {v} rows")
    print(f"# year backfilled  : {year_backfilled}")
    if want_fulltext:
        print(f"# fulltext matched : {ft_found}/{n}")
        print(f"# fulltext 3rd-party: {ft_third_party} (flagged, not dropped)")
    print(f"# licences         : {lic_counts}")
    print(f"# provenance models: {models}")
    print(f"# wrote card       : {card}")
    return 0


def _card(version: str, n_papers: int, totals: Dict[str, int],
          lic: Dict[str, int], models: Dict[str, int], fulltext: bool,
          ft_third_party: int = 0, scrubbed_emails: bool = False,
          extra_cfgs: Optional[Dict[str, tuple]] = None) -> str:
    cfgs: list = ["papers", "affiliations", "topics"]
    if fulltext and totals.get("fulltext"):
        cfgs.append("fulltext")

    def yaml_entry(name: str, splits: Optional[list] = None) -> str:
        if not splits:
            return f"  - config_name: {name}\n    data_files: {name}/*.parquet"
        lines = [f"  - config_name: {name}", "    data_files:"]
        for sp in splits:
            lines.append(f"      - split: {sp}")
            lines.append(f"        path: {name}/{sp}-*.parquet")
        return "\n".join(lines)

    entries = [yaml_entry(c) for c in cfgs]
    for name, (_, splits) in (extra_cfgs or {}).items():
        cfgs.append(name)
        entries.append(yaml_entry(name, splits))
    cfg_yaml = "\n".join(entries)
    lic_rows = "\n".join(f"| {k} | {v:,} |" for k, v in sorted(lic.items()))
    model_rows = "\n".join(f"| {k} | {v:,} |" for k, v in
                           sorted(models.items(), key=lambda t: -t[1]))
    email_note = ("E-mail addresses were removed from the parsed text and "
                  "replaced with `<email-removed>`."
                  if scrubbed_emails else
                  "The `fulltext` config is a faithful parse of the published "
                  "PDFs, so where a paper printed author e-mail addresses in "
                  "its header, the TEI and OCR text contain them too. They are "
                  "not separated into a column, and no contact details beyond "
                  "what the papers themselves print are added.")
    extra_rows = "\n".join(f"| `{c}` | {n:,} | {EXTRA_CFG_DESC[c]} |"
                           for c, (n, _) in (extra_cfgs or {}).items()
                           if c in EXTRA_CFG_DESC)
    ft_row = ("| `fulltext` | " + f"{totals.get('fulltext', 0):,}"
              + " | parsed text per paper: GROBID TEI, PyMuPDF/LLM Markdown, OCR |"
              if totals.get("fulltext") else "")
    withheld = (f"\n{ft_third_party:,} papers were published at third-party "
                "(ELRA-published) venues such as LREC. The ACL copyright policy "
                "states that it does not cover third-party materials, so these "
                "rows carry `third_party_venue = true`; filter on it if your use "
                "needs only ACL-licensed material.\n"
                if ft_third_party else "")
    return f"""---
license: cc-by-nc-sa-4.0
language: [en]
size_categories: [100K<n<1M]
task_categories: [question-answering, text-retrieval]
tags: [scientometrics, affiliations, ror, acl-anthology, cso, bibliometrics]
configs:
{cfg_yaml}
---

# ANTHEM-{version} — an Enriched ACL Anthology for Scholarly Comparison

{n_papers:,} ACL Anthology papers linked to ROR-normalised author affiliations
and Computer Science Ontology (CSO) topic labels, with an evidence span
retained for every extraction decision.

Code: <https://github.com/MoritzStaudinger/anthem>

## Configs

| Config | Rows | What |
|---|---:|---|
| `papers` | {totals.get('papers', 0):,} | one row per paper; nested fields kept as JSON strings |
| `affiliations` | {totals.get('affiliations', 0):,} | flat: paper x author x affiliation, with ROR id and evidence |
| `topics` | {totals.get('topics', 0):,} | flat: paper x CSO label, `source` = syntactic \\| semantic |
{ft_row}
{extra_rows}

Nested structures are stored as JSON strings because record shape varies across
1952--2025; the flat `affiliations` and `topics` configs are the ones to query.

```python
from datasets import load_dataset

aff = load_dataset("Molarix/anthem", "affiliations", split="train")
```

`fulltext` is the large config — load it only when you need the text itself.

## Licensing

ANTHEM's own layer — the extracted affiliations, ROR links, CSO labels and
evidence spans in `papers`, `affiliations` and `topics` — is released under
**CC-BY-NC-SA-4.0**.

The `license` column is provenance, not the release licence: it records the
terms the underlying paper carries in the Anthology. Per the
[ACL Anthology copyright policy](https://aclanthology.org/faq/copyright/),
materials prior to 2016 are CC BY-NC-SA 3.0 and materials from 2016 onward are
CC BY 4.0. Papers whose year could not be determined are assigned the more
restrictive pre-2016 term.
{withheld}
Source PDFs are not redistributed; the Anthology serves those. `fulltext`
contains parsed renderings (TEI, Markdown, OCR text) produced by this pipeline.

| Underlying licence | Papers |
|---|---:|
{lic_rows}

## Construction

Affiliations were extracted from parsed PDFs by an LLM pipeline over combined
parser outputs (GROBID TEI + PyMuPDF/LLM Markdown + OCR), then linked to ROR
with country context. The model recorded per paper:

| `provenance_model` | Papers |
|---|---:|
{model_rows}

## Personal and sensitive information

The dataset joins author names to institutions and publication years. All of it
is already public in the ACL Anthology, and affiliations are as printed at
publication time rather than current.

{email_note}

## Known limitations

Affiliation extraction and linking reach F1 0.760 (hierarchy-aware, parent and
child ROR ids treated as equivalent) against a 781-paper decade-stratified gold
set; ROR linking alone reaches 0.857 on hand-checked affiliation strings; CSO
topic-label precision is 0.713. Roughly 90% of remaining false negatives are
linking rather than extraction failures, concentrated in city-level institution
clusters and geographically branched corporate entities.

This is a fixed snapshot (ANTHEM-{version}); it is not updated in place.

## Citation

Please cite the ANTHEM paper; the current reference is kept in the
[code repository](https://github.com/MoritzStaudinger/anthem).
"""


if __name__ == "__main__":
    raise SystemExit(main())
