"""Add the benchmark and the human annotation sets to the ANTHEM HF release.

`build_hf_release.py` writes the corpus configs (papers / affiliations / topics
/ fulltext). This script writes the four evaluation artefacts alongside them, so
everything the paper reports against is loadable with `load_dataset`:

    <out>/
    ├── benchmark_questions/*.parquet   99 comparative-QA questions + gold spec
    ├── affiliation_gold/*.parquet      781-paper hand-annotated affiliation gold
    ├── linking_adjudication/*.parquet  806 hand-verified ROR linking decisions
    └── cso_validation/*.parquet        104 hand-judged CSO topic labels

Nested gold structures stay as JSON strings: `classifier_gold` and
`topic_metadata` on the questions, and the per-author affiliation lists on the
gold set. Parquet cannot hold their varying shape as a struct without losing
rows, and json.loads on a column is cheap.

Usage:

    python scripts/build_hf_annotations.py \\
        --benchmark  ~/PycharmProjects/Comparer_flask/data/eval/scico_testset.jsonl \\
        --gold-csv   .../sample_data/affiliation_ground_truth.csv \\
        --adjudication-csv reports/relink_compare/adjudication_sheet.csv \\
        --cso-csv    reports/cso_validation/cso_validation.csv \\
        --out ~/anthem_build/release
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    print("# ERROR: pyarrow is required -> pip install 'pyarrow<19'", file=sys.stderr)
    raise SystemExit(1)

# csv fields here hold whole JSON documents, well past the 128 KB default.
csv.field_size_limit(16 * 1024 * 1024)

QUESTION_SCHEMA = pa.schema([
    ("id", pa.string()), ("query", pa.string()), ("type", pa.string()),
    ("persona", pa.string()), ("depth", pa.string()),
    ("question_type", pa.string()), ("question_category", pa.string()),
    ("uni1", pa.string()), ("uni1_ror", pa.string()),
    ("uni2", pa.string()), ("uni2_ror", pa.string()),
    ("country", pa.string()), ("country_b", pa.string()),
    ("topic", pa.string()), ("cso_url", pa.string()),
    ("cso_closure_size", pa.int32()),
    ("year_start", pa.int32()), ("year_end", pa.int32()),
    ("gold_paper_count", pa.int32()),
    ("gold_paper_ids", pa.list_(pa.string())),
    ("gold_summary", pa.string()),
    ("classifier_gold_json", pa.string()), ("topic_metadata_json", pa.string()),
])

GOLD_SCHEMA = pa.schema([
    ("anthology_id", pa.string()), ("year", pa.int32()), ("url", pa.string()),
    ("source", pa.string()),
    ("n_authors", pa.int32()), ("n_affiliations", pa.int32()),
    ("n_ror_linked", pa.int32()),
    ("authors_raw_json", pa.string()), ("authors_normalized_json", pa.string()),
])

ADJUDICATION_SCHEMA = pa.schema([
    ("raw_string", pa.string()), ("n_papers", pa.int32()),
    ("agree", pa.string()),
    ("multi_ror", pa.string()), ("multi_name", pa.string()),
    ("multi_country", pa.string()), ("multi_score", pa.float64()),
    ("single_ror", pa.string()), ("single_name", pa.string()),
    ("single_country", pa.string()), ("single_score", pa.float64()),
    ("correct_ror", pa.string()), ("notes", pa.string()),
])

# The LoRA training data the paper says is released alongside ANTHEM.
# `output` is a list of author dicts and `source` a dict, so both are kept as
# JSON strings for the same reason the corpus configs do it.
LORA_SCHEMA = pa.schema([
    ("input", pa.string()), ("output_json", pa.string()),
    ("source_json", pa.string()),
])

CSO_SCHEMA = pa.schema([
    ("paper_id", pa.string()), ("title", pa.string()), ("authors", pa.string()),
    ("abstract", pa.string()), ("cso_topic", pa.string()),
    ("source", pa.string()), ("applies", pa.string()), ("notes", pa.string()),
])


def s(v: Any) -> Optional[str]:
    """Normalise a CSV/JSON scalar to a non-empty string or None."""
    if v is None:
        return None
    t = str(v).strip()
    return t or None


def i(v: Any) -> Optional[int]:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def f(v: Any) -> Optional[float]:
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def write(out: Path, name: str, schema: pa.Schema, rows: List[Dict],
          split: Optional[str] = None) -> int:
    """Write one single-shard config; these sets are all small.

    `split` names a HuggingFace split, which the shard filename carries so the
    card generator can discover it without a sidecar file.
    """
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    stem = split or name
    cols = {fld.name: [r.get(fld.name) for r in rows] for fld in schema}
    pq.write_table(pa.table(cols, schema=schema), d / f"{stem}-00000.parquet",
                   compression="zstd")
    return len(rows)


def iter_json_objects(text: str):
    """Yield JSON objects from a concatenation of them.

    Not every record in the LoRA training files is on a single line -- some
    `input` values contain literal newlines -- so splitlines() tears records in
    half. raw_decode walks object by object instead and is agnostic to how they
    happen to be wrapped.
    """
    dec = json.JSONDecoder()
    i, n = 0, len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n":
            i += 1
        if i >= n:
            return
        obj, i = dec.raw_decode(text, i)
        yield obj


def load_lora(p: Path) -> List[Dict]:
    out = []
    for r in iter_json_objects(p.read_text(encoding="utf-8")):
        out.append({
            "input": s(r.get("input")),
            "output_json": json.dumps(r["output"], ensure_ascii=False)
                           if r.get("output") is not None else None,
            "source_json": json.dumps(r["source"], ensure_ascii=False)
                           if r.get("source") is not None else None,
        })
    return out


def load_questions(p: Path) -> List[Dict]:
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        cg = r.get("classifier_gold") or {}
        tm = r.get("topic_metadata") or {}
        out.append({
            "id": s(r.get("id")), "query": s(r.get("query")),
            "type": s(r.get("type")), "persona": s(r.get("persona")),
            "depth": s(r.get("depth")),
            "question_type": s(cg.get("question_type")),
            "question_category": s(cg.get("question_category")),
            "uni1": s(cg.get("uni1")), "uni1_ror": s(cg.get("uni1_ror")),
            "uni2": s(cg.get("uni2")), "uni2_ror": s(cg.get("uni2_ror")),
            "country": s(cg.get("country")), "country_b": s(cg.get("country_b")),
            "topic": s(tm.get("cso_label")) or s(cg.get("topic")),
            "cso_url": s(tm.get("cso_url")),
            "cso_closure_size": i(tm.get("cso_closure_size")),
            "year_start": i(cg.get("year_start")), "year_end": i(cg.get("year_end")),
            "gold_paper_count": i(r.get("gold_paper_count")),
            "gold_paper_ids": [x for x in (r.get("gold_paper_ids") or []) if x],
            "gold_summary": s(r.get("gold_summary")),
            "classifier_gold_json": json.dumps(cg, ensure_ascii=False) if cg else None,
            "topic_metadata_json": json.dumps(tm, ensure_ascii=False) if tm else None,
        })
    return out


def load_gold(p: Path) -> List[Dict]:
    out = []
    with p.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            # Count authors/affiliations so the set is summarisable without
            # parsing the JSON columns.
            n_auth = n_aff = n_ror = 0
            try:
                authors = json.loads(r.get("authors_normalized_json") or "[]")
                n_auth = len(authors)
                for a in authors:
                    for af in (a.get("affiliations") or []):
                        n_aff += 1
                        if af.get("ror_id"):
                            n_ror += 1
            except (json.JSONDecodeError, AttributeError, TypeError):
                pass
            out.append({
                "anthology_id": s(r.get("anthology_id")),
                "year": i(r.get("year")), "url": s(r.get("url")),
                "source": s(r.get("_source")),
                "n_authors": n_auth, "n_affiliations": n_aff, "n_ror_linked": n_ror,
                "authors_raw_json": s(r.get("authors_raw_json")),
                "authors_normalized_json": s(r.get("authors_normalized_json")),
            })
    return out


def load_adjudication(p: Path) -> List[Dict]:
    out = []
    with p.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            out.append({
                "raw_string": s(r.get("string")), "n_papers": i(r.get("n_papers")),
                "agree": s(r.get("agree")),
                "multi_ror": s(r.get("multi_ror")), "multi_name": s(r.get("multi_name")),
                "multi_country": s(r.get("multi_country")),
                "multi_score": f(r.get("multi_score")),
                "single_ror": s(r.get("single_ror")),
                "single_name": s(r.get("single_name")),
                "single_country": s(r.get("single_country")),
                "single_score": f(r.get("single_score")),
                "correct_ror": s(r.get("correct_ror")), "notes": s(r.get("notes")),
            })
    return out


def load_cso(p: Path) -> List[Dict]:
    out = []
    with p.open(encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            out.append({
                "paper_id": s(r.get("paper_id")), "title": s(r.get("title")),
                "authors": s(r.get("authors")), "abstract": s(r.get("abstract")),
                "cso_topic": s(r.get("cso_topic")), "source": s(r.get("source")),
                "applies": s(r.get("applies?")), "notes": s(r.get("notes")),
            })
    return out


JOBS = (
    ("benchmark_questions", "benchmark", QUESTION_SCHEMA, load_questions),
    ("affiliation_gold", "gold_csv", GOLD_SCHEMA, load_gold),
    ("linking_adjudication", "adjudication_csv", ADJUDICATION_SCHEMA, load_adjudication),
    ("cso_validation", "cso_csv", CSO_SCHEMA, load_cso),
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", type=Path, help="scico/anthem RAG testset .jsonl")
    ap.add_argument("--gold-csv", type=Path, help="affiliation_ground_truth.csv")
    ap.add_argument("--adjudication-csv", type=Path, help="adjudication_sheet.csv")
    ap.add_argument("--cso-csv", type=Path, help="cso_validation.csv")
    ap.add_argument("--lora-train", type=Path, help="data/lora_train/train.jsonl")
    ap.add_argument("--lora-val", type=Path, help="data/lora_train/val.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    wrote = {}
    for name, arg, schema, loader in JOBS:
        src = getattr(args, arg)
        if not src:
            print(f"# SKIP {name} (no --{arg.replace('_', '-')})")
            continue
        if not src.is_file():
            print(f"# ERROR {name}: not a file: {src}", file=sys.stderr)
            return 1
        rows = loader(src)
        wrote[name] = write(args.out, name, schema, rows)
        print(f"# {name:22s}: {wrote[name]:>6,} rows  <- {src.name}")

    # lora_train is the one config with splits.
    for split, src in (("train", args.lora_train), ("validation", args.lora_val)):
        if not src:
            print(f"# SKIP lora_train/{split} (not given)")
            continue
        if not src.is_file():
            print(f"# ERROR lora_train/{split}: not a file: {src}", file=sys.stderr)
            return 1
        n = write(args.out, "lora_train", LORA_SCHEMA, load_lora(src), split=split)
        wrote[f"lora_train/{split}"] = n
        print(f"# {'lora_train/' + split:22s}: {n:>6,} rows  <- {src.name}")

    if not wrote:
        print("# nothing written", file=sys.stderr)
        return 1
    print(f"\n# wrote {len(wrote)} configs into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
