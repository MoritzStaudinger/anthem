"""Export the per-paper metadata corpus to one gzipped JSONL file.

Run this ON THE CLUSTER. It walks the 117k-file metadata directory, scrubs
machine-local provenance paths, and streams one JSON object per line into a
single .jsonl.gz — so the transfer is one ~150 MB file instead of 117,386
separate rsync round-trips.

Pure I/O, stdlib only: run it on the login node, not via sbatch.

Usage (cluster):

    python3 scripts/export_corpus_jsonl.py \\
        --src /share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta \\
        --out ~/anthem-202605.jsonl.gz

Then, locally:

    rsync -avP moritz.staudinger@cluster.datalab.tuwien.ac.at:~/anthem-202605.jsonl.gz ./release/
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

# Keys under _provenance that embed absolute cluster paths / usernames.
PROVENANCE_DROP = ("paths", "path", "src", "cwd")


def scrub(rec: dict) -> dict:
    """Remove machine-local paths so they don't end up in a public release."""
    prov = rec.get("_provenance")
    if isinstance(prov, dict):
        for k in PROVENANCE_DROP:
            prov.pop(k, None)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True,
                    help="Directory of <paper_id>.json metadata records.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output .jsonl.gz path.")
    ap.add_argument("--no-scrub", action="store_true",
                    help="Keep _provenance paths (not for public releases).")
    args = ap.parse_args()

    if not args.src.is_dir():
        print(f"# ERROR: not a directory: {args.src}", file=sys.stderr)
        return 1

    n = bad = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as fh:
        for p in sorted(args.src.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                bad += 1
                print(f"# skip {p.name}: {exc}", file=sys.stderr)
                continue
            rec.setdefault("paper_id", p.stem)
            if not args.no_scrub:
                rec = scrub(rec)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 10000 == 0:
                print(f"# {n} records", flush=True)

    size_mb = args.out.stat().st_size / 1e6
    print(f"# wrote {n} records ({bad} unreadable) -> {args.out} ({size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
