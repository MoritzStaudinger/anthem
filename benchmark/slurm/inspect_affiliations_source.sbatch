#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --job-name=inspect_affs
#SBATCH --output=/home/%u/logs/inspect_affs-%j.out
#
# Print the top-level keys + values of a few affiliation JSONs to find
# out what fields the source actually carries. If `year` is missing or
# named differently, the index build is reading the wrong key — fix is
# in scico_indexing.py:_meta_to_document. If `year` is genuinely absent
# from the source, the affiliation enrichment pipeline upstream needs a
# bibtex/ACL Anthology join.

set -euo pipefail
mkdir -p "$HOME/logs"

AFFS_DIR="/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics/affiliations"
[ -d "$AFFS_DIR" ] || { echo "missing $AFFS_DIR"; exit 1; }

python3 -u <<PY
import json
import os
from pathlib import Path
from collections import Counter

aff_dir = Path("$AFFS_DIR")

# Sample 5 papers: 2 old-format, 2 modern, 1 random
candidates = []
files = sorted(aff_dir.glob("*.json"))
print(f"total affiliation files: {len(files)}")

# Pick representative samples
picks = []
for fp in files:
    pid = fp.stem
    if pid.startswith("1952") or pid.startswith("P85") or pid.startswith("C82"):
        picks.append(fp)
    elif pid.startswith("2024.acl-") or pid.startswith("2023.emnlp-"):
        picks.append(fp)
    if len(picks) >= 5:
        break

print()
print("=" * 80)
print("Sample raw JSON contents (top-level keys + first-level values):")
print("=" * 80)
for fp in picks[:5]:
    print()
    print(f"FILE: {fp.name}")
    print("-" * 60)
    try:
        d = json.loads(fp.read_text())
    except Exception as e:
        print(f"  ERROR loading: {e}")
        continue
    print(f"keys: {sorted(d.keys())}")
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, (list, dict)):
            preview = (json.dumps(v, ensure_ascii=False)[:120])
            print(f"  {k}: {type(v).__name__}[{len(v)}]  preview={preview}...")
        else:
            print(f"  {k}: {v!r}")

# Aggregate: how many files have 'year', 'venue', 'publication_year', 'pub_year'?
print()
print("=" * 80)
print("Aggregate: how often does each candidate key appear, across 1000 random files?")
print("=" * 80)
import random
random.seed(0)
sample = random.sample(files, min(1000, len(files)))
key_counts = Counter()
year_field_present = Counter()
year_field_nonempty = Counter()
for fp in sample:
    try:
        d = json.loads(fp.read_text())
    except Exception:
        continue
    for k in d:
        key_counts[k] += 1
    for k in ("year", "publication_year", "pub_year", "Year",
              "venue", "Venue", "month", "title", "authors"):
        if k in d:
            year_field_present[k] += 1
            if d[k] not in (None, "", [], {}):
                year_field_nonempty[k] += 1

print(f"sampled {len(sample)} files")
print()
print("All top-level keys observed (count):")
for k, n in key_counts.most_common():
    print(f"  {k:<30} {n}")
print()
print("Candidate 'year-ish' / 'venue-ish' fields:")
for k in ("year", "publication_year", "pub_year", "Year",
          "venue", "Venue", "month", "title", "authors"):
    p = year_field_present.get(k, 0)
    n = year_field_nonempty.get(k, 0)
    print(f"  {k:<20} present={p:>4}  non-empty={n:>4}")
PY
