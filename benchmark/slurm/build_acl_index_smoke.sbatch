#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=build_idx_smoke
#SBATCH --output=/home/%u/logs/build_idx_smoke-%j.out
#
# Smoke-test the index build with 100 papers from the ENRICHED corpus.
# Writes to data/local_index_acl_smoke/ (separate from the canonical
# index), then queries the result to verify that publication_year +
# venue + filters actually work end-to-end.
#
# Run this BEFORE the full ~30 min rebuild. Total wall: ~2-3 min.

set -euo pipefail

SCICO_ROOT="${1:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta}"
PERSIST="data/local_index_acl_smoke"

cd ~/compare_acl 2>/dev/null || cd ~/PycharmProjects/Comparer_flask 2>/dev/null || cd ~/Comparer_flask
mkdir -p "$HOME/logs"

unset VIRTUAL_ENV PYTHONHOME
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv/bin' | paste -sd: -)
export PATH

if [ -d ~/miniconda3/envs/compare_eval ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate compare_eval
elif [ -d ~/miniconda3/envs/scico ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate scico
fi
if [ -f .env ]; then set -a; . .env; set +a; fi

echo "[smoke] scico root  : $SCICO_ROOT"
echo "[smoke] persist dir : $PERSIST (will be wiped)"
echo ""

# Wipe any prior smoke index so the build is fresh
rm -rf "$PERSIST"

# Step 1: build the small index
echo "[smoke] building 100-paper index ..."
python3 -u scripts/build_acl_index.py \
    --scico-root "$SCICO_ROOT" \
    --persist-dir "$PERSIST" \
    --resume \
    --batch-size 100 \
    --limit-papers 100

echo ""
echo "[smoke] post-build sanity check ..."

# Step 2: load the smoke index and verify metadata + filters
python3 -u <<'PY'
import os
os.environ["ACL_INDEX_DIR"] = "data/local_index_acl_smoke"

from backend.rag.local_acl_indexing import load_acl_index
from backend.rag.rag_utils import retrieve
import backend.rag.rag_utils as ru

idx = load_acl_index("data/local_index_acl_smoke")
print(f"loaded smoke index: {len(idx.docstore.docs)} docs")

# Show metadata of first 3 docs — confirm year/venue populated
print()
print("Sample metadata (first 3 docs):")
for i, (k, node) in enumerate(idx.docstore.docs.items()):
    if i >= 3: break
    md = node.metadata or {}
    print(f"  {md.get('paper_id')}  year={md.get('publication_year')}  "
          f"venue={md.get('venue')!r}  doc_type={md.get('doc_type')}")

# Now exercise retrieve() with a year filter — the failing case in
# the big-index test.
ru.static_index = idx
print()
print("retrieve(query='language', year_start=1950, year_end=2000)")
docs = retrieve(query="language", year_start=1950, year_end=2000, top_k=5)
print(f"  returned {len(docs)} docs")
for d in docs[:3]:
    md = getattr(d, "metadata", {}) or {}
    print(f"    - {md.get('paper_id')}  year={md.get('publication_year')}  venue={md.get('venue')}")

print()
print("retrieve(query='translation', year_start=1990, year_end=2025)")
docs = retrieve(query="translation", year_start=1990, year_end=2025, top_k=5)
print(f"  returned {len(docs)} docs")

print()
print("retrieve(query='language', country='US')")
docs = retrieve(query="language", country="US", top_k=5)
print(f"  returned {len(docs)} docs")

PY

echo ""
echo "[smoke] done."
