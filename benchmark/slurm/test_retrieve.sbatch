#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --job-name=test_retrieve
#SBATCH --output=/home/%u/logs/test_retrieve-%j.out
#
# Isolated diagnostic: confirms whether retrieve() from the local ACL
# index actually returns documents under the filters the pipeline uses.
# No eval scaffolding, no LLM, no agent — just the retrieval path.
#
# If this shows zero results for big-name institutions, the bug is in
# the index filter logic (SimpleVectorStore's CONTAINS operator on
# string fields) and we know to fix that instead of the prompt path.

set -euo pipefail

cd ~/compare_acl 2>/dev/null || cd ~/PycharmProjects/Comparer_flask 2>/dev/null || cd ~/Comparer_flask
mkdir -p "$HOME/logs"

unset VIRTUAL_ENV PYTHONHOME
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv/bin' | paste -sd: -)
export PATH
. ~/miniconda3/etc/profile.d/conda.sh && conda activate compare_eval

# Stage index to RAM-tmpfs so the test isn't dominated by NFS load.
SRC_INDEX="data/local_index_acl"
STAGED="/dev/shm/acl_index_${SLURM_JOB_ID:-$$}_$USER"
if [ -d "$SRC_INDEX" ] && [ -w /dev/shm ]; then
    echo "staging index to $STAGED ..."
    t0=$(date +%s)
    cp -r "$SRC_INDEX" "$STAGED"
    echo "staged in $(($(date +%s) - t0))s ($(du -sh $STAGED | cut -f1))"
    export ACL_INDEX_DIR="$STAGED"
    trap 'rm -rf "$STAGED" 2>/dev/null || true' EXIT
fi

python3 -u <<'PY'
import time
t0 = time.time()
def s(msg): print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)

s("loading index")
from backend.rag.local_acl_indexing import load_acl_index, DEFAULT_ACL_PERSIST_DIR
idx = load_acl_index(DEFAULT_ACL_PERSIST_DIR)
s(f"index loaded; docs={len(idx.docstore.docs)}")

s("import retrieve()")
from backend.rag.rag_utils import retrieve
import backend.rag.rag_utils as ru
ru.static_index = idx

def run(label, **kwargs):
    s(f"---- {label} ----")
    s(f"  args: {kwargs}")
    try:
        docs = retrieve(**kwargs)
        s(f"  returned {len(docs)} docs")
        for d in docs[:3]:
            meta = getattr(d, "metadata", {}) or {}
            pid = meta.get("paper_id")
            yr = meta.get("publication_year")
            affs = (meta.get("affiliations_string") or "")[:80]
            ccs = (meta.get("countries_string") or "")[:40]
            s(f"    - {pid} ({yr})  affs={affs!r}  ccs={ccs!r}")
    except Exception as e:
        s(f"  EXCEPTION: {type(e).__name__}: {e}")

# 1) Bare query — should return top-k matches regardless of filter.
run("bare query, no filters",
    query="natural language generation", top_k=5)

# 2) Big-name institution filter — Stanford is THE largest ACL contributor.
run("institution=Stanford University",
    query="natural language generation",
    institution="Stanford University", top_k=5)

# 3) Country filter only.
run("country=US (ISO)",
    query="machine translation", country="US", top_k=5)

# 4) Combined filter — institution AND country (university_country_comparison).
run("institution=Utrecht University, country=NL",
    query="large language models",
    institution="Utrecht University", country="NL",
    year_start=2023, year_end=2025, top_k=5)

# 5) Year window only (field_evolution style).
run("year window 2020-2024 only",
    query="instruction tuning", year_start=2020, year_end=2024, top_k=5)

# 6) Inspect one raw metadata blob to confirm what the filter is matching against.
s("---- raw metadata of first 2 docs (to see actual `affiliations_string` shape) ----")
ds = list(idx.docstore.docs.values())[:2]
for d in ds:
    md = d.metadata or {}
    keep = {k: md.get(k) for k in ("paper_id", "publication_year", "venue",
                                    "affiliations_string", "countries_string")}
    s(f"  {keep}")

s("done")
PY
