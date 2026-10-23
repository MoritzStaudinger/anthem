#!/bin/bash
#SBATCH --job-name=scico-enrich
#SBATCH --partition=GPU-a100s              # CSO is CPU-bound, no GPU request below
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64                 # CSO is CPU-bound (spaCy + Word2Vec + n-gram match)
#SBATCH --mem=64G                          # 32 workers × ~500MB + headroom
#SBATCH --time=08:00:00                    # CSO-only finishes in ~3-5h; some slack
#SBATCH --output=logs/enrich-%j.out
#SBATCH --error=logs/enrich-%j.err
#
# No --gres line — CSO Classifier doesn't use a GPU; requesting one would
# just block on GPU availability for nothing. If the cluster rejects CPU-only
# jobs on GPU-* partitions, add: #SBATCH --gres=gpu:1

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — override via env vars on the sbatch invocation if needed.
# ---------------------------------------------------------------------------
PROJECT_DIR="${PROJECT_DIR:-/share/moritz.staudinger/acl_processor}"
AFFILIATIONS_DIR="${AFFILIATIONS_DIR:-/share/moritz.staudinger/datastructure/affiliations}"
# Abstract sources for papers whose affiliations JSON has no abstract field.
# Primary: ACL Anthology metadata clone (clean author-provided abstracts).
# Fallback: PyMu_LLM Markdown export (parsed from PDF).
ANTHOLOGY_REPO="${ANTHOLOGY_REPO:-/share/moritz.staudinger/acl_data/acl-anthology}"
MD_DIR="${MD_DIR:-/home/$USER/data/acl_pdfs_pymu_llm}"
# If set, enriched JSONs are written here as <paper_id>.json. If unset, the
# script overwrites the source files in AFFILIATIONS_DIR in place.
OUT_DIR="${OUT_DIR:-}"
# OpenAlex disabled by default — it doesn't contribute to topic labels and
# its 5 req/sec polite-pool cap would bottleneck the whole job to ~6.5h for
# 114k papers. Title backfill (its only remaining purpose) is a separate
# concern that can run offline from ACL Anthology metadata.
MAILTO="${MAILTO:-momo.staudinger@gmail.com}"
SKIP_OA="${SKIP_OA:-1}"

# ---------------------------------------------------------------------------
# Worker count: each CSO worker holds spaCy + Word2Vec (~500 MB RAM).
# 32 workers on 64 cores leaves cache/I/O headroom and avoids RAM thrash.
# Override with CSO_WORKERS=N if you've sized memory accordingly.
# ---------------------------------------------------------------------------
CSO_WORKERS="${CSO_WORKERS:-32}"
CSO_BATCH="${CSO_BATCH:-500}"

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR"
mkdir -p logs

# Activate the Python environment. Try a conda env first (default name 'scico'),
# fall back to a local venv. Override with PY_ENV=<conda_env_name>.
PY_ENV="${PY_ENV:-scico}"
if command -v conda >/dev/null 2>&1 && conda env list 2>/dev/null | grep -qE "^${PY_ENV}\s"; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$PY_ENV"
    echo "[$(date '+%F %T')] activated conda env: $PY_ENV ($(python --version))"
elif [ -f "venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source venv/bin/activate
    echo "[$(date '+%F %T')] activated venv ($(python --version))"
else
    echo "ERROR: no Python env found (looked for conda '$PY_ENV' and venv/)." >&2
    echo "Create it on the login node first — see prepare_env.sh." >&2
    exit 1
fi

# Stop intra-process BLAS threads from oversubscribing the multiprocessing pool.
# Without these caps, OpenBLAS spawns 64 threads inside each of 32 workers
# (~2k threads total) and everything thrashes.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Belt-and-braces pre-warm in case the cached models aren't on this node yet.
# If compute nodes have outbound internet (typical for TU Wien DataLab), this
# is a no-op after the first run. If not, run prepare_env.sh on a login node
# first to populate ~/.cso_classifier/ and the spaCy/NLTK caches.
echo "[$(date '+%F %T')] Pre-warming CSO + spaCy + NLTK..."
python - <<'PY'
from cso_classifier.ontology import Ontology
Ontology(load_ontology=True)
import spacy
spacy.load("en_core_web_sm")
# NLTK 3.6.2 (pinned by cso-classifier 3.1) ships punkt + stopwords; no
# punkt_tab (that's NLTK 3.9+). find() can raise either LookupError or
# OSError depending on the resource, so catch both.
import nltk
for pkg, path in (("punkt", "tokenizers/punkt"),
                  ("stopwords", "corpora/stopwords")):
    try:
        nltk.data.find(path)
    except (LookupError, OSError):
        nltk.download(pkg, quiet=True)
print("warmup ok")
PY

# ---------------------------------------------------------------------------
# Run enrichment
# ---------------------------------------------------------------------------
SKIP_OA_FLAG=""
if [ "$SKIP_OA" = "1" ]; then
    SKIP_OA_FLAG="--skip-openalex"
    echo "[$(date '+%F %T')] OpenAlex disabled (--skip-openalex)"
else
    echo "[$(date '+%F %T')] OpenAlex enabled — note this caps throughput at ~5 req/sec"
fi

OUT_DIR_FLAG=()
if [ -n "$OUT_DIR" ]; then
    OUT_DIR_FLAG=(--out-dir "$OUT_DIR")
    echo "[$(date '+%F %T')] writing enriched JSONs to: $OUT_DIR"
else
    echo "[$(date '+%F %T')] overwriting source JSONs in: $AFFILIATIONS_DIR"
fi

echo "[$(date '+%F %T')] Starting enrichment"
echo "  PROJECT_DIR    = $PROJECT_DIR"
echo "  AFFILIATIONS   = $AFFILIATIONS_DIR"
echo "  CSO_WORKERS    = $CSO_WORKERS"
echo "  CSO_BATCH      = $CSO_BATCH"
echo "  SLURM_JOB_ID   = ${SLURM_JOB_ID:-(none)}"
echo "  Host           = $(hostname)"

ABSTRACT_FLAGS=()
[ -n "$ANTHOLOGY_REPO" ] && [ -d "$ANTHOLOGY_REPO" ] && ABSTRACT_FLAGS+=(--anthology-repo "$ANTHOLOGY_REPO")
[ -n "$MD_DIR"         ] && [ -d "$MD_DIR"         ] && ABSTRACT_FLAGS+=(--md-dir         "$MD_DIR")
echo "[$(date '+%F %T')] abstract sources: ${ABSTRACT_FLAGS[*]:-(none)}"

python scripts/enrich_with_topics.py \
    --affiliations-dir "$AFFILIATIONS_DIR" \
    --mailto "$MAILTO" \
    --cso-batch "$CSO_BATCH" \
    --cso-workers "$CSO_WORKERS" \
    "${OUT_DIR_FLAG[@]}" \
    "${ABSTRACT_FLAGS[@]}" \
    $SKIP_OA_FLAG

echo "[$(date '+%F %T')] Enrichment done"
