#!/bin/bash
# One-shot environment prep for the SciCo CSO enrichment job.
# Run on the LOGIN node (needs internet for pip / spaCy / NLTK / CSO).
#
# Uses conda-forge prebuilt binaries for gensim 3.8.3, spacy, nltk —
# avoiding pip's broken source build of gensim 3.8.3 on modern Python.
# cso-classifier itself comes from pip (it isn't on conda-forge).
#
# Idempotent: re-running is safe.
#
# Usage:
#     bash scripts/prepare_env.sh
#
# To start over from a broken env:
#     conda env remove -n scico -y && bash scripts/prepare_env.sh

set -euo pipefail

PY_ENV="${PY_ENV:-scico}"
# Python 3.9 — conda-forge's gensim 3.8.3 only ships builds for 3.6-3.9.
PY_VER="${PY_VER:-3.9}"

echo "[prepare_env] target conda env: $PY_ENV (Python $PY_VER)"

# ---- Sanity: need conda --------------------------------------------------
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---- Create / reuse the conda env ---------------------------------------
if conda env list | grep -qE "^${PY_ENV}\s"; then
    echo "[prepare_env] conda env exists: $PY_ENV"
    echo "[prepare_env] NOTE: if a previous run left it half-broken (e.g."
    echo "[prepare_env]       cso-classifier 3.1 + gensim 4.x), run:"
    echo "[prepare_env]         conda env remove -n $PY_ENV -y"
    echo "[prepare_env]       and re-run this script for a clean rebuild."
else
    echo "[prepare_env] creating conda env: $PY_ENV (python=$PY_VER)"
    conda create -n "$PY_ENV" "python=$PY_VER" -y
fi
conda activate "$PY_ENV"

PY_CHECK=$(python --version 2>&1)
echo "[prepare_env] active env: $PY_ENV  ($PY_CHECK)"
case "$PY_CHECK" in
    *Python\ 3.9.*) ;;
    *)
        echo "ERROR: env has $PY_CHECK; conda-forge's gensim 3.8.3 only builds" >&2
        echo "       for Python 3.6-3.9. Need exactly Python 3.9. Remove + recreate:" >&2
        echo "         conda env remove -n $PY_ENV -y" >&2
        echo "         PY_VER=3.9 bash scripts/prepare_env.sh" >&2
        exit 1
        ;;
esac

# ---- conda-forge: ONLY the binaries pip can't build ---------------------
# gensim 3.8.3 source build fails on Python 3.9+ (__NUMPY_SETUP__ bug);
# conda-forge has a prebuilt wheel. Everything else (spacy, nltk, click,
# kneed, rapidfuzz, ...) is left to pip via cso-classifier's own pins
# below — this avoids version-clash games between conda and pip.
echo "[prepare_env] installing gensim 3.8.3 + numpy from conda-forge..."
conda install -n "$PY_ENV" -c conda-forge -y \
    "gensim=3.8.3" \
    "numpy=1.21"

echo "[prepare_env] upgrading pip..."
pip install --upgrade pip --quiet

# Let pip resolve cso-classifier 3.1's exact dep pins automatically
# (spacy==3.0.5, nltk==3.6.2, click==7.1.2, kneed==0.3.1, rapidfuzz==2.11.1,
# strsimpy==0.2.0, requests==2.25.1, python-igraph==0.10.4, etc.).
# We pre-install gensim above so pip sees the requirement satisfied without
# trying its own source build.
echo "[prepare_env] installing cso-classifier 3.1 (pip resolves its pins)..."
pip install --quiet "cso-classifier==3.1"

# Extras for our own scripts (independent of cso-classifier).
pip install --quiet tqdm

# ---- spaCy model --------------------------------------------------------
# spaCy 3.0.5 (pinned by cso-classifier 3.1) has a broken `spacy download`
# command that emits an old `#egg=...` URL fragment that modern pip rejects.
# Install the matching wheel directly from the spacy-models GitHub release.
echo "[prepare_env] installing spaCy en_core_web_sm-3.0.0 wheel directly..."
pip install --quiet \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0-py3-none-any.whl"

# ---- NLTK data ----------------------------------------------------------
# nltk==3.6.2 (cso-classifier 3.1's pin) doesn't have punkt_tab — that's
# only an NLTK 3.9+ artifact. Download only what 3.6.2 actually ships.
echo "[prepare_env] downloading NLTK punkt / stopwords..."
python - <<'PY'
import nltk
for pkg in ("punkt", "stopwords"):
    nltk.download(pkg, quiet=True)
print("[prepare_env] nltk: ok")
PY

# ---- CSO warmup ---------------------------------------------------------
# Downloads ontology + Word2Vec model (~350 MB) to ~/.cso_classifier/
# Runs a 1-paper inference to confirm everything plays nicely.
echo "[prepare_env] warming CSO classifier..."
python - <<'PY'
from cso_classifier import CSOClassifier
clf = CSOClassifier(modules="both", enhancement="first", explanation=False)
out = clf.batch_run(papers={
    "warmup": {
        "title": "Neural Machine Translation with Attention",
        "abstract": "We propose a transformer-based approach to NMT with self-attention."
    }
}, workers=1)
labels = list((out.get("warmup") or {}).get("union") or [])
if not labels:
    raise SystemExit("CSO warmup returned no labels — something is wrong")
print(f"[prepare_env] cso warmup ok — sample labels: {labels[:5]}")
PY

cat <<EOF

[prepare_env] DONE.

Activate in any shell with:
    conda activate $PY_ENV

Submit the enrichment job:
    PROJECT_DIR=\$(pwd) \\
    AFFILIATIONS_DIR=/share/moritz.staudinger/<your-affiliations-path> \\
    PY_ENV=$PY_ENV \\
    sbatch scripts/enrich_topics.sbatch

EOF
