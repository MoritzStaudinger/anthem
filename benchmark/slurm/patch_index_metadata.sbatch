#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=patch_idx
#SBATCH --output=/home/%u/logs/patch_idx-%j.out
#
# In-place patch the existing ACL vector index with year/venue/title/doi
# /url/language fields from the enriched per-paper JSONs. No re-
# embedding, no model load on GPU. ~3-5 min.
#
# Prereq: scripts/enrich_affiliations_with_acl_metadata.py was run and
# produced an enriched directory at
#   /share/.../ror_parsed_affiliations_90_with_topics_and_bibmeta
#
# Args (positional, optional):
#   $1  index dir       (default: data/local_index_acl)
#   $2  enriched dir    (default: /share/.../<...>_and_bibmeta)

set -euo pipefail

INDEX_DIR="${1:-data/local_index_acl}"
ENRICHED_DIR="${2:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta}"

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

# Stage index to /dev/shm so the in-place patch reads and writes fast.
# We then sync the patched dir back to the persistent location after.
SRC_INDEX="$INDEX_DIR"
STAGED="/dev/shm/acl_index_${SLURM_JOB_ID:-$$}_$USER"
if [ -d "$SRC_INDEX" ] && [ -w /dev/shm ]; then
    echo "[patch] staging index $SRC_INDEX -> $STAGED ..."
    cp -r "$SRC_INDEX" "$STAGED"
    export ACL_INDEX_DIR="$STAGED"
    SYNC_BACK=1
    trap 'rm -rf "$STAGED" 2>/dev/null || true' EXIT
else
    SYNC_BACK=0
fi

PATCH_TARGET="${ACL_INDEX_DIR:-$INDEX_DIR}"
echo "[patch] index dir   : $PATCH_TARGET"
echo "[patch] enriched dir: $ENRICHED_DIR"

python3 -u scripts/patch_index_metadata.py \
    --index-dir "$PATCH_TARGET" \
    --enriched-dir "$ENRICHED_DIR" \
    -v

# Sync ONLY docstore.json back to NFS — the vector store / index store
# weren't modified by this patch, so rsync them is wasted I/O. Plain
# `cp` is fine because we're only writing one file atomically.
if [ "$SYNC_BACK" = "1" ] && [ "$PATCH_TARGET" != "$SRC_INDEX" ]; then
    echo "[patch] copying patched docstore back to $SRC_INDEX ..."
    t0=$(date +%s)
    cp "$PATCH_TARGET/docstore.json" "$SRC_INDEX/docstore.json.tmp"
    mv "$SRC_INDEX/docstore.json.tmp" "$SRC_INDEX/docstore.json"
    echo "[patch] copied in $(($(date +%s) - t0))s"
fi

echo "[patch] done."
