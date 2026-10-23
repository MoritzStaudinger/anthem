#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=build_acl_index
#SBATCH --output=/home/%u/logs/build_acl_index/slurm-%j.out
#
# One-off build of the local ACL/SciCo abstracts index used by
# search_local_acl_papers. Resumable: re-submit on failure and it picks
# up from the last checkpoint in <persist-dir>/processed.txt.
#
# Embedding model: sentence-transformers/all-MiniLM-L6-v2 (CPU is fine
# in a pinch, but GPU brings ~5-15 min total wall time vs ~60-90 on CPU).
#
# Args (positional, override defaults):
#   $1  scico root dir  (default: /share/.../scico_root_topics)
#   $2  mode            (default: abstracts)         [abstracts|fulltext]
#   $3  batch size      (default: 500)
#   $4  persist dir     (default: data/local_index_acl)
#
# Smoke pattern:
#   First run can be tested with a smaller --batch-size and Ctrl-C after
#   1-2 checkpoints to confirm the resume path works.

set -euo pipefail

SCICO_ROOT="${1:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta}"
MODE="${2:-abstracts}"
# 10000 papers per checkpoint. Persist of the full 5 GB vector_store
# scales ~linearly with index size and dominates wall time. Going from
# batch=500 (242 persists) to batch=10000 (12 persists) cuts ~25 min
# of pure I/O. Trade-off: a mid-build crash loses up to 10000 papers
# (resumable but you re-do that batch).
BATCH_SIZE="${3:-10000}"
PERSIST_DIR="${4:-data/local_index_acl}"

cd ~/compare_acl 2>/dev/null || cd ~/PycharmProjects/Comparer_flask 2>/dev/null || cd ~/Comparer_flask
mkdir -p "$HOME/logs/build_acl_index"

# Strip inherited venv (same reason as eval_scico.sbatch — prevents the
# wrong python ending up first on PATH).
unset VIRTUAL_ENV PYTHONHOME
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv/bin' | grep -v "${HOME}/\.venv" | paste -sd: -)
export PATH

if [ -d ~/miniconda3/envs/compare_eval ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate compare_eval
elif [ -d ~/miniconda3/envs/scico ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate scico
elif [ -f .venv/bin/activate ]; then
    . .venv/bin/activate
fi

if [ -f .env ]; then set -a; . .env; set +a; fi

echo "[build_acl_index] scico root  : $SCICO_ROOT"
echo "[build_acl_index] mode        : $MODE"
echo "[build_acl_index] batch size  : $BATCH_SIZE papers"
echo "[build_acl_index] persist dir : $PERSIST_DIR"
echo "[build_acl_index] python      : $(which python3)"
echo "[build_acl_index] gpu         : $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo ""

python3 -u scripts/build_acl_index.py \
    --scico-root "$SCICO_ROOT" \
    --mode "$MODE" \
    --persist-dir "$PERSIST_DIR" \
    --resume \
    --batch-size "$BATCH_SIZE" \
    --no-checkpoint

echo ""
echo "[build_acl_index] DONE. Persisted to $PERSIST_DIR"
ls -la "$PERSIST_DIR" | head -10
