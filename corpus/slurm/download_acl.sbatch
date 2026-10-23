#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --job-name=acl_download
#SBATCH --output=/home/%u/logs/full_pipeline/download-%j.out
#
# Phase 0 of the full pipeline: download PDFs from the ACL Anthology.
# Reads paper URLs by walking the cloned acl-anthology repo's
# data/xml/ directory. Idempotent — skips PDFs already present in
# --output_dir.
#
# Args (positional, optional):
#   $1  output dir        (default: /share/.../acl_pdfs)
#   $2  anthology repo    (default: /share/.../acl-anthology)
#   $3  max_workers       (default: 32)

set -euo pipefail

OUTPUT_DIR="${1:-/share/moritz.staudinger/acl_data/acl_pdfs}"
ANTHOLOGY_REPO="${2:-/share/moritz.staudinger/acl_data/acl-anthology}"
MAX_WORKERS="${3:-32}"

cd ~/acl_processor 2>/dev/null || cd ~/PycharmProjects/acl_processor
mkdir -p "$HOME/logs/full_pipeline" "$OUTPUT_DIR" logs

unset VIRTUAL_ENV PYTHONHOME
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv/bin' | grep -v "${HOME}/\.venv" | paste -sd: -)
export PATH

if [ -d ~/miniconda3/envs/compare_eval ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate compare_eval
elif [ -d ~/miniconda3/envs/scico ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate scico
fi

echo "[download] output_dir     : $OUTPUT_DIR"
echo "[download] anthology_repo : $ANTHOLOGY_REPO"
echo "[download] max_workers    : $MAX_WORKERS"
echo ""

python3 -u scripts/acl_downloader.py \
    --output_dir "$OUTPUT_DIR" \
    --anthology_repo "$ANTHOLOGY_REPO" \
    --max_workers "$MAX_WORKERS"

echo ""
echo "[download] DONE. PDFs in: $OUTPUT_DIR"
find "$OUTPUT_DIR" -name '*.pdf' | wc -l
