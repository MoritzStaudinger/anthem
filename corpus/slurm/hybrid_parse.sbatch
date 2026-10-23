#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=23:59:00
#SBATCH --job-name=hybrid_parse
#SBATCH --output=/home/%u/logs/full_pipeline/hybrid-%j.out
#
# Phase 1c — Hybrid PDF parse (PyMu + OCR fallback) → raw text/jsonl.
# CPU-bound; OCR fallback uses Tesseract if installed on the node.
#
# Args (positional, optional):
#   $1  input dir   (default: /share/.../acl_pdfs)
#   $2  output dir  (default: /share/.../acl_pdfs_hybrid)
#   $3  workers     (default: 16)
#   $4  mode        (default: md) [md|jsonl]

set -euo pipefail

INPUT_DIR="${1:-/share/moritz.staudinger/acl_data/acl_pdfs}"
OUTPUT_DIR="${2:-/share/moritz.staudinger/acl_data/acl_pdfs_hybrid}"
WORKERS="${3:-16}"
MODE="${4:-md}"

cd ~/acl_processor 2>/dev/null || cd ~/PycharmProjects/acl_processor
mkdir -p "$HOME/logs/full_pipeline" "$OUTPUT_DIR"

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

echo "[hybrid] input_dir  : $INPUT_DIR"
echo "[hybrid] output_dir : $OUTPUT_DIR"
echo "[hybrid] workers    : $WORKERS"
echo "[hybrid] mode       : $MODE"
echo ""

python3 -u scripts/hybrid_parse_pdfs.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_workers "$WORKERS" \
    --mode "$MODE"

echo ""
echo "[hybrid] DONE. Output in: $OUTPUT_DIR"
find "$OUTPUT_DIR" -name "*.${MODE}" | wc -l
