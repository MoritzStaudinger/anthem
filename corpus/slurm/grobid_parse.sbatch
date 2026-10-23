#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --time=23:59:00
#SBATCH --job-name=grobid_parse
#SBATCH --output=/home/%u/logs/full_pipeline/grobid-%j.out
#
# Phase 1a — GROBID parse PDFs → TEI XML. CPU-bound, no GPU.
# Requires GROBID server running (the script orchestrates it).
#
# Args (positional, optional):
#   $1  input dir   (default: /share/.../acl_pdfs)
#   $2  output dir  (default: /share/.../acl_pdfs_parsed)
#   $3  workers     (default: 32)

set -euo pipefail

INPUT_DIR="${1:-/share/moritz.staudinger/acl_data/acl_pdfs}"
OUTPUT_DIR="${2:-/share/moritz.staudinger/acl_data/acl_pdfs_parsed}"
WORKERS="${3:-32}"

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

echo "[grobid] input_dir  : $INPUT_DIR"
echo "[grobid] output_dir : $OUTPUT_DIR"
echo "[grobid] workers    : $WORKERS"
echo ""

python3 -u scripts/grobid_parser.py "$INPUT_DIR" "$OUTPUT_DIR" --workers "$WORKERS"

echo ""
echo "[grobid] DONE. TEI XML in: $OUTPUT_DIR"
find "$OUTPUT_DIR" -name '*.xml' | wc -l
