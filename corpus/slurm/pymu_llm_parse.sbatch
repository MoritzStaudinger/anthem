#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=pymu_parse
#SBATCH --output=/home/%u/logs/full_pipeline/pymu-%j.out
#
# Phase 1b — PyMu_LLM parse PDFs → Markdown. CPU-bound.
# Idempotent (skips existing outputs).
#
# Args (positional, optional):
#   $1  input dir   (default: /share/.../acl_pdfs)
#   $2  output dir  (default: /share/.../acl_pdfs_pymu_llm)
#   $3  workers     (default: 16)

set -euo pipefail

INPUT_DIR="${1:-/share/moritz.staudinger/acl_data/acl_pdfs}"
OUTPUT_DIR="${2:-/share/moritz.staudinger/acl_data/acl_pdfs_pymu_llm}"
WORKERS="${3:-16}"

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

echo "[pymu] input_dir  : $INPUT_DIR"
echo "[pymu] output_dir : $OUTPUT_DIR"
echo "[pymu] workers    : $WORKERS"
echo ""

python3 -u scripts/pymu_llm_parser.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --max_workers "$WORKERS"

echo ""
echo "[pymu] DONE. Markdown in: $OUTPUT_DIR"
find "$OUTPUT_DIR" -name '*.md' | wc -l
