#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=corpus_stats
#SBATCH --output=/home/%u/logs/corpus_stats-%j.out
#
# Compute corpus statistics over the bibmeta-enriched per-paper JSONs
# and write a markdown report. Suitable for inclusion in the paper.
#
# Args (positional, optional):
#   $1  corpus-dir  (default: /share/.../ror_parsed_..._and_bibmeta)
#   $2  out-md      (default: reports/corpus_stats.md)

set -euo pipefail

CORPUS_DIR="${1:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta}"
OUT_MD="${2:-reports/corpus_stats.md}"

cd ~/acl_processor 2>/dev/null || cd ~/PycharmProjects/acl_processor
mkdir -p "$HOME/logs" reports

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

echo "[corpus_stats] corpus_dir : $CORPUS_DIR"
echo "[corpus_stats] out_md     : $OUT_MD"
echo ""

python3 -u scripts/corpus_stats.py \
    --corpus-dir "$CORPUS_DIR" \
    --out-md "$OUT_MD" \
    --out-json "${OUT_MD%.md}.json" \
    -v

echo ""
echo "[corpus_stats] DONE."
ls -la "$OUT_MD" "${OUT_MD%.md}.json"
