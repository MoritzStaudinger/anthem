#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --job-name=fetch_oa_train
#SBATCH --output=/home/%u/logs/lora_aff/fetch-%j.out
#
# Phase 1 (no GPU): build the LoRA training dataset from OpenAlex
# authorships + local pymu_llm headers. Writes data/lora_train/
# {train,val}.jsonl. After this finishes, submit
# slurm/train_lora_affiliation.sbatch for the GPU training pass.
#
# Args (positional, optional):
#   $1  target rows       (default: 10000)
#   $2  pymu_llm dir      (default: /share/.../acl_pdfs_pymu_llm)
#   $3  enriched affs dir (default: /share/.../<...>_and_bibmeta)
#
# Resumable via reports/.../.oa_cache.json — re-run picks up where the
# OA lookups left off.

set -euo pipefail

TARGET_ROWS="${1:-10000}"
PYMU_DIR="${2:-/share/moritz.staudinger/acl_data/acl_pdfs_pymu_llm}"
ENRICHED_AFFS_DIR="${3:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta}"

cd ~/acl_processor 2>/dev/null || cd ~/PycharmProjects/acl_processor
mkdir -p "$HOME/logs/lora_aff" data/lora_train

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
if [ -f .env ]; then set -a; . .env; set +a; fi

echo "[fetch_oa] target rows  : $TARGET_ROWS"
echo "[fetch_oa] pymu_llm dir : $PYMU_DIR"
echo "[fetch_oa] enriched affs: $ENRICHED_AFFS_DIR"
echo ""

python3 -u scripts/fetch_oa_training.py \
    --pymu-dir "$PYMU_DIR" \
    --enriched-affs-dir "$ENRICHED_AFFS_DIR" \
    --gold-csv reports/ground_truth_merged.csv \
    --target-rows "$TARGET_ROWS" \
    --mailto "${OPENALEX_MAILTO:-}" \
    --out-dir data/lora_train \
    -v

echo ""
echo "[fetch_oa] DONE."
echo "Next:  sbatch slurm/train_lora_affiliation.sbatch"
wc -l data/lora_train/train.jsonl data/lora_train/val.jsonl 2>/dev/null || true
