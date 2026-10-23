#!/usr/bin/env bash
#SBATCH --partition=GPU-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --job-name=lora_aff
#SBATCH --output=/home/%u/logs/lora_aff/slurm-%j.out
#
# LoRA fine-tune Qwen2.5-7B-Instruct on OpenAlex-derived affiliation
# extraction. GPU-only training phase. Run
# slurm/fetch_oa_training_data.sbatch FIRST to build the training JSONL
# (no GPU needed for that — separate so the GPU isn't held idle for the
# ~45 min OpenAlex fetch).
#
# Args (positional, optional):
#   $1  target rows         (default: 10000)  — only used if a fallback
#                                                in-job fetch is needed
#   $2  pymu_llm dir        (default: /share/.../acl_pdfs_pymu_llm)
#   $3  enriched affs dir   (default: /share/.../<...>_and_bibmeta)
#   $4  base model          (default: Qwen/Qwen2.5-7B-Instruct)
#   $5  out dir             (default: models/qwen25_7b_aff_lora)
#
# Resumable: HF trainer checkpoints in $OUT_DIR/ — re-submit and it
# picks up the next epoch.
#
# Expected wall time: ~4-5 h for 10k rows × 3 epochs on a single A100.

set -euo pipefail

TARGET_ROWS="${1:-10000}"
PYMU_DIR="${2:-/share/moritz.staudinger/acl_data/acl_pdfs_pymu_llm}"
ENRICHED_AFFS_DIR="${3:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta}"
BASE_MODEL="${4:-Qwen/Qwen2.5-7B-Instruct}"
OUT_DIR="${5:-models/qwen25_7b_aff_lora}"

cd ~/acl_processor 2>/dev/null || cd ~/PycharmProjects/acl_processor
mkdir -p "$HOME/logs/lora_aff" "$OUT_DIR" data/lora_train

unset VIRTUAL_ENV PYTHONHOME
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv/bin' | grep -v "${HOME}/\.venv" | paste -sd: -)
export PATH

if [ -d ~/miniconda3/envs/lora_train ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate lora_train
elif [ -d ~/miniconda3/envs/compare_eval ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate compare_eval
elif [ -f .venv/bin/activate ]; then
    . .venv/bin/activate
fi

if [ -f .env ]; then set -a; . .env; set +a; fi

echo "[lora_aff] target rows  : $TARGET_ROWS"
echo "[lora_aff] pymu_llm dir : $PYMU_DIR"
echo "[lora_aff] enriched affs: $ENRICHED_AFFS_DIR"
echo "[lora_aff] base model   : $BASE_MODEL"
echo "[lora_aff] out dir      : $OUT_DIR"
echo "[lora_aff] python       : $(which python3)"
echo "[lora_aff] gpu          : $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo ""

# ---- Phase 1: data prep ----
if [ ! -s data/lora_train/train.jsonl ]; then
    if [ ! -d "$PYMU_DIR" ]; then
        echo "ERROR: $PYMU_DIR not found." >&2
        exit 2
    fi
    if [ ! -d "$ENRICHED_AFFS_DIR" ]; then
        echo "ERROR: $ENRICHED_AFFS_DIR not found. Run enrich_affiliations first." >&2
        exit 2
    fi
    echo "[lora_aff] Phase 1: building training data ..."
    python3 -u scripts/fetch_oa_training.py \
        --pymu-dir "$PYMU_DIR" \
        --enriched-affs-dir "$ENRICHED_AFFS_DIR" \
        --gold-csv reports/ground_truth_merged.csv \
        --target-rows "$TARGET_ROWS" \
        --mailto "${OPENALEX_MAILTO:-}" \
        --out-dir data/lora_train \
        -v
else
    echo "[lora_aff] data/lora_train/train.jsonl already exists, skipping prep"
    wc -l data/lora_train/train.jsonl data/lora_train/val.jsonl
fi

# ---- Phase 2: training ----
echo ""
echo "[lora_aff] Phase 2: training LoRA adapter..."
python3 -u scripts/train_lora_affiliation.py \
    --train data/lora_train/train.jsonl \
    --val data/lora_train/val.jsonl \
    --base-model "$BASE_MODEL" \
    --out-dir "$OUT_DIR"

echo ""
echo "[lora_aff] DONE. Adapter saved to $OUT_DIR/adapter"
ls -la "$OUT_DIR/adapter"
