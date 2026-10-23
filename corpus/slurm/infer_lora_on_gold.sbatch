#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --job-name=lora_infer_gold
#SBATCH --output=/home/%u/logs/lora_aff/infer-%j.out
#
# Run the trained LoRA adapter on the 781-paper gold set and write
# leaderboard-schema predictions so the row joins the aff-comparison
# table next to `ours`, `openalex`, etc.
#
# Args (positional, optional):
#   $1  adapter dir   (default: models/qwen25_7b_aff_lora/adapter)
#   $2  out file      (default: reports/pipeline_comparison/
#                                predictions_lora_qwen25_7b_aff.jsonl)
#   $3  base model    (default: Qwen/Qwen2.5-7B-Instruct)
#   $4  pymu_llm dir  (default: /share/.../acl_pdfs_pymu_llm)
#
# Expected wall time: 781 papers × ~3-6 s/paper on a single A40 in
# 4-bit ≈ 40-80 min. Add ~3 min for model load.
#
# After it finishes:
#   1) sbatch slurm/aff_compare_aggregate.sbatch    # re-link + re-score
#                                                   # all pipelines incl.
#                                                   # the new LoRA row

set -euo pipefail

ADAPTER="${1:-models/qwen25_7b_aff_lora/adapter}"
OUT="${2:-reports/pipeline_comparison/predictions_lora_qwen25_7b_aff.jsonl}"
BASE_MODEL="${3:-Qwen/Qwen2.5-7B-Instruct}"
PYMU_DIR="${4:-/share/moritz.staudinger/acl_data/acl_pdfs_pymu_llm}"

cd ~/acl_processor 2>/dev/null || cd ~/PycharmProjects/acl_processor
mkdir -p "$HOME/logs/lora_aff" "$(dirname "$OUT")"

unset VIRTUAL_ENV PYTHONHOME
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv/bin' | grep -v "${HOME}/\.venv" | paste -sd: -)
export PATH

if [ -d ~/miniconda3/envs/lora_train ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate lora_train
elif [ -d ~/miniconda3/envs/compare_eval ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate compare_eval
fi

if [ -f .env ]; then set -a; . .env; set +a; fi

# HF cache on /share so each job doesn't re-download the 14 GB base
# weights — same convention as the training sbatch.
export HF_HOME="${HF_HOME:-/share/moritz.staudinger/hf_cache}"
mkdir -p "$HF_HOME"

echo "[infer_lora] adapter      : $ADAPTER"
echo "[infer_lora] out file     : $OUT"
echo "[infer_lora] base model   : $BASE_MODEL"
echo "[infer_lora] pymu_llm dir : $PYMU_DIR"
echo "[infer_lora] gpu          : $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "[infer_lora] hf cache     : $HF_HOME"
echo ""

if [ ! -d "$ADAPTER" ]; then
    echo "ERROR: $ADAPTER not found. Is training finished?" >&2
    echo "Check the training job's log; the final line should be:" >&2
    echo "  [lora_aff] DONE. Adapter saved to models/qwen25_7b_aff_lora/adapter" >&2
    exit 2
fi

python3 -u scripts/infer_lora_on_gold.py \
    --gold-csv reports/ground_truth_merged.csv \
    --pymu-dir "$PYMU_DIR" \
    --adapter "$ADAPTER" \
    --base-model "$BASE_MODEL" \
    --out "$OUT" \
    -v

echo ""
echo "[infer_lora] DONE. wrote: $OUT"
wc -l "$OUT"
echo ""
echo "Next:  sbatch slurm/aff_compare_aggregate.sbatch"
echo "       (re-runs ROR linking + scoring across all pipelines,"
echo "        including the new lora_qwen25_7b_aff row)"
