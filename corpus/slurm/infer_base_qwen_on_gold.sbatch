#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --job-name=qwen_base_infer_gold
#SBATCH --output=/home/%u/logs/lora_aff/base-infer-%j.out
#
# Run the BASE Qwen2.5-7B-Instruct (no LoRA adapter) on the 781-paper
# gold set as the zero-shot control row for the LoRA ablation. Writes
# leaderboard-schema predictions so the row joins the aff-comparison
# table next to `ours`, `lora_qwen25_7b_aff`, etc.
#
# Args (positional, optional):
#   $1  out file      (default: reports/pipeline_comparison/
#                                predictions_qwen25_7b_instruct.jsonl)
#   $2  base model    (default: Qwen/Qwen2.5-7B-Instruct)
#   $3  pymu_llm dir  (default: /share/.../acl_pdfs_pymu_llm)
#   $4  pipeline id   (default: qwen25_7b_instruct)
#
# Submit with:
#   sbatch slurm/infer_base_qwen_on_gold.sbatch
#
# After it finishes:
#   sbatch slurm/aff_compare_aggregate.sbatch    # re-link + re-score all
#                                                # pipelines incl. the new
#                                                # qwen25_7b_instruct row

set -euo pipefail

OUT="${1:-reports/pipeline_comparison/predictions_qwen25_7b_instruct.jsonl}"
BASE_MODEL="${2:-Qwen/Qwen2.5-7B-Instruct}"
PYMU_DIR="${3:-/share/moritz.staudinger/acl_data/acl_pdfs_pymu_llm}"
PIPELINE_ID="${4:-qwen25_7b_instruct}"

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

export HF_HOME="${HF_HOME:-/share/moritz.staudinger/hf_cache}"
mkdir -p "$HF_HOME"

echo "[base_infer] out file     : $OUT"
echo "[base_infer] base model   : $BASE_MODEL"
echo "[base_infer] pymu_llm dir : $PYMU_DIR"
echo "[base_infer] pipeline id  : $PIPELINE_ID"
echo "[base_infer] gpu          : $(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "[base_infer] hf cache     : $HF_HOME"
echo ""

python3 -u scripts/infer_lora_on_gold.py \
    --gold-csv reports/ground_truth_merged.csv \
    --pymu-dir "$PYMU_DIR" \
    --no-adapter \
    --base-model "$BASE_MODEL" \
    --pipeline-id "$PIPELINE_ID" \
    --out "$OUT" \
    -v

echo ""
echo "[base_infer] DONE. wrote: $OUT"
wc -l "$OUT"
echo ""
echo "Next:  sbatch slurm/aff_compare_aggregate.sbatch"
echo "       (re-runs ROR linking + scoring across all pipelines,"
echo "        including the new $PIPELINE_ID row)"
