#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=aff_compare
#SBATCH --output=/home/%u/logs/aff_compare/slurm-%A_%x.out
#
# Note: 100% CPU/network-bound (LLM API + ROR lookups + JSON munging).
# --gres=gpu:0 explicitly releases the GPU so we don't waste an a40 slot.
# If your cluster has a CPU-only partition (e.g. `compute`, `cpu`), swap
# --partition=GPU-a40 for that — frees GPU nodes for GPU work.
#
# Runs ONE pipeline from configs/pipelines.yaml against the 781-paper gold
# set, writes predictions_<id>.jsonl to --out-dir, and STOPS without
# scoring. A separate aggregation job (submitted by aff_compare_submit_all.sh)
# scores everything together once the per-pipeline jobs finish.
#
# Args (positional):
#   $1  pipeline id (must exist in the YAML config)
#   $2  out_dir          (default: reports/pipeline_comparison/)
#   $3  config path      (default: configs/pipelines.yaml)
#   $4  gold csv         (default: reports/ground_truth_merged.csv)
#   $5  sources root     (default: /share/moritz.staudinger/acl_data)
#   $6  s2orc norm root  (default: /share/moritz.staudinger/acl_data/s2orc_affiliations)
#
# Submit directly:
#   sbatch slurm/aff_compare_one_pipeline.sbatch gpt4o_pymu5k
# or via the wrapper that submits every pipeline + the aggregator:
#   bash slurm/aff_compare_submit_all.sh

set -euo pipefail

PIPELINE_ID="${1:?pipeline id required (e.g. gpt4o_pymu5k)}"
OUT_DIR="${2:-reports/pipeline_comparison}"
CONFIG="${3:-configs/pipelines.yaml}"
GOLD_CSV="${4:-reports/ground_truth_merged.csv}"
SOURCES_ROOT="${5:-/share/moritz.staudinger/acl_data}"
S2ORC_NORM_ROOT="${6:-/share/moritz.staudinger/acl_data/s2orc_affiliations}"

cd ~/acl_processor
mkdir -p "$HOME/logs/aff_compare" "$OUT_DIR"

# Activate venv if present; fall back to whatever python is on PATH.
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    . .venv/bin/activate
fi

# Load .env (API keys for LLM backends + AQUEDUCT etc.)
if [ -f .env ]; then
    set -a; . .env; set +a
fi

echo "[aff_compare] pipeline       : $PIPELINE_ID"
echo "[aff_compare] out_dir        : $OUT_DIR"
echo "[aff_compare] config         : $CONFIG"
echo "[aff_compare] gold csv       : $GOLD_CSV"
echo "[aff_compare] sources_root   : $SOURCES_ROOT"
echo "[aff_compare] s2orc_norm_root: $S2ORC_NORM_ROOT"
echo "[aff_compare] python         : $(which python3)"
echo "[aff_compare] API keys set   : "
for k in OPENAI_API_KEY ANTHROPIC_API_KEY AQUEDUCT_API_KEY OPENROUTER_API_KEY; do
    [[ -n "${!k:-}" ]] && echo "    ✓ $k" || echo "    ✗ $k (not set — that pipeline will error if used)"
done

# --no-ror: extraction-only at the per-pipeline stage. ROR linking
# happens in one centralised serial pass during aggregation, so 24
# parallel pipelines don't contest the ROR API rate limit. Massive
# speed win at the cost of writing predictions without ror_id (they
# get filled in by ror_link_predictions.py before scoring).
python3 scripts/run_pipeline_comparison.py \
    --config "$CONFIG" \
    --pipelines "$PIPELINE_ID" \
    --gold-csv "$GOLD_CSV" \
    --sources-root "$SOURCES_ROOT" \
    --s2orc-norm-root "$S2ORC_NORM_ROOT" \
    --out-dir "$OUT_DIR" \
    --llm-workers 4 \
    --skip-score \
    --no-ror

echo "[aff_compare] done: $PIPELINE_ID"
