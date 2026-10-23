#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=aff_compare_agg
#SBATCH --output=/home/%u/logs/aff_compare/slurm-agg-%j.out
#
# Aggregation pass: reads all predictions_*.jsonl in OUT_DIR (produced by
# per-pipeline jobs), scores every pipeline against gold across all 4
# tracks (raw / ror / ror_hier / name), writes the comparison report +
# per-decade CSV + cross-pipeline merge. Submit with `--dependency=...`
# so it only runs after every per-pipeline job completes.
#
# Args (positional, same defaults as aff_compare_one_pipeline.sbatch):
#   $1  out_dir          (default: reports/pipeline_comparison/)
#   $2  config           (default: configs/pipelines.yaml)
#   $3  gold csv         (default: reports/ground_truth_merged.csv)
#   $4  sources root     (default: /share/moritz.staudinger/acl_data)
#   $5  s2orc norm root  (default: /share/moritz.staudinger/acl_data/s2orc_affiliations)

set -euo pipefail

OUT_DIR="${1:-reports/pipeline_comparison}"
CONFIG="${2:-configs/pipelines.yaml}"
GOLD_CSV="${3:-reports/ground_truth_merged.csv}"
SOURCES_ROOT="${4:-/share/moritz.staudinger/acl_data}"
S2ORC_NORM_ROOT="${5:-/share/moritz.staudinger/acl_data/s2orc_affiliations}"

cd ~/acl_processor
mkdir -p "$HOME/logs/aff_compare"
if [ -f .venv/bin/activate ]; then . .venv/bin/activate; fi
if [ -f .env ]; then set -a; . .env; set +a; fi

echo "[agg] step 1/3 — ROR-linking unique affiliation strings (serial) ..."
# Per-pipeline jobs ran with --no-ror; their predictions have raw
# affiliation strings but no ror_id yet. One serial pass de-duplicates
# across all pipelines and resolves through the shared cache.
python3 scripts/ror_link_predictions.py \
    --predictions-dir "$OUT_DIR" \
    --ror-cache cache/ror_lookup_cache.json \
    --hierarchy-cache cache/ror_hierarchy_cache.json

echo ""
echo "[agg] step 2/3 — scoring all pipelines (4 tracks) ..."
python3 scripts/run_pipeline_comparison.py \
    --config "$CONFIG" \
    --gold-csv "$GOLD_CSV" \
    --sources-root "$SOURCES_ROOT" \
    --s2orc-norm-root "$S2ORC_NORM_ROOT" \
    --out-dir "$OUT_DIR" \
    --score-from-dir "$OUT_DIR"

echo ""
echo "[agg] step 3/3 — building cross-pipeline per-paper merge view ..."
python3 scripts/merge_pipeline_predictions_per_paper.py \
    --predictions-dir "$OUT_DIR" \
    --gold-csv "$GOLD_CSV" \
    --out "$OUT_DIR/merged_per_paper.jsonl"

echo ""
echo "[agg] done. Artifacts in $OUT_DIR:"
ls -la "$OUT_DIR"
