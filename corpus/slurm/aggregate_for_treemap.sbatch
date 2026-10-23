#!/usr/bin/env bash
#SBATCH --partition=GPU-a40                # any partition works; --gres omitted
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=aggregate_treemap
#SBATCH --output=/home/%u/logs/aggregate_treemap/slurm-%j.out
#
# Walk the bibmeta-enriched corpus once and emit three long-form CSVs
# for squarified treemap / group-by plotting (institutions × country,
# topics × country, institutions × topic). CPU-only — `--gres` is
# intentionally omitted so the job doesn't block on GPU availability.
#
# Args (positional, optional):
#   $1  corpus dir (default: /share/.../<...>_and_bibmeta)
#   $2  out dir    (default: reports/aggregates)
#   $3  year min   (default: unset = no lower bound)
#   $4  year max   (default: unset = no upper bound)
#
# Examples:
#   # all years
#   sbatch slurm/aggregate_for_treemap.sbatch
#
#   # 2020–2025 slice into a dedicated output dir
#   sbatch slurm/aggregate_for_treemap.sbatch \
#       /share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta \
#       reports/aggregates/2020_2025 \
#       2020 2025
#
# Expected wall time: ~10-20 min for 117k JSONs (pure JSON I/O).

set -euo pipefail

CORPUS_DIR="${1:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta}"
OUT_DIR="${2:-reports/aggregates}"
YEAR_MIN="${3:-}"
YEAR_MAX="${4:-}"

cd ~/acl_processor 2>/dev/null || cd ~/PycharmProjects/acl_processor
mkdir -p "$HOME/logs/aggregate_treemap" "$OUT_DIR"

unset VIRTUAL_ENV PYTHONHOME
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv/bin' | grep -v "${HOME}/\.venv" | paste -sd: -)
export PATH

if [ -d ~/miniconda3/envs/scico ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate scico
elif [ -d ~/miniconda3/envs/compare_eval ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate compare_eval
fi

echo "[agg_tm] corpus dir : $CORPUS_DIR"
echo "[agg_tm] out dir    : $OUT_DIR"
echo "[agg_tm] year min   : ${YEAR_MIN:-<unset>}"
echo "[agg_tm] year max   : ${YEAR_MAX:-<unset>}"
echo ""

YEAR_ARGS=()
[ -n "$YEAR_MIN" ] && YEAR_ARGS+=(--year-min "$YEAR_MIN")
[ -n "$YEAR_MAX" ] && YEAR_ARGS+=(--year-max "$YEAR_MAX")

python3 -u scripts/aggregate_for_treemap.py \
    --corpus-dir "$CORPUS_DIR" \
    --out-dir "$OUT_DIR" \
    "${YEAR_ARGS[@]}" \
    -v

echo ""
echo "[agg_tm] DONE. Files:"
ls -la "$OUT_DIR"/*.csv
