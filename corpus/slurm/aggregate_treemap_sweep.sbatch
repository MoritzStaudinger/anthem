#!/usr/bin/env bash
#SBATCH --partition=GPU-a100               # any partition works; --gres omitted
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --job-name=agg_tm_sweep
#SBATCH --output=/home/%u/logs/aggregate_treemap/sweep-%j.out
#
# Run the corpus → long-form CSV aggregation four times in one job:
#   1) full corpus           → reports/aggregates/all_years/
#   2) 2010–2014             → reports/aggregates/2010_2014/
#   3) 2015–2019             → reports/aggregates/2015_2019/
#   4) 2020–2025             → reports/aggregates/2020_2025/
#
# Sequential because the aggregator is single-threaded (~15 min per
# pass over 117k JSONs); whole sweep ~1 h wall.
#
# Args (positional, optional):
#   $1  corpus dir (default: /share/.../<...>_and_bibmeta)
#   $2  out root   (default: reports/aggregates)

set -euo pipefail

CORPUS_DIR="${1:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta}"
OUT_ROOT="${2:-reports/aggregates}"

cd ~/acl_processor 2>/dev/null || cd ~/PycharmProjects/acl_processor
mkdir -p "$HOME/logs/aggregate_treemap" "$OUT_ROOT"

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

echo "[sweep] corpus dir : $CORPUS_DIR"
echo "[sweep] out root   : $OUT_ROOT"
echo ""

run_one() {
    local label="$1"
    local out="$OUT_ROOT/$label"
    shift
    echo "==============================================================="
    echo "[sweep] $label  →  $out"
    echo "==============================================================="
    mkdir -p "$out"
    python3 -u scripts/aggregate_for_treemap.py \
        --corpus-dir "$CORPUS_DIR" \
        --out-dir "$out" \
        "$@" \
        -v
    echo ""
}

run_one "all_years"
run_one "2010_2014" --year-min 2010 --year-max 2014
run_one "2015_2019" --year-min 2015 --year-max 2019
run_one "2020_2025" --year-min 2020 --year-max 2025

echo "[sweep] DONE. Outputs:"
ls -la "$OUT_ROOT"/*/*.csv
