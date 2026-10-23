#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --job-name=fetch_api_affs
#SBATCH --output=/home/%u/logs/fetch_api_affs-%j.out
#
# Fetch per-paper, per-author affiliations from the two graph APIs we
# care about, for every paper in the 781-paper gold set:
#   - OpenAlex   → predictions_openalex.jsonl
#   - S2 graph   → predictions_s2_graph.jsonl
#
# Both join the aff-comparison leaderboard as new pipeline rows once
# the aggregate sbatch (aff_compare_aggregate.sbatch) is re-run.
#
# These directly evaluate the affiliation quality of the two upstream
# data sources we depend on:
#   - OpenAlex is the LoRA training gold; if its predictions score
#     poorly against our gold, the LoRA can't be expected to do better.
#   - S2 graph API is the data S2's website actually serves (different
#     from the bulk s2orc dataset which mostly lacks affiliations).

set -euo pipefail

cd ~/acl_processor 2>/dev/null || cd ~/PycharmProjects/acl_processor
mkdir -p "$HOME/logs" reports/pipeline_comparison

unset VIRTUAL_ENV PYTHONHOME
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv/bin' | paste -sd: -)
export PATH

if [ -d ~/miniconda3/envs/compare_eval ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate compare_eval
elif [ -d ~/miniconda3/envs/scico ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate scico
fi
if [ -f .env ]; then set -a; . .env; set +a; fi

echo "============================================================"
echo "Step 1/2 — OpenAlex per-paper authorships for 781 gold papers"
echo "============================================================"
python3 -u scripts/fetch_openalex_affiliations.py \
    --gold-csv reports/ground_truth_merged.csv \
    --enriched-affs-dir /share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta \
    --out reports/pipeline_comparison/predictions_openalex.jsonl \
    --rate-limit-hz 10 \
    -v

echo ""
echo "============================================================"
echo "Step 2/2 — S2 graph API per-paper authorships for 781 papers"
echo "============================================================"
python3 -u scripts/fetch_s2_graph_affiliations.py \
    --gold-csv reports/ground_truth_merged.csv \
    --out reports/pipeline_comparison/predictions_s2_graph.jsonl \
    --rate-limit-hz 10 \
    -v

echo ""
echo "[fetch_api] DONE. Both predictions JSONLs written:"
ls -la reports/pipeline_comparison/predictions_openalex.jsonl \
       reports/pipeline_comparison/predictions_s2_graph.jsonl
echo ""
echo "Next: sbatch slurm/aff_compare_aggregate.sbatch"
echo "      (re-links + re-scores the whole leaderboard including the"
echo "      new openalex and s2_graph rows)"
