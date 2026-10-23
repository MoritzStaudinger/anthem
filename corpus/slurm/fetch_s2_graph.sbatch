#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=s2_graph_affs
#SBATCH --output=/home/%u/logs/s2_graph_affs-%j.out
#
# Fetch per-paper, per-author affiliations from the S2 graph API for
# every paper in the gold set. Writes a predictions_s2_graph.jsonl
# that joins the aff-compare leaderboard as a new pipeline row.
#
# ~15 min wall with S2_API_KEY set (10 RPS), ~13 min unauthenticated (1 RPS).

set -euo pipefail

cd ~/acl_processor 2>/dev/null || cd ~/PycharmProjects/acl_processor
mkdir -p "$HOME/logs"

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

python3 -u scripts/fetch_s2_graph_affiliations.py \
    --gold-csv reports/ground_truth_merged.csv \
    --out reports/pipeline_comparison/predictions_s2_graph.jsonl \
    --rate-limit-hz 10 \
    -v

echo ""
echo "[s2_graph] done. To fold into the leaderboard:"
echo "  sbatch slurm/aff_compare_aggregate.sbatch"
