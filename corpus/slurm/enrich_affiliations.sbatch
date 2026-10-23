#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=enrich_affs
#SBATCH --output=/home/%u/logs/enrich_affs-%j.out
#
# One-shot: enrich every affiliation JSON in
# /share/.../ror_parsed_affiliations_90_with_topics/affiliations/
# with bibliographic metadata (year, venue, title, doi) from the cloned
# ACL Anthology repo at /share/.../acl-anthology.
#
# Idempotent: rerunning skips files that already have `year`. After the
# enrichment finishes, rebuild the local ACL index via
# slurm/build_acl_index.sbatch — year filters in retrieve() and the
# load_affiliations_corpus path in eval_scico.py both start working.

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

AFFS_DIR="/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics"
OUT_DIR="/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta"
ACL_REPO_DATA="/share/moritz.staudinger/acl_data/acl-anthology/data"

mkdir -p "$OUT_DIR"

echo "[enrich] source affs : $AFFS_DIR ($(ls $AFFS_DIR | wc -l) files)"
echo "[enrich] output dir  : $OUT_DIR"
echo "[enrich] acl repo    : $ACL_REPO_DATA"
echo ""

python3 -u scripts/enrich_affiliations_with_acl_metadata.py \
    --affs-dir "$AFFS_DIR" \
    --out-dir "$OUT_DIR" \
    --acl-repo "$ACL_REPO_DATA" \
    -v

echo ""
echo "[enrich] done. Next step: rebuild the index."
echo "  cd ~/compare_acl && sbatch slurm/build_acl_index.sbatch"
