#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --job-name=ror_v2
#SBATCH --output=/home/%u/logs/extract_aff/rorv2-%j.out
#
# Phase 2 of "add new papers to the affiliations corpus":
#   - reads per-paper extraction JSONs from --in-dir (output of
#     slurm/extract_affiliations_gemini.sbatch)
#   - scans for unique raw_affiliation strings, queries ROR v2 once per
#     unique string (cached), MIN_AFF_SCORE = 0.90 by default
#   - writes ROR-linked JSONs (ror_id, clean_institution_name, country,
#     country_code, ror_score) to --out-dir
#
# CPU-only — ROR linking is HTTP-bound.
#
# After this, phase 3 = scripts/enrich_topics.sbatch with
#   AFFILIATIONS_DIR=$OUT_DIR  OUT_DIR=..._with_topics
#
# Args (positional, optional):
#   $1  in-dir       (default: /share/.../gemini_affiliations_NEW)
#   $2  out-dir      (default: /share/.../ror_parsed_affiliations_NEW_90)
#   $3  min-score    (default: 0.90)

set -euo pipefail

IN_DIR="${1:-/share/moritz.staudinger/acl_data/gemini_affiliations_NEW}"
OUT_DIR="${2:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_NEW_90}"
MIN_SCORE="${3:-0.90}"

cd ~/acl_processor 2>/dev/null || cd ~/PycharmProjects/acl_processor
mkdir -p "$HOME/logs/extract_aff" "$OUT_DIR" cache

unset VIRTUAL_ENV PYTHONHOME
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv/bin' | grep -v "${HOME}/\.venv" | paste -sd: -)
export PATH

if [ -d ~/miniconda3/envs/compare_eval ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate compare_eval
elif [ -d ~/miniconda3/envs/scico ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate scico
fi
if [ -f .env ]; then set -a; . .env; set +a; fi

echo "[rorv2] in_dir    : $IN_DIR"
echo "[rorv2] out_dir   : $OUT_DIR"
echo "[rorv2] min_score : $MIN_SCORE"
echo "[rorv2] cache     : cache/ror_v2_mapping_cache.json"
echo ""

if [ ! -d "$IN_DIR" ]; then
    echo "ERROR: $IN_DIR not found. Did phase 1 finish?" >&2
    echo "Expected output of slurm/extract_affiliations_gemini.sbatch." >&2
    exit 2
fi

python3 -u scripts/ror_link_v2.py \
    --in-dir  "$IN_DIR" \
    --out-dir "$OUT_DIR" \
    --mapping-file cache/ror_v2_mapping_cache.json \
    --min-score "$MIN_SCORE" \
    -v

echo ""
echo "[rorv2] DONE. ROR-linked JSONs in: $OUT_DIR"
ls "$OUT_DIR" | wc -l
echo ""
echo "Next (CSO topic classification):"
echo "  AFFILIATIONS_DIR=$OUT_DIR OUT_DIR=${OUT_DIR}_with_topics \\"
echo "    sbatch scripts/enrich_topics.sbatch"
