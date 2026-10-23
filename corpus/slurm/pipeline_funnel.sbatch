#!/usr/bin/env bash
#SBATCH --partition=GPU-a100
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --job-name=pipeline_funnel
#SBATCH --output=/home/%u/logs/pipeline_funnel/slurm-%j.out
#
# Walk every stage directory in the SciCo pipeline and report
# per-stage paper counts as a markdown funnel table. CPU-only —
# `--gres` is intentionally omitted so the job doesn't block on GPU
# availability. Pure os.listdir / globbing; finishes in minutes even
# across 100k+ files per stage.
#
# Edit the path overrides below if your layout differs.

set -euo pipefail

cd ~/acl_processor 2>/dev/null || cd ~/PycharmProjects/acl_processor
mkdir -p "$HOME/logs/pipeline_funnel" reports

unset VIRTUAL_ENV PYTHONHOME
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv/bin' | grep -v "${HOME}/\.venv" | paste -sd: -)
export PATH

if [ -d ~/miniconda3/envs/scico ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate scico
fi

ANTHOLOGY="${ANTHOLOGY:-/share/moritz.staudinger/acl-anthology}"
PDFS="${PDFS:-/share/moritz.staudinger/acl_pdfs}"
GROBID="${GROBID:-$HOME/data/acl_pdfs_parsed}"
PYMU="${PYMU:-/share/moritz.staudinger/acl_data/acl_pdfs_pymu_llm}"
HYBRID="${HYBRID:-$HOME/data/acl_pdfs_hybrid}"
GEMINI="${GEMINI:-/share/moritz.staudinger/acl_data/gemini_affiliations}"
ROR="${ROR:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90}"
TOPICS="${TOPICS:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics}"
FINAL="${FINAL:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta}"

echo "[funnel] anthology : $ANTHOLOGY"
echo "[funnel] pdfs      : $PDFS"
echo "[funnel] grobid    : $GROBID"
echo "[funnel] pymu      : $PYMU"
echo "[funnel] hybrid    : $HYBRID"
echo "[funnel] gemini    : $GEMINI"
echo "[funnel] ror       : $ROR"
echo "[funnel] topics    : $TOPICS"
echo "[funnel] final     : $FINAL"
echo ""

python3 -u scripts/pipeline_funnel.py \
    --anthology-dir "$ANTHOLOGY" \
    --pdfs-dir      "$PDFS" \
    --grobid-dir    "$GROBID" \
    --pymu-dir      "$PYMU" \
    --hybrid-dir    "$HYBRID" \
    --gemini-dir    "$GEMINI" \
    --ror-dir       "$ROR" \
    --topics-dir    "$TOPICS" \
    --final-dir     "$FINAL" \
    --out reports/pipeline_funnel.md \
    -v

echo ""
echo "[funnel] DONE. Report at reports/pipeline_funnel.md"
