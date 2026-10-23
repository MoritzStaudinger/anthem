#!/usr/bin/env bash
#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=evalsc_agg
#SBATCH --output=/home/%u/logs/eval_scico/slurm-agg-%j.out
#
# Aggregates per-agent eval reports produced by eval_scico_matrix.sh
# into a single cross-agent comparison table.
#
# Each per-agent run writes a directory under reports/eval_scico/<run_tag>/
# containing the JSON/MD report from eval_scico.py. This script reads all
# of them and emits a flat CSV + markdown summary.
#
# Args:
#   $1  out_root (default: most-recent dir under reports/eval_scico/)

set -euo pipefail

OUT_ROOT="${1:-}"
cd ~/compare_acl 2>/dev/null || cd ~/PycharmProjects/Comparer_flask 2>/dev/null || cd ~/Comparer_flask
mkdir -p "$HOME/logs/eval_scico"

if [ -z "$OUT_ROOT" ]; then
    OUT_ROOT=$(ls -td reports/eval_scico/matrix_* 2>/dev/null | head -1)
    if [ -z "$OUT_ROOT" ]; then
        echo "no out_root passed and no matrix_* dirs found in reports/eval_scico/" >&2
        exit 1
    fi
    echo "[agg] auto-detected most recent matrix run: $OUT_ROOT"
fi

if [ ! -d "$OUT_ROOT" ]; then
    echo "[agg] out_root not found: $OUT_ROOT" >&2
    exit 1
fi

# Strip inherited .venv (see eval_scico.sbatch for rationale)
unset VIRTUAL_ENV PYTHONHOME
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv/bin' | grep -v "${HOME}/\.venv" | paste -sd: -)
export PATH

if [ -d ~/miniconda3/envs/compare_eval ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate compare_eval
elif [ -d ~/miniconda3/envs/scico ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate scico
elif [ -f .venv/bin/activate ]; then
    . .venv/bin/activate
fi

echo "[agg] scanning per-agent reports under $OUT_ROOT"
python3 scripts/aggregate_eval_scico_matrix.py "$OUT_ROOT"
