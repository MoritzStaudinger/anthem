#!/usr/bin/env bash
#SBATCH --partition=GPU-a100s
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --job-name=s2orc_dl
#SBATCH --output=/home/%u/logs/s2orc/slurm-%j.out
#
# Runs the S2ORC download pipeline end-to-end. Each step resumes from
# state.json on failure, so re-submitting after a crash picks up where
# it left off.
#
# Usage:
#   sbatch scripts/s2orc_pipeline.sbatch [output_dir] [steps]
#
# Examples:
#   sbatch scripts/s2orc_pipeline.sbatch                           # default dir, all steps
#   sbatch scripts/s2orc_pipeline.sbatch /share/X/s2orc_raw        # custom dir, all steps
#   sbatch scripts/s2orc_pipeline.sbatch /share/X/s2orc_raw s2orc  # only the s2orc step
#
# Requires S2_API_KEY in .env (free key from semanticscholar.org/product/api/key).

set -euo pipefail

OUT_DIR="${1:-/share/moritz.staudinger/acl_data/s2orc_raw}"
STEPS="${2:-paper-ids papers s2orc authors}"

mkdir -p "$OUT_DIR"
mkdir -p "$HOME/logs/s2orc"

cd ~/acl_processor
# Activate venv if present; falls back to whatever python is on PATH.
if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  . .venv/bin/activate
fi

echo "[s2orc] output dir   : $OUT_DIR"
echo "[s2orc] steps        : $STEPS"
echo "[s2orc] python       : $(which python3)"
echo "[s2orc] S2_API_KEY set: $([ -n "${S2_API_KEY:-}" ] && echo yes || echo no)"

for step in $STEPS; do
  echo ""
  echo "============================================================"
  echo "[s2orc] step: $step"
  echo "============================================================"
  python3 -u scripts/s2orc_download.py "$step" --output-dir "$OUT_DIR"
done

echo ""
echo "[s2orc] all steps complete. Next: scripts/s2orc_normalize.py"
echo "[s2orc] suggested:"
echo "  python3 scripts/s2orc_normalize.py \\"
echo "    --input-dir  $OUT_DIR \\"
echo "    --output-dir /share/moritz.staudinger/acl_data/s2orc_affiliations \\"
echo "    --write-fulltext-md"
