#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=s2orc_dl
#SBATCH --output=/home/%u/logs/s2orc_dl/slurm-%j.out
#
# Run an S2ORC download step as a batch job (rather than on the login
# node). No GPU needed — this is bandwidth + JSON-decode bound.
#
# If your cluster has a dedicated CPU/general partition, change
# --partition and drop --gres entirely. Sticking with GPU-a40 + gres:0
# follows the pattern in eval_scico.sbatch and tends to be fine because
# the GPU isn't allocated.
#
# Args (positional, optional):
#   $1  step              (default: s2orc)             [paper-ids|papers|s2orc|authors]
#   $2  output dir        (default: /share/.../s2orc_raw)
#   $3  workers           (default: 4)                 — s2orc step only
#   $4  limit shards      (default: empty = all)       — useful for smoke
#
# Examples:
#   # Smoke test the fixed paper-ids step (1 shard, ~10 min)
#   sbatch slurm/s2orc_download.sbatch paper-ids \
#       /share/moritz.staudinger/acl_data/s2orc_raw 4 1
#
#   # Full s2orc fulltext download (4 workers, ~2-3 hours)
#   sbatch slurm/s2orc_download.sbatch s2orc \
#       /share/moritz.staudinger/acl_data/s2orc_raw 4
#
# Resumable: state.json + acl_to_s2.json on disk; re-submit and skipped
# shards are picked up from `done`.

set -euo pipefail

STEP="${1:-s2orc}"
OUT_DIR="${2:-/share/moritz.staudinger/acl_data/s2orc_raw}"
WORKERS="${3:-4}"
LIMIT_SHARDS="${4:-}"

cd ~/acl_processor 2>/dev/null || cd ~/PycharmProjects/acl_processor
mkdir -p "$HOME/logs/s2orc_dl" "$OUT_DIR"

# Strip any inherited venv (same defensive pattern as other sbatch scripts)
unset VIRTUAL_ENV PYTHONHOME
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv/bin' | grep -v "${HOME}/\.venv" | paste -sd: -)
export PATH

# Use whatever python env has `requests` + `tqdm` — these are tiny deps,
# the download script doesn't need torch or llama-index. Prefer scico/
# compare_eval since they're already populated.
if [ -d ~/miniconda3/envs/compare_eval ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate compare_eval
elif [ -d ~/miniconda3/envs/scico ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate scico
elif [ -f .venv/bin/activate ]; then
    . .venv/bin/activate
fi

if [ -f .env ]; then set -a; . .env; set +a; fi

echo "[s2orc_dl] step      : $STEP"
echo "[s2orc_dl] output    : $OUT_DIR"
echo "[s2orc_dl] workers   : $WORKERS"
echo "[s2orc_dl] limit     : ${LIMIT_SHARDS:-(all shards)}"
echo "[s2orc_dl] python    : $(which python3)"
echo "[s2orc_dl] S2_API_KEY: $([[ -n "${S2_API_KEY:-}" ]] && echo ✓ || echo MISSING)"
echo ""

ARGS=("$STEP" "--output-dir" "$OUT_DIR" "--workers" "$WORKERS")
if [ -n "$LIMIT_SHARDS" ]; then
    ARGS+=("--limit-shards" "$LIMIT_SHARDS")
fi

python3 scripts/s2orc_download.py "${ARGS[@]}"

echo ""
echo "[s2orc_dl] DONE."
ls -la "$OUT_DIR" | head -20
