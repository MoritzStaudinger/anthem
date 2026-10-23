#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=23:59:00
#SBATCH --job-name=evalsc_pipe
#SBATCH --output=/home/%u/logs/eval_scico/slurm-%j.out
#
# Non-agentic RAG pipeline: classify → retrieve → summarize → synthesize.
# 3-4 fixed LLM calls per row, no dynamic agent loop. The middle row in
# the closed_book / pipeline / agent comparison — measures what a
# vanilla chained RAG flow recovers vs both extremes.
#
# Args (positional):
#   $1  agent_model    (required, e.g. claude-sonnet-4-6)
#   $2  out_dir        (required — answers land at $out_dir/answers.jsonl)
#   $3  testset        (default: data/eval/scico_testset.jsonl)

set -euo pipefail

AGENT_MODEL="${1:?agent model required}"
OUT_DIR="${2:?out_dir required}"
TESTSET="${3:-data/eval/scico_testset.jsonl}"

cd ~/compare_acl 2>/dev/null || cd ~/PycharmProjects/Comparer_flask 2>/dev/null || cd ~/Comparer_flask
mkdir -p "$HOME/logs/eval_scico" "$OUT_DIR"

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
if [ -f .env ]; then set -a; . .env; set +a; fi
# Pick up keys the user keeps in the sibling acl_processor repo (e.g.
# OPEN_ROUTER) so we don't need to duplicate them. Local .env wins on
# overlap because it was sourced first.
for _sib in ~/acl_processor/.env ~/PycharmProjects/acl_processor/.env; do
    if [ -f "$_sib" ]; then
        set -a; . "$_sib"; set +a
        break
    fi
done

echo "[pipeline] agent     : $AGENT_MODEL"
echo "[pipeline] out_dir   : $OUT_DIR"
echo "[pipeline] testset   : $TESTSET ($(wc -l < $TESTSET) rows)"

# Stage index to RAM-backed tmpfs (/dev/shm) — universally writable
# on Linux. Fits in 48 GB --mem allocation. ~5-15s sequential copy
# vs 9-min random-access NFS load.
SRC_INDEX="data/local_index_acl"
STAGED_INDEX="/dev/shm/acl_index_${SLURM_JOB_ID:-$$}_$USER"
if [ -d "$SRC_INDEX" ] && [ -w /dev/shm ]; then
    echo "[pipeline] staging index $SRC_INDEX -> $STAGED_INDEX ..."
    t0=$(date +%s)
    if cp -r "$SRC_INDEX" "$STAGED_INDEX" 2>/dev/null; then
        echo "[pipeline] staged in $(($(date +%s) - t0))s ($(du -sh $STAGED_INDEX | cut -f1))"
        export ACL_INDEX_DIR="$STAGED_INDEX"
        trap 'rm -rf "$STAGED_INDEX" 2>/dev/null || true' EXIT
    else
        echo "[pipeline] /dev/shm copy failed — falling back to NFS"
        rm -rf "$STAGED_INDEX" 2>/dev/null || true
    fi
else
    echo "[pipeline] /dev/shm not writable — index will load from NFS (slow)"
fi

python3 -u scripts/eval_scico.py \
    --testset "$TESTSET" \
    --mode pipeline \
    --model "$AGENT_MODEL" \
    --answers-out "$OUT_DIR/answers.jsonl" \
    --generate-only \
    -v 2>&1 | tee "$OUT_DIR/answer.log"

echo "[pipeline] done — $(wc -l < $OUT_DIR/answers.jsonl) saved answers"
