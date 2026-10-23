#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=evalsc_retr
#SBATCH --output=/home/%u/logs/eval_scico/slurm-%j.out
#
# Pure retrieval mode: runs search_local_acl_papers per row, saves
# records JSONL. No LLM generation, no agent loop. Records the retrieved
# paper_ids so the retrieval-only baseline P@k/R@k can be computed
# without paying for any generation.
#
# Same answer-then-judge schema as the other sbatch scripts, so judging
# / aggregation in eval_scico_judge.sbatch reads it transparently.
#
# Args (positional):
#   $1  out_dir       (required — answers land at $out_dir/answers.jsonl)
#   $2  testset       (default: data/eval/scico_testset.jsonl)
#
# Note: this mode is model-independent — the index does the retrieval
# via MiniLM embeddings. We still tag the JSONL with a synthetic model
# label ("retrieval_only") so the eval table can include it as a row.

set -euo pipefail

OUT_DIR="${1:?out_dir required}"
TESTSET="${2:-data/eval/scico_testset.jsonl}"

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

echo "[retrieval] out_dir   : $OUT_DIR"
echo "[retrieval] testset   : $TESTSET ($(wc -l < $TESTSET) rows)"

# Stage index to RAM-backed tmpfs (/dev/shm).
SRC_INDEX="data/local_index_acl"
STAGED_INDEX="/dev/shm/acl_index_${SLURM_JOB_ID:-$$}_$USER"
if [ -d "$SRC_INDEX" ] && [ -w /dev/shm ]; then
    echo "[retrieval] staging index $SRC_INDEX -> $STAGED_INDEX ..."
    t0=$(date +%s)
    if cp -r "$SRC_INDEX" "$STAGED_INDEX" 2>/dev/null; then
        echo "[retrieval] staged in $(($(date +%s) - t0))s ($(du -sh $STAGED_INDEX | cut -f1))"
        export ACL_INDEX_DIR="$STAGED_INDEX"
        trap 'rm -rf "$STAGED_INDEX" 2>/dev/null || true' EXIT
    else
        echo "[retrieval] /dev/shm copy failed — falling back to NFS"
        rm -rf "$STAGED_INDEX" 2>/dev/null || true
    fi
else
    echo "[retrieval] /dev/shm not writable — index will load from NFS (slow)"
fi

python3 -u scripts/eval_scico.py \
    --testset "$TESTSET" \
    --mode retrieval \
    --model retrieval_only \
    --answers-out "$OUT_DIR/answers.jsonl" \
    --generate-only \
    -v 2>&1 | tee "$OUT_DIR/answer.log"

echo "[retrieval] done — $(wc -l < $OUT_DIR/answers.jsonl) saved answers"
