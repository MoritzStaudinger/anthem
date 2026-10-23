#!/usr/bin/env bash
#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=evalsc_answer
#SBATCH --output=/home/%u/logs/eval_scico/slurm-%j.out
#
# Phase 1 of the answer-then-judge split: runs the agent over every
# testset row, saves its retrieved records + answer to a JSONL. Does
# NOT call any judges. Pair with eval_scico_judge.sbatch which loads
# this file and runs judges against the saved answers.
#
# Args (positional):
#   $1  agent_model        (required)
#   $2  out_dir            (required — answers land at $out_dir/answers.jsonl)
#   $3  testset            (default: data/eval/scico_testset.jsonl)
#   $4  affiliations dir   (default: /share/.../ror_parsed_affiliations_90_with_topics_and_bibmeta)
#   $5  ground-truth csv   (default: data/datastructure/ground_truth.csv)

set -euo pipefail

AGENT_MODEL="${1:?agent model required}"
OUT_DIR="${2:?out_dir required}"
TESTSET="${3:-data/eval/scico_testset.jsonl}"
AFFILIATIONS_DIR="${4:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta}"
GROUND_TRUTH_CSV="${5:-data/datastructure/ground_truth.csv}"

cd ~/compare_acl 2>/dev/null || cd ~/PycharmProjects/Comparer_flask 2>/dev/null || cd ~/Comparer_flask
mkdir -p "$HOME/logs/eval_scico" "$OUT_DIR"

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
if [ -f .env ]; then set -a; . .env; set +a; fi

echo "[answer] agent     : $AGENT_MODEL"
echo "[answer] out_dir   : $OUT_DIR"
echo "[answer] testset   : $TESTSET ($(wc -l < $TESTSET) rows)"

# ---- Stage the vector index to RAM-backed tmpfs ----
# /dev/shm is a RAM-backed tmpfs writable by any user on standard
# Linux. The 5.24 GB index sits comfortably inside the 48 GB --mem
# allocation. Copy is sequential (~5-15s), subsequent index load is
# RAM-speed (~5s) instead of 9-min NFS.
SRC_INDEX="data/local_index_acl"
STAGED_INDEX="/dev/shm/acl_index_${SLURM_JOB_ID:-$$}_$USER"
if [ -d "$SRC_INDEX" ] && [ -w /dev/shm ]; then
    echo "[answer] staging index $SRC_INDEX -> $STAGED_INDEX ..."
    t0=$(date +%s)
    if cp -r "$SRC_INDEX" "$STAGED_INDEX" 2>/dev/null; then
        echo "[answer] staged in $(($(date +%s) - t0))s ($(du -sh $STAGED_INDEX | cut -f1))"
        export ACL_INDEX_DIR="$STAGED_INDEX"
        trap 'rm -rf "$STAGED_INDEX" 2>/dev/null || true' EXIT
    else
        echo "[answer] /dev/shm copy failed — falling back to NFS"
        rm -rf "$STAGED_INDEX" 2>/dev/null || true
    fi
else
    echo "[answer] /dev/shm not writable — index will load from NFS (slow)"
fi

# --generate-only skips the 92k-file affiliations corpus load.
# The dense vector index (loaded by the agent's retrieval tool) has
# everything the agent needs. All scoring happens in a later pass via
# eval_scico_judge.sbatch reading --answers-in.
#
# -u: unbuffered stdout so progress is visible in the slurm log in
# real time rather than only when the page-buffer flushes.
python3 -u scripts/eval_scico.py \
    --testset "$TESTSET" \
    --mode agent \
    --model "$AGENT_MODEL" \
    --answers-out "$OUT_DIR/answers.jsonl" \
    --generate-only \
    -v 2>&1 | tee "$OUT_DIR/answer.log"

echo "[answer] done — $(wc -l < $OUT_DIR/answers.jsonl) saved answers"
