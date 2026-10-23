#!/usr/bin/env bash
#SBATCH --partition=GPU-a100s
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=evalsc_judge
#SBATCH --output=/home/%u/logs/eval_scico/slurm-%j.out
#
# Phase 2: re-grade saved agent answers under a chosen judge panel.
# Reads $answers_dir/answers.jsonl (written by eval_scico_answer.sbatch)
# and writes $answers_dir/judge_report.{log,json}. Re-runnable any time
# to add/swap judges without re-paying for the agent.
#
# GPU is for BERTScore (set EVAL_NO_BERT=1 to drop the GPU and switch
# partition).
#
# Args (positional):
#   $1  out_dir         (required — reads answers.jsonl here, writes report)
#   $2  judges          (optional — space-separated judge model ids; pass ""
#                        or omit to skip LLM judges and run ONLY the
#                        deterministic metrics + BERTScore. Useful when
#                        you want retrieval/BERTScore without paying judge
#                        tokens.)
#   $3  testset         (default: data/eval/scico_testset.jsonl)
#   $4  affiliations    (default: /share/.../ror_parsed_affiliations_90_with_topics_and_bibmeta)
#   $5  ground-truth    (default: data/datastructure/ground_truth.csv)

set -euo pipefail

OUT_DIR="${1:?out_dir required}"
JUDGES="${2:-}"
TESTSET="${3:-data/eval/scico_testset.jsonl}"
AFFILIATIONS_DIR="${4:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta}"
GROUND_TRUTH_CSV="${5:-data/datastructure/ground_truth.csv}"

cd ~/compare_acl 2>/dev/null || cd ~/PycharmProjects/Comparer_flask 2>/dev/null || cd ~/Comparer_flask
mkdir -p "$HOME/logs/eval_scico"

ANSWERS="$OUT_DIR/answers.jsonl"
if [ ! -f "$ANSWERS" ]; then
    echo "[judge] missing $ANSWERS — did the answer phase complete?" >&2
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
if [ -f .env ]; then set -a; . .env; set +a; fi

BERT_FLAG=""
if [ "${EVAL_NO_BERT:-0}" = "1" ]; then BERT_FLAG="--no-bertscore"; fi

# Build --judges only when a non-empty judge list was passed. Empty
# JUDGES → deterministic metrics + BERTScore only (no LLM tokens spent).
JUDGES_FLAG=""
if [ -n "$JUDGES" ]; then JUDGES_FLAG="--judges $JUDGES"; fi

echo "[judge] out_dir : $OUT_DIR"
echo "[judge] judges  : ${JUDGES:-(none — deterministic + BERTScore only)}"
echo "[judge] answers : $ANSWERS ($(wc -l < $ANSWERS) rows)"

# shellcheck disable=SC2086
python3 scripts/eval_scico.py \
    --testset "$TESTSET" \
    --mode agent \
    --affiliations-dir "$AFFILIATIONS_DIR" \
    --ground-truth-csv "$GROUND_TRUTH_CSV" \
    --answers-in "$ANSWERS" \
    $JUDGES_FLAG \
    $BERT_FLAG \
    -v 2>&1 | tee "$OUT_DIR/judge.log"

echo "[judge] done — report saved to $OUT_DIR/"
