#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --job-name=evalsc_closed
#SBATCH --output=/home/%u/logs/eval_scico/slurm-%j.out
#
# Closed-book baseline: same testset, same models, NO retrieval tool.
# The LLM is asked the question cold and must answer from its
# parametric knowledge alone. This is the comparison point for the
# agentic / retrieval-augmented runs.
#
# Saves answers in the same per-row JSONL schema as eval_scico_answer.sbatch,
# so the judge phase (eval_scico_judge.sbatch) and any post-hoc metric
# scripts can re-grade these without code changes. `records` will be []
# for every row by design — retrieval metrics will all be zero, but
# generation metrics + phantom rate are the interesting numbers here.
#
# Args (positional):
#   $1  agent_model        (required, e.g. claude-sonnet-4-6)
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
for _sib in ~/acl_processor/.env ~/PycharmProjects/acl_processor/.env; do
    if [ -f "$_sib" ]; then
        set -a; . "$_sib"; set +a
        break
    fi
done

echo "[closed-book] agent     : $AGENT_MODEL"
echo "[closed-book] out_dir   : $OUT_DIR"
echo "[closed-book] testset   : $TESTSET ($(wc -l < $TESTSET) rows)"

# -u: unbuffered stdout so progress prints land in the slurm log live,
# not when the (~8 KB) buffer happens to flush.
python3 -u scripts/eval_scico.py \
    --testset "$TESTSET" \
    --mode closed_book \
    --model "$AGENT_MODEL" \
    --answers-out "$OUT_DIR/answers.jsonl" \
    --generate-only \
    -v 2>&1 | tee "$OUT_DIR/answer.log"

echo "[closed-book] done — $(wc -l < $OUT_DIR/answers.jsonl) saved answers"
