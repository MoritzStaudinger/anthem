#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=eval_ragas
#SBATCH --output=/home/%u/logs/eval_scico/slurm-%j.out
#
# RAGAS additive judge pass over saved eval_scico answers. Reads
# $answers_dir/answers.jsonl and writes $answers_dir/ragas_<judge>.jsonl
# with Faithfulness + ResponseRelevancy scores. Re-run per judge.
#
# No GPU needed (judge LLM is remote; embeddings are remote too if the
# judge uses an OpenAI / Anthropic / Google model). CPU partition would
# also work — using GPU-a40 just for queue access parity with the other
# eval sbatches; set --gres=gpu:0 so we don't burn a GPU we won't use.
#
# Args (positional):
#   $1  out_dir         (required — reads answers.jsonl here, writes report)
#   $2  judge model id  (required)
#   $3  testset path    (default: data/eval/scico_testset.jsonl)

set -euo pipefail

OUT_DIR="${1:?out_dir required}"
JUDGE="${2:?judge model id required}"
TESTSET="${3:-data/eval/scico_testset.jsonl}"

cd ~/compare_acl 2>/dev/null || cd ~/PycharmProjects/Comparer_flask 2>/dev/null || cd ~/Comparer_flask
mkdir -p "$HOME/logs/eval_scico"

ANSWERS="$OUT_DIR/answers.jsonl"
if [ ! -f "$ANSWERS" ]; then
    echo "[ragas] missing $ANSWERS — did the answer phase complete?" >&2
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

# Load API keys (.env + sibling acl_processor/.env for OPEN_ROUTER)
if [ -f .env ]; then set -a; . .env; set +a; fi
for _sib in ~/acl_processor/.env ~/PycharmProjects/acl_processor/.env; do
    if [ -f "$_sib" ]; then
        set -a; . "$_sib"; set +a
        break
    fi
done

echo "[ragas] out_dir : $OUT_DIR"
echo "[ragas] judge   : $JUDGE"
echo "[ragas] testset : $TESTSET ($(wc -l < $TESTSET) rows)"
echo "[ragas] answers : $ANSWERS ($(wc -l < $ANSWERS) rows)"
echo "[ragas] python  : $(which python3)"
echo "[ragas] extra   : ${RAGAS_EXTRA_FLAGS:-(none)}"
echo ""

# RAGAS_EXTRA_FLAGS lets you forward script flags (e.g. --retry-failed)
# without editing this sbatch. Use:  RAGAS_EXTRA_FLAGS=--retry-failed sbatch ...
python3 -u scripts/eval_ragas.py \
    --answers-dir "$OUT_DIR" \
    --judge "$JUDGE" \
    --testset "$TESTSET" \
    ${RAGAS_EXTRA_FLAGS:-} \
    -v 2>&1 | tee "$OUT_DIR/ragas.log"

echo ""
echo "[ragas] DONE."
ls -la "$OUT_DIR"/ragas_*.jsonl 2>/dev/null || echo "[ragas] no ragas_*.jsonl files written"
