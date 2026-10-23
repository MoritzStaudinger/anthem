#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=eval_ir
#SBATCH --output=/home/%u/logs/eval_scico/slurm-%j.out
#
# Standard IR metrics (nDCG/MAP/P/R/MRR/Bpref) over saved eval_scico
# answers via ir_measures. Reads $answers_dir/answers.jsonl, joins to
# gold via compute_gold_paper_ids, writes $answers_dir/ir_metrics.jsonl.
#
# No GPU, no LLM API — deterministic + cheap. The 1h wall is generous;
# typical run is <5 min once the affiliations corpus loads (~30s).
#
# Args (positional):
#   $1  out_dir            (required — reads answers.jsonl here)
#   $2  testset            (default: data/eval/scico_testset.jsonl)
#   $3  affiliations dir   (default: /share/.../ror_parsed_affiliations_90_with_topics_and_bibmeta)

set -euo pipefail

OUT_DIR="${1:?out_dir required}"
TESTSET="${2:-data/eval/scico_testset.jsonl}"
AFFILIATIONS_DIR="${3:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta}"

cd ~/compare_acl 2>/dev/null || cd ~/PycharmProjects/Comparer_flask 2>/dev/null || cd ~/Comparer_flask
mkdir -p "$HOME/logs/eval_scico"

ANSWERS="$OUT_DIR/answers.jsonl"
if [ ! -f "$ANSWERS" ]; then
    echo "[ir] missing $ANSWERS — did the answer phase complete?" >&2
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

echo "[ir] out_dir      : $OUT_DIR"
echo "[ir] testset      : $TESTSET ($(wc -l < $TESTSET) rows)"
echo "[ir] affiliations : $AFFILIATIONS_DIR"
echo "[ir] answers      : $ANSWERS ($(wc -l < $ANSWERS) rows)"
echo "[ir] python       : $(which python3)"
echo ""

EXTRA=""
if [ "${EVAL_BERTSCORE:-0}" = "1" ]; then EXTRA="--bertscore"; fi

python3 -u scripts/eval_ir_measures.py \
    --answers-dir "$OUT_DIR" \
    --testset "$TESTSET" \
    --affiliations-dir "$AFFILIATIONS_DIR" \
    $EXTRA \
    -v 2>&1 | tee "$OUT_DIR/ir_metrics.log"

echo ""
echo "[ir] DONE."
ls -la "$OUT_DIR"/ir_metrics*.jsonl 2>/dev/null || echo "[ir] no output file written"
