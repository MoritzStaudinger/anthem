#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --job-name=eval_scico
#SBATCH --output=/home/%u/logs/eval_scico/slurm-%j.out
#
# Runs eval_scico.py end-to-end against a curated SciCo testset:
#   - retrieval mode OR agent mode
#   - one or more LLM-as-judge models
#   - per-category breakdown vs gold paper IDs
#   - optional classifier-accuracy + affiliation correctness
#
# Why GPU: BERTScore (roberta-large) runs locally on a GPU. Set
# EVAL_NO_BERT=1 to skip and switch to a CPU partition for cheaper runs.
#
# Args (positional, override defaults):
#   $1  testset path     (default: data/eval/scico_testset.jsonl)
#   $2  mode             (default: agent)        [retrieval|agent]
#   $3  agent/model id   (default: gpt-4o-mini)
#   $4  judge id(s)      (default: "gpt-4o claude-sonnet-4-6 qwen-3.5-397b")
#   $5  affiliations dir (default: /share/.../ror_parsed_affiliations_90_with_topics_and_bibmeta)
#   $6  ground-truth csv (default: data/datastructure/ground_truth.csv)
#   $7  out tag          (default: timestamp)
#
# Smoke pattern:
#   See slurm/eval_scico_smoke.sh — submits this with a 5-row subset
#   testset for ~5 min total. Use that BEFORE the full 100-q run.

set -euo pipefail

TESTSET="${1:-data/eval/scico_testset.jsonl}"
MODE="${2:-agent}"
AGENT_MODEL="${3:-gpt-4o-mini}"
JUDGES="${4:-gpt-4o claude-sonnet-4-6 qwen-3.5-397b}"
AFFILIATIONS_DIR="${5:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta}"
GROUND_TRUTH_CSV="${6:-data/datastructure/ground_truth.csv}"
OUT_TAG="${7:-$(date +%Y%m%d-%H%M%S)}"

OUT_DIR="reports/eval_scico/${OUT_TAG}"

cd ~/compare_acl 2>/dev/null || cd ~/PycharmProjects/Comparer_flask 2>/dev/null || cd ~/Comparer_flask
mkdir -p "$HOME/logs/eval_scico" "$OUT_DIR"

# Strip any inherited venv from the submitting shell. slurm inherits
# the env by default, and if a .venv was active at submit time its
# bin/ ends up earlier in PATH than the conda env we activate below,
# causing ModuleNotFoundError on heavy deps (llama_index, bert-score).
unset VIRTUAL_ENV PYTHONHOME
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv/bin' | grep -v "${HOME}/\.venv" | paste -sd: -)
export PATH

# Activate the eval conda env. Prefer the dedicated `compare_eval` env
# (Python 3.11 + modern llama_index); fall back to `scico` for legacy
# runs, then to .venv as a last resort.
if [ -d ~/miniconda3/envs/compare_eval ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate compare_eval
elif [ -d ~/miniconda3/envs/scico ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate scico
elif [ -f .venv/bin/activate ]; then
    . .venv/bin/activate
fi

# Load API keys
if [ -f .env ]; then set -a; . .env; set +a; fi
for _sib in ~/acl_processor/.env ~/PycharmProjects/acl_processor/.env; do
    if [ -f "$_sib" ]; then
        set -a; . "$_sib"; set +a
        break
    fi
done

# Optional: skip BERTScore (drop GPU requirement entirely)
BERT_FLAG=""
if [ "${EVAL_NO_BERT:-0}" = "1" ]; then
    BERT_FLAG="--no-bertscore"
fi

echo "[eval_scico] testset         : $TESTSET ($(wc -l < $TESTSET) rows)"
echo "[eval_scico] mode            : $MODE"
echo "[eval_scico] agent model     : $AGENT_MODEL"
echo "[eval_scico] judges          : $JUDGES"
echo "[eval_scico] affiliations    : $AFFILIATIONS_DIR"
echo "[eval_scico] ground truth    : $GROUND_TRUTH_CSV"
echo "[eval_scico] out dir         : $OUT_DIR"
echo "[eval_scico] python          : $(which python3)"
echo "[eval_scico] API keys present:"
for k in OPENAI_API_KEY ANTHROPIC_API_KEY AQUEDUCT_API_KEY GOOGLE_API_KEY OPEN_ROUTER OPENROUTER_API_KEY; do
    [[ -n "${!k:-}" ]] && echo "    ✓ $k" || echo "    ✗ $k (not set)"
done
echo ""

# shellcheck disable=SC2086
python3 scripts/eval_scico.py \
    --testset "$TESTSET" \
    --mode "$MODE" \
    --model "$AGENT_MODEL" \
    --judges $JUDGES \
    --affiliations-dir "$AFFILIATIONS_DIR" \
    --ground-truth-csv "$GROUND_TRUTH_CSV" \
    $BERT_FLAG \
    -v 2>&1 | tee "$OUT_DIR/eval.log"

echo ""
echo "[eval_scico] DONE. Outputs in: $OUT_DIR"
ls -la "$OUT_DIR"
