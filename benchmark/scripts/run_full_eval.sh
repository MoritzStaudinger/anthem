#!/bin/bash
# End-to-end pipeline: generate the question set + run the Comparer eval
# on it + emit a report. Designed to be re-runnable.
#
# Usage:
#     bash scripts/run_full_eval.sh [--oversample N] [--skip-generate]
#
# Required env (put in .env):
#     AQUEDUCT_API_KEY=...        # for Qwen-3.5-397B (PhD generation + judges)
#     OPENAI_API_KEY=...          # gpt-4o-mini judge
#     ANTHROPIC_API_KEY=...       # claude-sonnet judge
#     GOOGLE_API_KEY=...          # gemini-flash judge
#
# Paths assumed by default — override via env vars on invocation:
#     SCICO_ROOT      enriched JSONs dir   (default /share/.../with_topics)
#     MD_DIR          PyMu_LLM Markdown    (default ~/data/acl_pdfs_pymu_llm)
#     QUESTIONS_OUT   question JSONL path  (default data/eval/scico_testset.jsonl)
#     EVAL_LOG        eval log path        (default logs/eval-<timestamp>.log)
#     MODEL           agent LLM            (default gpt-4o-mini)
#     JUDGES          space-separated      (default "gpt-4o-mini claude-sonnet-4-6 gemini-2.5-flash")
#     OVERSAMPLE      question oversample  (default 1.0 — pass 2-3 to over-generate)

set -euo pipefail

# ---- Paths --------------------------------------------------------------
SCICO_ROOT="${SCICO_ROOT:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta}"
MD_DIR="${MD_DIR:-/home/$USER/data/acl_pdfs_pymu_llm}"
QUESTIONS_OUT="${QUESTIONS_OUT:-data/eval/scico_testset.jsonl}"
EVAL_LOG="${EVAL_LOG:-logs/eval-$(date +%Y%m%d-%H%M%S).log}"
MODEL="${MODEL:-gpt-4o-mini}"
JUDGES="${JUDGES:-gpt-4o-mini claude-sonnet-4-6 gemini-2.5-flash}"
OVERSAMPLE="${OVERSAMPLE:-1.0}"
TOTAL_BUDGET="${TOTAL_BUDGET:-100}"
PHD_COUNT="${PHD_COUNT:-24}"
PHD_MODEL="${PHD_MODEL:-qwen-3.5-397b}"

# ---- Parse flags --------------------------------------------------------
SKIP_GENERATE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --skip-generate) SKIP_GENERATE=1; shift ;;
        --oversample=*) OVERSAMPLE="${1#*=}"; shift ;;
        --oversample) OVERSAMPLE="$2"; shift 2 ;;
        *) shift ;;
    esac
done

mkdir -p logs data/eval

# ---- Load env / API keys ------------------------------------------------
if [ -f .env ]; then
    set -a; . ./.env; set +a
fi

# ---- 1. Generate questions ----------------------------------------------
if [ "$SKIP_GENERATE" -eq 0 ]; then
    echo "[$(date '+%F %T')] === STEP 1/2: generate questions ==="
    echo "  SCICO_ROOT  = $SCICO_ROOT"
    echo "  MD_DIR      = $MD_DIR"
    echo "  OVERSAMPLE  = $OVERSAMPLE"
    echo "  output      = $QUESTIONS_OUT"
    python scripts/generate_scico_questions.py \
        --scico-root "$SCICO_ROOT" \
        --md-dir "$MD_DIR" \
        --out "$QUESTIONS_OUT" \
        --total-budget "$TOTAL_BUDGET" \
        --phd-count "$PHD_COUNT" \
        --phd-model "$PHD_MODEL" \
        --oversample "$OVERSAMPLE" \
        --no-backup
    echo "[$(date '+%F %T')] generated $(wc -l < "$QUESTIONS_OUT") questions"
else
    echo "[$(date '+%F %T')] === STEP 1/2 SKIPPED (--skip-generate) ==="
    if [ ! -s "$QUESTIONS_OUT" ]; then
        echo "ERROR: $QUESTIONS_OUT not found; cannot skip generation." >&2
        exit 1
    fi
fi

# ---- 2. Run the eval ----------------------------------------------------
echo "[$(date '+%F %T')] === STEP 2/2: run agent eval ==="
echo "  testset       = $QUESTIONS_OUT"
echo "  agent model   = $MODEL"
echo "  judges        = $JUDGES"
echo "  log           = $EVAL_LOG"

# shellcheck disable=SC2086
python scripts/eval_scico.py \
    --testset "$QUESTIONS_OUT" \
    --affiliations-dir "$SCICO_ROOT" \
    --mode agent --model "$MODEL" \
    --judges $JUDGES \
    --no-bertscore \
    2>&1 | tee "$EVAL_LOG"

echo "[$(date '+%F %T')] === DONE ==="
echo "  log: $EVAL_LOG"
