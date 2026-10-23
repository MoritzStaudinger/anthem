#!/usr/bin/env bash
# Submit the four eval passes (deterministic + retrieval, IR measures,
# RAGAS, BERTScore promotion) over a curated DIRS list. Each pass has
# its own guards so re-running is safe — only missing/incomplete work
# gets resubmitted.
#
# Usage:
#     bash scripts/submit_eval_passes.sh                 # submit all four passes
#     bash scripts/submit_eval_passes.sh --phase=1       # only pass N (1|2|3|4)
#     bash scripts/submit_eval_passes.sh --dry-run       # print what would submit
#     DIRS_OVERRIDE="dir1 dir2" bash scripts/submit_eval_passes.sh
#
# Passes:
#   1. Deterministic + retrieval (no BERTScore, no LLM judges, no API)
#      → GPU-a40 gpu:0  → writes judge_report.json
#   2. Standard IR metrics via ir_measures (nDCG, MAP, P/R/MRR/Bpref)
#      → no GPU, no API → writes ir_metrics.jsonl
#   3. RAGAS (faithfulness + response_relevancy, 3 OpenRouter judges)
#      → OpenRouter API → writes ragas_<judge_slug>.jsonl per judge
#   4. BERTScore promotion (re-runs pass 1 with BERTScore enabled)
#      → GPU-a40 gpu:1  → renames pass-1 output to judge_report_nobert.json,
#                         writes new judge_report.json WITH BERTScore.
#      → REQUIRES pass 1 to have completed first for the same dir.
#
# Typical workflow:
#   bash scripts/submit_eval_passes.sh --phase=1   # cheap, fast
#   bash scripts/submit_eval_passes.sh --phase=2   # cheap, fast (parallel to 1)
#   bash scripts/submit_eval_passes.sh --phase=3   # OpenRouter, parallel
#   # ... wait for pass 1 to finish across all dirs ...
#   bash scripts/submit_eval_passes.sh --phase=4   # BERTScore promotion

set -euo pipefail

# ---------------------------------------------------------------------------
# Configurable
# ---------------------------------------------------------------------------

# Override via DIRS_OVERRIDE env var (space-separated list) or edit below.
DEFAULT_DIRS=(
    reports/eval_scico/agent_claude_sonnet_4_6
    reports/eval_scico/agent_gemini_2_5_pro
    reports/eval_scico/agent_gpt_5_mini
    reports/eval_scico/agent_deepseek_v4_flash
    reports/eval_scico/agent_qwen3_6_35b_a3b_openrouter
    reports/eval_scico/agent_gemma_4_31b_it_openrouter
    reports/eval_scico/pipeline_claude_sonnet_4_6
    reports/eval_scico/pipeline_gemini_2_5_pro
    reports/eval_scico/pipeline_gpt_5_mini
    reports/eval_scico/pipeline_deepseek_v4_flash
    reports/eval_scico/pipeline_qwen3_6_35b_a3b_openrouter
    reports/eval_scico/pipeline_gemma_4_31b_it_openrouter
    reports/eval_scico/closed_claude_sonnet_4_6
    reports/eval_scico/closed_gemini_2_5_pro
    reports/eval_scico/closed_gpt_5_mini
)

JUDGES_RAGAS="${JUDGES_RAGAS:-qwen/qwen3.5-397b-a17b openai/gpt-5 anthropic/claude-opus-4-7}"
EXPECTED_ROWS="${EXPECTED_ROWS:-99}"
# Drop dirs where more than this fraction of rows are agent errors
# (e.g. an OpenRouter 402 storm during generation). 0.1 = drop if > 10% errored.
MAX_ERROR_FRACTION="${MAX_ERROR_FRACTION:-0.1}"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
PHASE="all"
DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --phase=*)  PHASE="${1#*=}"; shift ;;
        --phase)    PHASE="$2"; shift 2 ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '2,30p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

case "$PHASE" in
    1|2|3|4|all) ;;
    *) echo "--phase must be 1, 2, 3, 4, or all (got: $PHASE)" >&2; exit 2 ;;
esac

# Resolve DIRS: env override beats default
if [ -n "${DIRS_OVERRIDE:-}" ]; then
    # shellcheck disable=SC2206
    DIRS=( $DIRS_OVERRIDE )
else
    DIRS=( "${DEFAULT_DIRS[@]}" )
fi

# ---------------------------------------------------------------------------
# Filter DIRS to complete runs
# ---------------------------------------------------------------------------
COMPLETE_DIRS=()
echo "# Filtering ${#DIRS[@]} dirs (expecting $EXPECTED_ROWS rows each)"
for d in "${DIRS[@]}"; do
    if [ ! -f "$d/answers.jsonl" ]; then
        printf '  SKIP (no answers.jsonl)         %s\n' "$d"
        continue
    fi
    rows=$(wc -l < "$d/answers.jsonl" 2>/dev/null || echo 0)
    if [ "$rows" -ne "$EXPECTED_ROWS" ]; then
        printf '  SKIP (incomplete: %d/%d rows)   %s\n' "$rows" "$EXPECTED_ROWS" "$d"
        continue
    fi
    # Count error rows — a row with .error set means the agent failed
    # during generation (e.g. OpenRouter 402, timeout). High counts mean
    # this dir has stale junk that judges/IR/RAGAS will (correctly) skip,
    # so it's pointless to submit them.
    err_rows=$(python3 -c "
import json, sys
n = 0
for line in open(sys.argv[1]):
    line = line.strip()
    if not line: continue
    try: rec = json.loads(line)
    except json.JSONDecodeError: continue
    if rec.get('error'): n += 1
print(n)
" "$d/answers.jsonl" 2>/dev/null || echo 0)
    err_frac=$(awk -v e="$err_rows" -v r="$rows" 'BEGIN{ printf "%.3f", (r>0 ? e/r : 0) }')
    if awk -v f="$err_frac" -v m="$MAX_ERROR_FRACTION" 'BEGIN{exit !(f > m)}'; then
        printf '  SKIP (errored: %d/%d rows, frac=%s > %s)  %s\n' \
            "$err_rows" "$rows" "$err_frac" "$MAX_ERROR_FRACTION" "$d"
        continue
    fi
    COMPLETE_DIRS+=( "$d" )
done
echo "# ${#COMPLETE_DIRS[@]} complete dirs to evaluate"
echo ""

if [ "${#COMPLETE_DIRS[@]}" -eq 0 ]; then
    echo "Nothing to do." >&2
    exit 0
fi

# ---------------------------------------------------------------------------
# submit() — print or actually sbatch depending on --dry-run
# ---------------------------------------------------------------------------
submit() {
    if [ "$DRY_RUN" -eq 1 ]; then
        echo "  [dry-run] $*"
    else
        eval "$@"
    fi
}

# ---------------------------------------------------------------------------
# Pass 1 — deterministic + retrieval (no BERTScore, no judges)
# ---------------------------------------------------------------------------
if [ "$PHASE" = "1" ] || [ "$PHASE" = "all" ]; then
    echo "=== Pass 1: deterministic + retrieval (no BERTScore, no judges) ==="
    n=0
    for d in "${COMPLETE_DIRS[@]}"; do
        if [ -f "$d/judge_report.json" ]; then
            printf '  SKIP (already has judge_report.json)  %s\n' "$d"
            continue
        fi
        submit "EVAL_NO_BERT=1 sbatch --partition=GPU-a40 --gres=gpu:0 slurm/eval_scico_judge.sbatch \"$d\" \"\""
        n=$((n + 1))
    done
    echo "  -> submitted $n pass-1 jobs"
    echo ""
fi

# ---------------------------------------------------------------------------
# Pass 2 — standard IR metrics (ir_measures)
# ---------------------------------------------------------------------------
if [ "$PHASE" = "2" ] || [ "$PHASE" = "all" ]; then
    echo "=== Pass 2: standard IR metrics (ir_measures, no API, no GPU) ==="
    n=0
    for d in "${COMPLETE_DIRS[@]}"; do
        # always re-run so we don't get stuck on stale partial outputs
        rm -f "$d/ir_metrics.jsonl"
        submit "sbatch slurm/eval_ir_measures.sbatch \"$d\""
        n=$((n + 1))
    done
    echo "  -> submitted $n pass-2 jobs"
    echo ""
fi

# ---------------------------------------------------------------------------
# Pass 3 — RAGAS (3 judges × N dirs via OpenRouter)
# ---------------------------------------------------------------------------
if [ "$PHASE" = "3" ] || [ "$PHASE" = "all" ]; then
    echo "=== Pass 3: RAGAS (3 OpenRouter judges per dir) ==="
    n=0
    for d in "${COMPLETE_DIRS[@]}"; do
        for j in $JUDGES_RAGAS; do
            submit "sbatch slurm/eval_ragas.sbatch \"$d\" \"$j\""
            n=$((n + 1))
        done
    done
    echo "  -> submitted $n pass-3 jobs ($((${#COMPLETE_DIRS[@]})) dirs × $(echo $JUDGES_RAGAS | wc -w) judges)"
    echo ""
fi

# ---------------------------------------------------------------------------
# Pass 4 — BERTScore promotion (requires pass 1 done for the same dir)
# ---------------------------------------------------------------------------
if [ "$PHASE" = "4" ] || [ "$PHASE" = "all" ]; then
    echo "=== Pass 4: BERTScore promotion (re-runs pass 1 WITH BERTScore) ==="
    n=0
    for d in "${COMPLETE_DIRS[@]}"; do
        if [ ! -f "$d/judge_report.json" ]; then
            printf '  SKIP (pass 1 not done yet)            %s\n' "$d"
            continue
        fi
        if [ -f "$d/judge_report_nobert.json" ]; then
            printf '  SKIP (already promoted)               %s\n' "$d"
            continue
        fi
        # Preserve the no-BERT version, then re-submit with BERTScore on
        if [ "$DRY_RUN" -eq 1 ]; then
            echo "  [dry-run] mv $d/judge_report.json $d/judge_report_nobert.json"
            echo "  [dry-run] sbatch --partition=GPU-a40 --gres=gpu:1 slurm/eval_scico_judge.sbatch $d \"\""
        else
            mv "$d/judge_report.json" "$d/judge_report_nobert.json"
            sbatch --partition=GPU-a40 --gres=gpu:1 slurm/eval_scico_judge.sbatch "$d" ""
        fi
        n=$((n + 1))
    done
    echo "  -> submitted $n pass-4 jobs"
    echo ""
fi

echo "=== Done. Monitor with: squeue -u \$USER ==="
