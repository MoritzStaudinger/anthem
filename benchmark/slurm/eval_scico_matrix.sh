#!/usr/bin/env bash
# Two-phase agent-vs-judge eval matrix:
#
#   PHASE 1 (parallel) — answer phase
#     For each AGENT, submit eval_scico_answer.sbatch which runs the
#     RAG agent over all testset rows and persists per-row answers to
#     answers.jsonl. NO judges called. Pure agent-cost phase.
#
#   PHASE 2 (parallel, depends on PHASE 1 per-agent) — judge phase
#     For each AGENT, after its answer job lands, submit
#     eval_scico_judge.sbatch which loads answers.jsonl and grades it
#     under the configured judge panel. Re-runnable any time with
#     different judges (cheap, no re-agent cost).
#
#   PHASE 3 — aggregation
#     Depends on every PHASE 2 job. Builds cross-agent comparison.
#
# Why split: 'i want to re-judge with a stronger panel later' is
# basically free in this layout — re-submit only PHASE 2 against the
# same answers/, swap JUDGES. Also makes per-agent failures isolated
# (judging gpt-5 doesn't depend on claude finishing its agent run).
#
# Env overrides:
#   AGENTS    space-separated agent model ids
#   JUDGES    space-separated judge ids (used by ALL phase-2 jobs)
#   TESTSET   path to curated JSONL
#   AFF_DIR   affiliations dir
#   RUN_TAG   subdirectory under reports/eval_scico (default: matrix_<ts>)
#
# Usage:
#   bash slurm/eval_scico_matrix.sh
#   AGENTS="qwen-3.5-397b gemini-2.5-pro" bash slurm/eval_scico_matrix.sh
#
# Default agents (chosen to minimise judge-self-preference):
#   qwen-3.5-397b      Alibaba   (sibling of qwen-3.6-35b judge — minor)
#   gemini-2.5-pro     Google    (no Google judge → cleanest)
#   claude-sonnet-4-6  Anthropic (sibling of claude-haiku judge — minor)
#   claude-opus-4-7    Anthropic (sibling of claude-haiku judge — minor)
#
# Default judges (cheap+fast panel; re-run phase 2 with stronger judges later):
#   gpt-5.4-nano       OpenAI    (no OpenAI agent — cleanest)
#   claude-haiku-4-5   Anthropic
#   qwen-3.6-35b       Aqueduct  (free)

set -euo pipefail

AGENTS="${AGENTS:-${1:-qwen-3.5-397b gemini-2.5-pro claude-sonnet-4-6 claude-opus-4-7}}"
JUDGES="${JUDGES:-gpt-5.4-nano claude-haiku-4-5 qwen-3.6-35b}"
TESTSET="${TESTSET:-data/eval/scico_testset.jsonl}"
AFF_DIR="${AFF_DIR:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta}"
GROUND_TRUTH_CSV="${GROUND_TRUTH_CSV:-data/datastructure/ground_truth.csv}"
RUN_TAG="${RUN_TAG:-matrix_$(date +%Y%m%d-%H%M%S)}"

OUT_ROOT="reports/eval_scico/${RUN_TAG}"

cd ~/compare_acl 2>/dev/null || cd ~/PycharmProjects/Comparer_flask 2>/dev/null || cd ~/Comparer_flask
mkdir -p "$OUT_ROOT" "$HOME/logs/eval_scico"

[ -f "$TESTSET" ] || { echo "missing $TESTSET" >&2; exit 1; }
N_QUESTIONS=$(wc -l < "$TESTSET")

echo "==================== EVAL MATRIX — 2-phase ===================="
echo "  testset    : $TESTSET ($N_QUESTIONS questions)"
echo "  agents     : $AGENTS"
echo "  judges     : $JUDGES"
echo "  aff dir    : $AFF_DIR"
echo "  out root   : $OUT_ROOT"
echo ""

JUDGE_JOB_IDS=()

for AGENT in $AGENTS; do
    AGENT_TAG=$(echo "$AGENT" | tr '/.' '__')
    AGENT_OUT="${OUT_ROOT}/agent_${AGENT_TAG}"
    mkdir -p "$AGENT_OUT"

    # --- PHASE 1: answer ---
    ANSWER_JID=$(sbatch --parsable \
        --job-name="evalsc-ans-${AGENT_TAG}" \
        slurm/eval_scico_answer.sbatch \
        "$AGENT" "$AGENT_OUT" "$TESTSET" "$AFF_DIR" "$GROUND_TRUTH_CSV")
    echo "  [P1 answer] $AGENT -> job $ANSWER_JID  (out=$AGENT_OUT)"

    # --- PHASE 2: judge (depends on PHASE 1 for this agent only) ---
    JUDGE_JID=$(sbatch --parsable \
        --job-name="evalsc-jdg-${AGENT_TAG}" \
        --dependency="afterok:$ANSWER_JID" \
        slurm/eval_scico_judge.sbatch \
        "$AGENT_OUT" "$JUDGES" "$TESTSET" "$AFF_DIR" "$GROUND_TRUTH_CSV")
    echo "  [P2 judge]  $AGENT -> job $JUDGE_JID  (after answer $ANSWER_JID)"
    JUDGE_JOB_IDS+=("$JUDGE_JID")
done

# --- PHASE 3: aggregation (depends on all PHASE 2 jobs) ---
DEP=$(IFS=:; echo "afterany:${JUDGE_JOB_IDS[*]}")
AGG_JID=$(sbatch --parsable \
    --dependency="$DEP" \
    --job-name="evalsc-agg" \
    slurm/eval_scico_aggregate.sbatch \
    "$OUT_ROOT")

echo ""
echo "==================== JOB GRAPH ===================="
echo "  Phase 1 (answer) jobs run in parallel:    one per agent"
echo "  Phase 2 (judge)  starts per agent once its answer lands"
echo "  Phase 3 (aggregate)  job=$AGG_JID  after all judges:"
echo "    afterany: ${JUDGE_JOB_IDS[*]}"
echo ""
echo "Final cross-agent comparison will land at:"
echo "  $OUT_ROOT/cross_agent_comparison.csv"
echo "  $OUT_ROOT/cross_agent_comparison.md"
echo ""
echo "Re-judge later with a different panel (no agent re-cost):"
echo "  for d in $OUT_ROOT/agent_*; do"
echo "    sbatch slurm/eval_scico_judge.sbatch \"\$d\" \"<new judges>\""
echo "  done"
echo ""
echo "Monitor: squeue -u \$USER"
