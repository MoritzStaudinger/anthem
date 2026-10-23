#!/usr/bin/env bash
# Smoke wrapper for eval_scico.sbatch.
#
# Carves out a 5-row subset from the curated testset, submits a quick
# slurm job that runs the FULL eval pipeline against just those 5
# questions. Lets you verify:
#   - environment + API keys work end-to-end on the cluster
#   - retriever/agent loop completes
#   - judges respond and aggregate
#   - per-row outputs land in the report dir
# ...without burning the cost of a full 100-question multi-judge run.
#
# Cost: ~$0.10-0.30 (5 questions × agent + 3 judges).
# Wall-clock: ~5-10 minutes.
#
# Usage:
#   bash slurm/eval_scico_smoke.sh
#   bash slurm/eval_scico_smoke.sh 10   # if you want 10 rows instead of 5
#   bash slurm/eval_scico_smoke.sh 5 retrieval gpt-4o-mini "gpt-4o"
#
# Args (positional, all optional):
#   $1  N rows (default: 5)
#   $2  mode  (default: agent)
#   $3  agent model (default: gpt-4o-mini)
#   $4  judges (default: gpt-4o claude-sonnet-4-6 qwen-3.5-397b)

set -euo pipefail

N_ROWS="${1:-5}"
MODE="${2:-agent}"
AGENT_MODEL="${3:-gpt-4o-mini}"
JUDGES="${4:-gpt-4o claude-sonnet-4-6 qwen-3.5-397b}"

cd ~/compare_acl 2>/dev/null || cd ~/PycharmProjects/Comparer_flask 2>/dev/null || cd ~/Comparer_flask

FULL_TESTSET="data/eval/scico_testset.jsonl"
[ -f "$FULL_TESTSET" ] || { echo "missing $FULL_TESTSET" >&2; exit 1; }

# Build a subset: pick rows stratified across category so we exercise
# every code path (institution-aware gold, country-pair gold, etc.)
SMOKE_TESTSET="data/eval/scico_testset.smoke_${N_ROWS}.jsonl"
python3 - "$FULL_TESTSET" "$SMOKE_TESTSET" "$N_ROWS" <<'PY'
import json, sys
from collections import defaultdict
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])

with open(src) as f:
    rows = [json.loads(l) for l in f if l.strip()]

# Stratify across categories — take ceil(n / n_cats) from each
by_cat = defaultdict(list)
for r in rows:
    cat = (r.get("classifier_gold") or {}).get("question_category") or "unknown"
    by_cat[cat].append(r)

cats = sorted(by_cat.keys())
per = max(1, n // len(cats))
sampled = []
for c in cats:
    sampled.extend(by_cat[c][:per])
    if len(sampled) >= n: break
sampled = sampled[:n]

with open(dst, "w") as f:
    for r in sampled:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"wrote {len(sampled)} rows -> {dst}")
print(f"categories covered: {sorted({(r.get('classifier_gold') or {}).get('question_category') for r in sampled})}")
PY

echo ""
echo "Submitting smoke job ..."
sbatch --job-name="eval_scico_smoke_${N_ROWS}" \
    --time=01:00:00 \
    slurm/eval_scico.sbatch \
    "$SMOKE_TESTSET" \
    "$MODE" \
    "$AGENT_MODEL" \
    "$JUDGES" \
    "/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta" \
    "data/datastructure/ground_truth.csv" \
    "smoke_${N_ROWS}"

echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    tail -f ~/logs/eval_scico/slurm-<jobid>.out"
echo "Report:  reports/eval_scico/smoke_${N_ROWS}/"
