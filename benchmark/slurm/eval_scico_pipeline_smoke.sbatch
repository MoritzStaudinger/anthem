#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --job-name=evalsc_pipe_smoke
#SBATCH --output=/home/%u/logs/eval_scico/slurm-%j.out
#
# Pipeline-mode smoke test: 5 rows, fast model, prints the classifier
# output per row so we can sanity-check that:
#   - classifier_gold.question_category gold matches predicted_category
#   - rewritten queries look reasonable
#   - the right handler fires
#   - JSONL schema is intact (records, summary_text, telemetry, error)
#
# Run BEFORE the full pipeline matrix to catch dispatch bugs cheaply.

set -euo pipefail

cd ~/compare_acl 2>/dev/null || cd ~/PycharmProjects/Comparer_flask 2>/dev/null || cd ~/Comparer_flask
mkdir -p "$HOME/logs/eval_scico" reports/eval_scico/pipeline_smoke

# Build a tiny stratified testset so we hit each of the 5 categories at
# least once. The carve-out picks the first row of each category from
# the full testset (so paper IDs are reproducible).
SMOKE_TESTSET="data/eval/scico_testset.pipeline_smoke.jsonl"
python3 - <<'PY'
import json
from collections import defaultdict
src = "data/eval/scico_testset.jsonl"
dst = "data/eval/scico_testset.pipeline_smoke.jsonl"
by_cat = defaultdict(list)
with open(src) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        cat = (r.get("classifier_gold") or {}).get("question_category") or "unknown"
        by_cat[cat].append(r)
picks = []
for cat in sorted(by_cat):
    picks.extend(by_cat[cat][:1])
with open(dst, "w") as f:
    for r in picks:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"wrote {len(picks)} rows covering {sorted(by_cat)}")
PY

unset VIRTUAL_ENV PYTHONHOME
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv/bin' | grep -v "${HOME}/\.venv" | paste -sd: -)
export PATH

if [ -d ~/miniconda3/envs/compare_eval ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate compare_eval
elif [ -d ~/miniconda3/envs/scico ]; then
    . ~/miniconda3/etc/profile.d/conda.sh
    conda activate scico
fi
if [ -f .env ]; then set -a; . .env; set +a; fi

# Stage index to RAM-backed tmpfs (/dev/shm).
SRC_INDEX="data/local_index_acl"
STAGED_INDEX="/dev/shm/acl_index_${SLURM_JOB_ID:-$$}_$USER"
if [ -d "$SRC_INDEX" ] && [ -w /dev/shm ]; then
    echo "[smoke] staging index $SRC_INDEX -> $STAGED_INDEX ..."
    t0=$(date +%s)
    if cp -r "$SRC_INDEX" "$STAGED_INDEX" 2>/dev/null; then
        echo "[smoke] staged in $(($(date +%s) - t0))s ($(du -sh $STAGED_INDEX | cut -f1))"
        export ACL_INDEX_DIR="$STAGED_INDEX"
        trap 'rm -rf "$STAGED_INDEX" 2>/dev/null || true' EXIT
    else
        echo "[smoke] /dev/shm copy failed — falling back to NFS"
        rm -rf "$STAGED_INDEX" 2>/dev/null || true
    fi
fi

echo "[smoke] running pipeline-mode smoke with gpt-5-mini on 5 rows..."
python3 -u scripts/eval_scico.py \
    --testset "$SMOKE_TESTSET" \
    --mode pipeline \
    --model gpt-5-mini \
    --answers-out reports/eval_scico/pipeline_smoke/answers.jsonl \
    --generate-only \
    -v 2>&1 | tee reports/eval_scico/pipeline_smoke/answer.log

echo ""
echo "[smoke] inspection: predicted vs gold categories"
python3 - <<'PY'
import json
with open("reports/eval_scico/pipeline_smoke/answers.jsonl") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        tele = r.get("telemetry") or {}
        rid = r.get("id")
        gold = tele.get("gold_category")
        pred = tele.get("predicted_category")
        match = "✓" if gold == pred else "✗"
        queries = tele.get("predicted_queries") or []
        n_records = len(r.get("records") or [])
        wall = tele.get("wall_seconds", 0)
        sumlen = len((r.get("summary_text") or ""))
        print(f"{match} {rid}")
        print(f"   gold={gold}  pred={pred}  wall={wall:.1f}s  "
              f"records={n_records}  summary_chars={sumlen}")
        print(f"   queries: {queries}")
        if r.get("error"):
            print(f"   ERROR: {r['error']}")
        print()
PY

echo "[smoke] done"
