#!/usr/bin/env bash
# Smoke-test every pipeline in the YAML config against ONE paper from
# the gold set. Confirms API keys are set + auth works + the model id
# exists at the backend, BEFORE you commit a full 781-paper run.
#
# Total cost: ~$0.10 (one call per LLM pipeline at ~$0.01–$0.02 per
# call for frontier models). Free pipelines (Aqueduct, passthroughs)
# cost $0.
#
# Usage (on the cluster login node — no slurm needed):
#   bash scripts/smoke_test_pipelines.sh
#
# Optional args:
#   $1  config path (default: configs/pipelines.yaml)
#   $2  paper_id to smoke-test (default: first paper in gold CSV)

set -uo pipefail   # NOT -e — we want to keep going even when a pipeline fails

CONFIG="${1:-configs/pipelines.yaml}"
PAPER_ID="${2:-}"
GOLD_CSV="reports/ground_truth_merged.csv"
SOURCES_ROOT="/share/moritz.staudinger/acl_data"
S2ORC_NORM_ROOT="/share/moritz.staudinger/acl_data/s2orc_affiliations"
SMOKE_OUT="/tmp/aff_smoke_$$"

cd ~/acl_processor
mkdir -p "$SMOKE_OUT"

# Load .env so we see what's set
if [ -f .env ]; then set -a; . .env; set +a; fi
. .venv/bin/activate 2>/dev/null || true

echo "==================== ENV CHECK ===================="
for k in OPENAI_API_KEY ANTHROPIC_API_KEY AQUEDUCT_API_KEY OPENROUTER_API_KEY GOOGLE_API_KEY; do
    if [ -n "${!k:-}" ]; then
        len="${#k}"
        val="${!k}"
        masked="${val:0:6}...${val: -4}"
        echo "  ✓ $k = $masked"
    else
        echo "  ✗ $k (not set)"
    fi
done
echo ""

# Validate the paper_id (or pick a default that exists in gold)
PAPER_ID=$(python3 - "$GOLD_CSV" "$PAPER_ID" <<'PY'
import csv, sys
gold_csv, requested = sys.argv[1], sys.argv[2]
ids = []
with open(gold_csv) as f:
    for row in csv.DictReader(f):
        pid = (row.get("anthology_id") or "").strip()
        if pid:
            ids.append(pid)
if not ids:
    sys.stderr.write(f"ERROR: no paper ids in gold csv\n"); sys.exit(2)

if requested:
    if requested in ids:
        print(requested); sys.exit(0)
    # Not in gold — suggest closest recent papers
    sys.stderr.write(f"WARN: '{requested}' is NOT in the gold CSV.\n")
    recent = sorted([p for p in ids if p[:4].isdigit() and int(p[:4]) >= 2020],
                    reverse=True)[:5]
    sys.stderr.write(f"Suggested recent papers that ARE in gold:\n")
    for p in recent:
        sys.stderr.write(f"  - {p}\n")
    sys.stderr.write(f"Falling back to the first gold paper: {ids[0]}\n\n")
print(ids[0])
PY
)
PAPER_ID=$(echo "$PAPER_ID" | head -1 | tr -d '\r\n')
echo "smoke-testing with paper: $PAPER_ID"
echo ""

# Pull pipeline ids from YAML
PIPELINE_IDS=$(grep -E "^\s*-\s*id:\s*" "$CONFIG" | sed -E "s/^\s*-\s*id:\s*//" | tr -d '"')

echo "==================== PER-PIPELINE SMOKE ===================="
PASS=()
FAIL=()
for pid in $PIPELINE_IDS; do
    printf "  %-35s " "$pid"
    log="$SMOKE_OUT/${pid}.log"
    # Run the pipeline against exactly the requested paper.
    python3 scripts/run_pipeline_comparison.py \
        --config "$CONFIG" \
        --pipelines "$pid" \
        --gold-csv "$GOLD_CSV" \
        --sources-root "$SOURCES_ROOT" \
        --s2orc-norm-root "$S2ORC_NORM_ROOT" \
        --out-dir "$SMOKE_OUT" \
        --paper-ids "$PAPER_ID" \
        --skip-score \
        --no-ror \
        > "$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        echo "✗ FAIL (exit $rc)"
        FAIL+=("$pid")
        # Snip the most informative error: prefer grep matches, fall back to
        # the last non-empty line so "SystemExit: ..." messages aren't lost.
        last_err=$(tail -30 "$log" | grep -iE "error|exception|http|denied|auth|404|401|403|fatal|SystemExit|requested" | tail -1)
        if [ -z "$last_err" ]; then
            last_err=$(grep -v '^\s*$' "$log" | tail -1)
        fi
        echo "      → ${last_err:0:160}"
        continue
    fi
    # Check the written prediction has at least one author OR has an error field
    pred_file="$SMOKE_OUT/predictions_${pid}.jsonl"
    if [ ! -f "$pred_file" ]; then
        echo "✗ FAIL (no predictions file)"
        FAIL+=("$pid")
        continue
    fi
    n_authors=$(python3 -c "
import json
with open('$pred_file') as f:
    for line in f:
        rec = json.loads(line)
        err = (rec.get('_pipeline') or {}).get('error')
        if err:
            print('ERR:'+err[:80])
            break
        print(len(rec.get('authors') or []))
        break
")
    if [[ "$n_authors" == ERR:* ]]; then
        err_msg="${n_authors#ERR:}"
        # Coverage gap: "S2ORC has no record" / "missing JSON" / "no TEI XML"
        # are data-availability issues, NOT pipeline/auth failures.
        if echo "$err_msg" | grep -qiE "no record|missing|no TEI XML|no PyMu"; then
            echo "⚠  data gap (pipeline OK, this paper not in source)"
            echo "      → ${err_msg:0:120}"
            PASS+=("$pid")
        else
            echo "✗ runtime error"
            echo "      → $err_msg"
            FAIL+=("$pid")
        fi
    elif [ "$n_authors" = "0" ]; then
        echo "⚠  ok but 0 authors extracted (model may be too small or paper too sparse)"
        PASS+=("$pid")
    else
        echo "✓ ok ($n_authors authors extracted)"
        PASS+=("$pid")
    fi
done
echo ""

echo "==================== SUMMARY ===================="
echo "✓ PASS (${#PASS[@]}):"
for p in "${PASS[@]}"; do echo "    $p"; done
echo ""
echo "✗ FAIL (${#FAIL[@]}):"
for p in "${FAIL[@]}"; do echo "    $p"; done
echo ""
if [ "${#FAIL[@]}" -gt 0 ]; then
    echo "Per-pipeline logs in: $SMOKE_OUT/<pipeline_id>.log"
    echo "Inspect a failure with: cat $SMOKE_OUT/<pipeline_id>.log"
    exit 1
fi
echo "All pipelines passed smoke test."
echo "Predictions cleaned up — they're not used. ($SMOKE_OUT will be removed.)"
rm -rf "$SMOKE_OUT"
