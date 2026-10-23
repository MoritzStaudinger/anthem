#!/usr/bin/env bash
# Submits one slurm job per pipeline in the YAML config, then a final
# aggregation job that depends on every per-pipeline job. End result:
# total wall-clock = slowest single pipeline (≈30-60 min for the heaviest
# LLM), instead of sum-of-all (5-10 hours).
#
# Usage:
#   bash slurm/aff_compare_submit_all.sh [config] [out_dir]
#
# Defaults:
#   config  = configs/pipelines.yaml
#   out_dir = reports/pipeline_comparison/
#
# Requires:
#   - .env on the cluster with API keys for whichever LLM backends are
#     referenced in the config.
#   - cache/ror_lookup_cache.json (auto-created if missing).

set -euo pipefail

CONFIG="${1:-configs/pipelines.yaml}"
OUT_DIR="${2:-reports/pipeline_comparison}"
GOLD_CSV="${3:-reports/ground_truth_merged.csv}"
SOURCES_ROOT="${4:-/share/moritz.staudinger/acl_data}"
S2ORC_NORM_ROOT="${5:-/share/moritz.staudinger/acl_data/s2orc_affiliations}"

if [ ! -f "$CONFIG" ]; then
    echo "config not found: $CONFIG" >&2
    exit 1
fi

# Parse pipeline ids from the YAML config (very dumb parser — depends on
# the per-entry id field being on its own line, which our generator
# follows). For anything more complex, use python -c "import yaml; ...".
PIPELINE_IDS=$(grep -E "^\s*-\s*id:\s*" "$CONFIG" | sed -E "s/^\s*-\s*id:\s*//" | tr -d '"')
if [ -z "$PIPELINE_IDS" ]; then
    echo "no '- id: ...' entries found in $CONFIG" >&2
    exit 1
fi

echo "Submitting one job per pipeline in $CONFIG:"
echo "$PIPELINE_IDS" | sed 's/^/  - /'
echo ""

JOB_IDS=()
# Fully parallel extraction. The per-pipeline sbatch passes --no-ror,
# so no pipeline calls the ROR API during this phase — there's no
# rate-limit contention to worry about. ROR linking is centralised in
# the aggregation phase via scripts/ror_link_predictions.py:
#   - 24 pipelines × ~3 affs per paper × 781 papers ≈ 60k lookups
#   - But only ~2,500 of those are UNIQUE strings → 25x reduction
#   - And the aggregation runs them serially through the shared cache
# Net wall-clock: max(per-pipeline LLM time) + ~5-10 min ROR linking
# + ~5 min scoring ≈ 45-90 min total (vs 4-6h sequential chained).
for pid in $PIPELINE_IDS; do
    JID=$(sbatch --parsable \
        --job-name="affc-$pid" \
        slurm/aff_compare_one_pipeline.sbatch \
        "$pid" "$OUT_DIR" "$CONFIG" "$GOLD_CSV" "$SOURCES_ROOT" "$S2ORC_NORM_ROOT")
    echo "  submitted $pid -> job $JID"
    JOB_IDS+=("$JID")
done

# Build the dependency string: aggregation runs after EVERY per-pipeline
# job finishes (regardless of success — we still want to aggregate
# whatever predictions made it through).
DEP=$(IFS=:; echo "afterany:${JOB_IDS[*]}")

AGG_JID=$(sbatch --parsable \
    --dependency="$DEP" \
    --job-name="affc-aggregate" \
    slurm/aff_compare_aggregate.sbatch \
    "$OUT_DIR" "$CONFIG" "$GOLD_CSV" "$SOURCES_ROOT" "$S2ORC_NORM_ROOT")

echo ""
echo "Aggregation job $AGG_JID will run after all per-pipeline jobs finish."
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Final report will land at: $OUT_DIR/comparison_report.json"
echo "                          $OUT_DIR/comparison_summary.csv"
echo "                          $OUT_DIR/merged_per_paper.jsonl"
