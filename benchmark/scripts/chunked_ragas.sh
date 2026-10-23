#!/usr/bin/env bash
# Chunked RAGAS judging. Splits a dir's answers.jsonl into N pieces and
# submits N sbatches, each judging one slice with the same judge. After
# all chunks finish, concat the per-chunk ragas_<slug>.jsonl into one
# file in the parent dir.
#
# Usage:
#     bash scripts/chunked_ragas.sh <dir> <judge> [n_chunks]
#
#     dir       : reports/eval_scico/<your_dir>   (must contain answers.jsonl)
#     judge     : judge model id (e.g., qwen/qwen3.5-397b-a17b)
#     n_chunks  : default 10
#
# Example:
#     bash scripts/chunked_ragas.sh \
#         reports/eval_scico/agent_gemini_2_5_pro \
#         qwen/qwen3.5-397b-a17b \
#         10
#
# After all chunks finish, script prints the exact merge command.

set -euo pipefail

DIR="${1:?dir required}"
JUDGE="${2:?judge required}"
N_CHUNKS="${3:-10}"

answers="$DIR/answers.jsonl"
[ -f "$answers" ] || { echo "ERROR: $answers not found" >&2; exit 1; }

# Derive the same slug eval_ragas.py uses (non-alphanumeric → underscore)
SLUG=$(echo "$JUDGE" | sed 's/[^a-zA-Z0-9_]/_/g; s/^_*//; s/_*$//')
RAGAS_FILE="ragas_${SLUG}.jsonl"

CHUNKS_DIR="$DIR/_ragas_chunks_${SLUG}"
mkdir -p "$CHUNKS_DIR"

total=$(wc -l < "$answers")
per_chunk=$(( (total + N_CHUNKS - 1) / N_CHUNKS ))

echo "# dir       : $DIR"
echo "# judge     : $JUDGE   (slug: $SLUG)"
echo "# answers   : $answers ($total rows)"
echo "# n_chunks  : $N_CHUNKS  ($per_chunk rows/chunk)"
echo ""

submitted=0
for i in $(seq 0 $((N_CHUNKS - 1))); do
    start=$((i * per_chunk + 1))
    end=$(( (i + 1) * per_chunk ))
    [ "$end" -gt "$total" ] && end=$total
    [ "$start" -gt "$total" ] && break

    tag=$(printf 'chunk_%02d' "$i")
    chunk_dir="$CHUNKS_DIR/$tag"
    mkdir -p "$chunk_dir"

    sed -n "${start},${end}p" "$answers" > "$chunk_dir/answers.jsonl"
    rows=$(wc -l < "$chunk_dir/answers.jsonl")

    jid=$(sbatch --parsable slurm/eval_ragas.sbatch "$chunk_dir" "$JUDGE")
    echo "  submitted $tag  rows=$rows  jid=$jid"
    submitted=$((submitted + 1))
done

echo ""
echo "# $submitted chunks submitted. Monitor: squeue -u \$USER --name=eval_ragas"
echo "# When all chunks finish, merge with:"
echo "#"
echo "#   cat $CHUNKS_DIR/chunk_*/$RAGAS_FILE > $DIR/$RAGAS_FILE"
echo "#"
echo "# Then optionally clean up: rm -rf $CHUNKS_DIR"
