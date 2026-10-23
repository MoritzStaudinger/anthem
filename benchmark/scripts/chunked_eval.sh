#!/usr/bin/env bash
# Chunked-parallel eval. Splits the testset into N pieces and submits
# N sbatches that each process one chunk. Each chunk writes to its
# own subdir so concurrent appends to the same answers.jsonl are
# impossible. After all chunks finish, concat into one answers.jsonl.
#
# Usage:
#     bash scripts/chunked_eval.sh <mode> <model> <out_dir> [n_chunks] [testset]
#
#     mode      : agent | pipeline | closed_book
#     model     : LLM model id (e.g., gemini-2.5-pro, deepseek/deepseek-v4-flash)
#     out_dir   : reports/eval_scico/<your_name>
#     n_chunks  : default 10 (10 rows per chunk for the 99-row testset)
#     testset   : default data/eval/scico_testset.jsonl
#
# Examples:
#     bash scripts/chunked_eval.sh pipeline gemini-2.5-pro \
#         reports/eval_scico/pipeline_gemini_2_5_pro_topk5
#
#     bash scripts/chunked_eval.sh agent gpt-5-mini \
#         reports/eval_scico/agent_gpt_5_mini_v2 8
#
# After all chunks finish:
#     cat <out_dir>/_chunks/chunk_*/answers.jsonl > <out_dir>/answers.jsonl
# (the script prints this exact command at the end.)

set -euo pipefail

MODE="${1:?mode required (agent|pipeline|closed_book)}"
MODEL="${2:?model required}"
OUT_DIR="${3:?out_dir required}"
N_CHUNKS="${4:-10}"
TESTSET="${5:-data/eval/scico_testset.jsonl}"

case "$MODE" in
    agent)       SBATCH=slurm/eval_scico_answer.sbatch ;;
    pipeline)    SBATCH=slurm/eval_scico_pipeline.sbatch ;;
    closed_book) SBATCH=slurm/eval_scico_closed_book.sbatch ;;
    *) echo "ERROR: mode must be one of {agent,pipeline,closed_book}, got: $MODE" >&2; exit 2 ;;
esac

[ -f "$TESTSET" ] || { echo "ERROR: testset not found: $TESTSET" >&2; exit 1; }
[ -f "$SBATCH" ]  || { echo "ERROR: sbatch not found: $SBATCH" >&2; exit 1; }

CHUNK_DIR="$OUT_DIR/_chunks"
mkdir -p "$CHUNK_DIR"

total=$(wc -l < "$TESTSET")
per_chunk=$(( (total + N_CHUNKS - 1) / N_CHUNKS ))

echo "# mode      : $MODE"
echo "# model     : $MODEL"
echo "# out_dir   : $OUT_DIR"
echo "# testset   : $TESTSET ($total rows)"
echo "# n_chunks  : $N_CHUNKS  ($per_chunk rows/chunk)"
echo ""

submitted=0
for i in $(seq 0 $((N_CHUNKS - 1))); do
    start=$((i * per_chunk + 1))             # 1-indexed for sed
    end=$(( (i + 1) * per_chunk ))
    [ "$end" -gt "$total" ] && end=$total
    [ "$start" -gt "$total" ] && break       # last chunk may be empty

    tag=$(printf 'chunk_%02d' "$i")
    chunk_file="$CHUNK_DIR/${tag}.jsonl"
    chunk_out="$CHUNK_DIR/${tag}"
    mkdir -p "$chunk_out"

    # Slice the testset for this chunk
    sed -n "${start},${end}p" "$TESTSET" > "$chunk_file"
    rows_in_chunk=$(wc -l < "$chunk_file")

    # Submit. The 3rd positional arg in both eval_scico_answer.sbatch
    # and eval_scico_pipeline.sbatch is the testset path — that's how
    # we point each chunk at its own slice.
    jid=$(sbatch --parsable "$SBATCH" "$MODEL" "$chunk_out" "$chunk_file")
    echo "  submitted $tag  rows=$rows_in_chunk  jid=$jid  out=$chunk_out"
    submitted=$((submitted + 1))
done

echo ""
echo "# $submitted chunks submitted."
echo "# Monitor: squeue -u \$USER"
echo "# When all chunks complete, merge into a single answers.jsonl with:"
echo "#"
echo "#     cat $CHUNK_DIR/chunk_*/answers.jsonl > $OUT_DIR/answers.jsonl"
echo "#"
echo "# Then run the IR metrics pass on the merged dir:"
echo "#"
echo "#     EVAL_BERTSCORE=1 sbatch --partition=GPU-a40 --gres=gpu:1 \\"
echo "#         slurm/eval_ir_measures.sbatch $OUT_DIR"
