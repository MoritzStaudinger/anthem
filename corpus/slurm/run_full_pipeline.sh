#!/usr/bin/env bash
# End-to-end pipeline orchestrator: download → parse (3 ways) →
# Gemini extract → ROR-link → CSO topics → ACL bibmeta enrich →
# corpus stats. Each phase is a slurm sbatch with --dependency=afterok
# on the previous, so the whole chain submits in seconds and runs
# unattended over multiple days of wall time.
#
# Phases (each writes to a distinct dir to keep prior runs intact):
#   0. download_acl.sbatch         → /share/.../acl_pdfs/<id>.pdf
#   1a. grobid_parse.sbatch        → /share/.../acl_pdfs_parsed/<id>.xml
#   1b. pymu_llm_parse.sbatch      → /share/.../acl_pdfs_pymu_llm/<id>.md
#   1c. hybrid_parse.sbatch        → /share/.../acl_pdfs_hybrid/<id>.md
#       (1a/1b/1c run in PARALLEL — they all depend on phase 0 only)
#   2. extract_affiliations_gemini.sbatch
#       → /share/.../gemini_affiliations_NEW/<id>.json
#       (depends on 1a + 1b + 1c — all three source types feed it)
#   3. ror_link_v2.sbatch          → /share/.../ror_parsed_affiliations_NEW_90/
#   4. enrich_topics.sbatch        → ..._NEW_90_with_topics/
#   5. enrich_affiliations.sbatch  → ..._NEW_90_with_topics_and_bibmeta/
#   6. corpus_stats.sbatch         → reports/corpus_stats.md
#
# Skip phases with SKIP=download,grobid,pymu,hybrid,extract,ror,topics,bibmeta,stats
# (comma-separated). Useful when re-running after a fix mid-pipeline.
#
# Usage:
#   bash slurm/run_full_pipeline.sh
#   SKIP=download,grobid bash slurm/run_full_pipeline.sh
#   DATA_ROOT=/share/me/acl_data bash slurm/run_full_pipeline.sh

set -euo pipefail

cd ~/acl_processor 2>/dev/null || cd ~/PycharmProjects/acl_processor

DATA_ROOT="${DATA_ROOT:-/share/moritz.staudinger/acl_data}"
SKIP="${SKIP:-}"

PDF_DIR="$DATA_ROOT/acl_pdfs"
TEI_DIR="$DATA_ROOT/acl_pdfs_parsed"
PYMU_DIR="$DATA_ROOT/acl_pdfs_pymu_llm"
HYBRID_DIR="$DATA_ROOT/acl_pdfs_hybrid"
GEMINI_DIR="$DATA_ROOT/gemini_affiliations_NEW"
ROR_DIR="$DATA_ROOT/ror_parsed_affiliations_NEW_90"
TOPICS_DIR="$DATA_ROOT/ror_parsed_affiliations_NEW_90_with_topics"
BIBMETA_DIR="$DATA_ROOT/ror_parsed_affiliations_NEW_90_with_topics_and_bibmeta"
EXISTING_CORPUS="$DATA_ROOT/ror_parsed_affiliations_90_with_topics_and_bibmeta"
ANTHOLOGY_REPO="$DATA_ROOT/acl-anthology"

skip()    { echo ",${SKIP}," | grep -q ",$1,"; }
submit()  {
    local name="$1"; shift
    local depstr="$1"; shift
    local depflag=()
    [ -n "$depstr" ] && depflag=(--dependency="$depstr")
    local jid
    jid=$(sbatch --parsable "${depflag[@]}" "$@" | tr -d '[:space:]')
    echo "  [$name] submitted job $jid (dep=${depstr:-none})" >&2
    echo "$jid"
}

echo "[orchestrator] data_root        : $DATA_ROOT"
echo "[orchestrator] skip phases      : ${SKIP:-(none)}"
echo ""

# ---- Phase 0: download ----
JID_DOWNLOAD=""
if ! skip download; then
    JID_DOWNLOAD=$(submit download "" \
        slurm/download_acl.sbatch "$PDF_DIR" "$ANTHOLOGY_REPO")
fi

# ---- Phase 1a/1b/1c: parse (parallel, all depend on download) ----
DEP_PARSE="${JID_DOWNLOAD:+afterok:$JID_DOWNLOAD}"

JID_GROBID=""; JID_PYMU=""; JID_HYBRID=""
if ! skip grobid; then
    JID_GROBID=$(submit grobid "$DEP_PARSE" \
        slurm/grobid_parse.sbatch "$PDF_DIR" "$TEI_DIR")
fi
if ! skip pymu; then
    JID_PYMU=$(submit pymu "$DEP_PARSE" \
        slurm/pymu_llm_parse.sbatch "$PDF_DIR" "$PYMU_DIR")
fi
if ! skip hybrid; then
    JID_HYBRID=$(submit hybrid "$DEP_PARSE" \
        slurm/hybrid_parse.sbatch "$PDF_DIR" "$HYBRID_DIR")
fi

# ---- Phase 2: Gemini extract (depends on all three parsers) ----
PARSE_DEPS=()
[ -n "$JID_GROBID" ] && PARSE_DEPS+=("afterok:$JID_GROBID")
[ -n "$JID_PYMU"   ] && PARSE_DEPS+=("afterok:$JID_PYMU")
[ -n "$JID_HYBRID" ] && PARSE_DEPS+=("afterok:$JID_HYBRID")
DEP_EXTRACT=$(IFS=','; echo "${PARSE_DEPS[*]:-}")

JID_EXTRACT=""
if ! skip extract; then
    JID_EXTRACT=$(submit extract "$DEP_EXTRACT" \
        slurm/extract_affiliations_gemini.sbatch \
        "$PYMU_DIR" "$TEI_DIR" "$HYBRID_DIR" "$EXISTING_CORPUS" "$GEMINI_DIR")
fi

# ---- Phase 3: ROR link ----
DEP_ROR="${JID_EXTRACT:+afterok:$JID_EXTRACT}"
JID_ROR=""
if ! skip ror; then
    JID_ROR=$(submit ror "$DEP_ROR" \
        slurm/ror_link_v2.sbatch "$GEMINI_DIR" "$ROR_DIR")
fi

# ---- Phase 4: CSO topics ----
DEP_TOPICS="${JID_ROR:+afterok:$JID_ROR}"
JID_TOPICS=""
if ! skip topics; then
    # Existing CSO sbatch reads paths from env vars; pass via --export so
    # they reach the job and overrides the defaults.
    JID_TOPICS=$(submit topics "$DEP_TOPICS" \
        --export=ALL,AFFILIATIONS_DIR="$ROR_DIR",OUT_DIR="$TOPICS_DIR" \
        scripts/enrich_topics.sbatch)
fi

# ---- Phase 5: ACL bibmeta enrichment ----
DEP_BIBMETA="${JID_TOPICS:+afterok:$JID_TOPICS}"
JID_BIBMETA=""
if ! skip bibmeta; then
    # enrich_affiliations.sbatch's defaults assume the full pipeline
    # was run against the canonical paths — we override its in-script
    # AFFS_DIR/OUT_DIR via env (the sbatch uses local vars but slurm
    # exports propagate to the script's bash). If the script doesn't
    # accept env overrides, swap to the inline python call.
    JID_BIBMETA=$(submit bibmeta "$DEP_BIBMETA" \
        --export=ALL,AFFS_DIR="$TOPICS_DIR",OUT_DIR="$BIBMETA_DIR" \
        slurm/enrich_affiliations.sbatch)
fi

# ---- Phase 6: corpus stats (on the union of old + new) ----
DEP_STATS="${JID_BIBMETA:+afterok:$JID_BIBMETA}"
JID_STATS=""
if ! skip stats; then
    JID_STATS=$(submit stats "$DEP_STATS" \
        slurm/corpus_stats.sbatch "$BIBMETA_DIR")
fi

echo ""
echo "[orchestrator] all phases submitted. Watch progress with:"
echo "  squeue -u \$USER"
echo "  tail -f ~/logs/full_pipeline/*-<jobid>.out"
echo ""
echo "Job IDs (latest at end):"
for var in JID_DOWNLOAD JID_GROBID JID_PYMU JID_HYBRID JID_EXTRACT \
           JID_ROR JID_TOPICS JID_BIBMETA JID_STATS; do
    eval "v=\$$var"
    [ -n "$v" ] && printf "  %-15s %s\n" "$var" "$v"
done
