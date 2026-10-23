#!/usr/bin/env bash
#SBATCH --partition=GPU-a40
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=23:59:00
#SBATCH --job-name=extract_gemini
#SBATCH --output=/home/%u/logs/extract_aff/gemini-%j.out
#
# Phase 1 of "add new papers to the affiliations corpus":
#   - diffs pymu_llm dir against the existing enriched corpus to find
#     paper_ids that haven't been processed yet
#   - runs scripts/extract_affiliations_gemini.py on JUST those papers
#     (async, API-bound, no GPU needed)
#   - writes per-paper JSONs (with raw_affiliation + evidence + conflicts)
#     to a NEW output dir — raw extraction, NO ROR fields yet
#
# Phase 2 is slurm/ror_link_v2.sbatch (CPU, HTTP).
# Phase 3 is the existing scripts/enrich_topics.sbatch for CSO labels.
#
# Args (positional, optional):
#   $1  pymu-dir              (default: /share/.../acl_pdfs_pymu_llm)
#   $2  tei-dir               (default: /share/.../acl_pdfs_parsed)
#   $3  raw-dir               (default: /share/.../acl_pdfs_hybrid)
#   $4  existing-corpus-dir   (default: /share/.../ror_parsed_..._and_bibmeta)
#   $5  out-dir               (default: /share/.../gemini_affiliations_NEW)
#   $6  model                 (default: gemini-1.5-flash)
#   $7  max_concurrent        (default: 50)

set -euo pipefail

PYMU_DIR="${1:-/share/moritz.staudinger/acl_data/acl_pdfs_pymu_llm}"
TEI_DIR="${2:-/share/moritz.staudinger/acl_data/acl_pdfs_parsed}"
RAW_DIR="${3:-/share/moritz.staudinger/acl_data/acl_pdfs_hybrid}"
EXISTING_CORPUS_DIR="${4:-/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta}"
OUT_DIR="${5:-/share/moritz.staudinger/acl_data/gemini_affiliations_NEW}"
MODEL="${6:-gemini-1.5-flash}"
MAX_CONCURRENT="${7:-50}"

cd ~/acl_processor 2>/dev/null || cd ~/PycharmProjects/acl_processor
mkdir -p "$HOME/logs/extract_aff" "$OUT_DIR" reports

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

# Auto-install runtime deps the cluster env may not carry. Quiet + idempotent
# — pip skips already-satisfied packages, so warm envs add ~1s overhead.
echo "[gemini] verifying runtime deps ..."
python3 -m pip install -q --disable-pip-version-check \
    google-generativeai json_repair \
    || { echo "[gemini] pip install failed" >&2; exit 2; }
python3 -c "import google.generativeai, json_repair" \
    || { echo "[gemini] post-install import failed" >&2; exit 2; }

echo "[gemini] pymu_dir            : $PYMU_DIR"
echo "[gemini] tei_dir             : $TEI_DIR"
echo "[gemini] raw_dir             : $RAW_DIR"
echo "[gemini] existing_corpus_dir : $EXISTING_CORPUS_DIR"
echo "[gemini] out_dir             : $OUT_DIR"
echo "[gemini] model               : $MODEL"
echo "[gemini] max_concurrent      : $MAX_CONCURRENT"
if [ -n "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]; then
    echo "[gemini] API key             : detected in env"
else
    echo "[gemini] API key             : MISSING — set GEMINI_API_KEY in .env" >&2
    exit 2
fi
echo ""

# Identify new papers: paper_ids in pymu_dir not yet in the existing
# enriched corpus. Stem-only diff (no extension).
LIST_FILE="reports/new_papers_to_extract.txt"
echo "[gemini] identifying new papers ..."
python3 -u -c "
import sys
from pathlib import Path

def stem_strip(name, suffixes):
    # Iteratively strip known suffix chains so '2023.acl-long.123.tei.xml'
    # → '2023.acl-long.123' (matches the extractor's _get_clean_key).
    lower = name.lower()
    changed = True
    while changed:
        changed = False
        for suf in suffixes:
            if lower.endswith(suf):
                name = name[:-len(suf)]
                lower = lower[:-len(suf)]
                changed = True
                break
    return name

TEI_SUFFIXES  = ('.tei.xml', '.xml')
PYMU_SUFFIXES = ('.md', '.jsonl')
RAW_SUFFIXES  = ('.txt', '.text', '.jsonl')

pymu   = Path('$PYMU_DIR')
tei    = Path('$TEI_DIR')
raw    = Path('$RAW_DIR')
corpus = Path('$EXISTING_CORPUS_DIR')
out    = Path('$LIST_FILE')
for label, d in (('pymu', pymu), ('tei', tei), ('raw', raw), ('corpus', corpus)):
    if not d.is_dir():
        sys.exit(f'{label} dir not found: {d}')

def index(d, exts, suffixes):
    ids = set()
    for p in d.rglob('*'):
        if p.is_file() and p.name.lower().endswith(exts):
            ids.add(stem_strip(p.name, suffixes))
    return ids

tei_ids  = index(tei,  ('.xml',),           TEI_SUFFIXES)
pymu_ids = index(pymu, ('.md', '.jsonl'),   PYMU_SUFFIXES)
raw_ids  = index(raw,  ('.txt',),           RAW_SUFFIXES)
union_ids = tei_ids | pymu_ids | raw_ids
existing_ids = {p.stem for p in corpus.glob('*.json')}
new_ids = sorted(union_ids - existing_ids)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text('\n'.join(new_ids) + ('\n' if new_ids else ''))

print(f'tei  paper_ids                : {len(tei_ids)}')
print(f'pymu paper_ids                : {len(pymu_ids)}')
print(f'raw  paper_ids                : {len(raw_ids)}')
print(f'union (any source available)  : {len(union_ids)}')
print(f'existing corpus paper_ids     : {len(existing_ids)}')
print(f'NEW (in union, not in corpus) : {len(new_ids)}')
# Diagnostic split on what's MISSING from union vs corpus
only_tei  = tei_ids  - pymu_ids - raw_ids
only_pymu = pymu_ids - tei_ids  - raw_ids
only_raw  = raw_ids  - tei_ids  - pymu_ids
all_three = tei_ids & pymu_ids & raw_ids
print(f'  papers with TEI only        : {len(only_tei)}')
print(f'  papers with PyMu only       : {len(only_pymu)}')
print(f'  papers with RAW only        : {len(only_raw)}')
print(f'  papers with all three       : {len(all_three)}')
print(f'wrote {out}')
"
N_NEW=$(wc -l < "$LIST_FILE")
if [ "$N_NEW" -eq 0 ]; then
    echo "[gemini] nothing new to extract — exiting."
    exit 0
fi

echo ""
echo "[gemini] running Gemini extraction on $N_NEW new papers ..."
python3 -u scripts/extract_affiliations_gemini.py \
    --tei-dir  "$TEI_DIR" \
    --pymu-dir "$PYMU_DIR" \
    --raw-dir  "$RAW_DIR" \
    --out-dir  "$OUT_DIR" \
    --list-file "$LIST_FILE" \
    --model "$MODEL" \
    --max-concurrent "$MAX_CONCURRENT" \
    -v

echo ""
echo "[gemini] DONE. Raw per-paper extractions in: $OUT_DIR"
ls "$OUT_DIR" | wc -l
echo ""
echo "Next: sbatch slurm/ror_link_v2.sbatch $OUT_DIR"
