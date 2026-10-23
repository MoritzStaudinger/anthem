#!/usr/bin/env bash
# Build a fully anonymised sibling repo for conference submission.
# Idempotent: nukes any existing target dir and recreates from scratch.
#
# Usage:
#   bash scripts/build_anon_release.sh [target_dir]
#
# Default target: ~/PycharmProjects/Comparer_flask_anon

set -euo pipefail

SRC="$(cd "$(dirname "$0")/.." && pwd)"
DST="${1:-$HOME/PycharmProjects/Comparer_flask_anon}"

echo "# src: $SRC"
echo "# dst: $DST"
[ -d "$DST" ] && { echo "# nuking existing $DST"; rm -rf "$DST"; }
mkdir -p "$DST"

echo "# 1. rsync (exclude private/derived files)"
rsync -a \
    --exclude='.git/' \
    --exclude='.claude/' \
    --exclude='.idea/' \
    --exclude='.venv/' \
    --exclude='.env' \
    --exclude='.pytest_cache/' \
    --exclude='.ipynb_checkpoints/' \
    --exclude='__pycache__/' \
    --exclude='legacy/' \
    --exclude='reports/' \
    --exclude='nginx/' \
    --exclude='data/local_index_acl/' \
    --exclude='data/local_index_scico_fulltext/' \
    --exclude='data/cache/' \
    --exclude='data/temp/' \
    --exclude='backend/tuwien/' \
    --exclude='static/' \
    --exclude='templates/' \
    --exclude='docker-compose.yml' \
    --exclude='backend/Dockerfile' \
    --exclude='frontend/Dockerfile' \
    --exclude='*.pdf' \
    --exclude='test.py' \
    --exclude='_*.py' \
    "$SRC/" "$DST/"

echo "# 2. bulk path/identifier scrub (across source files)"
find "$DST" -type f \( -name "*.py" -o -name "*.sh" -o -name "*.sbatch" \
    -o -name "*.yml" -o -name "*.yaml" -o -name "*.md" -o -name "*.txt" \
    -o -name "*.json" -o -name "Dockerfile" -o -name "*.cfg" -o -name "*.toml" \
    -o -name "*.ini" -o -name "*.def" \) \
  -not -path "*/data/datastructure/*" \
  -not -path "*/data/eval/scico_testset*" \
  -not -path "*/data/eval/acl_testset*" \
  -not -path "*/__pycache__/*" \
  -print0 | xargs -0 sed -i \
    -e 's|/share/moritz\.staudinger/acl_data|${DATA_ROOT:-./data/acl_data}|g' \
    -e 's|/share/moritz\.staudinger|${DATA_ROOT}|g' \
    -e 's|moritz\.staudinger@tuwien\.ac\.at|anon@example.com|g' \
    -e 's|compare\.ds-ifs\.tuwien\.ac\.at|<deployment-host>|g' \
    -e 's|aqueduct\.ai\.datalab\.tuwien\.ac\.at/|${AQUEDUCT_BASE_URL_VALUE}|g' \
    -e 's|https://aqueduct\.ai\.datalab\.tuwien\.ac\.at|https://<aqueduct-host>|g' \
    -e 's|aqueduct\.ai\.datalab\.tuwien\.ac\.at|<aqueduct-host>|g' \
    -e 's|repositum\.tuwien\.at|<repositum-host>|g' \
    -e 's|tiss\.tuwien\.ac\.at|<tiss-host>|g' \
    -e 's|/home/moritz\.staudinger|${HOME}|g' \
    -e 's|/home/moritz/PycharmProjects/Comparer_flask|.|g' \
    -e 's|/home/moritz|${HOME}|g' \
    -e 's|MoritzStaudinger|Anonymous Authors|g' \
    -e 's|Moritz Staudinger|Anonymous Authors|g' \
    -e 's|moritz\.staudinger|anon|g' \
    -e 's|~/acl_processor|<acl-processor-repo>|g' \
    -e 's|Technische Universität Wien|[Home Institution]|g' \
    -e 's|Vienna University of Technology|[Home Institution]|g' \
    -e 's|TU Vienna|[Home Institution]|g' \
    -e 's|TU Wien|[Home Institution]|g' \
    -e 's|TUWien|HomeInstitution|g' \
    -e 's|TU_Wien|Home_Institution|g' \
    -e 's|tuwien|home_inst|g'

# Fix LICENSE
sed -i 's|MoritzStaudinger|Anonymous Authors|g; s|Copyright (c) 2025 Anonymous Authors|Copyright (c) 2026 Anonymous Authors|g' "$DST/LICENSE"

echo "# 3. surgical code cleanup (TU-Wien tool + handler dead code)"

# agent_tools.py: drop the imported helper, the tool function, and the registration
DST_DIR="$DST" python3 - <<'PY'
from pathlib import Path
import os, re
DST = os.environ.get("DST_DIR")
p = Path(DST) / "backend/rag/agent_tools.py"
src = p.read_text()

# Drop the home_inst import line
src = re.sub(
    r"^from backend\.home_inst\.utils import .*$",
    "# Removed for anonymization: import of an institution-specific publication-archive helper.",
    src, flags=re.MULTILINE,
)

# Drop the entire search_tu_wien_papers_tool function (def line through its closing return)
src = re.sub(
    r"def search_tu_wien_papers_tool\([^)]*\)[^:]*:.*?\n\n\n",
    "# NOTE: a `search_tu_wien_papers_tool` function (institution-specific\n"
    "# publication-archive search) was removed for double-blind submission.\n\n\n",
    src, flags=re.DOTALL,
)

# Drop the tool registration
src = re.sub(
    r"\s*FunctionTool\.from_defaults\(\s*\n\s*fn=search_tu_wien_papers_tool,\s*name=\"search_tu_wien_papers\"\s*\n\s*\),",
    "",
    src,
)

# Fix the docstring crumb
src = src.replace(
    "    Don't use for: [Home Institution]-only queries (use search_tu_wien_papers) or",
    "    Don't use for:",
)
p.write_text(src)
PY

# pipeline.py: drop the home_inst import, wrapper, handler, and dispatch case
DST_DIR="$DST" python3 - <<'PY'
from pathlib import Path
import os, re
DST = os.environ.get("DST_DIR")
p = Path(DST) / "backend/rag/pipeline.py"
src = p.read_text()

src = re.sub(
    r"^from backend\.home_inst\.utils import .*$",
    "# Removed for anonymization: helpers from `backend.home_inst.utils`\n# (an institution-specific publication-archive client).",
    src, flags=re.MULTILINE,
)

# Drop handle_home_inst_wrapper through the end of handle_home_inst
src = re.sub(
    r"def handle_home_inst_wrapper\(.*?(?=\ndef process_input\b|\ndef [a-zA-Z_]+\(.*?\):\s*\n\s*\"\"\"\s*\n?\s*Run )",
    "# NOTE: `handle_home_inst_wrapper` and `handle_home_inst` (handlers for\n"
    "# the authors' home-institution publication archive) removed for\n"
    "# double-blind submission.\n\n",
    src, flags=re.DOTALL,
)

# Drop the dispatch case
src = re.sub(
    r'\n\s*elif data_type == "home_inst":\s*\n\s*yield from handle_home_inst_wrapper\([^)]*\)\n',
    '\n    # Removed: dispatch case for `data_type == "home_inst"`.\n',
    src,
)
p.write_text(src)
PY

# rag_utils.py: drop the [Home Institution]-named classifier rules
DST_DIR="$DST" python3 - <<'PY'
from pathlib import Path
import os
DST = os.environ.get("DST_DIR")
p = Path(DST) / "backend/rag/rag_utils.py"
src = p.read_text()
for old, new in [
    ('"- Use \'internal_kg\' if [Home Institution] is in the query, and no other university is given. The goal is to fall back on internal datasources. For this replace single_university if [Home Institution], [Home Institution] is mentioned (e.g. What has [Home Institution] contributed to Information Retrieval in 2024?)"',
     '# (removed for anonymization)'),
    ('"- Use \'home_inst\' if the query specifically asks about [Home Institution], [Home Institution], [Home Institution], or [Home Institution] research, publications, or projects.\\n"',
     '# (removed for anonymization)'),
    ('"- Use \'internal_kg\' if [Home Institution] is in the query, and no other university is given. The goal is to fall back on internal datasources. Do this just for [Home Institution]"',
     '# (removed for anonymization)'),
    ('"- For [Home Institution] queries, set this to \'[Home Institution]\'.\\n"',
     '# (removed for anonymization)'),
    ('"- For [Home Institution] queries, set this to \'AT\'.\\n"',
     '# (removed for anonymization)'),
    ('(e.g., "[Home Institution]")', '(e.g., "MIT")'),
    ("'**[Home Institution]'s Data Science unit produced 47 IR papers in 2024, with",
     "'**Example University X produced 47 IR papers in 2024, with"),
]:
    src = src.replace(old, new)
p.write_text(src)
PY

# agent.py: prompt cleanup
DST_DIR="$DST" python3 - <<'PY'
from pathlib import Path
import os
DST = os.environ.get("DST_DIR")
p = Path(DST) / "backend/rag/agent.py"
src = p.read_text()
src = src.replace(
    "- search_tu_wien_papers(query, year_start?, org_units?)\n"
    "    — [Home Institution] Repositum (preferred for [Home Institution]-specific queries).\n",
    "",
)
src = src.replace("OpenAlex / CORE / [Home Institution] / Neo4j KG", "OpenAlex / CORE / Neo4j KG")
src = src.replace("internal Neo4j KG of [Home Institution] research units (Data Science, Knowledge",
                  "internal Neo4j KG of internal research units (Data Science, Knowledge")
src = src.replace("a research unit, or a [Home Institution] query.", "or a research unit query.")
src = src.replace("- [Home Institution] queries → prefer the [Home Institution] / KG tools first.\n", "")
src = src.replace("**[Home Institution]'s Data Science unit produced 47 IR papers in 2024, with",
                  "**Example University X produced 47 IR papers in 2024, with")
src = src.replace("from [Home Institution]\"-style accusations", "from Institution X\"-style accusations")
src = src.replace("'citation [4] attributes paper X to [Home Institution] but evidence E12 lists",
                  "'citation [4] attributes paper X to Institution X but evidence E12 lists")
p.write_text(src)
PY

# Misc small files
sed -i '/"home_inst",/d' "$DST/tests/test_lookup_live.py"
sed -i 's|"institution": "\[Home Institution\]"|"institution": "MIT"|g' "$DST/scripts/eval_acl_retrieval.py" 2>/dev/null || true
sed -i 's|\[Home Institution\] DataLab|the Aqueduct gateway|g' \
    "$DST/scripts/paraphrase_scico_questions.py" \
    "$DST/scripts/generate_phd_questions.py" \
    "$DST/backend/eval/metrics.py" \
    "$DST/README.md" 2>/dev/null || true
sed -i 's|OpenAlex / CORE / \[Home Institution\] Repositum / Neo4j|OpenAlex / CORE / Neo4j|g' \
    "$DST/scripts/eval_scico.py" 2>/dev/null || true
sed -i 's|a [Home Institution] colleague or external|a local colleague or external|g' "$DST/data/eval/PROTOCOL.md" 2>/dev/null || true
sed -i 's|\*\*Owner:\*\* Moritz Staudinger|**Owner:** Anonymous Authors|g; s|"reviewed_by": "moritz"|"reviewed_by": "<reviewer>"|g' "$DST/data/eval/PROTOCOL.md" 2>/dev/null || true

echo "# 4. write .gitignore + clean README"
cat > "$DST/.gitignore" <<'EOF'
# env / secrets
.env
.env.local

# python cruft
__pycache__/
*.pyc
*.pyo
.venv/
.pytest_cache/
.ipynb_checkpoints/

# IDE
.idea/
.vscode/
.claude/

# eval outputs + indexes (reproducible from scripts)
reports/
data/local_index_*/
data/cache/
data/temp/

# build / dist
dist/
build/
*.egg-info/
EOF

# Drop the docker-style empty dirs that survived rsync (if any)
rm -rf "$DST/static" "$DST/templates" 2>/dev/null || true

cat > "$DST/README.md" <<'EOF'
# Comparer — Agentic + Pipelined RAG Evaluation for Institutional Research Comparison

Anonymous code release accompanying the submitted paper. The repository implements and evaluates two retrieval-augmented-generation (RAG) variants — a deterministic **pipeline** mode and an **agentic** mode — on a question-answering task over the ACL Anthology, plus a closed-book baseline. The evaluation reports retrieval, NLG, and LLM-as-judge metrics across multiple generator and judge models.

## Layout

```
backend/                    # Flask backend + RAG core
  api/                      # HTTP routes (not used by the eval)
  core/                     # CORE Open Research fulltext client
  eval/                     # in-house metrics + gold-set construction
  neo4j/                    # internal-KG client (optional, eval-irrelevant)
  openalex/                 # OpenAlex client (PyAlex wrapper)
  rag/                      # agent (FunctionAgent + critic), pipeline,
                            #   tool wrappers, local ACL index loader
  utils/                    # misc helpers (dedup-by-DOI, etc.)
frontend/                   # Streamlit UI (not used by the eval)
scripts/                    # eval drivers + per-pass scripts
slurm/                      # cluster wrappers for the scripts in scripts/
data/eval/                  # testset (scico_testset.jsonl) + smaller
                            #   acl_testset.jsonl for retrieval smoke tests
data/datastructure/         # gold artifacts (ground_truth.csv, affiliations/*.json)
tests/                      # pytest fixtures + unit tests
backend/requirements.txt    # runtime deps
dev-requirements.txt        # eval-only deps (RAGAS, ir_measures, BERTScore, ...)
```

## Modes evaluated

- **closed_book** — model answers the question with no retrieval.
- **pipeline** — deterministic chained RAG: classifier → category-specific handler that retrieves docs from the local ACL index, summarises per doc, synthesises an answer.
- **agent** — FunctionAgent over a restricted toolset (`search_local_acl_papers`, `drill_into_papers_fulltext`), with a critic-refine loop bounded by `MAX_REFINE_ITERATIONS`.

## Setup

1. Python 3.11.
2. `pip install -r backend/requirements.txt && pip install -r dev-requirements.txt`
3. Copy `.env.example` → `.env`, fill API keys for the providers you want.
4. Point `DATA_ROOT` at the enriched ACL corpus dir (per-paper JSONs with CSO topics + ROR-normalised affiliations + bib metadata).
5. Build the local ACL vector index once:
   ```bash
   python scripts/build_acl_index.py \
       --scico-root "$DATA_ROOT/ror_parsed_affiliations_90_with_topics_and_bibmeta"
   ```

## Reproducing the eval (per model)

```bash
# Generation (chunked = N parallel sbatches)
bash scripts/chunked_eval.sh agent    <model> <out_dir> 20
bash scripts/chunked_eval.sh pipeline <model> <out_dir> 20
cat <out_dir>/_chunks/chunk_*/answers.jsonl > <out_dir>/answers.jsonl

# Retrieval + NLG (set-level P/R/F1, P@k/nDCG@k/MAP/MRR/Bpref, ROUGE-L, optional BERTScore)
EVAL_BERTSCORE=1 sbatch --partition=GPU-a40 --gres=gpu:1 \
    slurm/eval_ir_measures.sbatch <out_dir>

# LLM-as-judge (per judge, chunked)
bash scripts/chunked_ragas.sh <out_dir> <judge_model_id> 10

# Reports
python scripts/build_eval_report.py --min-ok 50 --min-judge-n 30
python scripts/build_category_report.py --min-ok 50
```

## Testset

`data/eval/scico_testset.jsonl` — 99 questions across 5 categories:

- `overview_university` (20)
- `university_to_university_comparison` (20)
- `cross_country_method` (28)
- `university_country_comparison` (13)
- `field_evolution` (18)

`gold_summary` is empty (the gold is a set of paper IDs, not a reference summary), so ROUGE-L-vs-gold and BERTScore-vs-gold appear as NaN.

## Caveats disclosed in the paper

- **Evidence records are metadata-only** (title + authors + year + DOI; no abstracts). RAGAS Faithfulness is therefore a lower bound — it can't verify substantive claims from titles alone. Especially affects `field_evolution`.
- **`field_evolution` failure**: pipeline mode produces near-zero faithfulness on this category for every model tested. Report builders split per-category so this is visible.
- **Uneven per-category sample sizes** (n = 6–62 per cell). Cells with small n are filtered via `--min-judge-n`.

## What this release does NOT include

For anonymity and to keep the repository focused on the eval contribution:

- `backend/tuwien/` and its companion tool/handler — institution-specific publication-archive code. Eval pipeline does not depend on it. All references in surviving code carry a `# Removed for anonymization` note.
- Deployment-specific configuration (docker-compose, Nginx, certbot).
- Generated eval outputs (`reports/`) and pre-built indexes (`data/local_index_*`). Reproducible from the scripts above.
- Git history of the working repository (this is a fresh `git init`).

## License

MIT (see `LICENSE`).
EOF

echo "# 5. git init + single anon commit"
cd "$DST"
git init -q -b main
git add -A
git -c user.email=anon@example.com -c user.name="Anonymous Authors" \
    commit -q -m "Initial anonymous release accompanying paper submission."

echo ""
echo "# 6. final verification"
echo "--- repo size ---"
du -sh "$DST"
echo "--- tree top-level ---"
ls "$DST"
echo "--- committer history ---"
git log --format='%an <%ae>' | sort -u
echo "--- identifier leaks (excluding explanatory notes) ---"
grep -rniE "moritz|staudinger|tuwien|@tuwien|/share/moritz" "$DST" 2>/dev/null \
  | grep -v __pycache__ \
  | grep -v "removed for anonymization" \
  | grep -v "data/datastructure" \
  | grep -v "search_tu_wien_papers_tool" \
  | head -10
echo "--- testset location ---"
ls -la "$DST/data/eval/" 2>/dev/null

echo ""
echo "Done. Anon repo at: $DST"
