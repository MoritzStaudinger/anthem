# Comparer — Scientometric Question Answering over the ACL Anthology

Comparer is a RAG-style agent that answers scientometric questions over
the SciCo dataset: papers from the full ACL Anthology with normalized
author affiliations, CSO topic labels, and parsed full text. This repo
contains the agent (`backend/`), the question-generation pipeline
(`scripts/generate_scico_questions.py`), and an end-to-end evaluation
harness (`scripts/eval_scico.py`).

The companion repository [`acl_processor`](../acl_processor/) builds the
SciCo dataset: downloads PDFs, parses them, extracts/normalizes
affiliations, and runs CSO topic classification. See that repo's README
for the data side.

---

## Quickstart

```bash
# 1. Conda env with all deps (llama-index, openai, anthropic, google-genai,
#    bert-score, rouge-score, ...). See "Setup" below.
conda activate scico-eval

# 2. Make sure ~/.env has the required API keys (see "API keys" below).

# 3. One-shot generate-and-evaluate:
bash scripts/run_full_eval.sh
```

That runs:
1. `generate_scico_questions.py` — 100 questions across 5 categories,
   grounded in CSO topic closures
2. `eval_scico.py` — agent answers each, retrieval + LLM-as-judge scoring

Output: `data/eval/scico_testset.jsonl` (questions) and
`logs/eval-<timestamp>.log` (per-question metrics + aggregates).

---

## Setup

### Conda env

```bash
conda create -n scico-eval python=3.11 -y
conda activate scico-eval
pip install -r requirements.txt
# heavy deps not in requirements.txt:
pip install llama-index llama-index-embeddings-huggingface \
    openai anthropic google-genai \
    bert-score rouge-score sentence-transformers tqdm
```

### API keys

`.env` at the project root (loaded automatically by `run_full_eval.sh`):

```
AQUEDUCT_API_KEY=...        # TU Wien gateway for Qwen-3.5-397B (PhD question generation)
OPENAI_API_KEY=...          # gpt-4o-mini (agent + one judge)
ANTHROPIC_API_KEY=...       # claude-sonnet-4-6 (judge)
GOOGLE_API_KEY=...          # gemini-2.5-flash (judge)
```

**Loading `.env` for individual script runs** (anytime you're NOT using
`run_full_eval.sh`):

```bash
set -a && . ./.env && set +a
```

The `set -a` flag turns on auto-export so subprocesses inherit the
variables. Without it, the variables exist in your shell but Python's
`os.environ` can't see them — and you get `RuntimeError: AQUEDUCT_API_KEY
not set in environment` (or similar) even though the file looks right.

Quick verify after sourcing:

```bash
echo "key length: ${#AQUEDUCT_API_KEY}"   # must be > 30
env | grep _API_KEY                        # all keys should be listed
```

### Data prerequisites

The eval reads the **enriched SciCo affiliations** dir (one JSON per
paper, with `topics.cso.union` populated by the CSO classifier). Build
it via the `acl_processor` pipeline:

```bash
cd ../acl_processor
bash scripts/enrich_topics.sbatch   # or: sbatch on the cluster
```

See `acl_processor/README.md` for details. Once built, point Comparer at
the resulting directory via `SCICO_ROOT` (default
`/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics`).

The eval also needs the **PyMu_LLM Markdown** dir for `paper_vs_field`
question generation (full paper bodies). Default `MD_DIR` =
`~/data/acl_pdfs_pymu_llm`.

### Building the local ACL retrieval index

The agent and the non-agentic pipeline both query a persisted
`VectorStoreIndex` built over the SciCo affiliations dir. Build it once
per corpus snapshot:

```bash
# Default config: enriched corpus → data/local_index_acl, batch=10000,
# no-checkpoint mode (single persist at end; ~halves wall vs per-batch
# checkpointing). ~7-12 min on a GPU-a40 node.
sbatch slurm/build_acl_index.sbatch

# Equivalent with explicit args:
sbatch slurm/build_acl_index.sbatch \
    /share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics_and_bibmeta \
    abstracts \
    10000 \
    data/local_index_acl

# Smoke test on 100 papers first (~2-3 min wall, separate persist dir):
sbatch slurm/build_acl_index_smoke.sbatch
```

The build:
- Reads affiliation JSONs from `$SCICO_ROOT` (flat or `<root>/affiliations/`)
- Emits two Documents per paper: `doc_type=abstract` and `doc_type=people`
- Stages itself to `/dev/shm` for fast random-access JSON parse
- Persists to `data/local_index_acl/` then rsyncs back to NFS

Re-run after changing the source corpus (e.g., enriched with new
bibmeta from ACL Anthology):

```bash
rm -rf data/local_index_acl
sbatch slurm/build_acl_index.sbatch
```

---

## What gets evaluated

### Question schema

Each question in `scico_testset.jsonl` is a JSON record:

```json
{
  "id": "scico-overview_university-001",
  "type": "free_form",
  "persona": "head",
  "depth": "overview",
  "query": "What did MIT contribute to machine translations research between 1952 and 2026?",
  "gold_summary": "",
  "gold_paper_ids": [],
  "classifier_gold": {
    "question_type": "single_university",
    "question_category": "overview_university",
    "uni1": "Massachusetts Institute of Technology",
    "uni2": null,
    "country": null,
    "country_b": null,
    "topic": "machine translations",
    "year_start": 1952,
    "year_end": 2026,
    "years": null
  },
  "topic_metadata": {
    "cso_label": "machine translations",
    "cso_url": "https://cso.kmi.open.ac.uk/topics/machine_translations",
    "cso_closure_size": 8,
    "cso_closure": ["machine translations", "neural machine translation", ...],
    "cso_version": "3.5"
  }
}
```

Gold paper-id lists are deliberately **not** embedded — they're computed
at eval time against the current ACL Anthology snapshot, so the question
file stays valid as new proceedings are added.

### Categories

| Category                              | Persona  | Question shape                                                                                       | # questions |
|---------------------------------------|----------|------------------------------------------------------------------------------------------------------|-------------|
| `overview_university`                 | head     | "What did X contribute to T between Y1 and Y2?"                                                      | 16          |
| `university_country_comparison`       | head     | "...compared to others in country C?"                                                                | 10          |
| `university_to_university_comparison` | head     | "...compared to X2?"                                                                                 | 18          |
| `cross_country_method`                | postdoc  | "Which approaches to T dominated in country A vs country B between Y1–Y2, and what does the gap suggest?" | 26    |
| `field_evolution`                     | postdoc  | "How has T research evolved?" (year window varies — full span, recent half, era slice, decade)        | 14         |
| `paper_vs_field`                      | postdoc  | "How does this paper compare to other work on T?" (target paper full-text included)                  | 16          |

**All 100 questions are scientometric cross-paper aggregation queries.**
Single-paper QA (the QASPER domain) is deliberately out of scope so
the eval cleanly differentiates from existing benchmarks. The companion
script `scripts/import_qasper_subset.py` can pull a ~20-question QASPER
subset for *literature calibration* if you want to additionally report
how the system performs on the established single-paper QA task — kept
as an auxiliary eval, not part of the headline 100.

**Topics** are 29 CSO ontology labels picked from the empirical
top-100 most-frequent labels in the enriched corpus (≥~2000 papers
each). See `DEFAULT_TOPICS` in `scripts/generate_scico_questions.py`.

**16 of the 100 questions (paper_vs_field) require the paper's full
text** — not just the abstract. This is why SciCo's full-text release
is load-bearing for the evaluation, not future-work infrastructure.

#### Quality guards in the templated generators

The head/postdoc categories sample `(institution, topic, year window)`
tuples from precomputed corpus indices. Three guards protect against
the common failure modes:

- **Per-(institution, topic) year clamp.** The year window is sampled
  only from years where the chosen institution actually published on
  the chosen topic — not the global 1979–2026 corpus range. This is
  what prevents anachronistic combinations like "deep learning in 1985"
  or "Huawei's NLP research in 1989".
- **Random pair sampling for `university_to_university_comparison`.**
  Pairs are drawn uniformly from the eligible institution set rather
  than via a nested loop, which previously caused one institution to
  dominate `uni1` in every comparison row.
- **Institution-junk filter** (`_looks_junky` in
  `scripts/generate_scico_questions.py`). Drops affiliations that ROR
  surfaces but that aren't research entities: regional-subsidiary
  parens ("Google (Switzerland)", "Microsoft (China)"), fundraising
  NGOs ("Friends of …", "… Foundation", "… Fund"), department-only
  strings, and overlong multi-institution concatenations.

#### Future-work question categories

These would meaningfully extend the dataset's scientometric depth but
require either non-trivial new extraction passes or schema changes
beyond what the EMNLP-revision timeline allows. They are listed here
as the natural follow-ups the SciCo corpus *enables* rather than
shortcomings of the current release.

- **Cross-institution diffusion.**
  *"Which institution first published on T (among ACL papers), and
  which N institutions adopted the topic within the M years that
  followed?"* Needs a per-topic first-publication-year index per
  institution; structured gold scoring (verifying the answer's
  ordering) is more involved than the corpus-snapshot-derived gold the
  other categories use.

- **Author mobility and idea transfer.**
  *"Author A moved from institution I1 to I2 in year Y. Did the topic
  distribution of I2's NLP output shift toward A's prior topics in the
  following 3 years?"* Requires author-identity disambiguation across
  papers (no ORCID present in the corpus); name-collision handling is
  a paper-sized problem on its own.

- **Method genealogy.**
  *"What was the dominant approach to coreference resolution at ACL
  before 2017, and which paper(s) marked the transition to neural
  methods?"* Requires per-paper method extraction beyond what CSO
  topics provide (CSO labels are taxonomic, not method-level).

- **Counterfactual field shape.**
  *"If we exclude papers from the top-3 US institutions, how does the
  rank-ordering of methods for sentiment analysis between 2015–2020
  change?"* Same blocker as method genealogy — needs a method-extraction
  pipeline plus a defensible "ranking" metric.

- **Comparative methodology across papers.**
  *"How does the reward modeling in [paper A] differ from [paper B],
  and which assumption do they share?"* A multi-paper extension of
  `phd_methodology`. Feasible via the Qwen generator with two paper
  bodies as input; the cost is finding paper pairs that share enough
  topic + methodology to be comparable.

- **Negative-result and ablation reasoning.**
  *"In [paper], removing component X caused a Y-point drop on benchmark
  Z. What does the paper attribute this to?"* Doable with the existing
  LLM-generation infrastructure once ablation tables / negative-result
  spans are reliably identifiable in the parsed papers.

### Eval metrics

Per question (in `agent` mode):

- **Retrieval**: Recall@k, Precision@k, F1@k, MRR over the runtime-computed gold set
- **Generation, abstractive**: ROUGE-L vs. retrieved evidence
- **Generation, rubric**: 3-judge LLM-as-judge on 4 axes — faithfulness,
  coverage, attribution_accuracy, relevance (1-5 Likert)
- **Inter-judge agreement**: Spearman ρ across the 3 judges per axis
- **Phantom citation rate**: fraction of cited papers not in the agent's evidence buffer
- **Affiliation correctness**: F1 between system's per-paper affiliations and SciCo gold (`data/datastructure/ground_truth.csv`)
- **Telemetry**: wall-clock, prompt/completion tokens, est. USD cost

### Runtime gold computation

`scico_testset.jsonl` carries the *question metadata* but not the gold
paper ids. At eval time `backend/eval/gold.py:compute_gold_paper_ids()`
builds the gold set per question by intersecting:

1. CSO closure (from `row["topic_metadata"]["cso_closure"]`)
2. Institution(s) (from `row["classifier_gold"]["uni1"]` / `uni2`)
3. Year window (from `row["classifier_gold"]["year_start"]` / `year_end`)

against the enriched-affiliations corpus passed via
`--affiliations-dir`. This keeps the question file corpus-agnostic — the
same questions evaluate cleanly against any SciCo snapshot.

---

## Running the eval

### One-shot (recommended)

```bash
bash scripts/run_full_eval.sh
```

Env-var overrides (see top of `run_full_eval.sh` for the full list):

```bash
OVERSAMPLE=3 MODEL=gpt-4o-mini bash scripts/run_full_eval.sh
```

`OVERSAMPLE=3` generates ~300 candidate questions instead of 100 — for
manual curation before evaluating.

### Skip generation, re-evaluate existing questions

```bash
bash scripts/run_full_eval.sh --skip-generate
```

### Just generate, don't evaluate

```bash
# Required if you're calling the PhD generator (Qwen via Aqueduct):
set -a && . ./.env && set +a

python scripts/generate_scico_questions.py \
    --scico-root /share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics \
    --md-dir /home/$USER/data/acl_pdfs_pymu_llm \
    --out data/eval/scico_testset.jsonl \
    --total-budget 100 --phd-count 24 --phd-model qwen-3.5-397b \
    --no-backup
```

### Just evaluate, don't generate

```bash
# Required: agent + judges all hit external LLM APIs:
set -a && . ./.env && set +a

python scripts/eval_scico.py \
    --testset data/eval/scico_testset.jsonl \
    --affiliations-dir /share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics \
    --mode agent --model gpt-4o-mini \
    --judges gpt-4o-mini claude-sonnet-4-6 gemini-2.5-flash \
    --no-bertscore
```

### Modes

- `--mode retrieval` — calls the SciCo retrieval tool directly; no agent,
  no LLM. Fast and cheap; only retrieval metrics computed.
- `--mode agent` — runs the full SciCo agent (process_input_agent_sync)
  with only SciCo tools exposed. Slower (~2-3 min/question) and costs
  LLM tokens, but produces the full metric suite.

---

## Curation

`OVERSAMPLE=N` over-generates questions for manual curation. A typical workflow:

```bash
# 1. Over-generate 3×
OVERSAMPLE=3 bash scripts/run_full_eval.sh --skip-generate=false \
    QUESTIONS_OUT=data/eval/scico_candidates.jsonl

# 2. Export to CSV for spreadsheet review
python -c "
import json, csv
rows = [json.loads(l) for l in open('data/eval/scico_candidates.jsonl')]
with open('data/eval/scico_candidates.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'category', 'query', 'topic', 'uni1', 'uni2', 'keep', 'notes'])
    for r in rows:
        cg = r.get('classifier_gold') or {}
        w.writerow([r['id'], cg.get('question_category'), r['query'],
                    cg.get('topic'), cg.get('uni1') or '', cg.get('uni2') or '',
                    '', ''])
"

# 3. Review in Excel/Sheets, mark `keep` = Y for the best 100, then
#    filter the JSONL back to keep just those rows:
python -c "
import json, csv
keep_ids = set()
for row in csv.DictReader(open('data/eval/scico_candidates_curated.csv')):
    if (row.get('keep') or '').strip().upper() in ('Y','YES','1'):
        keep_ids.add(row['id'])
rows = [json.loads(l) for l in open('data/eval/scico_candidates.jsonl')]
with open('data/eval/scico_testset.jsonl', 'w') as f:
    for r in rows:
        if r['id'] in keep_ids:
            f.write(json.dumps(r, ensure_ascii=False) + '\\n')
print(f'curated to {len(keep_ids)} questions')
"

# 4. Run eval on curated set
bash scripts/run_full_eval.sh --skip-generate
```

What to look for in curation:

- Drop questions where the topic doesn't match the institution
  (`overview_university` for an institution with no papers on that
  topic)
- Drop questions where `cso_closure_size` is unreasonably broad and
  pulls in cross-domain noise (e.g., IR including CV retrieval)
- Drop LLM-generated PhD questions that are vague or off-topic
- Keep institution + topic diversity (don't end up with 30 questions
  about MIT)

---

## Output format

`eval_scico.py` prints per-row lines and end-of-run aggregates. Per row:

```
[scico-overview_university-001] free_form  R@10=0.42 P@10=0.18 MAP=0.31 mrr=0.50
        rougeL_vs_evidence=0.314
        judge[gpt-4o-mini]: faithfulness=4, coverage=3, attribution_accuracy=4, relevance=5  — ...
        judge[claude-sonnet-4-6]: ...
        judge[gemini-2.5-flash]: ...
        phantom=0.10 (1/10 refs not in evidence)
        aff_correctness mean_F1=0.85 (3 graded, 7 skipped)
        telemetry: 45.2s prompt_tok=4200 completion_tok=380 cost=$0.0042
```

End of run:

```
# free_form.precision@k: 0.187  (n=100)
# free_form.recall@k:    0.341  (n=98)
# judge_grounded.mean.faithfulness: 3.84  (n=300)
...

# Per-persona aggregates
#   [head] free_form.recall@k: ...
#   [postdoc] free_form.recall@k: ...
#   [phd] free_form.recall@k: ...

# Inter-judge agreement (Spearman ρ across 3 judges)
#   faithfulness          mean ρ = 0.612  (3 pairs)
...
```

---

## Files of interest

| Path | Purpose |
|---|---|
| `scripts/generate_scico_questions.py` | Generates the 100-question evaluation set |
| `scripts/generate_phd_questions.py` | LLM-generated PhD-tier questions (called by ↑) |
| `scripts/eval_scico.py` | The eval driver — runs agent, scores everything |
| `scripts/run_full_eval.sh` | One-shot generate + evaluate wrapper |
| `backend/eval/gold.py` | Runtime gold paper-id computation |
| `backend/eval/metrics.py` | Retrieval, NLG, faithfulness metrics |
| `backend/rag/scico_indexing.py` | Builds the local SciCo vector index for retrieval |
| `data/eval/scico_testset.jsonl` | Generated question file |
| `data/datastructure/affiliations/` | Symlink to enriched SciCo JSONs (or local copy) |
| `data/datastructure/ground_truth.csv` | Manually validated affiliations (the small held-out set) |

---

## Known limitations

- **CSO ontology coverage gaps**: no native nodes for multimodality,
  multilinguality, low-resource languages, or AI ethics (beyond
  `fairness in ai`). Those areas are excluded from the topic list.
- **Single-annotator gold for affiliations**: the manually validated
  subset has no inter-annotator agreement. Subset is held out — eval
  gold falls back to the full pipeline output for the rest.
- **Pre-1990 papers have low topic recall**: CSO's vocabulary is
  modern; historical terminology like "mechanical translation" doesn't
  match.
- **Year filters in questions are static**: a question scoped to
  "1952-2026" stays scoped to that range even after 2027 papers are
  added to the corpus. Intentional for benchmark stability.

See `Limitations` section of the paper draft for the full list.
