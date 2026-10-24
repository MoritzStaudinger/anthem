# ANTHEM

Code for *Singing from the Same ANTHEM: An Enriched ACL Anthology for Scholarly
Comparison*.

Dataset: <https://huggingface.co/datasets/Molarix/anthem>

## What the paper is about

Scientometric questions about NLP research — which institutions work on which
topics, how that has shifted, where collaboration happens — are hard to answer
over the ACL Anthology because the Anthology stores no affiliations. The
affiliation data that does exist elsewhere (OpenAlex, Semantic Scholar, S2ORC)
covers the Anthology unevenly, and coverage correlates with DOI availability,
so it thins out exactly on the older papers a longitudinal question needs.

ANTHEM closes that gap by extracting affiliations from the PDFs themselves and
normalising them against ROR, with Computer Science Ontology topic labels on
top. Two things follow from building the metadata rather than borrowing it.

First, the metadata has measurable reliability, and the paper reports it as
bounds rather than a single headline: extraction F1 0.760 against a 781-paper
decade-stratified gold set, ROR linking 0.857 on hand-checked strings, CSO
topic precision 0.713. Roughly 90% of the remaining errors are linking rather
than extraction failures, concentrated in city-level institution clusters and
branched corporate entities.

Second, because the corpus defines the ground truth, a comparative-QA benchmark
can be derived from it rather than annotated. A question such as *"what did
institution X contribute to topic Y between 2022 and 2025"* has a gold answer
computable from the corpus: the ROR id, the CSO topic closure, and the year
range determine the paper set. The benchmark therefore measures whether a
system can filter and rank over scholarly metadata, it is reproducible against
an archived snapshot, and it extends to future snapshots without re-annotation.

Evidence spans are retained for every extraction decision, so a disputed
affiliation can be traced back to the text it came from.

## Layout

| Directory | What |
|---|---|
| `corpus/` | builds the dataset: download, parse, extract affiliations, link to ROR, label topics, plus the evaluation in §3 and the appendices |
| `benchmark/` | the comparative-QA system and its evaluation: retrieval, pipeline and agent RAG modes, RAGAS judging |
| `release/` | packages the pipeline output into the published HuggingFace dataset |

`corpus/` and `benchmark/` grew as two repositories and keep their own READMEs,
requirements and Slurm scripts. They run independently — `benchmark/` reads
`corpus/` output from disk rather than importing it.

## Reproducing the paper

The reports land under `reports/` in each half.

### Corpus

| Result | Command |
|---|---|
| Table 2, affiliation extraction across 11 pipelines | `corpus/scripts/run_pipeline_comparison.py` |
| Table 2 rescored against the verified gold | `corpus/scripts/rescore_table2_corrected.py` |
| Paired bootstrap confidence intervals | `corpus/scripts/bootstrap_paired.py` |
| Per-decade breakdown | `corpus/scripts/rescore_per_decade.py` |
| ROR linking, four strategies vs. human adjudication | `corpus/scripts/relink_strategy_compare.py`, `corpus/scripts/score_relink_vs_human.py` |
| CSO topic-label precision | `corpus/scripts/score_cso.py` |
| LoRA baseline | `corpus/scripts/train_lora_affiliation.py`, `corpus/scripts/infer_lora_on_gold.py` |
| OpenAlex / S2ORC baselines | `corpus/scripts/fetch_openalex_affiliations.py`, `corpus/scripts/s2orc_normalize.py` |
| Extraction cost | `corpus/scripts/cost_extraction.py`, `corpus/scripts/cost_summary.py` |
| Corpus statistics and figures | `corpus/scripts/corpus_stats.py`, `corpus/scripts/aggregate_yearly_stats.py`, `corpus/scripts/plot_*.py` |

### Benchmark

| Result | Command |
|---|---|
| Question generation and curation | `benchmark/scripts/generate_scico_questions.py`, `benchmark/scripts/build_curated_testset.py` |
| Gold paper sets, precomputed from the corpus | `benchmark/scripts/precompute_gold_paper_ids.py` |
| End-to-end evaluation of a system | `benchmark/scripts/eval_scico.py` |
| Failures-as-zero rescoring | `benchmark/scripts/rescore_failures_as_zero.py` |
| Oracle retrieval ceiling | `benchmark/scripts/oracle_ceiling.py` |
| RAGAS judging and inter-judge agreement | `benchmark/scripts/eval_ragas.py`, `benchmark/scripts/ragas_judge_agreement.py` |
| Bootstrap intervals | `benchmark/scripts/bootstrap_scores.py` |
| Result tables | `benchmark/scripts/build_eval_report.py` |

### Release

```bash
cd release && make release && make check && make upload
```

## Configuration

Copy `.env.example` to `.env` and fill in the provider keys you need. Nothing
is read from anywhere else and nothing is hardcoded.

```bash
cp .env.example .env
```

The benchmark's LLM wiring is in `benchmark/backend/rag/rag_utils.py`
(`configure_llm`). Which pipelines the corpus aggregator accepts is declared in
`corpus/configs/pipelines.yaml` — a pipeline without an entry there is dropped
from the leaderboard without warning.

## Paths

Scripts and Slurm jobs carry the cluster layout they were run on
(`/share/moritz.staudinger/acl_data/...`) as default arguments. They are
defaults, not requirements: every path is overridable on the command line.
Point them at your own copy of the data, or at the HuggingFace dataset.

## Not included

- **Source PDFs.** The Anthology serves those, and ~196k of them is 80 GB.
- **Pipeline output.** Published as the HuggingFace dataset instead.
- **Exploratory notebooks.** `extractor.ipynb` and `analysis.ipynb` from the
  original working repositories are superseded by the scripts above.

## Licence

Code is MIT (`LICENSE`). The dataset is CC-BY-NC-SA-4.0; the underlying papers
keep their ACL Anthology terms, recorded per row in the dataset's `license`
column.
