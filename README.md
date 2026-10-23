# ANTHEM

Code for *Singing from the Same ANTHEM: An Enriched ACL Anthology for Scholarly
Comparison*.

ANTHEM links 117,386 ACL Anthology papers to ROR-normalised author affiliations
and Computer Science Ontology topic labels, keeping an evidence span for every
extraction decision, and ships a comparative-QA benchmark whose ground truth is
computed from that metadata rather than hand-annotated.

Dataset: <https://huggingface.co/datasets/Molarix/anthem>

## Layout

| Directory | What |
|---|---|
| `corpus/` | builds the dataset: download, parse, extract affiliations, link to ROR, label topics, and all the evaluation in §3–§4 and the appendices |
| `benchmark/` | the comparative-QA system and its evaluation: retrieval, pipeline and agent RAG modes, RAGAS judging |
| `release/` | packages the pipeline output into the published HuggingFace dataset |

`corpus/` and `benchmark/` were developed as two repositories and keep their own
READMEs, requirements and Slurm scripts. They run independently: `benchmark/`
consumes `corpus/` output from disk, not by import.

## Reproducing the paper

Every number has a script. The reports land under `reports/` in each half.

### Corpus (§3, §B)

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

### Benchmark (§4, §C)

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

Copy `.env.example` to `.env` and fill in whichever provider keys you need. No
key is read from anywhere else; nothing is hardcoded.

```bash
cp .env.example .env
```

The benchmark's LLM wiring lives in `benchmark/backend/rag/rag_utils.py`
(`configure_llm`), and which pipelines the corpus aggregator will accept is
declared in `corpus/configs/pipelines.yaml` — a pipeline without an entry there
is silently dropped from the leaderboard.

## Paths

Scripts and Slurm jobs carry the authors' cluster layout
(`/share/moritz.staudinger/acl_data/...`) as default arguments, because that is
what the reported runs actually used. They are defaults, not requirements:
every path is overridable on the command line. Point them at your own copy of
the data, or at the HuggingFace dataset.

## Not included

- **Source PDFs.** The ACL Anthology serves those; ~196k of them is 80 GB.
- **Pipeline output.** Published as the HuggingFace dataset instead.
- **The trained LoRA adapter** (Qwen2.5-7B-Instruct, r=16). Belongs in a model
  repository.
- **Exploratory notebooks.** `extractor.ipynb` and `analysis.ipynb` from the
  original working repositories are superseded by the scripts above.

## Licence

Code is MIT (`LICENSE`). The dataset is CC-BY-NC-SA-4.0; the underlying papers
keep their ACL Anthology terms, recorded per row in the dataset's `license`
column.
