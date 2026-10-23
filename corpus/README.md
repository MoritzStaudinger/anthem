# acl_processor — SciCo dataset construction pipeline

Builds the SciCo dataset: downloads PDFs from the ACL Anthology, parses
them three ways (GROBID / PyMuPDF+LLM / hybrid OCR), extracts and
normalizes author affiliations against ROR, and enriches each paper
with CSO topic classifications.

The companion repo [`Comparer_flask`](../Comparer_flask/) consumes the
output of this pipeline for question answering + evaluation.

---

## Pipeline overview

```
                        ┌─────────────────────┐
                        │ ACL Anthology       │
                        │ metadata (git)      │
                        └──────────┬──────────┘
                                   ▼
                        ┌─────────────────────┐
                        │ acl_downloader      │  → /share/.../acl_pdfs/
                        └──────────┬──────────┘
                                   ▼
                  ┌────────────────┴─────────────────┐
                  ▼                ▼                  ▼
            ┌──────────┐    ┌──────────┐       ┌──────────────┐
            │ GROBID   │    │ PyMu+LLM │       │ Hybrid OCR   │
            │ → TEI    │    │ → MD     │       │ → JSONL      │
            └────┬─────┘    └────┬─────┘       └──────┬───────┘
                 └────────────────┼────────────────────┘
                                  ▼
                       ┌─────────────────────┐
                       │ Gemini affiliation  │  → acl_affiliations_json/
                       │ extraction          │
                       └──────────┬──────────┘
                                  ▼
                       ┌─────────────────────┐
                       │ ROR normalization   │  → ror_parsed_affiliations_90/
                       └──────────┬──────────┘
                                  ▼
                       ┌─────────────────────┐
                       │ CSO topic           │  → ror_parsed_affiliations_90_with_topics/
                       │ enrichment          │
                       └─────────────────────┘
```

---

## Adding a new batch of ACL Anthology papers

The ACL Anthology adds papers continuously (NAACL/ACL/EMNLP each year +
workshops + findings). To pick up new papers:

### 1. Update the anthology metadata clone

```bash
cd /share/moritz.staudinger/acl_data/acl-anthology
git pull
```

If there's nothing new to pull, the anthology editors haven't published
new metadata yet. Conference proceedings typically land:
- ACL: June (cycle), with cleanup in following weeks
- EMNLP: November
- NAACL: June
- Findings + workshops: rolling

### 2. Submit the downloader

```bash
cd ~/acl_processor
sbatch apptainer-compose.sh acl_download \
    /share/moritz.staudinger/acl_data/acl-anthology \
    /share/moritz.staudinger/acl_pdfs/
```

The downloader walks the anthology metadata, identifies every paper
with a PDF URL, and downloads it as `<paper_id>.pdf`. Already-downloaded
PDFs are skipped, so only the new ones actually hit the network.

Watch progress:
```bash
squeue -u $USER
tail -f ~/acl_processor/slurm-<jobid>.out
```

Failures are logged to `~/logs/acl_download/acl_download_errors.log`.

### 3. Find which papers each parser still needs

`move_diff_to_directory.py` compares the PDF set against each parser's
output and stages the not-yet-processed PDFs in dedicated `*_diff` dirs:

```bash
cd ~/acl_processor
python scripts/move_diff_to_directory.py

# Confirm what's in the diff dirs (should be small after the first run,
# only the new papers from this download)
ls /share/moritz.staudinger/acl_pdfs_grobid_diff/ | wc -l
ls /share/moritz.staudinger/acl_pdfs_pymu_diff/   | wc -l
ls /share/moritz.staudinger/acl_pdfs_hybrid_diff/ | wc -l
```

### 4. Run the three parsers (in parallel)

GROBID → TEI XML:
```bash
ORDER_BY=name ORDER_DIR=desc sbatch apptainer-compose_watchdog.sh acl_parse 8 \
    /share/moritz.staudinger/acl_pdfs_grobid_diff/ \
    /home/$USER/data/acl_pdfs_parsed
```

PyMuPDF + LLM → Markdown:
```bash
sbatch apptainer-compose.sh acl_parse_pymu_llm \
    /share/moritz.staudinger/acl_pdfs_pymu_diff/ \
    /home/$USER/data/acl_pdfs_pymu_llm \
    /home/$USER/data/acl_lists_temp
```

Hybrid PyMuPDF + Tesseract OCR fallback → JSONL:
```bash
sbatch apptainer-compose.sh acl_parse_hybrid_sharded \
    /share/moritz.staudinger/acl_pdfs_hybrid_diff/ \
    /home/$USER/data/acl_pdfs_hybrid \
    /home/$USER/data/acl_lists_temp_hybrid
```

Watch all three; they run independently.

### 5. Affiliation extraction

Two paths:

**(a) Qwen-2.5-7B on GPU node** (local, no API cost):
```bash
sbatch apptainer-compose.sh acl_extract_affiliations_qwen \
    /home/$USER/data/acl_tei \
    /home/$USER/data/acl_pdfs_pymu_llm \
    /home/$USER/data/acl_pdfs_hybrid \
    /home/$USER/data/acl_affiliations \
    /home/$USER/data/acl_lists_temp
```

**(b) Gemini API** (matches the existing `_provenance.model:
"gemini-1.5-flash"` in production SciCo JSONs):
```bash
python run_affiliation_targets.py \
    --pymu-dir /home/$USER/data/acl_pdfs_pymu_llm \
    --out-dir /share/moritz.staudinger/acl_data/gemini_affiliations \
    --model gemini-1.5-flash
```
Needs `GOOGLE_API_KEY` in `.env`.

For continuity with the existing 2021-2025 data, prefer (b).

### 6. ROR normalization

(Pipeline step — produces `ror_parsed_affiliations_90/` from the raw
Gemini output. Run after step 5.)

### 7. CSO topic enrichment

```bash
# One-time env setup on the cluster login node
bash scripts/prepare_env.sh

# Submit the enrichment
PROJECT_DIR=~/acl_processor \
AFFILIATIONS_DIR=/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90 \
OUT_DIR=/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics \
ANTHOLOGY_REPO=/share/moritz.staudinger/acl_data/acl-anthology \
MD_DIR=/home/$USER/data/acl_pdfs_pymu_llm \
PY_ENV=scico \
sbatch scripts/enrich_topics.sbatch
```

The CSO enrichment:
- Loads abstracts from anthology metadata (primary) or the
  PyMu_LLM Markdown (fallback)
- Runs the CSO Classifier on title + abstract
- Writes the `topics.cso.union` field into each paper's JSON
- Idempotent: re-running skips papers that already have a
  `topics._provenance` block

Expected runtime: ~3-5h for 117k papers at 32 CSO workers on a single
node (CSO is CPU-bound; no GPU used).

After completion, generate the coverage report:
```bash
conda activate scico
python scripts/topic_coverage_report.py \
    --affiliations-dir /share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics \
    --top 100 --out reports/coverage_full.jsonl
```

---

## Going from raw output to the Comparer eval

Once the CSO-enriched JSONs are in
`/share/moritz.staudinger/acl_data/ror_parsed_affiliations_90_with_topics/`,
hop over to `../Comparer_flask/` and run:

```bash
cd ../Comparer_flask
bash scripts/run_full_eval.sh
```

See `Comparer_flask/README.md` for the eval side.

---

## Files of interest

| Path | Purpose |
|---|---|
| `scripts/acl_downloader.py` | Walks anthology metadata + downloads PDFs |
| `apptainer-compose.sh` | Orchestrates downloader + parsers + extractors via Slurm/Apptainer |
| `apptainer-compose_watchdog.sh` | Watchdog variant for long-running parser jobs |
| `scripts/move_diff_to_directory.py` | Stages new (not-yet-parsed) PDFs into `*_diff/` dirs |
| `scripts/hybrid_parse_pdfs.py` | PyMuPDF + Tesseract OCR fallback parser |
| `scripts/pymu_llm_parser.py` | PyMuPDF + LLM markdown parser |
| `scripts/extract_affiliations.py` | Local Qwen-based affiliation extractor |
| `run_affiliation_targets.py` | Gemini-API affiliation extractor (matches production) |
| `scripts/enrich_with_topics.py` | CSO topic classification |
| `scripts/enrich_topics.sbatch` | Slurm wrapper for the CSO enrichment |
| `scripts/topic_coverage_report.py` | Per-CSO-label coverage report over the enriched corpus |
| `scripts/sample_ground_truth.py` | Reproducible sampling for manual affiliation annotation |
| `scripts/export_authors_for_annotation.py` | Builds annotator-ready CSV from a sampled subset |
| `scripts/prepare_env.sh` | One-shot conda env setup for the CSO enrichment |

---

## Cluster setup notes (TU Wien DataLab)

- All visible Slurm partitions are GPU-named (`GPU-a100s`, `GPU-a40`,
  etc.) but the CSO enrichment doesn't use a GPU. The
  `enrich_topics.sbatch` script omits `--gres` to avoid blocking on GPU
  availability for nothing.
- `prepare_env.sh` uses Python 3.9 because conda-forge ships
  `gensim=3.8.3` (a cso-classifier dep) only for Python ≤ 3.9.
- Compute nodes typically have outbound internet, so the CSO ontology +
  Word2Vec model download works on first run.
