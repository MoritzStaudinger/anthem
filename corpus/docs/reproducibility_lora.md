# Appendix: LoRA fine-tuning for affiliation extraction

All scripts, configs, and intermediate artefacts referenced here are
shipped with the code release. File paths are repository-relative.

## A.1 Training data construction (OpenAlex)

**Goal.** Build supervised (input → JSON output) pairs of (paper-header
text → authors+affiliations+ROR+country) by joining three local
sources, with the 781-paper evaluation set excluded.

Script: `scripts/fetch_oa_training.py`. Slurm wrapper:
`slurm/fetch_oa_training_data.sbatch` (no GPU, ~2 h wall).

### A.1.1 Sources

1. **Paper header text** — first `--max-header-chars=3000` characters
   of `<pymu_llm_dir>/<paper_id>.md`. The pymu_llm parse is a
   PyMuPDF + LLM reformatting of the PDF and reliably contains
   title + author block + affiliation lines before the abstract in
   modern proceedings. We drop any paper whose header is shorter than
   `--min-header-chars=200` (junk or empty parses).
2. **DOI bridge** — `<enriched_affs_dir>/<paper_id>.json` from the
   ACL Anthology bibmeta merge. We require a non-empty `doi` field.
3. **Gold output** — OpenAlex `works/doi:<doi>` lookup. For each
   `authorships[i]`, we emit one author record containing
   `raw_affiliation_strings`, the canonical `institutions[j].display_name`,
   the ROR URL, and ISO 3166-1 alpha-2 country code.

### A.1.2 Filters and exclusions

- **Eval-set exclusion.** Every paper-id in
  `reports/ground_truth_merged.csv` (the 781-paper human-annotated
  gold set; see Appendix on annotation) is removed from the candidate
  pool *before* shuffling. This guarantees zero train/test
  contamination.
- **Per-paper ROR requirement.** Papers where OpenAlex returns
  authorships but no author has any ROR-linked institution are
  dropped (`n_with_ror == 0`). This keeps the training signal
  ROR-aware; otherwise the model learns to emit empty `ror_id`.
- **DOI requirement.** Papers without a DOI in the enriched JSON are
  silently skipped — older ACL Anthology proceedings without
  registered DOIs are therefore under-represented (see year
  distribution below).

### A.1.3 Sampling procedure

1. Read the enriched-JSON directory listing; subtract the eval-set
   IDs; shuffle the remainder with `random.seed(42)`.
2. Walk the shuffled list with a `ThreadPoolExecutor(max_workers=8)`
   submitting OpenAlex DOI lookups.
3. OpenAlex requests go through the polite pool (`mailto=` from
   `$OPENALEX_MAILTO`) with 4-retry exponential backoff on
   429/500/502/503/504.
4. Stop when `--target-rows=10000` usable rows are collected.
5. Deterministic 90/10 split (`random.Random(seed=42).shuffle` →
   first 10% to val, rest to train).
6. Resumable: per-DOI OpenAlex responses are slim-cached to
   `data/lora_train/.oa_cache.json` and re-used on rerun.

### A.1.4 Resulting corpus

|                              | train       | val      |
|------------------------------|-------------|----------|
| Papers                       | 9,000       | 1,000    |
| Author records               | 32,725      | 3,592    |
| Affiliation records          | 42,375      | 4,605    |
| Mean authors / paper         | 3.64        | 3.59     |
| Mean affiliations / paper    | 4.71        | 4.61     |
| % affiliations with ROR id   | 100 %       | 100 %    |
| Distinct ISO country codes   | 106         | 69       |
| Mean header length (chars)   | 2,996       | 2,996    |

The 100 % ROR coverage is by construction: any author without a
ROR-linked institution is dropped at the OpenAlex-gold step.

**Temporal distribution.** Training papers span 1979–2025 with a
strong recency skew driven by DOI availability in OpenAlex. ≥ 90 %
of training papers are 2014 or later; the largest single year is
2023 (1,639 papers). Pre-2010 coverage is sparse (≤ 70 papers per
year) and reflects the well-known DOI gap in early ACL Anthology
proceedings. The 781-paper evaluation set is decade-stratified
specifically to expose this gap; the LoRA's weaker score on pre-2000
papers (cf. main paper Table 3) is consistent with this training
distribution.

### A.1.5 Output schema

Each row of `data/lora_train/{train,val}.jsonl`:

```json
{"input": "<first 3000 chars of pymu_llm markdown>",
 "output": [
   {"name": "Jane Doe",
    "affiliations": [{
      "raw":          "raw_affiliation_strings[i] from OA",
      "institution":  "institutions[j].display_name",
      "ror_id":       "https://ror.org/...",
      "country_code": "US"}]}
 ],
 "source": {"paper_id": "...", "doi": "...", "work_id": "W..."}}
```

## A.2 LoRA fine-tuning

Script: `scripts/train_lora_affiliation.py`. Slurm wrapper:
`slurm/train_lora_affiliation.sbatch`. Adapter config saved to
`models/qwen25_7b_aff_lora/adapter/adapter_config.json`; training-step
log to `models/qwen25_7b_aff_lora/training_log.jsonl`.

### A.2.1 Base model and quantisation

- **Base:** `Qwen/Qwen2.5-7B-Instruct` (loaded with
  `attn_implementation="sdpa"`, `device_map="auto"`).
- **Quantisation:** 4-bit `nf4` with double quantisation; compute
  dtype `bfloat16`. Implemented via `bitsandbytes` ≥ 0.43
  (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
  bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)`).
- **KV cache** disabled during training (`model.config.use_cache =
  False`); gradient checkpointing on
  (`use_reentrant=False`).

### A.2.2 LoRA configuration

| Hyperparameter      | Value |
|---------------------|------:|
| `r`                 | 16    |
| `lora_alpha`        | 32    |
| `lora_dropout`      | 0.05  |
| `bias`              | none  |
| `task_type`         | CAUSAL_LM |
| `target_modules`    | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

All linear projections in both attention and MLP blocks are adapted.
Adapter file size on disk: 81 MB
(`models/qwen25_7b_aff_lora/adapter/adapter_model.safetensors`).
`peft` reports the adapter as `PEFT 0.19.1`.

### A.2.3 Optimisation

| Hyperparameter                  | Value |
|---------------------------------|------:|
| Optimiser                       | `paged_adamw_8bit` |
| Learning rate                   | 2 × 10⁻⁴ |
| LR scheduler                    | cosine |
| Warm-up ratio                   | 0.03 |
| Epochs                          | 3 |
| Per-device train batch          | 1 |
| Gradient accumulation steps     | 16 |
| **Effective batch size**        | **16** |
| Max sequence length             | 2 048 tokens |
| Mixed precision                 | bf16 |
| Gradient checkpointing          | on (non-reentrant) |
| Packing                         | off (one example per sequence) |
| Loss masking                    | assistant-only (TRL `assistant_only_loss=True`) |
| Random seed                     | 42 |
| Eval / save / log cadence       | every 200 / 200 / 20 steps |
| `save_total_limit`              | 3 (most recent checkpoints) |

### A.2.4 Prompt template

The model is supervised on Qwen-2.5 chat-template messages; only the
assistant turn's tokens contribute to the loss.

- **System:** "You are an expert at extracting author affiliations
  from scientific paper headers. Given the front-matter text of a
  paper (the title block, authors, and affiliation lines as they
  appear on the PDF before the abstract), produce a JSON array of
  authors. Each author has a `name` and a list of `affiliations`.
  Each affiliation has `raw` (the substring where the affiliation
  appears), `institution` (the normalised institution name),
  `ror_id` (the full ROR URL), and `country_code` (ISO 3166-1
  alpha-2). Output ONLY the JSON; no preamble or explanation."
- **User:** "Extract author affiliations from this paper header:\\n\\n{input}"
- **Assistant (target):** compact JSON serialisation of the `output`
  list (no whitespace, `separators=(",", ":")`) so the model is
  rewarded for emitting tight JSON.

### A.2.5 Training run

Run executed on the TU Wien DataLab cluster
(`--partition=GPU-a100`, single A100 80 GB, 64 GB RAM, 8 CPUs,
`--time=08:00:00` budget). HF cache pinned at
`/share/moritz.staudinger/hf_cache` to share base-weight downloads
across jobs.

| Metric                          | Value |
|---------------------------------|------:|
| Optimisation steps              | 1,689 (3 epochs × 9 000 / 16 effective) |
| Total tokens trained on         | 30.39 M |
| Train runtime                   | 12,694 s ≈ 3 h 31 min |
| Throughput                      | 2.13 samples/s · 0.133 steps/s |
| Total FLOPs                     | 1.30 × 10¹⁸ |
| Final training loss             | 0.0813 |
| Final eval loss (val n=1 000)   | 0.0826 |
| Final eval token accuracy       | 98.17 % |
| Final eval entropy              | 0.056 |
| Adapter on disk                 | 81 MB |

The loss curve in `training_log.jsonl` shows the standard SFT shape:
fast early descent (0.38 → 0.21 in the first 100 steps) followed by
a long plateau around 0.03–0.04 once the model has memorised the
schema; eval loss tracks train loss with the expected ~0.04 generalisation
gap.

### A.2.6 Software environment

Pinned in `requirements-lora.txt`; conda environment named
`lora_train`, Python 3.11.

```
torch>=2.4,<2.7
transformers>=4.45,<5.0
peft>=0.13            (actual run: 0.19.1)
trl>=0.12
accelerate>=1.0
datasets>=3.0
bitsandbytes>=0.43
sentencepiece>=0.2
```

## A.3 Inference protocol (gold-set evaluation)

Script: `scripts/infer_lora_on_gold.py`. Slurm wrapper:
`slurm/infer_lora_on_gold.sbatch` (A40, 24 GB RAM, ≤ 80 min wall for
781 papers).

- The base model is reloaded in the same 4-bit nf4 + bf16 compute
  configuration used for training; the LoRA adapter is loaded with
  `PeftModel.from_pretrained(...)` on top.
- Inputs reuse the same 3 000-character pymu_llm header window
  (identical to training).
- The prompt template is byte-identical to A.2.4.
- Generation is greedy single-shot JSON; the response is parsed by a
  bracket-matching JSON extractor (regex `_JSON_FENCE` strips
  Markdown fences, then balanced-bracket walk). Parse failures
  produce an empty author list with `_pipeline.error` populated so
  the row still counts against recall in the scorer (no
  silent-skip).
- Output is written in leaderboard schema to
  `reports/pipeline_comparison/predictions_lora_qwen25_7b_aff.jsonl`
  and consumed by `slurm/aff_compare_aggregate.sbatch`, which
  centralises ROR re-linking across all pipelines for apples-to-
  apples scoring against the human gold.

### A.3.1 Ablation control

To isolate the contribution of the LoRA adapter from base-model
zero-shot ability, the same script supports a `--no-adapter` flag.
The control run is launched via
`slurm/infer_base_qwen_on_gold.sbatch` and writes
`predictions_qwen25_7b_instruct.jsonl` (pipeline id
`qwen25_7b_instruct`), which appears alongside `lora_qwen25_7b_aff`
in the leaderboard for direct comparison.

## A.4 Reproduction recipe

Assuming the cluster environment described above, the full
pipeline is:

```bash
# 1. Build training data (no GPU)
sbatch slurm/fetch_oa_training_data.sbatch 10000

# 2. Train the LoRA adapter (~3.5 h on A100)
sbatch slurm/train_lora_affiliation.sbatch

# 3. Inference on the 781-paper gold set (~1 h on A40)
sbatch slurm/infer_lora_on_gold.sbatch

# 4. (Optional) Inference with the base model only
sbatch slurm/infer_base_qwen_on_gold.sbatch

# 5. Re-aggregate the leaderboard
sbatch slurm/aff_compare_aggregate.sbatch
```

All five steps are deterministic given `seed=42`, the same
OpenAlex snapshot, and the same input pymu_llm parses. The OpenAlex
cache (`data/lora_train/.oa_cache.json`) is shipped with the release
to allow re-builds without re-querying OpenAlex.
