# Fine-tuning an affiliation extractor on OpenAlex (roadmap)

Goal: train a small open-weight model on OpenAlex's paper → author-affiliation
data, then evaluate it on the 781-paper SciCo gold using the same
`run_pipeline_comparison.py` harness as the API-based LLMs.

Why this is worth doing — but later:

- A 7-8B model fine-tuned on millions of paper-affiliation pairs may
  rival or beat GPT-4o on this narrow task at <1% of the inference cost.
- Reviewer-friendly: shows your pipeline's contribution isn't bottlenecked
  by closed-source API access.
- Out of EMNLP revision scope as a contribution; in scope as future work.

## Data source

OpenAlex `works` dump has author → institution affiliation per paper for
~250 M papers. The relevant fields:

```
work.id
work.title
work.abstract_inverted_index   (reconstruct to plain text)
work.authorships[].author.display_name
work.authorships[].institutions[].display_name
work.authorships[].institutions[].country_code
work.authorships[].institutions[].ror
```

Bulk-download:
- HuggingFace: `datasets load_dataset("OpenAlex/works", streaming=True)`
- Or the OpenAlex snapshot: `https://docs.openalex.org/download-all-data`

ACL Anthology overlap: filter `work.host_venue.id` or DOI prefix to ACL
publications (~150k papers), drop the rest. Or skip the filter and train
on all-of-science — more data, generalises better but slower.

## Training recipe

- **Base**: Llama-3.1-8B-Instruct (or Qwen2.5-7B, Mistral-7B)
- **Method**: LoRA (rank=16) for speed; full FT if you have the GPU
- **Format**:
    ```
    Input  : title + abstract + first 5k chars of full text (when available)
    Output : JSON {authors: [{name, affiliations: [str, ...]}, ...]}
    ```
- **Loss**: cross-entropy on the JSON output, with a small format-validity
  reward (parse-rate × 0.1 added during eval, not training)
- **Hardware**: 1× A100 40GB ≈ 12-24 hours for 100k training pairs
- **Library**: `axolotl` or `trl.SFTTrainer`

## Integration with the eval harness

Add a `huggingface` backend to `scripts/aff_pipelines/llm_extract.py`:

```python
class HuggingFaceBackend(_Backend):
    name = "huggingface"
    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._cache = {}
    def call(self, system_prompt, user_prompt, model, ...):
        # model = local path to LoRA-merged checkpoint
        if model not in self._cache:
            tok = AutoTokenizer.from_pretrained(model)
            mod = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
            self._cache[model] = (tok, mod)
        ...
```

Then a config entry like:

```yaml
- id: ours_finetuned_llama8b
  type: llm
  backend: huggingface
  model: /share/moritz.staudinger/models/llama3-8b-affiliation-lora-merged
  input_template: pymu_first_n
  input_kwargs: {n: 5000}
```

drops it into the same comparison table.

## Stub scripts to create

- `scripts/download_openalex_affiliations.py` — stream OpenAlex works,
  filter to (paper has affiliations), write parquet shards.
- `scripts/format_openalex_training_data.py` — build (input_text, json_output)
  pairs in axolotl / SFT format.
- `scripts/train_affiliation_lora.py` — wraps axolotl/trl invocation.
- `scripts/merge_lora.py` — merge LoRA into base for inference.

## Quick-and-dirty alternative (no training)

If full fine-tuning is too much, an intermediate step: **prompt-tune** a
small base model with few-shot OpenAlex examples in the system prompt.
Costs nothing (just inference), shows how much of the "training" benefit
is just exemplar exposure. Could ship as a baseline alongside the
fine-tuned variant.

## Reviewer framing

> *"In addition to our retrieval-based extraction pipeline, we report results
> from a Llama-3.1-8B model fine-tuned on N OpenAlex paper-affiliation pairs.
> This open-weight baseline approaches the F1 of frontier proprietary models
> at <1% of the inference cost, demonstrating that the affiliation-extraction
> task is amenable to small models when domain-tuned."*

(Only make this claim if the numbers actually hold up. Run the comparison
first.)
