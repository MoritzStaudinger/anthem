# ANTHEM release tooling

Builds the HuggingFace dataset [`Molarix/anthem`](https://huggingface.co/datasets/Molarix/anthem)
from the pipeline output, and pushes it.

This repo holds only the packaging step. The extraction pipeline itself lives in
`acl_processor` (affiliations, ROR linking, CSO topics) and `Comparer_flask`
(the comparative-QA benchmark and its evaluation).

## What gets published

Eight Parquet configs, all loadable with `load_dataset`:

| Config | What |
|---|---|
| `papers` | one row per paper; nested fields as JSON strings |
| `affiliations` | flat: paper × author × affiliation, ROR id + evidence span |
| `topics` | flat: paper × CSO label |
| `fulltext` | GROBID TEI, PyMuPDF/LLM Markdown and OCR text per paper |
| `benchmark_questions` | the comparative-QA questions and their gold specification |
| `affiliation_gold` | the 781-paper hand-annotated affiliation gold set |
| `linking_adjudication` | hand-verified ROR linking decisions, four strategies |
| `cso_validation` | hand-judged CSO topic labels |
| `lora_train` | LoRA fine-tuning data for the affiliation adapter (`train`/`validation`) |

Source PDFs are **not** published. The ACL Anthology serves those, and
redistributing ~196k of them is both a copyright problem and 80 GB nobody needs.

## Getting the data off the cluster

`scripts/pack_anthem.sh` tars each source tree with zstd. Write the archives to
`/share`, never to `$HOME` — home is quota-limited and tar dies partway through
with `Disk quota exceeded`, leaving a useless file that still occupies the quota.

```bash
# on the cluster
SRC=/share/moritz.staudinger/acl_data OUT=/share/moritz.staudinger/anthem_pack \
  nohup bash scripts/pack_anthem.sh > /share/moritz.staudinger/pack.log 2>&1 &

# locally
rsync -avP user@cluster:/share/moritz.staudinger/anthem_pack/ ~/anthem_transfer/
cd ~/anthem_transfer && sha256sum -c SHA256SUMS
```

Extract into `$SRC` (default `~/anthem_build/src`), skipping the PDFs that sit
in subdirectories of the TEI tree:

```bash
mkdir -p ~/anthem_build/src && cd ~/anthem_build/src
tar -I zstd -xf ~/anthem_transfer/anthem-acl_pdfs_parsed.tar.zst \
    --exclude='*.pdf' --exclude='*.sif' --exclude='*.err' --exclude='*.log'
for t in acl_pdfs_pymu_llm acl_pdfs_hybrid \
         ror_parsed_affiliations_90_with_topics_and_bibmeta; do
  tar -I zstd -xf ~/anthem_transfer/anthem-$t.tar.zst
done
```

## Build, check, publish

```bash
pip install -r requirements.txt
make release    # annotations, then corpus (the corpus build writes the card)
make check      # gates the upload: leaked paths, orphan ids, counts
hf auth login   # needs a write token
HF_HUB_DISABLE_XET=1 make upload
make tag
```

`HF_HUB_DISABLE_XET=1` matters. With HF's Xet backend enabled the 4 GB
upload ran at 0.005 MB/s and effectively stalled -- one file in seven
minutes -- while the same link sustained 1.9 MB/s with Xet off. The
progress counter keeps reporting workers as busy either way, so check
the interface counters rather than the status line if it looks slow.

`make check` fails the build on an absolute path anywhere in a text column or a
`paper_id` in `affiliations`/`topics`/`fulltext` that is absent from `papers`.
It reports e-mail addresses in the parsed text as a note rather than a failure —
see below.

## Two decisions worth knowing about

**Third-party venues.** The ACL copyright policy covers pre-2016 material as
CC BY-NC-SA 3.0 and 2016-onward as CC BY 4.0, but states that it "does not
cover third-party materials". LREC and its sister venues are ELRA-published.
Rather than silently dropping them, every row carries `third_party_venue`, so
excluding them is a filter:

```python
ds = load_dataset("Molarix/anthem", "fulltext", split="train")
acl_only = ds.filter(lambda r: not r["third_party_venue"])
```

**E-mail addresses.** GROBID lifts author contact details out of the PDF
header, so the TEI and OCR text contain the addresses the papers themselves
print. They are kept by default, because `fulltext` is meant to be a faithful
parse and scrubbing would damage the TEI's `<email>` elements. Pass
`--scrub-emails` to `build_hf_release.py` to replace them instead.

## Still outstanding

The paper's appendix states that all scripts, configurations, training data,
the OpenAlex query cache and the trained adapter are released alongside ANTHEM.
The training data ships as the `lora_train` config; the OpenAlex query cache is
not packaged yet.

## Licensing

ANTHEM's own layer — the extracted affiliations, ROR links, CSO labels and
evidence spans — is released under CC-BY-NC-SA-4.0, matching the paper. The
`license` column on `papers` and `fulltext` is provenance: it records the terms
the underlying paper carries in the Anthology, not the release terms.

Papers whose year is missing from the metadata have it recovered from the legacy
Anthology id (`W18-0902` → 2018); only papers where that also fails fall back to
the more restrictive pre-2016 term.
