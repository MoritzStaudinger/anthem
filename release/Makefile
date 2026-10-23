# Build and publish the ANTHEM HuggingFace dataset release.
#
# Prerequisites: the four source trees extracted under $(SRC), and `hf auth
# login` with a write token.
#
#   make release    # build the Parquet configs + dataset card
#   make check      # row counts and sizes per config
#   make upload     # push to $(REPO)
#   make tag        # tag the pushed revision as $(VERSION)

SRC     ?= $(HOME)/anthem_build/src
OUT     ?= $(HOME)/anthem_build/release
REPO    ?= Molarix/anthem
VERSION ?= 202605
PY      ?= python3

# The 781-paper gold set and the 99-question benchmark live outside this repo.
GOLD_CSV ?= $(HOME)/Downloads/11688_Singing_from_the_Same_AN_Data/sample_data/affiliation_ground_truth.csv
BENCHMARK ?= $(HOME)/PycharmProjects/Comparer_flask/data/eval/scico_testset.jsonl
ADJUDICATION ?= $(HOME)/PycharmProjects/acl_processor/reports/relink_compare/adjudication_sheet.csv
CSO ?= $(HOME)/PycharmProjects/acl_processor/reports/cso_validation/cso_validation.csv
LORA_TRAIN ?= $(HOME)/PycharmProjects/acl_processor/data/lora_train/train.jsonl
LORA_VAL ?= $(HOME)/PycharmProjects/acl_processor/data/lora_train/val.jsonl

.PHONY: release corpus annotations check upload tag clean

# annotations first: the corpus build writes the dataset card, and the card
# lists whatever configs it finds in the output directory.
release: annotations corpus

corpus:
	$(PY) scripts/build_hf_release.py \
	  --metadata $(SRC)/ror_parsed_affiliations_90_with_topics_and_bibmeta \
	  --tei-dir  $(SRC)/acl_pdfs_parsed \
	  --md-dir   $(SRC)/acl_pdfs_pymu_llm \
	  --hybrid-dir $(SRC)/acl_pdfs_hybrid \
	  --out $(OUT) --version $(VERSION)

annotations:
	$(PY) scripts/build_hf_annotations.py \
	  --benchmark $(BENCHMARK) \
	  --gold-csv $(GOLD_CSV) \
	  --adjudication-csv $(ADJUDICATION) \
	  --cso-csv $(CSO) \
	  --lora-train $(LORA_TRAIN) --lora-val $(LORA_VAL) \
	  --out $(OUT)

check:
	@$(PY) scripts/check_release.py $(OUT)

# `hf upload` chunks and resumes by itself, which matters for the multi-GB
# fulltext shards; plain `git push` over LFS does not resume as gracefully.
upload:
	HF_HUB_DISABLE_XET=1 hf upload $(REPO) $(OUT) . --repo-type=dataset \
	  --commit-message "ANTHEM-$(VERSION): corpus, fulltext, benchmark, annotations"

tag:
	hf repo tag create $(REPO) $(VERSION) --repo-type=dataset

clean:
	rm -rf $(OUT)
