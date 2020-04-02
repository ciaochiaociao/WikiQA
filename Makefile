.PHONY: all downlad filter process get_qa clean merge

all: install download filter get_qa merge

install:
	git clone

# default inputs
FGC_VER?=1.7.8
TRAIN_FNAME?=FGC_release_all_train.json
DEV_FNAME?=FGC_release_all_dev.json
TEST_FNAME?=FGC_release_all_test.json
ALL_FNAME?=merged.json
RAW_DATASET_DIR?=data/raw/$(FGC_VER)
PROC_DATASET_DIR?=data/processed/$(FGC_VER)

# variables
RAW_TRAIN_FPATH = $(RAW_DATASET_DIR)/$(TRAIN_FNAME)
RAW_DEV_FPATH = $(RAW_DATASET_DIR)/$(DEV_FNAME)
RAW_TEST_FPATH = $(RAW_DATASET_DIR)/$(TEST_FNAME)
RAW_ALL_FPATH = $(RAW_DATASET_DIR)/$(ALL_FNAME)
PROC_TRAIN_FPATH = $(PROC_DATASET_DIR)/$(TRAIN_FNAME:.json=_filtered.json)
PROC_DEV_FPATH = $(PROC_DATASET_DIR)/$(DEV_FNAME:.json=_filtered.json)
PROC_TEST_FPATH = $(PROC_DATASET_DIR)/$(TEST_FNAME:.json=_filtered.json)
PROC_ALL_FPATH = $(PROC_DATASET_DIR)/$(ALL_FNAME:.json=_filtered.json)


# download and extract dataset
download: $(RAW_TRAIN_FPATH) $(RAW_DEV_FPATH) $(RAW_TEST_FPATH)
FGC_DATASET_7z_URL?='https://drive.google.com/uc?id=1ofd5U0q7-eUJrs6c4n0AZz2PLgc0KY7q&export=download'
$(RAW_TRAIN_FPATH) $(RAW_DEV_FPATH) $(RAW_TEST_FPATH) :
	{ \
	wget -O data/raw/data.7z $(FGC_DATASET_7z_URL) && \
	7z x -o$(RAW_DATASET_DIR)/ data/raw/data.7z && \
	rm data/raw/data.7z; \
	}

# merge
merge: $(RAW_ALL_FPATH)
$(RAW_ALL_FPATH) : $(RAW_TRAIN_FPATH) $(RAW_DEV_FPATH) $(RAW_TEST_FPATH)
	python3 -m fgc_wiki_qa.data.merge $^ $@

# filter
filter: $(PROC_TRAIN_FPATH) $(PROC_DEV_FPATH) $(PROC_TEST_FPATH) $(PROC_ALL_FPATH)

$(PROC_TRAIN_FPATH) : $(RAW_TRAIN_FPATH)
	python3 -m fgc_wiki_qa.data.filter_dataset $< $@
$(PROC_DEV_FPATH) : $(RAW_DEV_FPATH)
	python3 -m fgc_wiki_qa.data.filter_dataset $< $@
$(PROC_TEST_FPATH) : $(RAW_TEST_FPATH)
	python3 -m fgc_wiki_qa.data.filter_dataset $< $@
$(PROC_ALL_FPATH) : $(RAW_ALL_FPATH)
	python3 -m fgc_wiki_qa.data.filter_dataset $< $@

# get_qa
get_qa: $(PROC_DATASET_DIR)/qa_train.tsv $(PROC_DATASET_DIR)/qa_dev.tsv $(PROC_DATASET_DIR)/qa_test.tsv $(PROC_DATASET_DIR)/qa_all.tsv

$(PROC_DATASET_DIR)/qa_train.tsv: $(PROC_TRAIN_FPATH)
	python3 -m fgc_wiki_qa.data.get_qa_tsv $< $@

$(PROC_DATASET_DIR)/qa_dev.tsv: $(PROC_DEV_FPATH)
	python3 -m fgc_wiki_qa.data.get_qa_tsv $< $@

$(PROC_DATASET_DIR)/qa_test.tsv: $(PROC_TEST_FPATH)
	python3 -m fgc_wiki_qa.data.get_qa_tsv $< $@

$(PROC_DATASET_DIR)/qa_all.tsv: $(PROC_ALL_FPATH)
	python3 -m fgc_wiki_qa.data.get_qa_tsv $< $@

# get toy dataset
get_toy_dataset: $(PROC_DATASET_DIR)/toy_all.json
$(PROC_DATASET_DIR)/toy_all.json: $(PROC_ALL_FPATH)
	python3 -m fgc_wiki_qa.data.get_toy_dataset $< $@

.PHONY: run run_eval run_exp

#EXP_DIR?=experiments/new
PRED_INFER?=rule

EVAL_FPATH=$(EXP_DIR)/file4eval_all.tsv

run_exp:
	mkdir -p $(EXP_DIR)
	$(MAKE) run
	$(MAKE) run_eval

RUN_LOG = $(EXP_DIR)/run.log

run: $(EVAL_FPATH)

$(EVAL_FPATH): $(PROC_ALL_FPATH)
	python3 -m fgc_wiki_qa.commands.run_on_fgc \
		--fgc_fpath $< \
		--pred_infer $(PRED_INFER) \
		--use_se $(USE_SE) \
		--eval_fpath $@ | tee $(RUN_LOG)


# inputs
QA_FPATH ?= $(PROC_DATASET_DIR)/qa_all.tsv

# outputs
ERROR_ANALYSIS_FPATH ?= $(EXP_DIR)/error_analysis_all.xlsx
REPORT_FPATH ?= $(EXP_DIR)/report_all.txt
QIDS_FPATH ?= $(EXP_DIR)/qids_all.json

run_eval: $(ERROR_ANALYSIS_FPATH) $(REPORT_FPATH) $(QIDS_FPATH)

$(ERROR_ANALYSIS_FPATH) $(REPORT_FPATH) $(QIDS_FPATH): $(PROC_ALL_FPATH) $(EVAL_FPATH) $(QA_FPATH)
	python3 -m fgc_wiki_qa.commands.evaluate \
		--fgc_fpath $(PROC_ALL_FPATH) \
		--fgc_qa_fpath $(QA_FPATH) \
		--eval_fpath $(EVAL_FPATH) \
		--wiki_benchmark data/external/fgc_wiki_benchmark_v0.1.tsv \
		--result_fpath $(REPORT_FPATH) \
		--error_analysis $(ERROR_ANALYSIS_FPATH) \
		--qids_fpath $(QIDS_FPATH)

