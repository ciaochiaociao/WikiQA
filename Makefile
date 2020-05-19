.PHONY: all downlad filter get_qa merge extract get_toy_dataset compare

all: merge filter get_qa get_toy_dataset
get: download extract


#inputs: FGC_VER

RAW_DATASET_DIR = data/raw/$(FGC_VER)
PROC_DATASET_DIR = data/processed/$(FGC_VER)

# default contants
TRAIN_FNAME = FGC_release_all_train.json
DEV_FNAME = FGC_release_all_dev.json
TEST_FNAME = FGC_release_all_test.json
ALL_FNAME = merged.json

# variables
RAW_TRAIN_FPATH = $(RAW_DATASET_DIR)/$(TRAIN_FNAME)
RAW_DEV_FPATH = $(RAW_DATASET_DIR)/$(DEV_FNAME)
RAW_TEST_FPATH = $(RAW_DATASET_DIR)/$(TEST_FNAME)
RAW_ALL_FPATH = $(RAW_DATASET_DIR)/$(ALL_FNAME)
PROC_TRAIN_FPATH = $(PROC_DATASET_DIR)/$(TRAIN_FNAME:.json=_filtered.json)
PROC_DEV_FPATH = $(PROC_DATASET_DIR)/$(DEV_FNAME:.json=_filtered.json)
PROC_TEST_FPATH = $(PROC_DATASET_DIR)/$(TEST_FNAME:.json=_filtered.json)
PROC_ALL_FPATH = $(PROC_DATASET_DIR)/$(ALL_FNAME:.json=_filtered.json)


# download
# BUGFIX: below does not work
ifeq ($(FGC_VER), 1.7.8)
	FGC_DATASET_7z_URL ?= 'https://drive.google.com/uc?id=1ofd5U0q7-eUJrs6c4n0AZz2PLgc0KY7q&export=download'
endif
ifeq ($(FGC_VER), 1.7.9)
	FGC_DATASET_7z_URL ?= 'https://drive.google.com/u/0/uc?id=1K0QVORNDgEI6tF8e-SXa--UdjkzAZa2d&export=download'
endif

DOWNLOAD_FPATH = data/download/$(FGC_VER).7z

download: $(DOWNLOAD_FPATH)

$(DOWNLOAD_FPATH):
	echo 'downloading dataset ...'
	chmod u+w $(dir $(DOWNLOAD_FPATH)) && \
	wget -O $(DOWNLOAD_FPATH) $(FGC_DATASET_7z_URL) && \
	chmod 444 $(DOWNLOAD_FPATH);
	chmod u-w $(dir $(DOWNLOAD_FPATH));

# extract
extract: $(RAW_DATASET_DIR)

$(RAW_DATASET_DIR) : $(DOWNLOAD_FPATH)
	{ \
	chmod u+w $(dir $(RAW_DATASET_DIR)) && \
	7z x -o$(RAW_DATASET_DIR)/ $(DOWNLOAD_FPATH) && \
	chmod 444 $(RAW_DATASET_DIR)/*.json && \
	chmod 555 $(RAW_DATASET_DIR); \
	chmod u-w $(dir $(RAW_DATASET_DIR)); \
	}

# merge
merge: $(RAW_ALL_FPATH)
$(RAW_ALL_FPATH) :
	chmod u+w $(@D) && \
	python3 -m fgc_wiki_qa.data.merge $(RAW_TRAIN_FPATH) $(RAW_DEV_FPATH) $(RAW_TEST_FPATH) $@ &&\
	chmod 444 $@;
	chmod u-w $(@D)

# filter
filter: $(PROC_TRAIN_FPATH) $(PROC_DEV_FPATH) $(PROC_TEST_FPATH) $(PROC_ALL_FPATH)

$(PROC_TRAIN_FPATH) :
	mkdir -p $(@D)
	python3 -m fgc_wiki_qa.data.filter_dataset $(RAW_TRAIN_FPATH) $@
$(PROC_DEV_FPATH) :
	mkdir -p $(@D)
	python3 -m fgc_wiki_qa.data.filter_dataset $(RAW_DEV_FPATH) $@
$(PROC_TEST_FPATH) :
	mkdir -p $(@D)
	python3 -m fgc_wiki_qa.data.filter_dataset $(RAW_TEST_FPATH) $@
$(PROC_ALL_FPATH) :
	mkdir -p $(@D)
	python3 -m fgc_wiki_qa.data.filter_dataset $(RAW_ALL_FPATH) $@

# get_qa
get_qa: $(PROC_DATASET_DIR)/qa_train.tsv $(PROC_DATASET_DIR)/qa_dev.tsv $(PROC_DATASET_DIR)/qa_test.tsv $(PROC_DATASET_DIR)/qa_all.tsv $(RAW_DATASET_DIR)/qa_all.tsv

$(PROC_DATASET_DIR)/qa_train.tsv: $(PROC_TRAIN_FPATH)
	python3 -m fgc_wiki_qa.data.get_qa_tsv $< $@

$(PROC_DATASET_DIR)/qa_dev.tsv: $(PROC_DEV_FPATH)
	python3 -m fgc_wiki_qa.data.get_qa_tsv $< $@

$(PROC_DATASET_DIR)/qa_test.tsv: $(PROC_TEST_FPATH)
	python3 -m fgc_wiki_qa.data.get_qa_tsv $< $@

$(PROC_DATASET_DIR)/qa_all.tsv: $(PROC_ALL_FPATH)
	python3 -m fgc_wiki_qa.data.get_qa_tsv $< $@

$(RAW_DATASET_DIR)/qa_all.tsv: $(RAW_ALL_FPATH)
	chmod u+w $(@D) && \
	python3 -m fgc_wiki_qa.data.get_qa_tsv $< $@
	chmod 444 $@;
	chmod u-w $(@D)

# get toy dataset
get_toy_dataset: $(PROC_DATASET_DIR)/toy_all.json
$(PROC_DATASET_DIR)/toy_all.json: $(PROC_ALL_FPATH)
	python3 -m fgc_wiki_qa.data.get_toy_dataset $< $@

.PHONY: run run_eval run_exp

#EXP_DIR?=experiments/new
PRED_INFER?=rule

EVAL_FPATH=$(EXP_DIR)/file4eval_all.tsv
CONFIG_FPATH=$(EXP_DIR)/config.json

# examples: make run_exp EXP_DIR=experiments/v1.02_on_1.7.8_revise_sp PRED_INFER=rule FGC_VER=1.7.8-revise-sp RUN_ON=raw USE_SE=pred
# inputs: EXP_DIR RUN_ON PRED_INFER FGC_VER USE_SE

run_exp:
	mkdir -p $(EXP_DIR)
	$(MAKE) run
	$(MAKE) run_eval

RUN_LOG = $(EXP_DIR)/run.log

run: $(EVAL_FPATH)

$(info 'run_on' $(RUN_ON))

ifeq ($(RUN_ON), proc)
	DATASET_FPATH = $(PROC_ALL_FPATH)
	QA_FPATH = $(PROC_DATASET_DIR)/qa_all.tsv
else ifeq ($(RUN_ON), raw)
	DATASET_FPATH = $(RAW_ALL_FPATH)
	QA_FPATH = $(RAW_DATASET_DIR)/qa_all.tsv
else
$(info 'unrecognized RUN_ON:' $(RUN_ON))
endif

$(EVAL_FPATH): $(DATASET_FPATH)
	python3 -m fgc_wiki_qa.commands.run_on_fgc \
		--fgc_fpath $< \
		--pred_infer $(PRED_INFER) \
		--use_se $(USE_SE) \
		--eval_fpath $@ \
		--config_fpath $(CONFIG_FPATH) | tee $(RUN_LOG)


# inputs: QA_FPATH

# outputs
ERROR_ANALYSIS_FPATH ?= $(EXP_DIR)/error_analysis_all.xlsx
REPORT_FPATH ?= $(EXP_DIR)/report_all.txt
QIDS_FPATH ?= $(EXP_DIR)/qids_all.json

run_eval: $(ERROR_ANALYSIS_FPATH) $(REPORT_FPATH) $(QIDS_FPATH)

$(ERROR_ANALYSIS_FPATH) $(REPORT_FPATH) $(QIDS_FPATH): $(DATASET_FPATH) $(EVAL_FPATH) $(QA_FPATH)
	python3 -m fgc_wiki_qa.commands.evaluate \
		--fgc_fpath $(DATASET_FPATH) \
		--fgc_qa_fpath $(QA_FPATH) \
		--eval_fpath $(EVAL_FPATH) \
		--wiki_benchmark data/external/fgc_wiki_benchmark_v0.1.tsv \
		--result_fpath $(REPORT_FPATH) \
		--error_analysis $(ERROR_ANALYSIS_FPATH) \
		--qids_fpath $(QIDS_FPATH)

# [compare]
# examples: make compare EXP_DIR=... COMPARED_EXP_DIR=...
COMPARED ?= $(COMPARED_EXP_DIR)/qids_all.json
COMPARE_FPATH_NAME ?= $(EXP_DIR)/$(notdir $(EXP_DIR))_VS_$(notdir $(COMPARED_EXP_DIR))

compare: $(COMPARE_FPATH_NAME).json $(COMPARE_FPATH_NAME).txt

$(COMPARE_FPATH_NAME).json $(COMPARE_FPATH_NAME).txt:
	python3 -m fgc_wiki_qa.metrics.analyze_qids \
		--qids_fpath $(QIDS_FPATH) \
		--already_qids_fpath $(COMPARED) \
		--save_fpath $(COMPARE_FPATH_NAME).json \
		--report_fpath $(COMPARE_FPATH_NAME).txt
