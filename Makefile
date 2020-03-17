all: download filter

FGC_DATASET_7z_URL?='https://drive.google.com/uc?id=1ofd5U0q7-eUJrs6c4n0AZz2PLgc0KY7q&export=download'
FGC_VER?=1.7.8
TRAIN_FNAME?=FGC_release_all_train.json
DEV_FNAME?=FGC_release_all_dev.json
TEST_FNAME?=FGC_release_all_test.json

export TRAIN_FNAME DEV_FNAME TEST_FNAME

download:
	{ \
	wget -O data/raw/data.7z $(FGC_DATASET_7z_URL); \
	7z x -odata/raw/$(FGC_VER)/ data/raw/data.7z; \
	rm data/raw/data.7z; \
	}

process: filter get_qa

filter:
	mkdir -p data/processed/$(FGC_VER)
	python3 filter_dataset.py data/raw/$(FGC_VER)/$(TRAIN_FNAME) "data/processed/$(FGC_VER)/$${TRAIN_FNAME%.*}_filtered.json";
	python3 filter_dataset.py data/raw/$(FGC_VER)/$(DEV_FNAME) "data/processed/$(FGC_VER)/$${DEV_FNAME%.*}_filtered.json";
	python3 filter_dataset.py data/raw/$(FGC_VER)/$(TEST_FNAME) "data/processed/$(FGC_VER)/$${TEST_FNAME%.*}_filtered.json";

get_qa: data/processed/$(FGC_VER)/qa_train.tsv data/processed/$(FGC_VER)/qa_dev.tsv data/processed/$(FGC_VER)/qa_test.tsv

data/processed/$(FGC_VER)/qa_train.tsv: data/processed/$(FGC_VER)/$(TRAIN_FNAME:%.json=%_filtered.json)
	python3 get_qa_tsv.py $< $@

data/processed/$(FGC_VER)/qa_dev.tsv: data/processed/$(FGC_VER)/$(DEV_FNAME:%.json=%_filtered.json)
	python3 get_qa_tsv.py $< $@

data/processed/$(FGC_VER)/qa_test.tsv: data/processed/$(FGC_VER)/$(TEST_FNAME:%.json=%_filtered.json)
	python3 get_qa_tsv.py $< $@

clean:
	rm -rf data/processed/1.7.8
	rm -rf data/raw/1.7.8
