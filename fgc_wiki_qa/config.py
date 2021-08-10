#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
import os
from os import path

from opencc import OpenCC

cc = OpenCC('t2s')
FGC_WIKI_BENCHMARK_FPATH = 'data/external/fgc_wiki_benchmark_v0.1.tsv'
FGC_KB_PATH = 'files/fgc_knowledgebase.json'
DATASET_FPATH = 'files/predicate_inference_combined_v2_with_nones.csv'
TOKENIZER_FPATH = 'files/bert-base-chinese-vocab.txt'
MODEL_FPATH = 'files/v2_with_nones_early_stop.h5'