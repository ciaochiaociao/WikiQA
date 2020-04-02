#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
import os
from os import path

from opencc import OpenCC
from dotenv import load_dotenv, find_dotenv

# load_dotenv(find_dotenv())

cc = OpenCC('t2s')
FGC_WIKI_BENCHMARK_FPATH = 'data/external/fgc_wiki_benchmark_v0.1.tsv'

# DEFAULT_CORENLP_IP = os.getenv('DEFAULT_CORENLP_IP')
# DATASET_FPATH = os.getenv('DATASET_FPATH')
# TOKENIZER_FPATH = os.getenv('TOKENIZER_FPATH')
# MODEL_FPATH = os.getenv('MODEL_FPATH')

cur_path = path.dirname(path.abspath(__file__))
FGC_KB_PATH = path.join(cur_path, 'files', 'fgc_knowledgebase.json')  # TODO: to put into .env file???
DATASET_FPATH = path.join(cur_path, 'files', 'predicate_inference_combined_v2_with_nones.csv')
TOKENIZER_FPATH = path.join(cur_path, 'files', 'bert-base-chinese-vocab.txt')
MODEL_FPATH = path.join(cur_path, 'files', 'v2_with_nones_early_stop.h5')
DEFAULT_CORENLP_IP = 'http://140.109.19.51:9000'