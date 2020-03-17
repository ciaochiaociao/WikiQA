#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
import os

from opencc import OpenCC
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

cc = OpenCC('t2s')
FGC_KB_PATH = 'data/external/fgc_knowledgebase.json'  # TODO: to put into .env file???
FGC_WIKI_BENCHMARK_FPATH = 'data/external/fgc_wiki_benchmark_v0.1.tsv'

DEFAULT_CORENLP_IP = os.getenv('DEFAULT_CORENLP_IP')
DATASAT_FPATH = os.getenv('DATASAT_FPATH')
TOKENIZER_FPATH = os.getenv('TOKENIZER_FPATH')
MODEL_FPATH = os.getenv('MODEL_FPATH')