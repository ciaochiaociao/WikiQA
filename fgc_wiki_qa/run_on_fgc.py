#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

import json

from .config import DEFAULT_CORENLP_IP
from .models.wikiqa import WikiQA
from argparse import *

argparser = ArgumentParser()
argparser.add_argument('--fgc_fpath', default='../data/processed/FGC_release_all(cn)_filtered2.json')
argparser.add_argument('--corenlp_ip', default=DEFAULT_CORENLP_IP)
argparser.add_argument('--eval_fpath', default='../experiments/new/file4eval.tsv')
argparser.add_argument('--use_fgc_kb', action='store_true')
argparser.add_argument('--pred_infer', default='rule', choices=['rule', 'neural'])


args = argparser.parse_args()
fgc_fpath = args['fgc_fpath']
corenlp_ip = args['corenlp_ip']
file4eval_fpath = args['eval_fpath']
use_fgc_kb = args['use_fgc_kb']

if args['pred_infer'] == 'rule':
    neural_pred_infer = True
elif args['pred_infer'] == 'neural':
    neural_pred_infer = False
else:
    raise ValueError

with open(fgc_fpath, encoding='utf-8') as f:
    docs = json.load(f)

wiki_qa = WikiQA(server=corenlp_ip)
# docs = get_docs_with_certain_qs(['D002Q01'], docs)

wiki_qa.predict_on_docs(docs, file4eval_fpath, neural_pred_infer, use_fgc_kb)

