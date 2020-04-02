#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

import json
from argparse import *

from ..config import DEFAULT_CORENLP_IP
from ..models.wikiqa import WikiQA

argparser = ArgumentParser()
argparser.add_argument('--fgc_fpath', default='data/processed/1.7.8/FGC_release_all_train_filtered.json')
argparser.add_argument('--corenlp_ip', default=DEFAULT_CORENLP_IP)
argparser.add_argument('--use_fgc_kb', action='store_true')
argparser.add_argument('--pred_infer', default='rule', choices=['rule', 'neural'])
argparser.add_argument('--use_se', default='pred', choices=['gold', 'pred', 'pred_old', 'None'], help="""
gold: Use Gold Supporting Evidence in Dataset
pred: Use Predicted Supporting Evidence in Overall System (data/raw/predict)
pred_old: Use Predicted Supporting Evidence in Dataset provided by Meng-Tse (data/raw/1.7.8-revise-sp)
""")
argparser.add_argument('--qid_list', nargs='+', default=[])


argparser.add_argument('--eval_fpath', default='experiments/new/file4eval.tsv')

args = argparser.parse_args()
if args.pred_infer == 'rule':
    neural_pred_infer = False
elif args.pred_infer == 'neural':
    neural_pred_infer = True
else:
    raise ValueError

with open(args.fgc_fpath, encoding='utf-8') as f:
    docs = json.load(f)

wiki_qa = WikiQA(server=(args.corenlp_ip))
# docs = get_docs_with_certain_qs(['D002Q01'], docs)

wiki_qa.predict_on_docs(docs, args.eval_fpath,
                        neural_pred_infer=neural_pred_infer, use_fgc_kb=args.use_fgc_kb, use_se=args.use_se)
