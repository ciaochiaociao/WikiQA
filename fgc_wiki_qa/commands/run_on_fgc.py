#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

import json
from argparse import *

from ..config import DEFAULT_CORENLP_IP
from ..models.wikiqa import WikiQA
from ..utils.fgc_utils import get_docs_with_certain_qs

argparser = ArgumentParser()
argparser.add_argument('--fgc_fpath')
argparser.add_argument('--corenlp_ip', default=DEFAULT_CORENLP_IP)
argparser.add_argument('--use_fgc_kb', action='store_true')
argparser.add_argument('--pred_infer', default='rule', choices=['rule', 'neural'])
argparser.add_argument('--use_se', default='pred', choices=['gold', 'pred', 'pred_old', 'None'], help="""
gold: Use Gold Supporting Evidence in Dataset
pred: Use Predicted Supporting Evidence in Overall System (data/raw/predict)
pred_old: Use Predicted Supporting Evidence in Dataset provided by Meng-Tse (data/raw/1.7.8-revise-sp)
""")
argparser.add_argument('--mode', default='dev')
argparser.add_argument('--quiet', action='store_true')
argparser.add_argument('--qid_list', nargs='+', default=[])


argparser.add_argument('--eval_fpath', default='reports/file4eval.tsv')

args = argparser.parse_args()
if args.pred_infer == 'rule':
    neural_pred_infer = False
elif args.pred_infer == 'neural':
    neural_pred_infer = True
else:
    raise ValueError

with open(args.fgc_fpath, encoding='utf-8') as f:
    docs = json.load(f)

if args.qid_list:
    docs = get_docs_with_certain_qs(args.qid_list, docs)

wiki_qa = WikiQA(corenlp_ip=args.corenlp_ip, mode=args.mode, file4eval_fpath=args.eval_fpath,
                 pred_infer=args.pred_infer, use_fgc_kb=args.use_fgc_kb, use_se=args.use_se, verbose=not args.quiet)

wiki_qa.config.to_json_file(args.config_fpath)

wiki_qa.predict_on_docs(docs, args.eval_fpath,
                        neural_pred_infer=neural_pred_infer, use_fgc_kb=args.use_fgc_kb, use_se=args.use_se)
