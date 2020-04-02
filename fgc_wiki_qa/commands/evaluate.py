#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

from argparse import ArgumentParser

from ..utils.utils import TeeLogger

from ..config import FGC_WIKI_BENCHMARK_FPATH
from ..metrics.error_analysis import error_analysis
from ..metrics.evaluation import evaluate


argparser = ArgumentParser()
# inputs
argparser.add_argument('--fgc_fpath', default='data/processed/1.7.8/FGC_release_all_train_filtered.json')
argparser.add_argument('--fgc_qa_fpath', default='data/processed/1.7.8/qa_train.tsv')
argparser.add_argument('--eval_fpath', default='experiments/new/file4eval.tsv')
argparser.add_argument('--wiki_benchmark', default=FGC_WIKI_BENCHMARK_FPATH)

# outputs
argparser.add_argument('--result_fpath', default='experiments/new/result.txt')
argparser.add_argument('--error_analysis', default='experiments/new/error_analysis.xlsx')


args = argparser.parse_args()
# from rule-based model v0.7
already_corrects = \
    ['D004Q05', 'D004Q06', 'D071Q01', 'D071Q02', 'D071Q12', 'D071Q13', 'D072Q04', 'D274Q01', 'D275Q01', 'D284Q01', 'D293Q01', 'D305Q01', 'D305Q06', 'D305Q08', 'D001Q01', 'D001Q03', 'D001Q06', 'D097Q04', 'D097Q05', 'D285Q01']
    # ['D001Q01', 'D001Q03', 'D001Q06', 'D004Q06', 'D071Q12', 'D072Q04', 'D097Q04', 'D097Q05',
    #                 'D274Q01', 'D275Q01', 'D284Q01', 'D285Q01', 'D293Q01', 'D305Q01', 'D305Q06', 'D305Q08']\
    # + ['D004Q05'] # v0.7_on_1.5

already_errors = \
    ['D302Q01', 'D001Q09']
    # ['D001Q09', 'D071Q01', 'D071Q13', 'D077Q10', 'D254Q01', 'D247Q09', 'D275Q03', 'D302Q01']

# already_corrects = set(already_corrects) & set(qids)
evaluate(already_corrects, already_errors, args.fgc_fpath, args.eval_fpath, args.fgc_pred_infer_fpath,
         TeeLogger(args.result_fpath), args.qids_fpath, args.qids_by_stage_fpath)
error_analysis(args.fgc_qa_fpath, args.wiki_benchmark, args.eval_fpath, args.error_analysis)
