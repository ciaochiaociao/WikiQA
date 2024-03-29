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
argparser.add_argument('--eval_fpath', default='reports/file4eval.tsv')
argparser.add_argument('--wiki_benchmark', default=FGC_WIKI_BENCHMARK_FPATH)
argparser.add_argument('--fgc_pred_infer_fpath', default='data/external/fgc_predicate_inference_v0.1.tsv')
# outputs
argparser.add_argument('--result_fpath', default='reports/result.txt')
argparser.add_argument('--error_analysis', default='reports/error_analysis.xlsx')
argparser.add_argument('--qids_fpath', default='reports/qids.json')
argparser.add_argument('--qids_by_stage_fpath', default='reports/qids_by_stage.json')


args = argparser.parse_args()
# from rule-based model v0.7
# already_corrects = \
#     ['D004Q05', 'D004Q06', 'D071Q01', 'D071Q02', 'D071Q12', 'D071Q13', 'D072Q04', 'D274Q01', 'D275Q01', 'D284Q01', 'D293Q01', 'D305Q01', 'D305Q06', 'D305Q08', 'D001Q01', 'D001Q03', 'D001Q06', 'D097Q04', 'D097Q05', 'D285Q01']
#     # ['D001Q01', 'D001Q03', 'D001Q06', 'D004Q06', 'D071Q12', 'D072Q04', 'D097Q04', 'D097Q05',
#     #                 'D274Q01', 'D275Q01', 'D284Q01', 'D285Q01', 'D293Q01', 'D305Q01', 'D305Q06', 'D305Q08']\
#     # + ['D004Q05'] # v0.7_on_1.5
#
# already_errors = \
#     ['D302Q01', 'D001Q09']
#     # ['D001Q09', 'D071Q01', 'D071Q13', 'D077Q10', 'D254Q01', 'D247Q09', 'D275Q03', 'D302Q01']

# already_corrects = set(already_corrects) & set(qids)
evaluate(args.fgc_fpath, args.eval_fpath, args.fgc_pred_infer_fpath,
         TeeLogger(args.result_fpath), args.qids_fpath, args.qids_by_stage_fpath)
error_analysis(args.fgc_qa_fpath, args.wiki_benchmark, args.eval_fpath, args.error_analysis)
