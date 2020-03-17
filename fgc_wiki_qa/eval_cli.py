#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

from argparse import ArgumentParser

from .config import FGC_WIKI_BENCHMARK_FPATH
from .metrics.error_analysis import error_analysis
from .metrics.evaluation import evaluate


argparser = ArgumentParser()
argparser.add_argument('--fgc_fpath', default='../data/processed/FGC_release_all(cn)_filtered2.json')
argparser.add_argument('--fgc_qa_fpath', default='../experiments/new/fgc_qa.tsv')
argparser.add_argument('--error_analysis', default='../experiments/new/error_analysis.xlsx')
argparser.add_argument('--eval_fpath', default='../experiments/new/file4eval.tsv')
argparser.add_argument('--wiki_benchmark', default=FGC_WIKI_BENCHMARK_FPATH)
args = argparser.parse_args()
fgc_fpath = args['fgc_fpath']
fgc_qa_fpath = args['fgc_qa_fpath']
error_analysis_fpath = args['error_analysis']
file4eval_fpath = args['eval_fpath']
fgc_wiki_benchmark_fpath = args['wiki_benchmark']
# from rule-based model v0.7
already_corrects = ['D001Q01', 'D001Q03', 'D001Q06', 'D004Q06', 'D071Q12', 'D072Q04', 'D097Q04', 'D097Q05',
                    'D247Q09',
                    'D274Q01', 'D275Q01', 'D284Q01', 'D285Q01', 'D293Q01', 'D305Q01', 'D305Q06', 'D305Q08']
# already_corrects = set(already_corrects) & set(qids)
evaluate(already_corrects,
         fgc_fpath,
         file4eval_fpath)
error_analysis(fgc_qa_fpath, fgc_wiki_benchmark_fpath, file4eval_fpath, error_analysis_fpath)

