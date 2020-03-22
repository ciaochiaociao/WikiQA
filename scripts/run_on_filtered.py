# Copyright (c) $today.year. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

import json

from fgc_wiki_qa.metrics.error_analysis import error_analysis
from fgc_wiki_qa.metrics.evaluation import evaluate
from fgc_wiki_qa.models.wikiqa import WikiQA

fgc_fpath = '../FGC_release_all(cn)_filtered2.json'
file4eval_fpath = 'file4eval_filtered.tsv'
fgc_qa_fpath = 'fgc_qa_filtered.tsv'
fgc_wiki_benchmark_fpath = 'fgc_wiki_benchmark_v0.1.tsv'
error_analysis_fpath = 'error_analysis.xlsx'

with open(fgc_fpath, encoding='utf-8') as f:
    docs = json.load(f)

wiki_qa = WikiQA(server='http://140.109.19.51:9000')

docs = get_docs_with_certain_qs(['D002Q01'], docs)

all_answers = []
with open(file4eval_fpath, 'w', encoding='utf-8') as file4eval:
    print('qid\tparsed_subj\tparsed_pred\tsid\tpretty_values\tproc_values\tanswers\tanswer', file=file4eval)
    for doc in docs:
        answers = wiki_qa.predict_on_qs_of_one_doc(doc, use_fgc_kb=False, file4eval=file4eval)
        all_answers.extend(answers)
        print(answers)
print(all_answers)

already_corrects = ['D001Q01', 'D001Q03', 'D001Q06', 'D072Q03', 'D285Q01', 'D305Q06', 'D305Q08']
# already_corrects = set(already_corrects) & set(qids)
evaluate(already_corrects, already_errors, fgc_fpath, file4eval_fpath, eval_result_fpath)

error_analysis(fgc_qa_fpath, fgc_wiki_benchmark_fpath, file4eval_fpath, error_analysis_fpath)
