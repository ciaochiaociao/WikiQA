# Copyright (c) $today.year. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

import json

import pandas

from fgc_utils import get_docs_with_certain_qs
from wikiqa import WikiQA

with open('../FGC_release_all(cn).json', encoding='utf-8') as f:
    docs = json.load(f)

wiki_qa = WikiQA(server='http://140.109.19.191:9000')

df = pandas.read_csv('fgc_wiki_benchmark_v0.1.tsv', sep='\t')

qids = df.id.to_list()
filtered_docs = get_docs_with_certain_qs(qids, docs)

with open('file4eval.tsv', 'w', encoding='utf-8') as file4eval:
    print('qid\tparsed_subj\tparsed_pred\tsid\tpretty_values\tproc_values\tanswers\tanswer', file=file4eval)
    for doc in filtered_docs:
        answers = wiki_qa.predict_on_qs_of_one_doc(doc, use_fgc_kb=False, file4eval=file4eval)
        
        # break