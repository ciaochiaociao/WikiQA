# Copyright (c) $today.year. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited
# Proprietary and confidential
# Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

import json

import pandas

from fgc_utils import get_docs_with_certain_qs
from wikiqa import WikiQA

with open('../FGC_release_all(cn)_filtered.json', encoding='utf-8') as f:
    docs = json.load(f)

wiki_qa = WikiQA(server='http://140.109.19.191:9000')

docs = get_docs_with_certain_qs(['D002Q01'], docs)

all_answers = []
with open('file4eval_filtered.tsv', 'w', encoding='utf-8') as file4eval:
    print('qid\tparsed_subj\tparsed_pred\tsid\tpretty_values\tproc_values\tanswers\tanswer', file=file4eval)
    for doc in docs:
        answers = wiki_qa.predict_on_qs_of_one_doc(doc, use_fgc_kb=False, file4eval=file4eval)
        all_answers.extend(answers)
        print(answers)
print(all_answers)