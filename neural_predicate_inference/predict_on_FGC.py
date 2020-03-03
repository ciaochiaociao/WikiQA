#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
import pandas

from neural_predicate_inference.predict import predicate_inference
from utils import load_json
from fgc_utils import *
from wikidata4fgc_v2 import *

docs = load_json('../FGC_release_all(cn)_filtered2.json')
g = q_doc_generator(docs)
qids, qtexts, pids, atexts, plabels = [], [], [], [], []

for q, _ in g:
    qids.append(q['QID'])
    qtexts.append(q['QTEXT_CN'])
    pid = predicate_inference(q['QTEXT_CN'])
    pids.append(pid)
    atexts.append(q['ANSWER'][0]['ATEXT_CN'])
    try:
        plabels.append(get_all_aliases_from_pid(pid)[0][0])
    except IndexError:
        plabels.append('None')

df = pandas.DataFrame({'qid': qids, 'qtext': qtexts, 'pid': pids, 'plabel': plabels, 'atext': atexts})