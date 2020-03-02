#   Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#   Unauthorized copying of this file, via any medium is strictly prohibited
#   Proprietary and confidential
#   Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
from copy import deepcopy


def get_doc(did, docs):
    for doc in docs:
        if doc['DID'] == did:
            return doc


def get_que(qid, docs):
    for doc in docs:
        for q in doc['QUESTIONS']:
            if q['QID'] == qid:
                return q


def get_doc_with_one_que(qid, docs):
    for doc in docs:
        for q in doc['QUESTIONS']:
            if q['QID'] == qid:
                doc = deepcopy(doc)
                doc['QUESTIONS'] = [q]
                return doc


def remove_docs_before_did(did, docs):
    results = deepcopy(docs)
    counter = 0
    for doc in docs:
        if doc['DID'] == did:
            print(f'{counter} docs are removed before {did}')
            return results
        else:
            counter += 1
            results.pop(0)
    print('All docs are removed')
    return results


def get_docs_with_certain_qs(qids, docs):

    results = []
    for doc in docs:
        keep_doc = False
        qs = []
        for q in doc['QUESTIONS']:
            if q['QID'] in qids:
                keep_doc = True
                qs.append(q)
        doc = deepcopy(doc)
        doc['QUESTIONS'] = qs
        if keep_doc:
            results.append(doc)

    return results


def data_to_csv(docs, f):
    print('qid', 'qtext', 'atext', 'qtype', 'atype', 'amode', sep='\t', file=f)
    for doc in docs:
        for q in doc['QUESTIONS']:
            print(q['QID'], q['QTEXT_CN'], [ans['ATEXT_CN'] for ans in q['ANSWER']], q['QTYPE'], q['ATYPE'], q['AMODE'], sep='\t', file=f)


def q_doc_generator(docs):
    for doc in docs:
        for q in doc['QUESTIONS']:
            yield q, doc


