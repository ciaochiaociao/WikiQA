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


def get_doc_with_one_que(qid, docs, copy=True):
    if copy:
        docs = deepcopy(docs)
    for doc in docs:
        for q in doc['QUESTIONS']:
            if q['QID'] == qid:
                doc['QUESTIONS'] = [q]
                return doc


def remove_docs_before_did(did, docs, copy=True):
    if copy:
        docs = deepcopy(docs)

    removed_ids = [doc['DID'] for doc in docs if doc['DID'] <= did]

    for i in range(len(docs)-1, -1, -1):
        doc = docs[i]
        if doc['DID'] in removed_ids:
            docs.pop(i)
    print(f'{len(removed_ids)} docs are removed before {did}')
    return docs


def get_docs_with_certain_qs(qids, docs, copy=True):
    if copy:
        docs = deepcopy(docs)

    results = []
    for doc in docs:
        keep_doc = False
        qs = []
        for q in doc['QUESTIONS']:
            if q['QID'] in qids:
                keep_doc = True
                qs.append(q)
        doc['QUESTIONS'] = qs
        if keep_doc:
            results.append(doc)

    return results


def data_to_csv(docs, f):
    print('qid', 'qtext', 'atext', 'qtype', 'atype', 'amode', sep='\t', file=f)
    for doc in docs:
        for q in doc['QUESTIONS']:
            try:
                answers = [ans['ATEXT_CN'] for ans in q['ANSWER']]
            except KeyError:
                answers = []
            print(q['QID'], q['QTEXT_CN'], answers, q['QTYPE'], q['ATYPE'], q['AMODE'], sep='\t', file=f)


def q_doc_generator(docs):
    for doc in docs:
        for q in doc['QUESTIONS']:
            yield q, doc


def get_amode(qid, amode, docs):
    return get_que(qid, docs)['AMODE'][amode]


def get_golds_from_qid(qid, docs):
    return [a['ATEXT'] for a in get_que(qid, docs)['ANSWER']]


def g_doc_errors_gen(docs):
    for q, doc in q_doc_generator(docs):
        golds = [a['ATEXT']for a in q['ANSWER']]
        pred = q['AFINAL']['ATEXT_TW']
        if pred not in golds:
            yield q, doc


all_ents = [
    'PERSON',
    'PER',
    'TITLE',
    'DEGREE',
    'RELIGION',
    'IDEOLOGY',
    'EMAIL',
    'URL',
    'SET',
    'CRIMINAL_CHARGE',
    'CAUSE_OF_DEATH',
    'MISC',
    'DEMONYM',
    'CITY',
    'STATE_OR_PROVINCE',
    'COUNTRY',
    'GPE',
    'LOC',
    'LOCATION',
    'NATIONALITY',
    'ORGANIZATION',
    'ORG',
    'FACILITY',
    'DATE',
    'TIME',
    'DURATION',
    'DYNASTY',
    'MONEY',
    'ORDINAL',
    'NUMBER',
    'PERCENT',
]


ent_ans_map = {
    'PERSON': ['Person'],
    'PER': ['Person'],
    'TITLE': ['Organization', 'Person'],

    'DEGREE': ['Object'],
    'RELIGION': ['Object'],
    'IDEOLOGY': ['Object'],  #?
    'EMAIL': ['Object'],
    'URL': ['Object'],
    'SET': ['Object'],  #?

    'CRIMINAL_CHARGE': ['Event'],  #?
    'CAUSE_OF_DEATH': ['Event'],

    'MISC': ['Object', 'Event'],

    'DEMONYM': ['Organization', 'Location'],
    'CITY': ['Organization', 'Location'],
    'STATE_OR_PROVINCE': ['Organization', 'Location'],
    'COUNTRY': ['Organization', 'Location'],
    'GPE': ['Organization', 'Location'],
    'LOC': ['Organization', 'Location'],
    'LOCATION': ['Organization', 'Location'],
    'NATIONALITY': ['Location'],

    'ORGANIZATION': ['Organization'],
    'ORG': ['Organization'],
    'FACILITY': ['Organization'],

    'DATE': ['Date-Duration'],
    'TIME': ['Date-Duration'],
    'DURATION': ['Date-Duration'],
    'DYNASTY': ['Date-Duration', 'Event', 'Location'],

    'MONEY': ['Num-Measure'],
    'ORDINAL': ['Num-Measure'],
    'NUMBER': ['Num-Measure'],
    'PERCENT': ['Num-Measure']
}

coarse2fines_map = {
    'PER': ['PERSON', 'PER'],
    'LOC': ['LOCATION', 'LOC', 'CITY', 'STATE_OR_PROVINCE', 'COUNTRY', 'NATIONALITY', 'GPE', 'DEMONYM'],
    'ORG': ['ORGANIZATION', 'ORG', 'FACILITY'],
    'TITLE': ['TITLE'],
    'DEGREE': ['DEGREE'],
    'RELIGION': ['RELIGION'],
    'IDEOLOGY': ['IDEOLOGY'],
    'EMAIL': ['EMAIL'],
    'URL': ['URL'],
    'SET': ['SET'],
    'CRIMINAL_CHARGE': ['CRIMINAL_CHARGE'],
    'CAUSE_OF_DEATH': ['CAUSE_OF_DEATH'],
    'MISC': ['MISC'],
    'DATE': ['DATE'],
    'TIME': ['TIME'],
    'DURATION': ['DURATION'],
    'DYNASTY': ['DYNASTY'],
    'MONEY': ['MONEY'],
    'ORDINAL': ['ORDINAL'],
    'NUMBER': ['NUMBER'],
    'PERCENT': ['PERCENT'],
}





fine2coarse_map = {fine: coarse for coarse, fines in coarse2fines_map.items() for fine in fines}

assert set(ent_ans_map.keys()) == set(all_ents)
assert set([fine for fines in coarse2fines_map.values() for fine in fines]) == set(all_ents)
assert set(fine2coarse_map.keys()) == set(all_ents)


def interesting_ne(ne, etype):
    cetype = fine2coarse_map[etype]
    return cetype in ['PER', 'LOC', 'ORG'] and len(ne) > 1 or \
        cetype in ['MISC'] and len(ne) >2


def ne_generator_from_doc(doc):

    for sent in doc['SENTS']:
        for ne in sent['IE']['NER']:
            yield ne


def interesting_ne_gen_from_doc(doc):
    for ne in ne_generator_from_doc(doc):
        if interesting_ne(ne['string'], ne['type']):
            yield ne

