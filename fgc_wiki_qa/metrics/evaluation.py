#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
import sys

import pandas
from ..utils.utils import load_json
from ..utils.fgc_utils import q_doc_generator, get_docs_with_certain_qs


def get_all_answers(q):
    return set([a['ATEXT_CN'] for a in q['ANSWER']] + [a['ATEXT'] for a in q['ANSWER']])


def get_ans_from_df(docs_df, qid):
    return docs_df[docs_df.qid == qid].iloc[0]['answer']


def gained_qids(already_corrects, pred):
    return set(pred) - set(already_corrects)


def sacrificed_qids(already_corrects, pred):
    return set(already_corrects) - set(pred)


def get_new_errors(already_errors, errors):
    return set(errors) - set(already_errors)


def get_corrected_errors(already_errors, errors):
    return set(already_errors) - set(errors)


def get_stubborn_errors(already_errors, errors):
    return set(already_errors) & set(errors)


def evaluate(already_corrects, already_errors, fgc_fpath, file4eval, tee_logger):
    sys.stdout = tee_logger
    docs = load_json(fgc_fpath)
    df_pred = pandas.read_csv(file4eval, sep='\t', header=0)
    predicted = df_pred[(df_pred.answer.notna()) & (df_pred.answer != 'None')]
    if len(predicted) == 0:
        print('No questions are successfully answered')
        return
    docs_pred = get_docs_with_certain_qs(predicted.qid.to_list(), docs)
    gen = q_doc_generator(docs_pred)
    corrects, errors = [], []
    for q, d in gen:
        atexts = get_all_answers(q)
        if get_ans_from_df(predicted, q['QID']) in atexts:
            corrects.append(q['QID'])
        else:
            errors.append(q['QID'])
    num_corrects, num_errors, num_answered = len(corrects), len(errors), len(corrects) + len(errors)
    prec = num_corrects / num_answered
    questions_filtered = list(q_doc_generator(docs))
    parsed = df_pred[df_pred.parsed_subj != 'not_parsed']
    ent_linked = df_pred[df_pred.sid.notna()]
    traversed = df_pred[(df_pred.pretty_values.notna()) & (df_pred.pretty_values != '[]')]
    matched = df_pred[(df_pred.answers != '[]') & (df_pred.answers.notna())]
    print('# questions trying to answer:', len(questions_filtered))
    print('# questions that activate WikiQA: {} ({:.1%}) (Parse Q / Predicate Inference)'.format(
        len(parsed),
        len(parsed) / len(questions_filtered)))
    print('\t# after entity linking: {} ({:.1%})'.format(
        len(ent_linked),
        len(ent_linked) / len(parsed)))
    print('\t# after traversing: {} ({:.1%})'.format(len(traversed), len(traversed) / len(ent_linked)))
    print('\t# after matching w/ passage: {} ({:.1%})'.format(len(matched), len(matched) / len(traversed)))
    print(
        '# questions answered: {0} ({0} / {1} = {2:.1%})'.format(num_answered, len(parsed), num_answered / len(parsed)))
    print('# corrects:', num_corrects)
    print('# errors:', num_errors)
    print('Correct\'s QIDs:', corrects)
    print('Error\'s QIDs:', errors)
    sacrificed = sacrificed_qids(already_corrects, corrects)
    gained = gained_qids(already_corrects, corrects)
    new_errors = get_new_errors(already_errors, errors)
    corrected_errors = get_corrected_errors(already_errors, errors)
    stubborn_errors = get_stubborn_errors(already_errors, errors)
    if sacrificed:
        print('[WARN] Sacrificed: ', sacrificed)
    if new_errors:
        print('[WARN] New Errors: ', new_errors)
    if stubborn_errors:
        print('[INFO] Stubborn Errors: ', stubborn_errors)
    if gained:
        print('[INFO] Gained: ', gained)
    if corrected_errors:
        print('[INFO] Corrected Errors: ', corrected_errors)
    print('Precision (excl. TN): {} / {} = {:.1%}'.format(num_corrects, num_answered, prec))



