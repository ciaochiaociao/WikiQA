#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
import json
import sys

import pandas
from ..utils.utils import load_json
from ..utils.fgc_utils import q_doc_generator, get_docs_with_certain_qs


def get_all_answers(q):
    return set([a['ATEXT_CN'] for a in q['ANSWER']] + [a['ATEXT'] for a in q['ANSWER']])


def get_ans_from_df(docs_df, qid):
    return docs_df[docs_df.qid == qid].iloc[0]['answer']


def get_stubborn_errors(already_errors, errors):
    return list(set(already_errors) & set(errors))


def get_robust_corrects(already_corrects, corrects):
    return list(set(already_corrects) & set(corrects))


def get_betrayed_corrects(already_corrects, errors):
    return list(set(already_corrects) & set(errors))


def get_corrected_errors(already_errors, corrects):
    return list(set(already_errors) & set(corrects))


def get_gained_corrects(already_corrects, corrects, already_errors):
    return list(set(corrects) - set(already_corrects) - set(get_corrected_errors(already_errors, corrects)))


def get_sacrificed_corrects(already_corrects, corrects, errors):
    return list((set(already_corrects) - set(corrects)) - set(get_betrayed_corrects(already_corrects, errors)))


def get_new_errors(already_errors, errors, already_corrects):
    return list((set(errors) - set(already_errors)) - set(get_betrayed_corrects(already_corrects, errors)))


def get_eliminated_errors(already_errors, errors, corrects):
    return list((set(already_errors) - set(errors)) - set(get_corrected_errors(already_errors, corrects)))


def get_tp_predicates(parsed, golds):
    return list(set(parsed) & set(golds))


def get_fn_predicates(parsed, golds):
    return list(set(golds) - set(parsed))


def get_fp_predicates(parsed, golds):
    return list(set(parsed) - set(golds))


def evaluate(already_corrects, already_errors, fgc_fpath, file4eval, fgc_pred_infer_fpath, tee_logger, qids_fpath,
             qids_by_stage_fpath):
    sys.stdout = tee_logger
    docs = load_json(fgc_fpath)
    df_pred = pandas.read_csv(file4eval, sep='\t', header=0)
    df_wiki_pred = pandas.read_csv(fgc_pred_infer_fpath, sep='\t', header=0)
    predicted = df_pred[(df_pred.answer.notna()) & (df_pred.answer != 'None')]
    if len(predicted) == 0:
        print('No questions are successfully answered')
        return
    docs_pred = get_docs_with_certain_qs(predicted.qid.to_list(), docs)
    gen = q_doc_generator(docs_pred)

    # evaluate
    corrects, errors = [], []
    for q, d in gen:
        atexts = get_all_answers(q)
        if get_ans_from_df(predicted, q['QID']) in atexts:
            corrects.append(q['QID'])
        else:
            errors.append(q['QID'])

    # evalute predicate inference
    # TODO

    # statistics
    num_corrects, num_errors, num_answered = len(corrects), len(errors), len(corrects) + len(errors)
    prec = num_corrects / num_answered
    questions_filtered = list(q_doc_generator(docs))
    parsed = df_pred[df_pred.parsed_subj != 'not_parsed']
    questions_has_gold_predicate = df_wiki_pred[df_wiki_pred.label != 'None']
    ent_linked = df_pred[df_pred.sid.notna()]
    traversed = df_pred[(df_pred.pretty_values.notna()) & (df_pred.pretty_values != '[]')]
    matched = df_pred[(df_pred.answers != '[]') & (df_pred.answers.notna())]
    print('# questions trying to answer:', len(questions_filtered))
    print('# questions that activate WikiQA: {} ({:.1%}) (Parse Q / Predicate Inference) (ratio to ideal: {:.1%})'.format(
        len(parsed),
        len(parsed) / len(questions_filtered)),
        len(parsed) / len(questions_has_gold_predicate))
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

    qids_by_stage_dict = {
        'fn_predicates': get_fn_predicates(parsed, questions_has_gold_predicate.qid.to_list()),
        'fp_predicates': get_fp_predicates(parsed, questions_has_gold_predicate.qid.to_list()),
        'tp_predicates': get_tp_predicates(parsed, questions_has_gold_predicate.qid.to_list()),
    }

    qids_dict = {
        'sacrificed_corrects' : get_sacrificed_corrects(already_corrects, corrects, errors),
        'gained_correts' : get_gained_corrects(already_corrects, corrects, already_errors),
        'new_errors' : get_new_errors(already_errors, errors, already_corrects),
        'eliminated_errors' : get_eliminated_errors(already_errors, errors, corrects),
        'stubborn_errors' : get_stubborn_errors(already_errors, errors),
        'robust_corrects' : get_robust_corrects(already_corrects, corrects),
        'betrayed_corrects' : get_betrayed_corrects(already_corrects, errors),
        'corrected_errors' : get_corrected_errors(already_errors, corrects),
    }

    if qids_dict['robust_corrects']:
        print('[INFO] Robust Corrects (C>C): ', qids_dict['robust_corrects'])
    if qids_dict['stubborn_errors']:
        print('[INFO] Stubborn Errors (E>E): ', qids_dict['stubborn_errors'])
    if qids_dict['betrayed_corrects']:
        print('[BAD] Betrayed Corrects (C>E): ', qids_dict['betrayed_corrects'])
    if qids_dict['sacrificed_corrects']:
        print('[BAD] Sacrificed (C>_): ', qids_dict['sacrificed_corrects'])
    if qids_dict['new_errors']:
        print('[BAD] New Errors (_>E): ', qids_dict['new_errors'])
    if qids_dict['corrected_errors']:
        print('[GOOD] Corrected Errors (E>C): ', qids_dict['corrected_errors'])
    if qids_dict['eliminated_errors']:
        print('[GOOD] Eliminated Errors (E>_): ', qids_dict['eliminated_errors'])
    if qids_dict['gained_correts']:
        print('[GOOD] Gained (_>C): ', qids_dict['gained_correts'])
    print('Precision (excl. TN): {} / {} = {:.1%}'.format(num_corrects, num_answered, prec))

    with open(qids_fpath, 'w', encoding='utf-8') as f:
        json.dump(qids_dict, f, ensure_ascii=False, indent=4)

    with open(qids_by_stage_fpath, 'w', encoding='utf-8') as f:
        json.dump(qids_by_stage_dict, f, ensure_ascii=False, indent=4)

