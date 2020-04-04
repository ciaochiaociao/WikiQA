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


def evaluate(fgc_fpath, file4eval, fgc_pred_infer_fpath, tee_logger, qids_fpath,
             qids_by_stage_fpath):
    sys.stdout = tee_logger
    docs = load_json(fgc_fpath)
    df_pred = pandas.read_csv(file4eval, sep='\t', header=0)
    df_wiki_pred = pandas.read_csv(fgc_pred_infer_fpath, header=0)
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
        len(parsed) / len(questions_filtered),
        len(parsed) / len(questions_has_gold_predicate)))
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

    qids_dict = {
        'corrects': corrects,
        'errors': errors
    }

    with open(qids_fpath, 'w', encoding='utf-8') as f:
        json.dump(qids_dict, f, ensure_ascii=False, indent=4)

    # with open(qids_by_stage_fpath, 'w', encoding='utf-8') as f:
    #     json.dump(qids_by_stage_dict, f, ensure_ascii=False, indent=4)

    print('Precision (excl. TN): {} / {} = {:.1%}'.format(num_corrects, num_answered, prec))
