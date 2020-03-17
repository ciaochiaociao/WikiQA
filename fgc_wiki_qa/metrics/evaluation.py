#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
import pandas
from ..utils.utils import load_json
from ..utils.fgc_utils import q_doc_generator, get_docs_with_certain_qs


def get_all_answers(q):
    return set([a['ATEXT_CN'] for a in q['ANSWER']] + [a['ATEXT'] for a in q['ANSWER']])


def get_ans_from_df(docs_df, qid):
    return docs_df[docs_df.qid == qid].iloc[0]['answer']


def gained_qids(already, pred):
    return set(pred) - set(already)


def sacrificed_qids(already, pred):
    return set(already) - set(pred)


def evaluate(already, fgc_fpath, file4eval):
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
    sacrificed = sacrificed_qids(already, corrects)
    gained = gained_qids(already, corrects)
    if sacrificed:
        print('[WARN] Sacrificed: ', sacrificed)
    if gained:
        print('[INFO] Gained: ', gained)
    print('Error\'s QIDs:', errors)
    print('Precision (excl. TN): {} / {} = {:.1%}'.format(num_corrects, num_answered, prec))


def main():
    already = ['D001Q01', 'D001Q03', 'D001Q06', 'D072Q03', 'D285Q01', 'D305Q06', 'D305Q08']
    fgc_fpath = 'FGC_release_all(cn)_filtered2.json'
    file4eval = 'experiments/model_v0.5_on_filtered/file4eval_filtered.tsv'
    evaluate(already, fgc_fpath, file4eval)


if __name__ == '__main__':
    main()

