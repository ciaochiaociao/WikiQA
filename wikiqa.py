#  Copyright (c) 2020. The Natural Language Understanding Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

from config import DEFAULT_CORENLP_IP, FGC_KB_PATH
from entity_linking import build_candidates_to_EL, entity_linking
from predicate_inference_rules import parse_question_by_regex
from value2ans import generate_answers_from_datavalue, preprocess_values
from wikidata4fgc_v2 import *
from stanfordnlp.server import CoreNLPClient
from os.path import join, abspath, dirname

# fix bug: google.protobuf.message.DecodeError: Error parsing message
# ref: https://github.com/stanfordnlp/stanfordnlp/issues/154
from google.protobuf.pyext._message import SetAllowOversizeProtos

from wikidata_utils import traverse_wikidata_by_attr_name

SetAllowOversizeProtos(True)


UNKNOWN_MESSAGE = 'Unknown in evaluation mode (if_evaluate=True)'


# global variables
prediction_results = []


class WikiQA:
    def __init__(self, server=DEFAULT_CORENLP_IP):
        self.corenlp_ip = server  # 'http://localhost:9000'
        self.if_evaluate = True  # evaluate the performance

        # FGC KB
        cur_path = dirname(abspath(__file__))
        with open(join(cur_path, FGC_KB_PATH), 'r', encoding='utf-8') as f:
            self.kbqa_sheet = json.load(f)

    def predict(self, fgc_data, save_result=False, use_fgc_kb=True):
        """
        :param Dict fgc_data: fgc data at the level of document, i.e., one document with multiple questions
        """
        global nlp
        with CoreNLPClient(endpoint=self.corenlp_ip, annotators="tokenize,ssplit,lemma,pos,ner",
                            start_server=False, properties='chinese') as nlp:
            return self._predict(fgc_data, save_result, use_fgc_kb)

    def _get_from_fgc_kb(self, qtext):
        if qtext in self.kbqa_sheet:
            return self.kbqa_sheet[qtext]

    def _predict(self, fgc_data: dict, if_save_result: bool, use_fgc_kb: bool) -> List[List[Dict]]:
        global q, wd_item, rel, attr, predicate_matched, question_data, passage_data, mentions_bracketed, \
            dtext, answers, if_evaluate, qtext, debug_info

        # if the gold answer provided for checking performance
        if_evaluate = self.if_evaluate

        # for debugging
        attribute_match_count = 0
        debug_infos = []
        did_shown = []

        # variable for others to access
        dtext = fgc_data['DTEXT_CN']

        # get IE data for passage
        passage_data = nlp.annotate(dtext, properties={'pipelineLanguage': 'zh'})

        # output answers
        all_answers = []

        # region questions for-loop [{'QID': ...}, {'QID': ...}, ...]
        for q in fgc_data['QUESTIONS']:

            # FGC KB runs first if used
            if use_fgc_kb:
                matched = self._get_from_fgc_kb(q['QTEXT_CN'])
                if matched is not None:
                    # q_anses = default_answer(q['DID'], 'Wiki-Kb-Inference', _match(q['ATEXT']), 1.0)
                    q_anses = [{
                        # 'QID': q['QID'],
                        'AMODULE': 'Wiki-Json-Inference',
                        'ATEXT': matched,
                        'score': 1.0
                    }]
                    all_answers.append(q_anses)
                    continue

            # filter out unwanted mode
            if q['AMODE'] == 'Yes-No':
                continue
            if q['ATYPE'] in ['Object']:
                continue

            # for evaluation
            if if_evaluate:
                answers = [a['ATEXT_CN'] for a in q['ANSWER']]
            else:
                answers = UNKNOWN_MESSAGE

            # variables
            qtext = q['QTEXT_CN']

            # get IE data for question
            # note: don't add ssplit because there is a bug in CoreNLP: the NERMention tokenStartInSentenceInclusive
            # uses the location index in the original text without sentence split rather than with split even with
            # ssplit annotator on
            question_data = nlp.annotate(qtext,
                                         properties={'pipelineLanguage': 'zh', 'annotators': "tokenize,lemma,pos,ner"})

            # for debugging
            if fgc_data['DID'] not in did_shown:
                did_shown.append(fgc_data['DID'])
                print('{} (tokenized, hans):'.format(fgc_data['DID']))
                for sent in passage_data.sentence:
                    print(f'(sent{sent.sentenceIndex})', end=' ')
                    for tok in sent.token:
                        print(tok.originalText, end=' ')
                print('\n')
            print('{} (tokenized, hans):'.format(q['QID']),
                  *[token.originalText for sent in question_data.sentence for token in sent.token],
                  '(Gold)', answers,
                  sep=' ')

            # ===== STEP A. parse question (parse entity name + predicate inference) =====
            parsed_result = parse_question_by_regex(qtext)
            if parsed_result:
                name, attr, span, matched_pattern = parsed_result
                # print('[matched pattern]name, attr, span, matched_pattern:', name, attr, span, matched_pattern)
            else:  # skip this question if not matched by our rules/regex
                continue

            # ===== STEP B. entity linking =====
            ent_link_cands = build_candidates_to_EL(name, passage_data, question_data, span)
            wd_items = entity_linking(ent_link_cands)

            # ===== STEP C. traverse Wikidata =====
            answers_from_ent_link_queries = []  # answers_from_one_question
            # loop thourgh all results from wikidata API, get all datavalues from all relations of all wd_items from
            # one ent_link_query
            wd_item_rel_datavalues_matched_tuples = []
            for wd_item in wd_items:

                rel_datavalues_matched_tuples = traverse_wikidata_by_attr_name(attr, wd_item)

                if not rel_datavalues_matched_tuples:
                    continue
                wd_item_rel_datavalues_matched_tuples.append((wd_item, rel_datavalues_matched_tuples))

            # ------------------------------------------- wd_item_rel_datavalues_matched_tuples

            answers_from_wd_items = []  # answers_from_one_ent_link_query
            for wd_item, rel_datavalues_matched_tuples in wd_item_rel_datavalues_matched_tuples:

                answers_from_rels = []  # answers_from_one_wd_item
                for rel, datavalues in rel_datavalues_matched_tuples:

                    attribute_match_count += 1

                    answers_from_datavalues = []  # answers_from_one_rel

                    # predicted_answers
                    for dvalue_comp in preprocess_values(datavalues):
                        answers_from_one_datavalue = generate_answers_from_datavalue(dvalue_comp, q, dtext, qtext,
                                                                                     passage_data, q)
                        print('(WikiQA)', answers_from_one_datavalue)
                        # output answers
                        answers_from_datavalues.extend(answers_from_one_datavalue)


                    # print('answers_from_datavalues', answers_from_datavalues)
                    answers_from_rels.extend(answers_from_datavalues)
                # print('answers_from_rels', answers_from_rels)
                answers_from_wd_items.extend(answers_from_rels)
            # print('answers_from_wd_items', answers_from_wd_items)

            # ============================================================

            answers_from_ent_link_queries.extend(answers_from_wd_items)
            # print(f'answers_from_ent_link_queries {q["QID"]}: {answers_from_ent_link_queries}')

            def filter_answers(answers):
                # rule 5: filter out duplicate answers -> deprecated
                # return list(set(answers))
                # answer
                # return [for answer in answers]
                results = []
                for answer in answers:
                    if answer not in qtext:
                        results.append(answer)
                return results

            final_answers = list(filter_answers(answers_from_ent_link_queries))

            # transfer to FGC output api format

            if final_answers:
                q_anses = [{
                    'QID': q['QID'],
                    # 'QTEXT': qtext,  # for debugging
                    'AMODULE': 'Wiki-Kb-Inference',
                    'ATEXT': max(final_answers, key=len),
                    'score': 1.0,
                    'start_score': 0,
                    'end_score': 0,
                    # 'gold': answers  # for debugging
                }]
            # if q_anses:
                all_answers.append(q_anses)

        # endregion questions for-loop

        # df = pd.DataFrame(prediction_results)
        # if if_save_result:
        #     print('saving results ...')
        #     df.to_csv('result.csv')
        return all_answers

    # nlp.close()


def evaluate(dtext, predicted_ans, answers) -> str:
    prediction_result = 'wrong'
    # print('==========================================')
    # print('data value matched (predicted answer):', predicted_ans, 'gold:', answers)
    # highlight(dtext, predicted_ans, True)
    # evaluate performance
    # rule 3: datavalue(answer) matching passage text
    if predicted_ans in answers:
        # print('Exactly Match Gold Answer :)')
        prediction_result = 'exact_match'
    else:
        for answer in answers:
            if predicted_ans in answer:
                # print('Partially Match Gold Answer (substring of gold answer) :|')
                prediction_result = 'partial_match (substr)'
            elif answer in predicted_ans:
                # print('Partially Match Gold Answer (super-string of gold answer) :|')
                prediction_result = 'partial_match (superstr)'
            elif set(predicted_ans) <= set(answer) or set(predicted_ans) >= set(answer):
                # print('Partially Match Gold Answer (set) :|')
                prediction_result = 'partial_match (set)'
            else:
                pass
    return prediction_result


def save_results(prediction_result, ans_cand, answer_match_way):
    # q, wd_item, matched_attr_names, predicate_matched are global variables
    prediction_results.append(
        {'QID': q['QID'],
         'QTEXT': q['QTEXT_CN'],
         'AMODE': q['AMODE'],
         'ATYPE': q['ATYPE'],
         'ANS': [a['ATEXT_CN'] for a in q['ANSWER']],
         'prediction_result': prediction_result,
         'wd_item': wd_item['id'],
         'wd_item_name': get_fallback_zh_label_from_dict(
             wd_item),
         'predicate': attr,
         'predicted_ans': ans_cand,
         # 'predicate_matched': predicate_matched,
         'answer_match_way': answer_match_way
         })


if __name__ == '__main__':
    import json
    with open('FGC_release_all(cn).json', encoding='utf-8') as f:
        data = json.load(f)

    wiki_qa = WikiQA(server='http://140.109.19.191:9000')

    # all_answers = []
    # for item in data:
    #     all_answers.append(wiki_qa.predict(item, if_save_result=False, use_fgc_kb=True))
    # print(all_answers)

    # use data[0] to just answer the first two passages for the pilot run
    answers = wiki_qa.predict(data[3], save_result=True, use_fgc_kb=False)
    # for datum in data:
    #     answers = wiki_qa.predict(datum, save_result=True, use_fgc_kb=False)
        # break
