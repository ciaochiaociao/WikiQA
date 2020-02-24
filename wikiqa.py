#  Copyright (c) 2020. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
from typing import List, Dict

from ansi.colour import fg
from google.protobuf.pyext._message import SetAllowOversizeProtos
from os.path import join, abspath, dirname

from stanfordnlp.server import CoreNLPClient

from stanfordnlp_utils import snp_pprint
from wikidata4fgc_v2 import get_fallback_zh_label_from_dict
from wikidata_utils import traverse_by_attr_name, postprocess_datavalue
from config import DEFAULT_CORENLP_IP, FGC_KB_PATH, UNKNOWN_MESSAGE

from entity_linking import build_candidates_to_EL, entity_linking
from predicate_inference_rules import parse_question_by_regex
from value2ans import gen_anses_from_postprocessed_value, remove_duplicates, longest_answer

# fix bug: google.protobuf.message.DecodeError: Error parsing message
# ref: https://github.com/stanfordnlp/stanfordnlp/issues/154
SetAllowOversizeProtos(True)

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

    def predict_on_qs_of_one_doc(self, fgc_data, save_result=False, use_fgc_kb=True):
        """
        :param Dict fgc_data: fgc data at the level of document, i.e., one document with multiple questions
        """
        global nlp
        with CoreNLPClient(endpoint=self.corenlp_ip, annotators="tokenize,ssplit,lemma,pos,ner",
                            start_server=False, properties='chinese') as nlp:
            return self._predict_on_qs_of_one_doc(fgc_data, save_result, use_fgc_kb)

    def _get_from_fgc_kb(self, qtext):
        if qtext in self.kbqa_sheet:
            return self.kbqa_sheet[qtext]

    def _predict_on_qs_of_one_doc(self, fgc_data: dict, if_save_result: bool, use_fgc_kb: bool) -> List[List[Dict]]:
        # global q_dict, wd_item, rel, attr, predicate_matched, question_ie_data, passage_ie_data, mentions_bracketed, \
        #     dtext, answers, if_evaluate, qtext, debug_info

        # if the gold answer provided for checking performance
        if_evaluate = self.if_evaluate

        # for debugging
        did_shown = []

        # variable for others to access
        dtext = fgc_data['DTEXT_CN']

        # get IE data for passage
        passage_ie_data = nlp.annotate(dtext, properties={'pipelineLanguage': 'zh'})

        # output answers
        all_answers = []

        # region questions for-loop [{'QID': ...}, {'QID': ...}, ...]
        for q_dict in fgc_data['QUESTIONS']:

            # FGC KB runs first if used
            if use_fgc_kb:
                matched = self._get_from_fgc_kb(q_dict['QTEXT_CN'])
                if matched is not None:
                    # q_anses = default_answer(q_dict['DID'], 'Wiki-Kb-Inference', _match(q_dict['ATEXT']), 1.0)
                    q_anses = [{
                        # 'QID': q_dict['QID'],
                        'AMODULE': 'Wiki-Json-Inference',
                        'ATEXT': matched,
                        'score': 1.0
                    }]
                    all_answers.append(q_anses)
                    continue

            # filter out unwanted mode
            if q_dict['AMODE'] in ['Yes-No', 'Comparing-Members', 'Kinship']:
                continue
            # if q_dict['ATYPE'] in ['Object']:
            #     continue

            # for evaluation
            if if_evaluate:
                answers = [a['ATEXT_CN'] for a in q_dict['ANSWER']]
            else:
                answers = UNKNOWN_MESSAGE

            # variables
            qtext = q_dict['QTEXT_CN']

            # get IE data for question
            # note: don't add ssplit because there is a bug in CoreNLP: the NERMention tokenStartInSentenceInclusive
            # uses the location index in the original text without sentence split rather than with split even with
            # ssplit annotator on
            question_ie_data = nlp.annotate(qtext,
                                         properties={'pipelineLanguage': 'zh', 'annotators': "tokenize,lemma,pos,ner"})

            # for debugging
            if fgc_data['DID'] not in did_shown:
                did_shown.append(fgc_data['DID'])
                print('{} (tokenized, hans):'.format(fgc_data['DID']))
                for sent in passage_ie_data.sentence:
                    print(f'(sent{sent.sentenceIndex})', end=' ')
                    snp_pprint(sent)
                print('\n')
            print('{} {} {}:'.format(q_dict['QID'], fg.brightgray(q_dict['AMODE']), fg.brightgray(q_dict['ATYPE'])), end=' ')
            if len(question_ie_data.sentence) > 1:
                print('[WARN] question split into two sentences during IE!')
            snp_pprint(question_ie_data.sentence[0], end='')
            print('(Gold)', answers)

            final_answers = self.predict(qtext, q_dict, question_ie_data, dtext, passage_ie_data)

            # ===== FINISHING STEP =====
            # transfer to FGC output api format

            if final_answers:
                q_anses = [{
                    'QID': q_dict['QID'],  # for debugging
                    'QTEXT': qtext,  # for debugging
                    'AMODULE': 'Wiki-Kb-Inference',
                    'ATEXT': final_answers,
                    'score': 1.0,
                    'start_score': 0,
                    'end_score': 0,
                    'gold': answers  # for debugging
                }]
                all_answers.append(q_anses)

        # endregion questions for-loop

        # df = pd.DataFrame(prediction_results)
        # if if_save_result:
        #     print('saving results ...')
        #     df.to_csv('result.csv')
        return all_answers

    def predict(self, qtext, q_dict, question_ie_data, dtext, passage_ie_data):
        # ===== STEP A. parse question (parse entity name + predicate inference) =====
        parsed_result = parse_question_by_regex(qtext)
        if parsed_result:
            name, attr, span, matched_pattern = parsed_result
            print('(ParseQ)', name, '|', attr)
        else:  # skip this question if not matched by our rules/regex
            return
        # ===== STEP B. entity linking =====
        ent_link_cands = build_candidates_to_EL(name, question_ie_data, span)
        wd_items = entity_linking(ent_link_cands)
        print('(EL)', [(get_fallback_zh_label_from_dict(i), i['id']) for i in wd_items])
        # ===== STEP C. traverse Wikidata =====
        datavalues = []
        for wd_item in wd_items:
            values = traverse_by_attr_name(wd_item, attr)
            if values:
                datavalues.extend(values)
        try:  # item
            print('(Traverse)', [d['value'] + ' ' + d['all_aliases'][0][0] for d in datavalues])
        except KeyError:  # date/time
            print('(Traverse)', [d['value'] for d in datavalues])
        except TypeError:  # string
            print('(Traverse)', datavalues)

        # ===== STEP D. Post Processing datavalues =====
        processed_datavalues = []
        for value in datavalues:
            processed_list: List[str] = postprocess_datavalue(value)
            processed_datavalues.extend(processed_list)
        print('(Post-Proc)', processed_datavalues)

        # ===== STEP E. Coordinating with Passage =====
        final_answers = []  # answers_from_one_ent_link_query
        for dvalue_comp in processed_datavalues:
            answers_from_one_datavalue = gen_anses_from_postprocessed_value(dvalue_comp, qtext, q_dict['ATYPE'], dtext,
                                                                            passage_ie_data)
            final_answers.extend(answers_from_one_datavalue)
        final_answers = list(remove_duplicates(final_answers, qtext))
        print('(VALID)', final_answers)

        # ===== STEP F. Final Decision =====
        final_answer = longest_answer(final_answers)
        print('(WikiQA)', final_answer)
        return final_answer


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


# def save_results(prediction_result, ans_cand, answer_match_way):
#     # q_dict, wd_item, matched_attr_names, predicate_matched are global variables
#     prediction_results.append(
#         {'QID': q_dict['QID'],
#          'QTEXT': q_dict['QTEXT_CN'],
#          'AMODE': q_dict['AMODE'],
#          'ATYPE': q_dict['ATYPE'],
#          'ANS': [a['ATEXT_CN'] for a in q_dict['ANSWER']],
#          'prediction_result': prediction_result,
#          'wd_item': wd_item['id'],
#          'wd_item_name': get_fallback_zh_label_from_dict(
#              wd_item),
#          'predicate': attr,
#          'predicted_ans': ans_cand,
#          # 'predicate_matched': predicate_matched,
#          'answer_match_way': answer_match_way
#          })


if __name__ == '__main__':
    import json
    with open('FGC_release_all(cn).json', encoding='utf-8') as f:
        docs = json.load(f)

    wiki_qa = WikiQA(server='http://140.109.19.191:9000')

    # all_answers = []
    # for item in data:
    #     all_answers.append(wiki_qa.predict_on_qs_of_one_doc(item, if_save_result=False, use_fgc_kb=True))
    # print(all_answers)

    # use data[0] to just answer the first two passages for the pilot run
    # print(wiki_qa.predict_on_qs_of_one_doc(get_doc_with_one_que('D002Q02', docs), save_result=True, use_fgc_kb=False))
    # docs = remove_docs_before_did('D274', docs)
    for doc in docs:
        answers = wiki_qa.predict_on_qs_of_one_doc(doc, save_result=True, use_fgc_kb=False)
        # break
