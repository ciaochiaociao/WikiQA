#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
from typing import List, Dict, TextIO, Match
import json

from ansi.colour import fg
from google.protobuf.pyext._message import SetAllowOversizeProtos
from os.path import abspath, dirname

from stanfordnlp.server import CoreNLPClient

from ..utils.fgc_utils import get_doc_with_one_que
from .predicate_inference_neural import parse_question_w_neural
from ..utils.stanfordnlp_utils import snp_pstr, snp_get_ents_by_overlapping_char_span_in_doc
from ..utils.wikidata4fgc_v2 import get_fallback_zh_label_from_dict
from ..utils.wikidata_utils import traverse_by_attr_name, postprocess_datavalue
from ..config import DEFAULT_CORENLP_IP, FGC_KB_PATH

from .entity_linking import build_candidates_to_EL, entity_linking, _get_text_from_token_entity_comp
from .predicate_inference_rules import parse_question_by_regex
from .value2ans import remove_duplicates, longest_answer, match_with_psg, match_type

# fix bug: google.protobuf.message.DecodeError: Error parsing message
# ref: https://github.com/stanfordnlp/stanfordnlp/issues/154
SetAllowOversizeProtos(True)

UNKNOWN_MESSAGE = 'Unknown in evaluation mode (if_evaluate=True)'

# global variables
prediction_results = []


class WikiQA:
    def __init__(self, server=DEFAULT_CORENLP_IP):
        self.corenlp_ip = server  # 'http://localhost:9000'
        self.if_evaluate = True  # evaluate the performance

        # FGC KB
        with open(FGC_KB_PATH, 'r', encoding='utf-8') as f:
            self.kbqa_sheet = json.load(f)

    def predict_on_docs(self, docs, file4eval_fpath, neural_pred_infer, use_fgc_kb):
        all_answers = []
        with open(file4eval_fpath, 'w', encoding='utf-8') as file4eval:
            print('qid\tparsed_subj\tparsed_pred\tsid\tpretty_values\tproc_values\tanswers\tanswer', file=file4eval)
            for doc in docs:
                answers = self.predict_on_qs_of_one_doc(doc, use_fgc_kb=use_fgc_kb, file4eval=file4eval,
                                                           neural_pred_infer=neural_pred_infer)
                all_answers.extend(answers)
                print(answers)
        print(all_answers)

    def predict_on_qs_of_one_doc(self, fgc_data, use_fgc_kb: bool = True, file4eval: TextIO = None,
                                 neural_pred_infer: bool =False) -> List[List[Dict]]:
        """
        :param bool neural_pred_infer: if using neural model to inference predicate (e.g. PID in wikidata)
        :param Dict fgc_data: fgc data at the level of document, i.e., one document with multiple questions
        :param bool use_fgc_kb: if using fgc kb to answer before the wiki-based QA module
        :param TextIO file4eval: the file to save stage-by-stage results for evaluation
        :return List[List[Dict]]: answers of all questions in this document (an answer is in the format of a `dict`
        """
        global nlp
        with CoreNLPClient(endpoint=self.corenlp_ip, annotators="tokenize,ssplit,lemma,pos,ner",
                           start_server=False) as nlp:
            return self._predict_on_qs_of_one_doc(fgc_data, use_fgc_kb, file4eval, neural_pred_infer)

    def _get_from_fgc_kb(self, qtext):
        if qtext in self.kbqa_sheet:
            return self.kbqa_sheet[qtext]

    def _predict_on_qs_of_one_doc(self, fgc_data: dict, use_fgc_kb: bool, file4eval, neural_pred_infer) -> List[List[Dict]]:
        # global q_dict, wd_item, rel, attr, predicate_matched, question_ie_data, passage_ie_data, mentions_bracketed, \
        #     dtext, answers, if_evaluate, qtext, debug_info

        # if the gold answer provided for checking performance
        if_evaluate = self.if_evaluate

        # for debugging
        did_shown = []

        # variable for others to access
        dtext = fgc_data['DTEXT_CN']

        # get IE data for passage
        passage_ie_data = nlp.annotate(dtext, properties={'pipelineLanguage': 'zh',
                                                          'ssplit.boundaryTokenRegex': '[。]|[!?！？]+'})

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
            if q_dict['AMODE'] in ['Yes-No', 'Comparing-Members', 'Kinship', 'Arithmetic-Operations',  'Multi-Spans-Extraction', 'Counting']:
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
            # TODO: don't add ssplit because there is a bug in CoreNLP: the NERMention tokenStartInSentenceInclusive
            # uses the location index in the original text without sentence split rather than with split even with
            # ssplit annotator on
            question_ie_data = nlp.annotate(qtext,
                                            properties={'ssplit.boundaryTokenRegex': '[。]|[!?！？]+',
                                                        'pipelineLanguage': 'zh'})

            # for debugging
            if fgc_data['DID'] not in did_shown:
                did_shown.append(fgc_data['DID'])
                print('{} (tokenized, hans):'.format(fgc_data['DID']))
                for sent in passage_ie_data.sentence:
                    print(f'(sent{sent.sentenceIndex})', end=' ')
                    print(snp_pstr(sent))
                print('\n')
            print('{} {} {}:'.format(q_dict['QID'], fg.brightgray(q_dict['AMODE']), fg.brightgray(q_dict['ATYPE'])), end=' ')
            if len(question_ie_data.sentence) > 1:
                print('[WARN] question split into two sentences during IE!')
                for sent in question_ie_data.sentence:
                    print(snp_pstr(sent), end='|')
            else:
                print(snp_pstr(question_ie_data.sentence[0]), end='')
            print('(Gold)', answers)

            final_answers = self.predict(qtext, q_dict, question_ie_data, dtext, passage_ie_data, file4eval,
                                         neural_pred_infer)

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
    def predict(self, qtext, q_dict, question_ie_data, dtext, passage_ie_data, file4eval, neural_pred_infer):
        # ===== STEP A. parse question (parse entity name + predicate inference) =====
        if neural_pred_infer:
            parsed_result = parse_question_w_neural(qtext)
        else:
            parsed_result = parse_question_by_regex(qtext)
        if parsed_result:
            name, attr, span = parsed_result
            print('(ParseQ)', name, '|', attr)
        else:  # skip this question if not matched by our rules/regex
            print(q_dict['QID'] + '\tnot_parsed\tnot_parsed\t\t\t\t\t', file=file4eval)
            return
        # ===== STEP B. entity linking =====
        ent_link_cands = build_candidates_to_EL(name, question_ie_data, span, attr)
        print('(EL Queries)', [_get_text_from_token_entity_comp(cand) for cand in ent_link_cands])
        wd_items = entity_linking(ent_link_cands, attr)
        print('(EL)', [(get_fallback_zh_label_from_dict(i), i['id']) for i in wd_items])
        # ===== STEP C. traverse Wikidata =====
        datavalues, traversed_items = [], []
        for wd_item in wd_items:
            values = traverse_by_attr_name(wd_item, attr)
            if values:
                datavalues.extend(values)
                traversed_items.append(wd_item)

        def _pretty_datavalues(datavalues):
            try:  # item
                return [d['value'] + ' ' + d['all_aliases'][0][0] for d in datavalues]
            except KeyError:  # date/time
                return [d['value'] for d in datavalues]
            except TypeError:  # string
                return datavalues

        print('(Traverse) {} from {}'.format(_pretty_datavalues(datavalues), [i['id'] for i in traversed_items]))

        # ===== STEP D. Post Processing datavalues =====
        processed_datavalues = []
        for value in datavalues:
            processed_list: List[str] = postprocess_datavalue(value, attr)
            processed_datavalues.extend(processed_list)
        print('(Post-Proc)', processed_datavalues)

        # ===== STEP E. Coordinating with Passage =====
        answers = []
        # rule 3-1: if fuzzy matching datavalue (answer) with passage text (edit distance <= 1) -> generate answer
        print('(Match)', end=' ')
        matches = []
        for dvalue_str in processed_datavalues:
            ms: List[Match] = match_with_psg(dvalue_str, dtext, fuzzy=False)
            matches.extend(ms)
            print(bool(ms), end=' ')
        print()

        # rule: Add matched text to answers (not matcher)
        answers.extend([match.group(0) for match in matches])
        print('(Answers 1)', answers)

        # rule: find the matched NE in the passage
        mentions = []
        if matches:
            for match in matches:
                mentions.extend(list(snp_get_ents_by_overlapping_char_span_in_doc(match.span(0), passage_ie_data)))
        print('(NE)', [mention.entityMentionText for mention in mentions])

        # rule: match ans type and mention type
        mentions = [mention for mention in mentions if match_type(mention, q_dict['ATYPE'], qtext)]
        print('(NE-ANS Type Match)', [mention.entityMentionText for mention in mentions])

        # rule: Use matched NE (expansion/diminishing) from psg as answers
        if attr not in ['朝代', '出生年份', '死亡年份']:
            answers.extend([mention.entityMentionText for mention in mentions])
        print('(Answers 2)', answers)

        final_answers = list(remove_duplicates(answers, qtext))
        print('(VALID)', final_answers)

        # ===== STEP F. Final Decision =====
        final_answer = longest_answer(final_answers)
        print('(WikiQA)', final_answer)

        # output stage-by-stage results for evaluation
        if file4eval:
            print(q_dict['QID'],
                  name,
                  attr,
                  [i['id'] for i in wd_items],
                  _pretty_datavalues(datavalues),
                  processed_datavalues,
                  final_answers,
                  final_answer, sep='\t', file=file4eval, flush=True)

        return final_answer


if __name__ == '__main__':
    with open('data/raw/FGC_release_all(cn).json', encoding='utf-8') as f:
        docs = json.load(f)

    wiki_qa = WikiQA(server='http://140.109.19.191:9000')

    # all_answers = []
    # for item in data:
    #     all_answers.append(wiki_qa.predict_on_qs_of_one_doc(item, if_save_result=False, use_fgc_kb=True))
    # print(all_answers)

    # use data[0] to just answer the first two passages for the pilot run
    print(wiki_qa.predict_on_qs_of_one_doc(get_doc_with_one_que('D302Q01', docs), use_fgc_kb=False))
    # docs = remove_docs_before_did('D274', docs)
    # for doc in docs:
    #     answers = wiki_qa.predict_on_qs_of_one_doc(doc, use_fgc_kb=False)
        # break
