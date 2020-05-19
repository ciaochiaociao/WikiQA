#  Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
import copy
import logging
import sys
from copy import deepcopy
from functools import partial
from typing import List, Dict, TextIO, Match, Tuple, Union
import json

from ansi.colour import fg
from os.path import abspath, dirname

from stanfordnlp.server import CoreNLPClient
# Below line should only be called after stanfordnlp to prevent seg fault
from google.protobuf.pyext._message import SetAllowOversizeProtos
# fix bug: google.protobuf.message.DecodeError: Error parsing message
# ref: https://github.com/stanfordnlp/stanfordnlp/issues/154
from ..utils.utils import TeeLogger

SetAllowOversizeProtos(True)

from ..utils.fgc_utils import get_doc_with_one_que
from .predicate_inference_neural import parse_question_w_neural
from ..utils.stanfordnlp_utils import snp_pstr, snp_get_ents_by_overlapping_char_span_in_doc
from ..utils.wikidata_utils import get_fallback_zh_label_from_dict
from ..utils.wikidata4fgc import traverse_by_attr_name, postprocess_datavalue
from ..config import DEFAULT_CORENLP_IP, FGC_KB_PATH

from .entity_linking import build_candidates_to_EL, entity_linking, _get_text_from_token_entity_comp
from .predicate_inference_rules import parse_question_by_regex
from .value2ans import remove_duplicates, longest_answer, match_with_psg, match_type

UNKNOWN_MESSAGE = 'Unknown in evaluation mode (if_evaluate=False)'


def is_span(span):
    return len(span) == 2 and span[1] >= span[0] and isinstance(span[0], int) and isinstance(span[1], int)


def in_span(span, limit_span):
    assert is_span(span)
    assert is_span(limit_span)
    return span[0] >= limit_span[0] and span[1] <= limit_span[1]


def in_spans(span, spans):
    return any([in_span(span, limit_span) for limit_span in spans])


def filter_psg_matches_w_shints(matches: List[Match], shint_spans: List[Tuple[int]]):
    """
    >>> filter_psg_matches_w_shints([(1, 3), (5, 7), (8, 10)], [(1, 2), (4, 7), (8, 10)])
    [(5, 7), (8, 10)]
    """
    filtered_matches = []
    for m in matches:
        if in_spans(m.span(), shint_spans):
            filtered_matches.append(m)

    return filtered_matches


def get_topn_amode_dicts(from_amode, amode_topn) -> dict:
    if isinstance(from_amode, dict):
        from_amode = sorted(from_amode.items(), key=lambda x: x[1]['score'], reverse=True)[:amode_topn]
    elif isinstance(from_amode, list):
        pass
    else:
        raise TypeError
    return dict(from_amode)


def get_topn_atype_dicts(from_atype, atype_topn):
    if isinstance(from_atype, str):
        atypes = [from_atype]
    elif isinstance(from_atype, dict):
        atypes = sorted(from_atype.items(), key=lambda x: x[1], reverse=True)[:atype_topn]
    else:
        raise TypeError
    return dict(atypes)


def filter_ans_for_attrs(processed_datavalues, attr, qtext):
    if attr == '名字':
        def _ans_not_in_q(value, qtext):
            return value not in qtext

        _ans_not_in_q = partial(_ans_not_in_q, qtext=qtext)
        processed_datavalues = list(filter(_ans_not_in_q, processed_datavalues))
    elif attr == '朝代':  # TODO: Check with Wikidata item instead by using `instanceOf` predicate
        f = lambda value: not value.endswith('国')
        processed_datavalues = list(filter(f, processed_datavalues))
    return processed_datavalues


class WikiQA:
    def __init__(self, server=DEFAULT_CORENLP_IP, if_evaluate=False):
        self.corenlp_ip = server  # 'http://localhost:9000'
        self.if_evaluate = if_evaluate  # evaluate the performance

        # FGC KB
        with open(FGC_KB_PATH, 'r', encoding='utf-8') as f:
            self.kbqa_sheet = json.load(f)

    def predict_on_docs(self, docs, file4eval_fpath, **kwargs):
        all_answers = []
        with open(file4eval_fpath, 'w', encoding='utf-8') as file4eval:
            print('qid\tparsed_subj\tparsed_pred\tsid\tpretty_values\tproc_values\tanswers\tanswer', file=file4eval)
            for doc in docs:
                answers = self.predict_on_qs_of_one_doc(doc, file4eval, **kwargs)
                all_answers.extend(answers)
                print(answers)
        print(all_answers)

    def predict_on_qs_of_one_doc(self, fgc_data) -> List[List[Dict]]:
        """
        :param Dict fgc_data: fgc data at the level of document, i.e., one document with multiple questions
        :return List[List[Dict]]: answers of all questions in this document (an answer is in the format of a `dict`)
        """
        with self:
            return self._predict_on_qs_of_one_doc(fgc_data)

    def _get_from_fgc_kb(self, qtext):
        if qtext in self.kbqa_sheet:
            return self.kbqa_sheet[qtext]

    def _predict_on_qs_of_one_doc(self, fgc_data: dict, file4eval, use_fgc_kb: bool = True, amode_topn=1, atype_topn=1,
                                  **kwargs) -> List[List[Dict]]:
        # global q_dict, wd_item, rel, attr, predicate_matched, question_ie_data, passage_ie_data, mentions_bracketed, \
        #     dtext, answers, if_evaluate, qtext, debug_info

        # if the gold answer provided for checking performance
        if_evaluate = self.if_evaluate

        # for debugging
        did_shown = []

        dtext = fgc_data[self.dtext_attr]
        # get IE data for passage
        passage_ie_data = self.nlp.annotate(dtext, properties={'pipelineLanguage': 'zh',
                                                          'ssplit.boundaryTokenRegex': '[。]|[!?！？]+'})

        # output answers
        all_answers = []

        # region questions for-loop [{'QID': ...}, {'QID': ...}, ...]
        for q_dict in fgc_data['QUESTIONS']:

            # FGC KB runs first if used
            if self.config.use_fgc_kb:
                matched = self._get_from_fgc_kb(q_dict[self.qtext_attr])
                if matched is not None:
                    # q_anses = default_answer(q_dict['DID'], 'Wiki-Kb-Inference', _match(q_dict['ATEXT']), 1.0)
                    ans_dict = {
                        # 'QID': q_dict['QID'],
                        'AMODULE': 'Wiki-Json-Inference',
                        'ATEXT': matched,
                        'score': 1.0,
                        'start_score': 0.0,
                        'end_score': 0.0,
                    }
                    all_answers.append(ans_dict)
                    if self.file4eval:
                        print(f"{q_dict['QID']}\tmatched_by_json\tmatched_by_json\t\t\t\t\t{matched}",
                              file=self.file4eval, flush=True)
                    continue

            from_amode: Union[List[str], dict] = q_dict['AMODE']
            from_atype: Union[str, dict] = q_dict['ATYPE']
            if isinstance(from_amode, dict):  # pred
                def _filter_simplify_dict(dict_with_score: dict):
                    return {k: v['score'] for k, v in dict_with_score.items()}

                amode_dict = get_topn_amode_dicts(from_amode, amode_topn)
                amode_dict = _filter_simplify_dict(amode_dict)
            elif isinstance(from_amode, list):  # gold
                amode_dict = {mode: 1 for mode in from_amode}
            else:
                raise TypeError
            if isinstance(from_atype, dict):  # pred
                atype_dict = get_topn_atype_dicts(from_atype, atype_topn)
            elif isinstance(from_atype, str):  # gold
                atype_dict = {from_atype: 1}
            else:
                raise TypeError
            # rule: filter out unwanted mode
            # rule: filter out unwanted qtype
            if q_dict["QTYPE"] == '申論' or \
                    set(amode_dict) & set(['Yes-No', 'Kinship', 'Multi-Spans-Extraction']):
                ans_dict = {
                    'AMODULE': 'Wiki-Kb-Inference',
                    'ATEXT': '',
                    'score': 0.0,
                    'start_score': 0.0,
                    'end_score': 0.0
                }
                all_answers.append(ans_dict)
                if self.file4eval:
                    print(q_dict['QID'] + '\tskipped\tskipped\t\t\t\t\t', file=self.file4eval, flush=True)
                continue
            # for evaluation
            if if_evaluate:
                try:
                    answers = [a[self.atext_attr] for a in q_dict['ANSWER']]
                except KeyError:
                    answers = f'NO ANSWER FOR {q_dict["QID"], q_dict["QTYPE"]}'
            else:
                answers = UNKNOWN_MESSAGE

            # variables
            qtext = q_dict[self.qtext_attr]

            # get IE data for question
            # don't add ssplit because there is a bug in CoreNLP: the NERMention tokenStartInSentenceInclusive
            # uses the location index in the original text without sentence split rather than with split even with
            # ssplit annotator on
            question_ie_data = self.nlp.annotate(qtext,
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
            print('{} {} {}:'.format(q_dict['QID'], fg.brightgray('/'.join(amode_dict)), fg.brightgray('/'.join(atype_dict))), end=' ')
            if len(question_ie_data.sentence) > 1:
                print('[WARN] question split into two sentences during IE!')
                for sent in question_ie_data.sentence:
                    print(snp_pstr(sent), end='|')
            else:
                print(snp_pstr(question_ie_data.sentence[0]), end='')
            print('(Gold)', answers)

            final_answer = self.predict(qtext, q_dict, question_ie_data, dtext, passage_ie_data, file4eval,
                                        fgc_data['SENTS'], atype_dict, **kwargs)

            # ===== FINISHING STEP =====
            # transfer to FGC output api format

            if final_answer:
                ans_dict = {
                    # 'QID': q_dict['QID'],  # for debugging
                    # 'QTEXT': qtext,  # for debugging
                    'AMODULE': 'Wiki-Kb-Inference',
                    'ATEXT': final_answer,
                    'score': 1.0,
                    'start_score': 0.0,
                    'end_score': 0.0,
                    # 'gold': answers  # for debugging
                }
            else:
                ans_dict = {
                    'AMODULE': 'Wiki-Kb-Inference',
                    'ATEXT': '',
                    'score': 0.0,
                    'start_score': 0.0,
                    'end_score': 0.0
                }

            all_answers.append(ans_dict)

        return all_answers

    def predict(self, qtext, q_dict, question_ie_data, dtext, passage_ie_data, file4eval, psg_sents, atype_dict: dict,
                neural_pred_infer=False, use_se='pred'):
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

        # ===== STEP D-E. Extra filtering rules =====
        processed_datavalues = filter_ans_for_attrs(processed_datavalues, attr, qtext)

        # ===== STEP E. Coordinating with Passage =====
        answers = []
        # rule 3-1: if fuzzy matching datavalue (answer) with passage text (edit distance <= 1)
        print('(Match)', end=' ')
        matches = []
        for dvalue_str in processed_datavalues:
            ms: List[Match] = match_with_psg(dvalue_str, dtext, fuzzy=False)
            matches.extend(ms)
            print(bool(ms), end=' ')
        print()
        print('(Passage Match)', [match.group(0) for match in matches])

        # rule: filter out the matched span not in supporting evidences (Updated on 1.1)
        print('(Passage Match)', [match.group(0) for match in matches])
        if use_se != 'None':
            if use_se == 'gold':
                shint_sids = q_dict['SHINT']
            elif use_se == 'pred':
                shint_sids = q_dict['SHINT'][0]
            elif use_se == 'pred_old':
                shint_sids = q_dict['sp']
            else:
                raise ValueError
            assert isinstance(shint_sids, list)
            if len(shint_sids):
                assert isinstance(shint_sids[0], int)
            print('(INFO) Supporting Evidence', use_se, 'is used')
            shint_spans = [(psg_sents[sid]['start'], psg_sents[sid]['end']) for sid in shint_sids]
            matches = filter_psg_matches_w_shints(matches, shint_spans)

        # rule: Add matched text to answers (not matcher)
        answers.extend([match.group(0) for match in matches])
        print('(Answers 1) (SE Match)', answers)

        # rule: find the matched NE in the passage
        mentions = []
        if matches:
            for match in matches:
                mentions.extend(list(snp_get_ents_by_overlapping_char_span_in_doc(match.span(0), passage_ie_data)))
        print('(NE Exp/Dim)', [mention.entityMentionText for mention in mentions])

        # rule: match ans type and mention type excluding some predicates
        # if attr not in ['出生年份', '死亡年份', '成立或建立年份']:
        mentions = [mention for mention in mentions if match_type(mention, atype_dict.keys(), qtext)]
        print('(NE-ANS Type Match)', [mention.entityMentionText for mention in mentions])

        # rule: Use matched NE (expansion/diminishing) from psg as answers
        if attr not in ['朝代', '出生年份', '死亡年份', '成立或建立年份']:
            answers.extend([mention.entityMentionText for mention in mentions])
        else:
            print(f'[INFO] NEs are not added to answers for {attr}')

        print('(Answers 2)', answers)

        # rule: add traversed values for '寿命' (no need to match with passage)
        if attr in ['寿命']:
            answers.extend(processed_datavalues)

        # clean with removing duplicates
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
    import doctest
    doctest.testmod(verbose=True)