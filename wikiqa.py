from functools import reduce
from typing import Tuple, Union

import logging
import regex
from stanfordnlp.protobuf import NERMention, Token

from utils import *
from wikidata4fgc_v2 import *
from stanfordnlp.server import CoreNLPClient
import json
import re
from pandas import pandas as pd
from os.path import join, abspath, dirname

# fix bug: google.protobuf.message.DecodeError: Error parsing message
# ref: https://github.com/stanfordnlp/stanfordnlp/issues/154
from google.protobuf.pyext._message import SetAllowOversizeProtos
SetAllowOversizeProtos(True)
UNKNOWN_MESSAGE = 'Unknown in evaluation mode (if_evaluate=True)'
# rule 2-1-0: verbs/nouns/others
birth = '(出生|诞生)'
birth_all = f'({birth}|生)'
death = f'(死亡|死掉|过世|逝世|去世|离世|离开人间)'
death_sim = f'死'
death_all = f'({death}|{death_sim})'
_in = '(在|于)'

# rule 2-1-1: Place/Location
place = '(地区|地方|地点|省份|省|城市)'
place_sim = '(地|省)'
place_all = f'({place}|{place_sim})'
country = '(国家|国)'

# rule 2-1-2: what/which
which = '哪(一个|个)?'
which_sim = '(哪|哪一)'
which_all = f'({which}|{which_sim})'
which_place_sim = f'{which_sim}{place_sim}'
whichplace = f'{which}{place}'
whichplace_all = f'({whichplace}|{which_place_sim})'
what = '(甚么|什么)'
what_which = f'({what}|{which_all})'
what_sim = '(啥|何|哪)'

# rule 2-1-3: wh-word for place (where, what place)
whatplace = f'{what}|{place}'
where = '哪(里)'
where_all = f'({whichplace_all}|{whatplace}|{where})'

# rule 2-1-4: date/time
date = '(日期|日子|天)'
date_sim = '日'
month = '月份'
month_sim = '月'
year = '年'
time = '(时后|时候|时间)'
time_sim = '时'
start = '(开工|建立|成立|签署|生效|开始|始于)'

# rule 2-1-5: wh-word for time/date (what time, when)
when = f'({what_which}({time}|{year}|{month}|{date})|{what_which}({time_sim}|{year}|{month_sim}|{date_sim}))'
human = '(人|人物)'
who = f'((({what_which}|{what_sim}).{0, 5}{human})|谁)'
found = f'(创立|创办|开始|创建|发起|创业|建立)'


# global variables
cc = OpenCC('t2s')
prediction_results = []


class WikiQA:
    def __init__(self, server='http://140.109.19.191:9000'):
        # 'http://localhost:9000'
        self.corenlp_ip = server
        self.if_evaluate = False  # evaluate the performance

        cur_path = dirname(abspath(__file__))
        with open(join(cur_path, 'fgc_knowledgebase.json'), 'r', encoding='utf-8') as f:
            self.kbqa_sheet = json.load(f)

    def predict(self, fgc_data, save_result=False, fgc_kb=True):
        global nlp
        with CoreNLPClient(endpoint=self.corenlp_ip, annotators="tokenize,ssplit,lemma,pos,ner",
                            start_server=False, properties='chinese') as nlp:
            return self._predict(fgc_data, save_result, fgc_kb)

    def _get_from_fgc_kb(self, qtext):
        if qtext in self.kbqa_sheet:
            return self.kbqa_sheet[qtext]

    def _predict(self, item, save_result, fgc_kb):
        global q, wd_item, rel, attr, predicate_matched, nlp_data, passage_data, mentions_bracketed, \
            dtext, answers, if_evaluate, qtext, ent_link_query, debug_info

        # if the gold answer provided for checking performance
        if_evaluate = self.if_evaluate

        # for debugging
        attribute_match_count = 0
        debug_infos = []
        did_shown = []

        dtext = item['DTEXT']
        passage_data = nlp.annotate(dtext, properties={'pipelineLanguage': 'zh'})

        all_answers = []
        # region questions for-loop [{'QID': ...}, {'QID': ...}, ...]
        for q in item['QUESTIONS']:

            if fgc_kb:
                matched = self._get_from_fgc_kb(q['QTEXT'])
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

            if q['AMODE'] == 'Yes-No':
                continue
            if q['ATYPE'] in ['Object']:
                continue

            # for debugging
            if item['DID'] not in did_shown:
                # print('did:', item['DID'])
                # print('original text (hant):', item['DTEXT'])
                # print('tokenized passage (hans):')
                did_shown.append(item['DID'])

            # for sent in passage_data.sentence:
            #     print(f'sent{sent.sentenceIndex}:', end=' ')
            #     for tok in sent.token:
            #         print(tok.originalText, end=' ')
            # print('\n')
            # debug run
            # if q['QID'] not in testqids:
            #     continue
            if if_evaluate:
                answers = [a['ATEXT_CN'] for a in q['ANSWER']]
            else:
                answers = UNKNOWN_MESSAGE
            # print(q['QID'])
            qtext = q['QTEXT_CN']
            # props={'annotators': 'tokenize,ssplit,pos,ner','pipelineLanguage':'zh','outputFormat':'json'}

            nlp_data = nlp.annotate(qtext,
                                    properties={'pipelineLanguage': 'zh', 'annotators': "tokenize,lemma,pos,ner"})
            # TODO: BUGFIX: still doing ssplit!!!!!
            # note: don't add ssplit because there is a bug in CoreNLP: the NERMention tokenStartInSentenceInclusive
            # uses the location index in the original text without sentence split rather than with split even with
            # ssplit annotator on

            # print('tokenized_question:', *[token.originalText for sent in nlp_data.sentence for token in sent.token],
            #       sep=' ', end='')
            # print(answers)
            # mentions = [mention for mention in nlp_data.mentions if mention.entityType]
            # print('unfiltered', [mention.entityMentionText for mention in mentions])

            # questions without any mentions
            # if not len(mentions):
            #     print(text)
            #     input()

            # rule 1-1: filtering out 'NUMBER', 'DATE', 'ORDINAL', 'MONEY', 'TIME', 'PERCENT' Entity Mention
            # mentions = [mention for mention in mentions if
            #             mention.entityType not in ['NUMBER', 'DATE', 'ORDINAL', 'MONEY', 'TIME', 'PERCENT', 'EMAIL',
            #                                        'URL']]
            #
            # print('filtered mentions:', [mention.entityMentionText for mention in mentions])
            #
            # if not len(mentions):
            #     no_qs_wo_ems += 1

            # rule 1-2: add Proper Nouns to the entity linking query set
            # sid_and_tokens_nr = [(sid, tok) for sid, sent in enumerate(nlp_data.sentence) for tok in sent.token if tok.pos == 'NR']
            # rule 1-3: further find mentions bracketed (max length 9)
            # finditer() returns iterator of `MatchObject`s while findall() returns list of tuples
            # matches = list(re.finditer(r'[<‹«《〈〔「『【〖"](.{1,9})[>›»》〉〕」』】〗"]', qtext))
            # mentions_bracketed = [(match[1], match.span(1)) for match in matches]  # save string and (tok_b, tok_e)

            # rule 1-4: expand the query text set from the original query with multi-gram techniques
            # build queries for entity linking
            # ent_link_cands = [gram for mention in mentions for gram in
            #                   expand_with_multi_gram(mention, nlp_data.sentence, mention.sentenceIndex, n_grams=3) +
            #                   expand_with_multi_gram(mention, nlp_data.sentence, mention.sentenceIndex, n_grams=2)] + \
            #                  [gram for sid, token in sid_and_tokens_nr for gram in
            #                    expand_with_multi_gram(token, nlp_data.sentence, sid, n_grams=3) +
            #                     expand_with_multi_gram(token, nlp_data.sentence, sid, n_grams=2)] + \
            #                  mentions + mentions_bracketed

            # rule 1-5: followed by '的' + Noun => second hop on that Noun
            # ent_link_cands_2nd_hop = followed_by_2nd_hop_cands(ent_link_cands, nlp_data)

            answers_from_ent_link_queries = []  # answers_from_one_question

            shown = set()
            debug_info = {}
            debug_infos.append(debug_info)
            debug_info.update({'QID': q['QID'],
                               'QTEXT': qtext,
                               'ATYPE': q['ATYPE'],
                               'AMODE': q['AMODE'],
                               })

            parsed_result = parse_question(qtext)
            if parsed_result:
                name, attr, span, matched_pattern = parsed_result
                # print('[matched pattern]name, attr, span, matched_pattern:', name, attr, span, matched_pattern)
            else:
                continue
            mentions = list(get_ent_from_tok_b_e(nlp_data.mentions, span))
            ent_link_cands = [name]
            if '.' in name:
                ent_link_cands.append(''.join(name.split('.')))
            if '·' in name:
                ent_link_cands.append(''.join(name.split('·')))
            if mentions:
                ent_link_cands.extend(mentions)

            for ix, ent_link_cand in enumerate(ent_link_cands):
                ent_link_query = _get_text_from_token_entity_comp(ent_link_cand)
                if ent_link_query in shown:  # skip duplicate
                    continue
                shown.add(ent_link_query)
                # print(f'ent_link query {ix}: {ent_link_query}')
                # print()
                debug_info.update({'ent_link_query': ent_link_query})

                # ============================================================ ent_link_cand

                # -------------------------------------------
                # rule 1: Entity Linking - get item from Wikidata
                ent_link_query = _get_text_from_token_entity_comp(ent_link_cand)

                wd_items = get_dicts_from_keyword(ent_link_query)
                # clean wikidata item and simplify wikidata item
                wd_items = [readable(filter_claims_in_dict(d)) for d in wd_items]

                # rest_tokens = [token for sent in nlp_data.sentence for token in sent.token if
                #                token.tokenBeginIndex < _get_tok_b_e_from_token_entity_comp(ent_link_cand)[0] or
                #                token.tokenBeginIndex >= _get_tok_b_e_from_token_entity_comp(ent_link_cand)[1]]
                # rest_text = ' '.join([token.originalText for token in rest_tokens])
                # print('mention', mention.entityMentionText, 'rest', ' '.join([token.originalText for token in rest_tokens]))
                # loop thourgh all results from wikidata API

                # get all datavalues from all relations of all wd_items from one ent_link_query
                wd_item_rel_datavalues_matched_tuples = []
                for ix, wd_item in enumerate(wd_items):
                    # print(f'obtained wikidata item {ix} {wd_item["id"]} {get_fallback_zh_label_from_dict(wd_item)}')
                    # all_rels = [set(rel_labels) | set(rel_aliases) for rel_labels, rel_aliases in
                    #             wd_item['claims'].keys()]
                    # all_rels = [str(ix) + ':' + ' '.join(rel) for ix, rel in enumerate(all_rels)]
                    # print(all_rels)
                    # print()
                    debug_info.update({
                            'wd_item': wd_item['id'],
                            'wd_item_name': get_fallback_zh_label_from_dict(wd_item)
                            })

                    # match predicates
                    # old version: rel_datavalues_matched_tuples = self.get_values_from_matched_predicate(wd_item, rest_text)
                    # new version:
                    def _get_value_from_attr_and_wd_item(wd_item, attr):
                        if attr == '名字':
                            aliases = get_all_aliases_from_dict(wd_item)
                            yield (('名字'), ), aliases[0] + aliases[1]
                        for rel, datavalues in wd_item['claims'].items():
                            labels, _ = rel
                            for label in labels:
                                if label == attr:
                                    # print('matched - rel, datavalues', rel, datavalues)
                                    yield attr, datavalues

                    rel_datavalues_matched_tuples = list(_get_value_from_attr_and_wd_item(wd_item, attr))

                    # print('rel, datavalues:', rel_datavalues_matched_tuples)
                    if not rel_datavalues_matched_tuples:
                        continue
                    wd_item_rel_datavalues_matched_tuples.append((wd_item, rel_datavalues_matched_tuples))

                # ------------------------------------------- wd_item_rel_datavalues_matched_tuples


                answers_from_wd_items = []  # answers_from_one_ent_link_query
                for wd_item, rel_datavalues_matched_tuples in wd_item_rel_datavalues_matched_tuples:

                    answers_from_rels = []  # answers_from_one_wd_item
                    for rel, datavalues in rel_datavalues_matched_tuples:

                        # print('-------------------------------------------------')
                        # print(f'labels of attributes matched the question text\n' +
                        #       'ent_link_query: {} | attr: {} | values: {} | answers: {}'.format(
                        #           ent_link_query, attr, datavalues,
                        #           answers))
                        # TODO
                        debug_info.update({
                        })

                        attribute_match_count += 1

                        answers_from_datavalues = []  # answers_from_one_rel

                        # predicted_answers
                        for dvalue_comp in preprocess_values(datavalues):
                            answers_from_one_datavalue = generate_answers_from_datavalue(dvalue_comp, q, dtext)
                            # output answers
                            answers_from_datavalues.extend(answers_from_one_datavalue)

                        # evaluation
                        if save_result:
                            for dvalue_comp in answers_from_datavalues:
                                ans_cand = dvalue_comp
                                if if_evaluate:
                                    prediction_result = evaluate(dtext, ans_cand, answers)
                                else:
                                    prediction_result = UNKNOWN_MESSAGE
                                # save results
                                save_results(prediction_result, ans_cand, None)

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

            # def transform_answers(answers):
            #     # rule 5-2: output tokens in passage, recover date loss
            #     def _check(t):
            #         return ('年' in qtext, '月' in qtext, '日' in qtext or '号' in qtext)
            #
            #     has_year, has_month, has_day = _check(qtext)
            #     for answer in answers:
            #         has_year_ans, has_month_ans, has_day_ans = _check(answer)
            #         year, month, day = answer.match('(\D*)(\d*)年(\d)月()(日|号)').groups()
            #         result = ''
            #         if has_year:
            #             year
            #             if has_month:
            #                 if has_day:
            #
            #                     if has_year_ans and has_month_ans and has_day_ans:
            #                         return answer
            #                 if has_year_ans and has_month_ans and has_day_ans:
            #                     return answer
            #     for answer in answers:
            #
            #     pass

            final_answers = list(filter_answers(answers_from_ent_link_queries))

            # transfer to FGC output api format

            # for ans in final_answers:
            #     q_anses.append({
            #         'QID': q['QID'],
            #         # 'QTEXT': qtext,  # for debugging
            #         'AMODULE': 'WikiQA',
            #         'ATEXT': ans,
            #         'score': None,  # TODO
            #         'start_score': 0,
            #         'end_score': 0,
            #         # 'gold': answers  # for debugging
            #     })

            if final_answers:
                q_anses = [{
                    # 'QID': q['QID'],
                    # 'QTEXT': qtext,  # for debugging
                    'AMODULE': 'Wiki-Kb-Inference',
                    'ATEXT': max(final_answers, key=len),
                    'score': 1.0,
                    'start_score': 0,
                    'end_score': 0,
                    # 'gold': answers  # for debugging
                }]
            if q_anses:
                all_answers.append(q_anses)

            # break  # for a pilot run
        # endregion questions for-loop

        df = pd.DataFrame(prediction_results)
        if save_result:
            df.to_csv('result.csv')

        return all_answers

    # nlp.close()

    def get_values_from_matched_predicate(self, wd_item, rest_text):
        global predicate_matched
        for rel, datavalues in wd_item['claims'].items():
            # print(rel, datavalue)
            labels, aliases = rel
            predicate_matched = self.predicate_matching_all_aliases_in_rest_text(aliases, labels, rest_text)

            debug_info.update({
                'rel_name': labels,
                'predicate_matched': predicate_matched
            })

            if predicate_matched:
                yield rel, datavalues



    #
    # def predicate_matching_all_aliases_in_rest_text(self, aliases, labels, rest_text):
    #     global matched_attr_names
    #     # predicate inference:
    #     # rule 2: match the relation name with the rest texts than the entity mention
    #     # def predicate_inference():
    #     matched_attr_names = []
    #     predicate_matched = ''
    #     for label in labels:
    #         # print(label)
    #
    #         # rule 2-1: fine-grained rules for predicate inference
    #         if predicate_matching(label, rest_text)[0]:  # if matched by rule
    #             # print(label)
    #             matched_attr_names.append(label)
    #
    #             predicate_matched += predicate_matching(label, rest_text)[1]
    #     for alias in aliases:
    #         # print(label)
    #         # rule 2-2: filter out the attributes aliases with length = 1 e.g. 子, 妻, 名
    #         if alias in rest_text and len(
    #                 alias) > 1:  # alias not in ['子', '妻', '名', '市']:  # and len(alias) > 1:
    #             # print(label)
    #             matched_attr_names.append(alias)
    #             predicate_matched += 'byalias'
    #
    #     return predicate_matched

# match datavalue(answer) with passage text
def get_ent_from_tok_b_e(mentions, span):
    for mention in mentions:
        # def tok_ix_2_char_ix

        all_tokens = [token for sent in passage_data.sentence for token in sent.token]
        len_all_tokens = list(filter(bool, [len(token.originalText) for token in all_tokens]))

        assert len(len_all_tokens) == len(all_tokens)
        s = 0
        tok_char_indexes = [s]
        for l in len_all_tokens:
            s += l
            tok_char_indexes.append(s)

        # solve the problem that tokenization does not include newlines
        for token in all_tokens:
            tok_ix = token.tokenBeginIndex
            char_ix = tok_char_indexes[tok_ix]
            text_by_char_ix = passage_data.text[char_ix: char_ix + len(token.originalText)]
            newlines = re.findall('[\n\s\t]', text_by_char_ix)
            if newlines:
                for ix in range(tok_ix, len(all_tokens)):
                    tok_char_indexes[ix] += len(newlines)

            # for debugging
            # tok_ix = token.tokenBeginIndex
            # char_ix = tok_char_indexes[tok_ix]
            # print(tok_ix, char_ix, token.originalText, passage_data.text[char_ix: char_ix + len(token.originalText)])
        mention_slice = (tok_char_indexes[mention.tokenStartInSentenceInclusive], tok_char_indexes[mention.tokenEndInSentenceExclusive])
        # print(mention_slice, span)
        # print(passage_data.text[slice(*mention_slice)], passage_data.text[slice(*span)])
        if set(range(*mention_slice)).intersection(range(*span)):
            yield mention
    # print(f'NER not found in passage {span[0]} to {span[1]}')


def followed_by_2nd_hop_cands(ent_link_cands, nlp_data):
    """
    # e.g. ['中国', '苏轼', '唐代', ...] => [False, Token(爸爸), False, ...]
    # in sentnece '在 中国 古代 ， 苏轼 的 爸爸 又 称呼 为 什么 ?'
    :return: a list of token or entity or False
    :rtype: List[Union[false, Token, NERMention]]
    """

    def followed_by_2nd_hop_cand(ent_link_cand, nlp_data):
        """
        :type ent_link_cand: Union[Token, NERMention]
        :type nlp_data: dict
        :return:
        """
        tok_b, tok_e = _get_tok_b_e_from_token_entity_comp(ent_link_cand)
        # tokens = get_tokens_from_tok_b_e(tok_b, tok_e, nlp_data)
        try:
            after = get_ent_or_token_from_ix(tok_e, nlp_data)
            try:
                if after.originalText == '的':
                    try:
                        nearest_noun_after = get_nearest_noun_after(tok_e + 1, nlp_data)
                        return nearest_noun_after
                    except KeyError:  # end of sentence
                        return False
            except AttributeError:  # not a token because it has no originalText attribute
                if isinstance(after, NERMention):  # a mention
                    return after
                else:  # not a mention
                    return False
        except KeyError:  # end of sentence
            return False

    followed_by_ent_link_cands = []

    for ent_link_cand in ent_link_cands:
        followed_by_ent_link_cands.append(followed_by_2nd_hop_cand(ent_link_cand, nlp_data))

    assert len(followed_by_ent_link_cands) == len(ent_link_cands)
    return followed_by_ent_link_cands


def get_tokens_from_tok_b_e(tok_b, tok_e, nlp_data):
    """get tokens from token begin index and token end index in stanford nlp serialized format
    :type tok_b: int
    :type tok_e: int
    :type nlp_data: dict
    :return: tokens obtained from tok_b and tok_e or \
    raise KeyError if token does not exist with tok_b and tok_e
    :rtype: List[Token]
    """
    # token = [token for sent in nlp_data.sentence for token in sent.token if
    #          (token.tokenBeginIndex, token.tokenEndIndex) == (tok_b, tok_e)]
    tokens = [token for sent in nlp_data.sentence for token in sent.token]
    tokens = tokens[tok_b:tok_e]
    return tokens


def get_ent_or_token_from_ix(ix, nlp_data):
    """get mention or token from index in stanford nlp serialized format
    :type ix: int
    :type nlp_data: dict
    :return: mention or token
    :rtype: Union[NERMention, Token]
    """
    for mention in nlp_data.mentions:
        if mention.tokenStartInSentenceInclusive == ix:
            return mention
    tokens = [token for sent in nlp_data.sentence for token in sent.token]
    return tokens[ix]


def get_nearest_noun_after(ix, nlp_data):
    """get nearest noun after ix in stanford nlp serialized format
    :param ix: int
    :param nlp_data: dict
    :return: nearest noun after
    :rtype: Token
    """
    tokens = [token for sent in nlp_data.sentence for token in sent.token]
    for token in tokens[ix:]:
        if token.pos in ['NN', 'NR']:
            return token


def match_type(mention, ans_type):
    # TODO
    anstype: ['Person', 'Date-Duration', 'Location', 'Organization', 'Num-Measure', 'YesNo', 'Kinship', 'Event',
              'Object', 'Misc']
    ansmode: ['Single-Span-Extraction', 'Multi-Spans-Extraction', 'YesNo', 'Arithmetic-Operations', 'Counting',
              'Comparing-Members', 'CommonSense', 'Date-Duration', 'Kinship']
    ent_ans_map = {
        'PERSON': ['Person'],
        'MISC': ['Object'],
        'TITLE': ['Organization'],
        'COUNTRY': ['Organization', 'Location'],
        'CITY': ['Organization', 'Location'],
        'GPE': ['Organization', 'Location'],
        'STATE_OR_PROVINCE': ['Organization', 'Location'],
        'ORGANIZATION': ['Organization'],
        'CAUSE_OF_DEATH': ['Event'],
        'RELIGION': ['Object'],
        'NATIONALITY': ['Organization'],
        'IDEOLOGY': ['Object'],
        'DATE': ['Date-Duration'],
        'TIME': ['Date-Duration'],
        'MONEY': ['Num-Measure'],
        'ORDINAL': ['Num-Measure'],
        'NUMBER': ['Num-Measure'],
        'FACILITY': ['Organization']
    }

    def _other_rules(qtext, ans_type, ent_ans_map):
        return '朝代' in qtext and ans_type in ent_ans_map['COUNTRY'] + ent_ans_map['TIME']

    def _bracketed_mention_match_ans_type():
        # mentions_bracketed
        # TODO
        pass
    if mention is not None:
        # if mention.entityType not in mention:
        #     logging.error('')
        if ans_type in ent_ans_map[mention.entityType] or _bracketed_mention_match_ans_type() or _other_rules(qtext, ans_type, ent_ans_map):
            # logging.warning(
            #     f'matching the type: ans_type - {ans_type}, mention - {mention.entityMentionText}({mention.entityType})')
            return True
        else:
            # logging.warning(f'Not matching the type: ans_type - {ans_type}, mention - {mention.entityMentionText}({mention.entityType})')
            pass
    else:
        # logging.warning(f'Not matching the type: ans_type - {ans_type}, mention - None')
        return False


def get_mention_with_matching_ans_type(span: tuple):
    # TODO
    mentions = list(get_ent_from_tok_b_e(passage_data.mentions, span))
    for mention in mentions:
        if match_type(mention, q['ATYPE']):
            return mention


def get_mention_with_matching_passage_and_answer(ans_cand: str, dtext: str, nlp_data):
    # rule 3-1: if fuzzy matching datavalue (answer) with passage text (edit distance <= 1) -> generate answer
    ans_cand = cc.convert(ans_cand)
    max_errors = 1
    if len(ans_cand) > 3:
        pattern = '(' + ans_cand + '){e<=' + str(max_errors) + '}'  # allowed max_errors
    else:
        pattern = ans_cand
    matches = list(regex.finditer(pattern, dtext))
    if matches:
        # print('All exact matched answers:', [match.group(0) for match in matches if sum(match.fuzzy_counts) == 0])
        # print('All fuzzy matched answers:', [match.group(0) for match in matches if sum(match.fuzzy_counts) > 0])
        # longest_match = max(matches, key=lambda match: len(
        #     match.group(0)))  # 0 returns the whole matched instead of the capturing group which starts at 1
        # ans_cand = longest_match.group(0)

        # rule 3-2: if predicted answer characters have longer than length 1
        for match in matches:
            span = match.span(0)
            if not span:
                continue
            fuzzy_matched = dtext[span[0]:span[1]]
            mention = get_mention_with_matching_ans_type(span)
            if len(ans_cand) > 1 and mention is not None:  # TODO: matching also non-mention tokens
                #if the leftmost word is inserted
                # if fuzzy_matched[0] != ans_cand[0] and fuzzy_matched[0] == ans_cand[1]:
                #     fuzzy_matched = fuzzy_matched[1:]
                return mention.entityMentionText
            elif mention is None and '朝代' in qtext:
                return ans_cand
    return False


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


def parse_question(qtext):
    name = '(?P<name>[^的]{2,10})'
    is_ = '(叫|是)'
    height = '((总)高度|身高)'
    have = '(有|拥有)'
    strict_label_map = {
        '名字': [f'^{name}{is_}{what}名字', f'^{name}{is_}名字{what}', f'^{name}(的)?名字{is_}{what}'],
        # rule 2-1-a: 地方
        '出生地': [f'^{name}的老家', f'^{name}(的)?{birth}.*{place_all}',
                f'^{name}{birth_all}.*{where_all}', f'{where_all}{is_}{name}的{birth_all}',
                f'^{name}{is_}{where_all}(的)?人', f'^{name}({is_})?{where_all}来'],
        '死亡地': [f'^{name}的死亡地', f'^{name}{death_all}{_in}{place_all}',
                f'^{name}{_in}{where_all}{death_all}',
                f'^{name}{death_all}{_in}{where_all}', f'{where_all}{is_}{name}的{death_all}'],
        '墓地': [],
        # rule 2-1-b: 时间
        '出生日期': [f'^{name}的生日', f'^{name}的出生日期', f'^{name}的出生日',
                 f'^{name}{when}.*{birth_all}', f'^{name}的{birth_all}.*{when}'],
        '死亡日期': [f'^{name}的死亡地',
                 f'^{name}{when}.*{death_all}', f'^{name}的{death_all}.*{when}'],
        '成立或建立時間': [f'^{name}的{start}.*{time}',
                    f'^{name}{when}.*{start}', f'^{name}{start}.*{when}'],
        # rule 2-1-c: 人物资讯
        '國籍': [f'^{name}的国籍', f'^{name}{is_}{what_which}(朝代|国籍)',
               f'^{name}(的)?{birth_all}.*{what_which}({country}|朝代|国籍)', f'^{name}({is_})?{what_which}({country}|朝代|国籍){birth_all}',
               f'^{name}{is_}{what_which}({country}|朝代|国籍)(的)?人', f'^{name}({is_})?{what_which}({country}|朝代)来'],
        '职业': ['职(位|业|务)', '工作', '职称', '岗位'],
        '創辦者': [f'{found}(人|者)', f'{who}.*{found}', f'{start}.*{found}'],
        '高度': [f'^{name}(的)?{height}'],
        '配偶': [f'^{name}(的)?(妻子|老婆|配偶){is_}{what}名字'],
        '父親': [f'^{name}(的)?(爸爸|父亲|老爸){is_}{what}名字'],
        '母親': [f'^{name}(的)?(妈妈|母亲|老母){is_}{what}名字'],
        '子女数目': [f'^{name}总共{have}几个(小孩|孩子)']

    }

    alias_map = {
        '配偶': ['妻子', '老婆', '配偶'],
        '父親': ['爸爸', '父亲', '老爸'],
        '母親': ['妈妈','母亲','老妈'],
        '兄弟姊妹': ['兄弟姐妹','兄弟姊妹']
    }

    for attr_name, alias_name in alias_map.items():
        attr_name, alias_name


    for attr, patterns in strict_label_map.items():
        for pattern in patterns:
            pattern = re.compile(pattern)
            result = pattern.search(qtext)
            if result and 'name' in result.groupdict():
                return result.groupdict()['name'], attr, result.span(1), pattern.pattern

    return False


# def predicate_matching(label, rest_text):
#
#     label_map = {
#         # rule 2-1-a: 地方
#         '出生地': ['老家', f'{birth}.*{place_all}',
#                 f'{birth_all}.*{where_all}', f'{where_all}.*{birth_all}',
#                 f'{where_all}(的)?人', f'({where_all})来'],
#         '死亡地': ['死亡地', f'{death}.*{place_all}',
#                 f'{death_all}.*{where_all}', f'{where_all}.*{death_all}'],
#         # rule 2-1-b: 时间
#         '出生日期': ['生日', '出生日期', '出生日',
#                  f'{when}.*{birth_all}', f'{birth_all}.*{when}'],
#         '死亡日期': ['死亡地',
#                  f'{when}.*{death_all}', f'{death_all}.*{when}'],
#         '成立或建立時間': [f'{start}.*{time}',
#                     f'{when}.*{start}', f'{start}.*{when}'],
#         # rule 2-1-c: 人物资讯
#         '國籍': ['国籍', '朝代',
#                f'{birth_all}.*{what_which}{country}', f'{what_which}{country}.*{birth_all}',
#                f'{what_which}{country}(的)?人', f'{what_which}{country}来'],
#         '职业': ['职(位|业|务)', '工作', '职称', '岗位'],
#         '創辦者': [f'{found}(人|者)', f'{who}.*{found}', f'{start}.*{found}']
#     }
#
#     # for pattern in label_map.get(label, []):  # trivial rules: matching this first
#     #     pattern = re.compile(pattern)
#     #     result = pattern.search(rest_text)
#     #     if result:
#     #         if '日期' in label:
#     #             print('日期 matched', rest_text)
#     #         return True, pattern.pattern
#
#     return label in rest_text, label  # matching the label with the rest of the text if no rules are matched above


def preprocess_values(datavalues):
    # preprocess datavalues (format conversion)
    # convert to simiplified chinese
    # if len(datavalues) > 1:
    #     print('[note] more than one value for this attribute:', rel)

    processed_datavalues = []
    for ix, datavalue in enumerate(datavalues):
        if isinstance(datavalue, str):
            datatype = 'str'
        else:
            datatype = datavalue['type']

        # if ix > 0:
        #     assert datatype == lastdatatype, f'[note] different data types in this attribute {rel}'

        if datatype == 'wikibase-item':
            object_aliases = datavalue['all_aliases']  # a tuple
            object_aliases = tuple([list(set([cc.convert(name) for name in names])) for names in object_aliases])
            processed_datavalues.append(object_aliases)
        elif datatype == 'time':
            sutime_format = r'(?P<year>\d+)-(?P<month>\d+)-(?P<day>\d+)' \
                            r'T(?P<hour>\d+):(?P<minute>\d+):(?P<second>\d+)\.(?P<millisecond>\d+)Z'
            m = re.match(sutime_format, datavalue['value'])
            year, month, day = int(m.groupdict()['year']), int(m.groupdict()['month']), int(m.groupdict()['day'])
            time_formats = [
                '{}年{}月{}日'.format(year, month, day),
                '{}年{}月{}日'.format((int(year)-1911), month, day),
                '{}年{}月{}號'.format(year, month, day),
                '{}年{}月{}號'.format((int(year)-1911), month, day),
                '{}年{}月'.format(year, month),
                '{}年{}月'.format((int(year) - 1911), month),
                '{}年'.format(year),
                '{}年'.format((int(year) - 1911))
                ]

            processed_datavalues = time_formats
        elif datatype == 'quantity':
            amount = datavalue['value']['amount']
            unit_id = datavalue['value']['unit']
            unit_labels, unit_aliases = get_all_aliases_from_id(unit_id)
            processed_datavalues = [str(amount) + unit_name for unit_name in unit_labels + unit_aliases]
        elif datatype == 'str':
            processed_datavalues = [datavalue]
        else:
            object_value = datavalue['value']
            processed_datavalues = [object_value]

        # print('object value:', processed_datavalues, datatype)
        # lastdatatype = datatype
    return processed_datavalues


def expand_with_multi_gram(word: Union[NERMention, Token], nlp_sent, sid=0, n_grams: int = 3) -> List[Union[NERMention, Token]]:
    """ rule 1-4: expand the query text set from the original query with multi-gram techniques
    expand from NER mention with the multi-gram technique to generate more candidates of entity linking

    :param sid:
    :type n_grams: int
    :param n_grams: the number of multigrams you want to expand with
    :type word: NERMention
    :type nlp_sent: dict
    :rtype:  List[Union[NERMention, Token]]
    """
    try:
        token_b = word.tokenStartInSentenceInclusive
    except AttributeError:
        token_b = word.tokenBeginIndex
    try:
        token_e = word.tokenEndInSentenceExclusive
    except AttributeError:
        token_e = word.tokenEndIndex
    expanded = []
    tokens = nlp_sent[sid].token
    start = token_b - n_grams if token_b - n_grams >= 0 else 0
    end = token_e + n_grams - 1 if token_e + n_grams - 1 <= len(tokens) else len(tokens)

    def _add(ix, offset):
        if 0 <= ix + offset < len(tokens):
            expanded.append(tokens[ix + offset])
        else:
            expanded.append(None)

    for offset in range(-(n_grams - 1), 0):
        _add(token_b, offset)

    expanded.append(word)  # TODO: adapt this line of code to the bug of the NERMention tokenStartInSentenceInclusive
    # not adding at token_b of  but should consider the whole original text

    for offset in range(1, n_grams):
        _add(token_e - 1, offset)

    all_expanded = []
    for start in range(n_grams):
        to_be_add = expanded[start:start + n_grams]
        if len(list(filter(lambda x: x is None, to_be_add))) == 0:
            all_expanded.append(to_be_add)

    return all_expanded


def _get_text_from_token_entity_comp(ent_link_cand: Union[List[Union[NERMention, Token]], str]) -> str:
    """
    get text of `NERMention` and `Token`
    :type ent_link_cand: Union[List[Union[NERMention, Token]], str]
    :rtype: str
    """
    if isinstance(ent_link_cand, tuple):
        return ent_link_cand[0]
    if isinstance(ent_link_cand, NERMention):
        return ent_link_cand.entityMentionText
    if isinstance(ent_link_cand, str):
        return ent_link_cand
    texts = []
    for one in ent_link_cand:
        if isinstance(one, NERMention):
            texts.append(one.entityMentionText)
        elif isinstance(one, Token):
            texts.append(one.originalText)
        else:
            raise ValueError("Must be either entity mention or token in the expanded entity linking set")
    return ''.join(texts)


def _get_tok_b_e_from_token_entity_comp(ent_link_cand: Union[List[Union[NERMention, Token]], str]) -> Tuple[int, int]:
    """An adapter to get begin token and end token of either `NERMention` or `Token`
    :type ent_link_cand: Union[List[Union[NERMention, Token]], str]
    :rtype: Tuple[int, int]
    :return: begin token and end token of either `NERMention` or `Token`
    """
    if isinstance(ent_link_cand, tuple):
        return ent_link_cand[1]
    try:
        return ent_link_cand.tokenStartInSentenceInclusive, ent_link_cand.tokenEndInSentenceExclusive
    except AttributeError:
        pass
    try:
        tok_b = ent_link_cand[0].tokenBeginIndex
    except AttributeError:
        tok_b = ent_link_cand[0].tokenStartInSentenceInclusive
    try:
        tok_e = ent_link_cand[-1].tokenEndIndex
    except AttributeError:
        tok_e = ent_link_cand[-1].tokenEndInSentenceExclusive
    return tok_b, tok_e


def generate_answers_from_datavalue(dvalue_comp, nlp_data, dtext):
    answers_from_one_datavalue = []  # all possible answers from this datavalue
    prediction_result = None
    if isinstance(dvalue_comp, tuple):  # if the datavalue is a wikidata item
        labels, aliases = dvalue_comp
        labels = zip(['bylabel'] * len(labels), labels)
        aliases = zip(['byalias'] * len(aliases), aliases)
        shown = set()
        for (answer_match_way, ans_cand) in list(labels) + list(aliases):  # all possible labels in the wikidata item
            if ans_cand in shown:  # skip duplicate values
                continue
            shown.add(ans_cand)
            mention = get_mention_with_matching_passage_and_answer(ans_cand, dtext, nlp_data)
            if mention:
                answers_from_one_datavalue.append(mention)
    else:  # if the datavalue is not a wikidata item
        if isinstance(dvalue_comp, list):
            # TODO
            pass
        elif isinstance(dvalue_comp, str):
            mention = get_mention_with_matching_passage_and_answer(dvalue_comp, dtext, nlp_data)
            if mention:
                answers_from_one_datavalue.append(mention)
        else:
            pass
    return answers_from_one_datavalue


# def save_results(q, prediction_result, wd_item, matched_attr_names, ans_cand, predicate_matched, answer_match_way):
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
         'answer_match_way': answer_match_way,
         'ent_link_query': ent_link_query,

         })


if __name__ == '__main__':
    import json
    with open('FGC_release_all(cn).json', encoding='utf-8') as f:
        data = json.load(f)

    wiki_qa = WikiQA(server='http://140.109.19.191:9000')

    # all_answers = []
    # for item in data:
    #     all_answers.append(wiki_qa.predict(item, save_result=False, fgc_kb=True))
    # print(all_answers)

    # use data[0] to just answer the first two passages for the pilot run
    answers = wiki_qa.predict(data[0], save_result=True, fgc_kb=True)
    print(answers)
