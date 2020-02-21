#  Copyright (c) 2020. The Natural Language Understanding Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
import re

import regex
from stanfordnlp.protobuf import Document

from wikidata4fgc_v2 import get_all_aliases_from_id
from config import cc
from qa_utils import get_ent_from_tok_b_e


def generate_answers_from_datavalue(dvalue_comp, nlp_data, dtext, qtext, passage_data, q):
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
            mention = get_mention_with_matching_passage_and_answer(ans_cand, dtext, nlp_data, qtext, passage_data, q)
            if mention:
                answers_from_one_datavalue.append(mention)
    else:  # if the datavalue is not a wikidata item
        if isinstance(dvalue_comp, list):
            # TODO
            pass
        elif isinstance(dvalue_comp, str):
            mention = get_mention_with_matching_passage_and_answer(dvalue_comp, dtext, nlp_data, qtext, passage_data, q)
            if mention:
                answers_from_one_datavalue.append(mention)
        else:
            pass
    return answers_from_one_datavalue


def get_mention_with_matching_passage_and_answer(ans_cand: str, dtext: str, nlp_data: Document, qtext, passage_data, q):
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

            # TODO: add the condition for using NE as answer, if the object is a Wikibase-item and matching types
            # rule: get only mention as answers, check if the entity type mathces what the question is asking
            mention = get_mention_with_matching_ans_type(span, passage_data, q, qtext)
            if len(ans_cand) > 1 and mention is not None:  # TODO: matching also non-mention tokens
                #if the leftmost word is inserted
                # if fuzzy_matched[0] != ans_cand[0] and fuzzy_matched[0] == ans_cand[1]:
                #     fuzzy_matched = fuzzy_matched[1:]
                return mention.entityMentionText
            elif mention is None and '朝代' in qtext:
                return ans_cand

            text = get_text_from_char_b_e(span, passage_data)
    return False


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


def get_text_from_char_b_e(span, passage_data):
    return passage_data.text[slice(*span)]


def get_mention_with_matching_ans_type(span: tuple, passage_data, q, qtext):

    mentions = list(get_ent_from_tok_b_e(span, passage_data.mentions, passage_data))
    for mention in mentions:
        if match_type(mention, q['ATYPE'], qtext):
            return mention


def match_type(mention, ans_type, qtext):
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