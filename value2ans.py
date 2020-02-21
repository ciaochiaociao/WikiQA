#  Copyright (c) 2020. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
import re
from typing import Union, List, Dict

from fuzzy_match import build_fuzzy_match_pattern, fuzzy_match
from wikidata4fgc_v2 import get_all_aliases_from_id
from config import cc
from stanfordnlp_utils import get_ent_from_stanford_by_char_span


def datavalues2answers(wd_item_rel_datavalues_matched_tuples, qtext, q_dict, dtext, passage_ie_data):
    answers_from_wd_items = []  # answers_from_one_ent_link_query
    for wd_item, rel_datavalues_matched_tuples in wd_item_rel_datavalues_matched_tuples:

        answers_from_rels = []  # answers_from_one_wd_item
        for rel, datavalues in rel_datavalues_matched_tuples:

            answers_from_datavalues = []  # answers_from_one_rel

            # predicted_answers
            for dvalue_comp in postprocess_values(datavalues):
                answers_from_one_datavalue = gen_anses_from_postprocessed_value(dvalue_comp, qtext, q_dict, dtext,
                                                                                passage_ie_data)
                print('(WikiQA)', answers_from_one_datavalue)
                # output answers
                answers_from_datavalues.extend(answers_from_one_datavalue)

            # print('answers_from_datavalues', answers_from_datavalues)
            answers_from_rels.extend(answers_from_datavalues)
        # print('answers_from_rels', answers_from_rels)
        answers_from_wd_items.extend(answers_from_rels)
    # print('answers_from_wd_items', answers_from_wd_items)
    final_answers = answers_from_wd_items

    def remove_duplicates(answers):
        # rule 5: filter out duplicate answers -> deprecated
        # return list(set(answers))
        # answer
        # return [for answer in answers]
        results = []
        for answer in answers:
            if answer not in qtext:
                results.append(answer)
        return results

    def longest_answer(answers):
        # rule 5-2: only get the answer that has the longest length
        return max(answers, key=len)

    final_answers = list(remove_duplicates(final_answers))
    if final_answers:
        final_answer = longest_answer(final_answers)
        return final_answer
    else:
        return


def postprocess_values(datavalues: List[Union[str, Dict]]):
    # postprocess datavalues (format conversion)
    # convert to simiplified chinese
    # if len(datavalues) > 1:
    #     print('[note] more than one value for this attribute:', rel)

    postprocessed_datavalues = []
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
            postprocessed_datavalues.append(object_aliases)
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

            postprocessed_datavalues = time_formats
        elif datatype == 'quantity':
            amount = datavalue['value']['amount']
            unit_id = datavalue['value']['unit']
            unit_labels, unit_aliases = get_all_aliases_from_id(unit_id)
            postprocessed_datavalues = [str(amount) + unit_name for unit_name in unit_labels + unit_aliases]
        elif datatype == 'str':
            postprocessed_datavalues = [datavalue]
        else:
            object_value = datavalue['value']
            postprocessed_datavalues = [object_value]

        # print('object value:', postprocessed_datavalues, datatype)
        # lastdatatype = datatype
    return postprocessed_datavalues


def gen_anses_from_postprocessed_value(dvalue_comp, qtext, q_dict, dtext, passage_ie_data):
    # TODO: add answers without NER
    answers_from_one_datavalue = []  # all possible answers from this datavalue
    if isinstance(dvalue_comp, tuple):  # if the datavalue is a wikidata item
        labels, aliases = dvalue_comp
        labels = zip(['bylabel'] * len(labels), labels)
        aliases = zip(['byalias'] * len(aliases), aliases)
        shown = set()
        for (answer_match_way, ans_cand) in list(labels) + list(aliases):  # all possible labels in the wikidata item
            if ans_cand in shown:  # skip duplicate values
                continue
            shown.add(ans_cand)
            mention = get_mention_with_matching_passage_and_answer(ans_cand, qtext, q_dict, dtext, passage_ie_data)
            if mention:
                answers_from_one_datavalue.append(mention)
    else:  # if the datavalue is not a wikidata item
        if isinstance(dvalue_comp, list):
            # TODO
            pass
        elif isinstance(dvalue_comp, str):
            mention = get_mention_with_matching_passage_and_answer(dvalue_comp, qtext, q_dict, dtext, passage_ie_data)
            if mention:
                answers_from_one_datavalue.append(mention)
        else:
            pass
    return answers_from_one_datavalue


def get_text_from_char_b_e(span, stanfordnlp_data):
    return stanfordnlp_data.text[slice(*span)]


def get_mention_with_matching_passage_and_answer(ans_cand: str, qtext, q_dict, dtext: str, passage_ie_data):
    # rule 3-1: if fuzzy matching datavalue (answer) with passage text (edit distance <= 1) -> generate answer
    ans_cand = cc.convert(ans_cand)
    pattern = build_fuzzy_match_pattern(ans_cand)
    matches = fuzzy_match(dtext, pattern)
    if matches:

        # rule 3-2: if predicted answer characters have longer than length 1
        for match in matches:
            span = match.span(0)
            if not span:
                continue
            # fuzzy_matched = dtext[span[0]:span[1]]

            # TODO: add the condition for using NE as answer, if the object is a Wikibase-item and matching types
            # rule: get only mention as answers, check if the entity type mathces what the question is asking
            mention = get_mention_with_matching_ans_type(span, passage_ie_data, q_dict, qtext)
            if len(ans_cand) > 1 and mention is not None:  # TODO: matching also non-mention tokens
                #if the leftmost word is inserted
                # if fuzzy_matched[0] != ans_cand[0] and fuzzy_matched[0] == ans_cand[1]:
                #     fuzzy_matched = fuzzy_matched[1:]
                return mention.entityMentionText
            elif mention is None and '朝代' in qtext:
                return ans_cand

    return False


def get_mention_with_matching_ans_type(span: tuple, passage_data, q, qtext):

    mentions = list(get_ent_from_stanford_by_char_span(span, passage_data.mentions, passage_data))
    for mention in mentions:
        if match_type(mention, q['ATYPE'], qtext):
            return mention


def match_type(mention, ans_type, qtext):
    # TODO
    anstype: ["Person", 'Date-Duration', 'Location', 'Organization', 'Num-Measure', 'YesNo', 'Kinship', 'Event',
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