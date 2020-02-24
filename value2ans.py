#   Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#   Unauthorized copying of this file, via any medium is strictly prohibited
#   Proprietary and confidential
#   Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

from fuzzy_match import build_fuzzy_match_pattern, fuzzy_match
from config import cc
from stanfordnlp_utils import snp_get_ents_by_char_span_in_doc


def remove_duplicates(answers, qtext):
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
    if answers:
        return max(answers, key=len)
    else:
        return


def gen_anses_from_postprocessed_value(dvalue_comp, qtext, atype, dtext, passage_ie_data):
    # TODO: add answers without NER
    answers_from_one_datavalue = []  # all possible answers from this datavalue
    if isinstance(dvalue_comp, tuple):  # if the datavalue is a wikidata item
        labels, aliases = dvalue_comp
        ans_cands = set(labels + aliases)
        for ans_cand in ans_cands:  # all possible labels in the wikidata item
            mention = get_mention_with_matching_passage_and_answer(ans_cand, qtext, atype, dtext, passage_ie_data)
            if mention:
                answers_from_one_datavalue.append(mention)
    else:  # if the datavalue is not a wikidata item
        if isinstance(dvalue_comp, list):
            # TODO
            pass
        elif isinstance(dvalue_comp, str):
            mention = get_mention_with_matching_passage_and_answer(dvalue_comp, qtext, atype, dtext, passage_ie_data)
            if mention:
                answers_from_one_datavalue.append(mention)
        else:
            pass
    return answers_from_one_datavalue


def get_text_from_char_b_e(span, stanfordnlp_data):
    return stanfordnlp_data.text[slice(*span)]


def get_mention_with_matching_passage_and_answer(ans_cand: str, qtext, atype, dtext: str, passage_ie_data, fuzzy=False):
    # rule 3-1: if fuzzy matching datavalue (answer) with passage text (edit distance <= 1) -> generate answer
    ans_cand = cc.convert(ans_cand)
    if fuzzy:
        pattern = build_fuzzy_match_pattern(ans_cand)
    else:
        pattern = ans_cand
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
            mention = get_mention_with_matching_ans_type(span, qtext, atype, passage_ie_data)
            if len(ans_cand) > 1 and mention is not None:  # TODO: matching also non-mention tokens
                #if the leftmost word is inserted
                # if fuzzy_matched[0] != ans_cand[0] and fuzzy_matched[0] == ans_cand[1]:
                #     fuzzy_matched = fuzzy_matched[1:]
                return mention.entityMentionText
            elif mention is None and '朝代' in qtext:
                return ans_cand

    return False


def get_mention_with_matching_ans_type(span: tuple, qtext, atype, passage_data):

    mentions = list(snp_get_ents_by_char_span_in_doc(span, passage_data))
    for mention in mentions:
        if match_type(mention, atype, qtext):
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
        # if snp_mention.entityType not in snp_mention:
        #     logging.error('')
        if ans_type in ent_ans_map[mention.entityType] or _bracketed_mention_match_ans_type() or _other_rules(qtext, ans_type, ent_ans_map):
            # logging.warning(
            #     f'matching the type: ans_type - {ans_type}, snp_mention - {snp_mention.entityMentionText}({snp_mention.entityType})')
            return True
        else:
            # logging.warning(f'Not matching the type: ans_type - {ans_type}, snp_mention - {snp_mention.entityMentionText}({snp_mention.entityType})')
            pass
    else:
        # logging.warning(f'Not matching the type: ans_type - {ans_type}, snp_mention - None')
        return False