#   Copyright (c) 2020. The NLU Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#   Unauthorized copying of this file, via any medium is strictly prohibited
#   Proprietary and confidential
#   Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>

from fuzzy_match import build_fuzzy_match_pattern, fuzzy_match


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


def get_text_from_char_b_e(span, stanfordnlp_data):
    return stanfordnlp_data.text[slice(*span)]


def match_with_psg(ans_cand, dtext, fuzzy):
    if fuzzy:
        pattern = build_fuzzy_match_pattern(ans_cand)
    else:
        pattern = ans_cand
    matches = fuzzy_match(dtext, pattern)
    return matches


def match_type(mention, ans_type, qtext):
    anstype: ["Person", 'Date-Duration', 'Location', 'Organization', 'Num-Measure', 'YesNo', 'Kinship', 'Event',
              'Object', 'Misc']
    ansmode: ['Single-Span-Extraction', 'Multi-Spans-Extraction', 'YesNo', 'Arithmetic-Operations', 'Counting',
              'Comparing-Members', 'CommonSense', 'Date-Duration', 'Kinship']
    ent_ans_map = {
        'PERSON': ['Person'],
        'PER': ['Person'],
        'TITLE': ['Organization', 'Person'],

        'DEGREE': ['Object'],
        'RELIGION': ['Object'],
        'IDEOLOGY': ['Object'],  #?
        'EMAIL': ['Object'],
        'URL': ['Object'],
        'SET': ['Object'],  #?

        'CRIMINAL_CHARGE': ['Event'],  #?
        'CAUSE_OF_DEATH': ['Event'],

        'MISC': ['Object', 'Event'],

        'DEMONYM': ['Organization', 'Location'],
        'CITY': ['Organization', 'Location'],
        'STATE_OR_PROVINCE': ['Organization', 'Location'],
        'COUNTRY': ['Organization', 'Location'],
        'GPE': ['Organization', 'Location'],
        'LOC': ['Organization', 'Location'],

        'ORGANIZATION': ['Organization'],
        'ORG': ['Organization'],
        'NATIONALITY': ['Organization'],
        'FACILITY': ['Organization'],

        'DATE': ['Date-Duration'],
        'TIME': ['Date-Duration'],
        'DURATION': ['Date-Duration'],

        'MONEY': ['Num-Measure'],
        'ORDINAL': ['Num-Measure'],
        'NUMBER': ['Num-Measure'],
        'PERCENT': ['Num-Measure']
    }

    def _other_rules(qtext, ans_type, ent_ans_map):
        return '朝代' in qtext and ans_type in ent_ans_map['COUNTRY'] + ent_ans_map['TIME']

    def _bracketed_mention_match_ans_type():
        # mentions_bracketed
        # TODO
        pass

    if ans_type in ent_ans_map[mention.entityType] or _bracketed_mention_match_ans_type() or _other_rules(qtext, ans_type, ent_ans_map):
        # logging.warning(
        #     f'matching the type: ans_type - {ans_type}, snp_mention - {snp_mention.entityMentionText}({snp_mention.entityType})')
        return True
    else:
        # logging.warning(f'Not matching the type: ans_type - {ans_type}, snp_mention - {snp_mention.entityMentionText}({snp_mention.entityType})')
        pass
