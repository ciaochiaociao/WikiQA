#  Copyright (c) 2020. The Natural Language Understanding Lab, Institute of Information Science, Academia Sinica - All Rights Reserved
#  Unauthorized copying of this file, via any medium is strictly prohibited
#  Proprietary and confidential
#  Written by Chiao-Wei Hsu <cwhsu@iis.sinica.edu.tw>
from typing import Union, List

from stanfordnlp.protobuf import NERMention, Token

from stanfordnlp_utils import get_ent_from_stanford_by_char_span
from wikidata4fgc_v2 import get_dicts_from_keyword, filter_claims_in_dict, readable


def entity_linking(ent_link_cands):
    answers_from_ent_link_queries = []  # answers_from_one_question
    ent_link_queries = build_queries_to_EL(ent_link_cands)

    all_wd_items = []
    for ent_link_query in ent_link_queries:
        # -------------------------------------------
        # rule 1: Entity Linking - get item from Wikidata

        wd_items = get_dicts_from_keyword(ent_link_query)
        # clean wikidata item and simplify wikidata item
        wd_items = [readable(filter_claims_in_dict(d)) for d in wd_items]
        all_wd_items.append(wd_items)

    return all_wd_items


def build_candidates_to_EL(name, passage_ie_data, question_ie_data, span):
    ent_link_cands = []
    # 1. NER mentions containing/occupying the span ??
    mentions = list(get_ent_from_stanford_by_char_span(span, question_ie_data.mentions, passage_ie_data))
    if mentions:
        ent_link_cands.extend(mentions)
    # 2. parsed name from regex
    ent_link_cands = [name]
    # 3. add more candidates by post-processing
    # '.' or '·' in name, e.g., 馬可.波羅
    if '.' in name:
        ent_link_cands.append(''.join(name.split('.')))
    if '·' in name:
        ent_link_cands.append(''.join(name.split('·')))
    return ent_link_cands


def build_queries_to_EL(ent_link_cands):
    shown_queries = set()
    ent_link_queries = []
    for ix, ent_link_cand in enumerate(ent_link_cands):
        ent_link_query = _get_text_from_token_entity_comp(ent_link_cand)
        if ent_link_query in shown_queries:  # skip duplicate
            continue
        shown_queries.add(ent_link_query)
        # print(f'ent_link query {ix}: {ent_link_query}')
        # print()
        # debug_info.update({'ent_link_query': ent_link_query})

        # ============================================================ ent_link_cand
        ent_link_queries.append(ent_link_query)
    return ent_link_queries


def _get_text_from_token_entity_comp(ent_link_cand: Union[List[Union[NERMention, Token]], str, NERMention]) -> str:
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