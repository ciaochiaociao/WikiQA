#  Copyright (c) 2020. Lorem ipsum dolor sit amet, consectetur adipiscing elit.
#  Morbi non lorem porttitor neque feugiat blandit. Ut vitae ipsum eget quam lacinia accumsan.
#  Etiam sed turpis ac ipsum condimentum fringilla. Maecenas magna.
#  Proin dapibus sapien vel ante. Aliquam erat volutpat. Pellentesque sagittis ligula eget metus.
#  Vestibulum commodo. Ut rhoncus gravida arcu.
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


def build_candidates_to_EL(name, question_ie_data, span, use_ner=True, split_dot=True):

    ent_link_cands = []

    # 1. parsed name from regex (default)
    ent_link_cands.append(name)

    # 2. add more candidates by post-processing (remove 'dot' in subject)
    # '.' or '·' in name, e.g., 馬可.波羅
    if split_dot:
        if '.' in name:
            ent_link_cands.append(''.join(name.split('.')))
        if '·' in name:
            ent_link_cands.append(''.join(name.split('·')))

    # 3. NER mentions containing/occupying the span
    if use_ner:
        mentions = list(snp_get_ents_by_char_span_in_doc(span, question_ie_data))
        if mentions:
            ent_link_cands.extend(mentions)

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